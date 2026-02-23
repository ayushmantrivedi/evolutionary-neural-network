"""
cloud_signal.py  —  EvoTrader AI · Self-Contained Cloud Runner
================================================================
Runs on GitHub Actions every weekday at 3:37 PM IST.

 · ZERO local package imports  (no evonet, no pandas_ta)
 · ALL 10 market indicators calculated inline with pure numpy/pandas
 · Only external deps: numpy, pandas, yfinance, requests

Required GitHub Secrets:
    TELEGRAM_TOKEN   — Bot HTTP API token
    TELEGRAM_CHAT_ID — Your personal Telegram chat ID (run get_chat_id.py once)
"""

import os, sys, json, pickle, datetime, warnings, io
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TICKER          = "^NSEI"
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
BRAIN_FILE      = os.path.join(ROOT_DIR, "brain_weights.pkl")   # portable numpy dict
TRADE_LOG_FILE  = os.path.join(ROOT_DIR, "paper_trades.json")
WINDOW_SIZE     = 20
LOOKBACK_DAYS   = 120          # fetch extra buffer for indicator warmup
FEE_PCT         = 0.0007
INITIAL_CAPITAL = 1_000_000    # Rs 10 Lakh

ACTION_NAMES = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
ACTION_EMOJI = {0: "🔴 SHORT", 1: "⚪ NEUTRAL", 2: "🟢 LONG"}
ACTION_TEXT  = {0: "SELL / SHORT", 1: "STAY IN CASH", 2: "BUY / LONG"}

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ── Indicators (pure numpy/pandas — no pandas_ta needed) ─────────────────────
def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate AlphaFactory.apply_all() using pure pandas math."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # 1. Stationarity — log returns
    df["Log_Ret"]    = np.log(c / c.shift(1))
    df["Log_Ret_5d"] = df["Log_Ret"].rolling(5).sum()

    # 2. Trend strength — ADX (Wilder's method)
    tr   = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    dm_p  = (h - h.shift()).clip(lower=0)
    dm_n  = (l.shift() - l).clip(lower=0)
    dm_p[dm_p < (l.shift() - l).clip(lower=0)] = 0
    dm_n[dm_n < (h - h.shift()).clip(lower=0)]  = 0
    di_p  = 100 * dm_p.ewm(alpha=1/14, adjust=False).mean() / atr14
    di_n  = 100 * dm_n.ewm(alpha=1/14, adjust=False).mean() / atr14
    dx    = (100 * (di_p - di_n).abs() / (di_p + di_n).replace(0, np.nan)).fillna(0)
    df["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean() / 100.0

    # 3. Volatility clustering — ATR%
    df["ATR"]     = atr14
    df["ATR_Pct"] = atr14 / c

    # 4. Smart money flow — OBV slope
    obv           = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["OBV_Slope"] = obv.pct_change(5).fillna(0)

    # 5. Momentum — MACD histogram
    ema12          = _ema(c, 12)
    ema26          = _ema(c, 26)
    macd_line      = ema12 - ema26
    signal_line    = _ema(macd_line, 9)
    df["MACD_Hist"] = np.tanh(macd_line - signal_line)

    # 6. Mean reversion — Bollinger %B
    sma20          = _sma(c, 20)
    std20          = c.rolling(20).std()
    upper          = sma20 + 2 * std20
    lower          = sma20 - 2 * std20
    df["BB_Pct"]   = (c - lower) / (upper - lower).replace(0, np.nan)
    df["BB_Pct"]   = df["BB_Pct"].fillna(0.5).clip(0, 1)

    # 7. Tail risk — kurtosis & distance from SMA200
    df["Kurtosis_20"]    = c.rolling(20).kurt()
    sma200               = _sma(c, 200)
    df["Distance_SMA200"] = ((c - sma200) / sma200.replace(0, np.nan)).fillna(0)

    # 8. Risk regime — vol ratio & RSI & spread
    atr30                  = atr14.rolling(30).mean()
    df["Vol_Regime_Ratio"] = atr14 / atr30.replace(0, np.nan)
    delta                  = c.diff()
    gain                   = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss                   = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    rs                     = gain / loss.replace(0, np.nan)
    df["RSI"]              = (100 - 100 / (1 + rs)).fillna(50)
    df["Est_Spread_Pct"]   = (h - l) / c

    df.fillna(0, inplace=True)
    return df

FEATURE_COLS = [
    "Log_Ret", "Log_Ret_5d", "ADX", "ATR_Pct", "OBV_Slope",
    "MACD_Hist", "BB_Pct", "Kurtosis_20", "Distance_SMA200"
]   # exactly 9 features — matches training environment


# ── Brain inference (pure numpy — no evonet needed) ──────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def forward_pass(neurons: list, x: np.ndarray) -> np.ndarray:
    """Run one EvoNet forward pass.
    Architecture: ReLU(L1) -> ReLU(L2) -> skip-concat -> softmax(L3)
    Exactly replicates MultiClassEvoNet.predict() with USE_SKIP_CONNECTIONS=True.
    """
    # Layer 1 — ReLU
    l1 = np.array([np.maximum(0.0, np.dot(x, n['weights']) + n['bias'])
                   for n in neurons[0]], dtype=np.float32)
    # Layer 2 — ReLU
    l2 = np.array([np.maximum(0.0, np.dot(l1, n['weights']) + n['bias'])
                   for n in neurons[1]], dtype=np.float32)
    # Skip connection: concat(l2, l1)
    l3_in = np.concatenate([l2, l1])
    # Layer 3 — linear + softmax
    l3 = np.array([np.dot(l3_in, n['weights']) + n['bias']
                   for n in neurons[2]], dtype=np.float32)
    return softmax(l3)

def get_signal(brain: dict, df: pd.DataFrame, prev_position: int) -> int:
    window = df.iloc[-WINDOW_SIZE:]
    obs    = window[FEATURE_COLS].values.astype(np.float32)          # (20, 9)
    pos_ch = np.full((WINDOW_SIZE, 1), float(prev_position - 1), dtype=np.float32)
    state  = np.nan_to_num(np.hstack([obs, pos_ch]).flatten())       # (200,)
    probs  = forward_pass(brain['neurons'], state)
    return int(np.argmax(probs))


# ── Trade log ─────────────────────────────────────────────────────────────────
def load_log():
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE) as f:
            return json.load(f)
    return {
        "initial_capital": INITIAL_CAPITAL,
        "current_capital": INITIAL_CAPITAL,
        "current_position": 1,
        "trades": [],
        "daily_signals": [],
        "equity_curve": [],
    }

def save_log(data):
    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def update_log(log, today: str, action: int, price: float):
    signals      = log["daily_signals"]
    prev_action  = log.get("current_position", 1)

    # Apply yesterday's P&L
    if signals:
        prev       = signals[-1]
        pos_map    = prev["action"] - 1          # -1 short, 0 neutral, +1 long
        daily_ret  = pos_map * ((price - prev["price"]) / prev["price"])
        if action != prev["action"]:
            daily_ret -= FEE_PCT                 # transaction cost
        log["current_capital"] = max(log["current_capital"] * (1.0 + daily_ret), 0.01)

    # Record trade if position changed
    if action != prev_action:
        log.setdefault("trades", []).append({
            "date": today, "from": ACTION_NAMES[prev_action],
            "to": ACTION_NAMES[action], "price": price,
            "capital": log["current_capital"],
        })
    log["current_position"] = action

    # Record signal (deduplicated by date)
    existing = {s["date"] for s in signals}
    if today not in existing:
        signals.append({
            "date": today, "action": action,
            "action_name": ACTION_NAMES[action], "price": price,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })

    log.setdefault("equity_curve", []).append({
        "date": today, "equity": log["current_capital"],
        "position": ACTION_NAMES[action], "price": price,
    })
    return log


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] No credentials — skipped.")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=15
        )
        print("[TELEGRAM] ✅ Sent." if r.ok else f"[TELEGRAM] ⚠️ {r.text}")
    except Exception as e:
        print(f"[TELEGRAM] ⚠️ Error: {e}")

def bah_pnl(log):
    s = log.get("daily_signals", [])
    if len(s) < 2:
        return 0.0
    return ((s[-1]["price"] / s[0]["price"]) - 1) * 100

def build_message(log, action, price, today):
    cap     = log["current_capital"]
    pnl_pct = ((cap / log["initial_capital"]) - 1) * 100
    pnl_rs  = cap - log["initial_capital"]
    edge    = pnl_pct - bah_pnl(log)
    streak  = sum(1 for s in reversed(log["daily_signals"]) if s["action"] == action)

    return "\n".join([
        "🧠 <b>EVOTRADER AI — DAILY SIGNAL</b>",
        f"📅 <b>Date:</b> {today}",
        "",
        f"  Signal:  <b>{ACTION_EMOJI[action]}</b>",
        f"  Action:  {ACTION_TEXT[action]}",
        f"  NIFTY:   ₹{price:,.2f}",
        "",
        f"{'📈' if pnl_pct >= 0 else '📉'} <b>Portfolio</b>",
        f"  Capital :  ₹{cap:>12,.2f}",
        f"  P&amp;L   :  {pnl_pct:>+.2f}%  (₹{pnl_rs:>+,.0f})",
        f"  vs B&amp;H :  {edge:>+.2f}% edge",
        "",
        f"📊 Days: {len(log['daily_signals'])}  ·  Trades: {len(log['trades'])}  ·  Streak: {streak}d",
        "",
        f"<i>Next signal after market close tomorrow (3:37 PM IST)</i>",
        f"<i>EvoTrader AI v1.1 · github.com/ayushmantrivedi</i>",
    ])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    today = datetime.date.today().isoformat()
    print(f"\n{'='*60}\n  EVOTRADER CLOUD SIGNAL — {today}\n{'='*60}\n")

    # 1. Brain
    print("[1/5] Loading brain...")
    if not os.path.exists(BRAIN_FILE):
        send_telegram(f"❌ Brain file not found! Date: {today}")
        sys.exit(1)
    with open(BRAIN_FILE, "rb") as f:
        brain = pickle.load(f)
    layers = brain['neurons']
    print(f"      Loaded — arch:{brain['input_dim']}->{brain['layer_sizes']}->{brain['output_dim']}")

    # 2. Market data
    print("[2/5] Fetching NIFTY50 data...")
    end   = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=int(LOOKBACK_DAYS * 1.6))
    df = yf.download(TICKER, start=str(start), end=str(end), interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    print(f"      {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")

    # 3. Indicators
    print("[3/5] Computing indicators...")
    df = add_indicators(df)
    df.dropna(inplace=True)
    if len(df) < WINDOW_SIZE + 5:
        send_telegram(f"❌ Not enough data ({len(df)} bars). Date: {today}")
        sys.exit(1)
    print(f"      {len(df)} clean bars · {len(FEATURE_COLS)} features OK")

    # 4. Signal
    print("[4/5] Getting AI signal...")
    log          = load_log()
    prev_action  = log.get("current_position", 1)
    price        = float(df.iloc[-1]["Close"])
    action       = get_signal(brain, df, prev_action)
    print(f"      Signal: {ACTION_NAMES[action]}  @  ₹{price:,.2f}")

    # 5. Update log, save, notify
    print("[5/5] Updating trade log & sending Telegram...")
    log = update_log(log, today, action, price)
    save_log(log)
    pnl = ((log["current_capital"] / log["initial_capital"]) - 1) * 100
    print(f"      Capital: ₹{log['current_capital']:,.2f}  P&L: {pnl:+.2f}%")

    send_telegram(build_message(log, action, price, today))

    print(f"\n{'─'*60}")
    print(f"  SIGNAL:  {ACTION_NAMES[action]}")
    print(f"  PRICE:   ₹{price:,.2f}")
    print(f"  CAPITAL: ₹{log['current_capital']:,.2f}")
    print(f"  P&L:     {pnl:+.2f}%")
    print(f"  DAYS:    {len(log['daily_signals'])}")
    print(f"{'─'*60}\n✅ Done.")


if __name__ == "__main__":
    main()
