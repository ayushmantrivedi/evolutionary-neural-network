"""
cloud_signal.py
================
Runs on GitHub Actions every weekday at 3:35 PM IST.
- Fetches live NIFTY50 close price
- Asks the AI brain for today's signal
- Updates paper_trades.json
- Sends a Telegram message to the owner's phone

Required Env Vars (GitHub Secrets):
    TELEGRAM_TOKEN   - Bot HTTP API token
    TELEGRAM_CHAT_ID - Your personal chat ID
"""

import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import json
import pickle
import datetime
import warnings
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from evonet.trader.alpha_factory import AlphaFactory
from evonet.core import layers
layers.EvoAttentionLayer.forward = lambda self, x, train=True: x

# ── Config ────────────────────────────────────────────────────────────────────
TICKER          = "^NSEI"
BRAIN_FILE      = os.path.join(ROOT_DIR, "nifty50_brain_validated.pkl")
TRADE_LOG_FILE  = os.path.join(ROOT_DIR, "paper_trades.json")
WINDOW_SIZE     = 20
LOOKBACK_DAYS   = 80
FEE_PCT         = 0.0007
SLIPPAGE_PCT    = 0.0003
INITIAL_CAPITAL = 1_000_000   # Rs 10 Lakh

ACTION_NAMES  = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
ACTION_EMOJI  = {0: "🔴 SHORT", 1: "⚪ NEUTRAL", 2: "🟢 LONG"}
ACTION_TEXT   = {0: "SELL / SHORT", 1: "STAY IN CASH", 2: "BUY / LONG"}

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_telegram(message: str):
    """Send a message via Telegram. Silently skip if creds missing."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] Skipped — no credentials in environment.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.ok:
            print("[TELEGRAM] ✅ Message sent.")
        else:
            print(f"[TELEGRAM] ⚠️  Failed: {resp.text}")
    except Exception as e:
        print(f"[TELEGRAM] ⚠️  Error: {e}")


# ── Trade Log ─────────────────────────────────────────────────────────────────
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


# ── Market Data ───────────────────────────────────────────────────────────────
def fetch_data():
    end   = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=int(LOOKBACK_DAYS * 1.6))
    print(f"[FETCH] {TICKER} from {start} to {end}...")
    df = yf.download(TICKER, start=str(start), end=str(end), interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = AlphaFactory.apply_all(df)
    df.dropna(inplace=True)
    print(f"[FETCH] Got {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── AI Signal ─────────────────────────────────────────────────────────────────
def get_signal(brain, df, prev_position):
    feature_cols = [c for c in df.columns if c not in
                    ("Open","High","Low","Close","Volume","Adj Close","Adj_Close","Date")]
    window = df.iloc[-WINDOW_SIZE:]
    obs    = window[feature_cols].values[:, :9].astype(np.float32)
    if obs.shape != (WINDOW_SIZE, 9):
        padded = np.zeros((WINDOW_SIZE, 9), dtype=np.float32)
        padded[:, :min(9, obs.shape[1])] = obs[:, :min(9, obs.shape[1])]
        obs = padded
    pos_val  = float(prev_position - 1)
    pos_ch   = np.full((WINDOW_SIZE, 1), pos_val, dtype=np.float32)
    state    = np.nan_to_num(np.hstack([obs, pos_ch]).flatten())
    action   = brain.get_action(state, 0)
    latest   = df.iloc[-1]
    features = {"close": round(float(latest["Close"]), 2)}
    for col in ("Log_Ret","ADX","MACD_Hist","BB_Pct","ATR_Pct"):
        if col in df.columns:
            features[col.lower()] = round(float(latest[col]), 4)
    return action, features


# ── P&L Update ────────────────────────────────────────────────────────────────
def update_pnl(log, today_str, action, price, prev_action):
    """Apply yesterday's return to capital, then record position change."""
    signals = log["daily_signals"]
    if len(signals) >= 1:
        prev = signals[-1]
        prev_price = prev["price"]
        pos_map    = prev["action"] - 1   # -1, 0, +1
        daily_ret  = pos_map * ((price - prev_price) / prev_price)
        daily_ret -= FEE_PCT if action != prev["action"] else 0   # trade cost
        log["current_capital"] *= (1.0 + daily_ret)
        log["current_capital"] = max(log["current_capital"], 0.01)

    # Record position change as a "trade"
    if action != prev_action:
        log["trades"].append({
            "date": today_str,
            "from": ACTION_NAMES[prev_action],
            "to":   ACTION_NAMES[action],
            "price": price,
            "capital": log["current_capital"],
        })
        log["current_position"] = action

    # Append daily signal
    existing = {s["date"] for s in log["daily_signals"]}
    if today_str not in existing:
        log["daily_signals"].append({
            "date": today_str,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "price": price,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })

    # Equity curve
    log.setdefault("equity_curve", []).append({
        "date": today_str,
        "equity": log["current_capital"],
        "position": ACTION_NAMES[action],
        "price": price,
    })

    return log


# ── Buy-and-Hold Reference ────────────────────────────────────────────────────
def bah_pnl(log):
    signals = log.get("daily_signals", [])
    if len(signals) < 2:
        return 0.0
    first_price = signals[0]["price"]
    last_price  = signals[-1]["price"]
    return ((last_price / first_price) - 1) * 100


# ── Message Builder ───────────────────────────────────────────────────────────
def build_message(log, action, price, today_str, is_market_day):
    cap       = log["current_capital"]
    init_cap  = log["initial_capital"]
    pnl_pct   = ((cap / init_cap) - 1) * 100
    pnl_rs    = cap - init_cap
    n_signals = len(log["daily_signals"])
    n_trades  = len(log["trades"])
    bah       = bah_pnl(log)
    edge      = pnl_pct - bah
    pos_emoji = ACTION_EMOJI[action]

    # Streak
    streak = 0
    for s in reversed(log["daily_signals"]):
        if s["action"] == action:
            streak += 1
        else:
            break

    status_icon = "📈" if pnl_pct > 0 else "📉"

    if not is_market_day:
        header = "⏰ <b>EVOTRADER AI — MARKET CLOSED (Preview Signal)</b>"
    else:
        header = "🧠 <b>EVOTRADER AI — DAILY SIGNAL</b>"

    lines = [
        header,
        f"📅 <b>Date:</b> {today_str}",
        f"",
        f"┌────────────────────────────┐",
        f"│  Signal:  {pos_emoji:<20}│",
        f"│  Action:  {ACTION_TEXT[action]:<20}│",
        f"│  NIFTY:   ₹{price:>10,.2f}         │",
        f"└────────────────────────────┘",
        f"",
        f"{status_icon} <b>Portfolio Performance</b>",
        f"  Capital:    ₹{cap:>12,.2f}",
        f"  P&amp;L:       {pnl_pct:>+.2f}%  (₹{pnl_rs:>+,.0f})",
        f"  vs B&amp;H:    {edge:>+.2f}% edge over buy-and-hold",
        f"",
        f"📊 <b>Stats</b>",
        f"  Days tracked: {n_signals}",
        f"  Trades made:  {n_trades}",
        f"  {pos_emoji} streak: {streak} days",
        f"",
        f"<i>Next signal: Tomorrow after market close (3:35 PM IST)</i>",
        f"<i>EvoTrader AI v1.1 · github.com/ayushmantrivedi/evolutionary-neural-network</i>",
    ]
    return "\n".join(lines)


# ── Weekend / Holiday Check ────────────────────────────────────────────────────
def is_today_market_day(df):
    """True if today's date appears in the fetched data (NSE was open)."""
    today = datetime.date.today()
    return any(d.date() == today for d in df.index)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    today_str = datetime.date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"  EVOTRADER CLOUD SIGNAL — {today_str}")
    print(f"{'='*60}\n")

    # 1. Load brain
    print("[1/5] Loading brain...")
    if not os.path.exists(BRAIN_FILE):
        send_telegram(f"❌ <b>EvoTrader Error</b>\nBrain file missing on GitHub runner!\nDate: {today_str}")
        sys.exit(1)
    with open(BRAIN_FILE, "rb") as f:
        brain = pickle.load(f)
    print(f"      Brain loaded — {brain.pop_size} genomes")

    # 2. Fetch market data
    print("[2/5] Fetching market data...")
    df = fetch_data()
    market_day = is_today_market_day(df)
    print(f"      Market open today: {market_day}")

    # Use latest available price (could be yesterday if today is holiday)
    latest_row   = df.iloc[-1]
    price        = float(latest_row["Close"])
    price_date   = df.index[-1].date()

    # 3. Load trade log
    print("[3/5] Loading trade log...")
    log          = load_log()
    prev_action  = log.get("current_position", 1)

    # 4. Get AI signal
    print("[4/5] Getting AI signal...")
    action, features = get_signal(brain, df, prev_action)
    print(f"      Signal: {ACTION_NAMES[action]}  Price: {price:,.2f}")

    # 5. Update P&L and save
    print("[5/5] Updating P&L and saving...")
    log = update_pnl(log, today_str, action, price, prev_action)
    save_log(log)
    print(f"      Capital: ₹{log['current_capital']:,.2f}  (P&L: {((log['current_capital']/log['initial_capital'])-1)*100:+.2f}%)")

    # 6. Send Telegram
    msg = build_message(log, action, price, today_str, market_day)
    send_telegram(msg)

    # 7. Print summary to stdout (visible in GitHub Actions log)
    print(f"\n{'─'*60}")
    print(f"  SIGNAL:  {ACTION_NAMES[action]}")
    print(f"  PRICE:   ₹{price:,.2f}  ({price_date})")
    print(f"  CAPITAL: ₹{log['current_capital']:,.2f}")
    print(f"  P&L:     {((log['current_capital']/log['initial_capital'])-1)*100:+.2f}%")
    print(f"  DAYS:    {len(log['daily_signals'])}")
    print(f"{'─'*60}\n")
    print("✅ Done.")


if __name__ == "__main__":
    main()
