"""
+============================================================================+
|                     NIFTY50 AI - LIVE PAPER TRADER                         |
|                     Real-Time Signal Generation & Tracking                 |
+============================================================================+
|                                                                            |
|  Modes:                                                                    |
|   1. SIGNAL MODE  - Get today's trading signal                             |
|   2. BACKFILL     - Simulate recent days as if trading live                |
|   3. DASHBOARD    - Show full portfolio and trade history                  |
|   4. SCHEDULER    - Run automatically every day at market close            |
|                                                                            |
+============================================================================+

Usage:
    python live_paper_trader.py                    # Today's signal
    python live_paper_trader.py --backfill 30      # Simulate last 30 days
    python live_paper_trader.py --dashboard        # Show portfolio
    python live_paper_trader.py --schedule         # Auto-run daily at 3:45 PM IST
"""

import io, sys
import os
import pickle
import json
import argparse
import time
import datetime
import warnings
import subprocess
import numpy as np
import pandas as pd

# Path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import yfinance as yf
from evonet.trader.alpha_factory import AlphaFactory
from evonet.trader.environment import FinancialRegimeEnv
from evonet.core import layers

# Bypass attention layer (same as training)
layers.EvoAttentionLayer.forward = lambda self, x, train=True: x

# ─── Configuration ────────────────────────────────────────────────────────────
TICKER = "^NSEI"
BRAIN_FILE = os.path.join(ROOT_DIR, "nifty50_brain_validated.pkl")
TRADE_LOG_FILE = os.path.join(ROOT_DIR, "paper_trades.json")
WINDOW_SIZE = 20
LOOKBACK_DAYS = 60  # extra days needed for indicators
FEE_PCT = 0.0007
SLIPPAGE_PCT = 0.0003
INITIAL_CAPITAL = 1000000  # Rs 10 Lakh

ACTION_NAMES = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
ACTION_EMOJI = {0: "[SHORT]", 1: "[CASH]", 2: "[LONG]"}


# ─── Trade Log Manager ────────────────────────────────────────────────────────
class TradeLog:
    """Persistent trade log stored as JSON."""

    def __init__(self, filepath=TRADE_LOG_FILE):
        self.filepath = filepath
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                return json.load(f)
        return {
            "initial_capital": INITIAL_CAPITAL,
            "current_capital": INITIAL_CAPITAL,
            "current_position": 1,  # Neutral
            "trades": [],
            "daily_signals": [],
            "equity_curve": [{"date": datetime.date.today().isoformat(), "equity": INITIAL_CAPITAL}],
        }

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_signal(self, date, action, price, confidence, features_summary):
        """Record a daily signal."""
        signal = {
            "date": str(date),
            "action": int(action),
            "action_name": ACTION_NAMES[action],
            "price": float(price),
            "confidence": float(confidence),
            "features": features_summary,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        # Avoid duplicate entries for the same date
        existing_dates = {s["date"] for s in self.data["daily_signals"]}
        if str(date) not in existing_dates:
            self.data["daily_signals"].append(signal)
        return signal

    def add_trade(self, date, from_pos, to_pos, price, cost):
        """Record a trade execution."""
        trade = {
            "date": str(date),
            "from": ACTION_NAMES[from_pos],
            "to": ACTION_NAMES[to_pos],
            "price": float(price),
            "cost": float(cost),
            "capital_before": float(self.data["current_capital"]),
        }
        self.data["trades"].append(trade)
        self.data["current_position"] = int(to_pos)
        return trade

    def update_equity(self, date, price):
        """Update daily equity based on current position and price movement."""
        if len(self.data["equity_curve"]) == 0:
            return

        last = self.data["equity_curve"][-1]
        if str(date) == last.get("date"):
            return  # Already updated today

        pos = self.data["current_position"]
        pos_map = pos - 1  # -1, 0, 1

        # Simple: track capital changes
        self.data["equity_curve"].append({
            "date": str(date),
            "equity": float(self.data["current_capital"]),
            "position": ACTION_NAMES[pos],
            "price": float(price),
        })

    @property
    def total_trades(self):
        return len(self.data["trades"])

    @property
    def total_signals(self):
        return len(self.data["daily_signals"])

    @property
    def current_pnl_pct(self):
        return ((self.data["current_capital"] / self.data["initial_capital"]) - 1) * 100


# ─── Core Functions ───────────────────────────────────────────────────────────

def load_brain():
    """Load the trained NIFTY50 brain."""
    if not os.path.exists(BRAIN_FILE):
        print(f"[ERROR] Brain file not found: {BRAIN_FILE}")
        print(f"        Place 'nifty50_brain_validated.pkl' in the project root.")
        sys.exit(1)
    with open(BRAIN_FILE, "rb") as f:
        return pickle.load(f)


def fetch_recent_data(days_back=LOOKBACK_DAYS + WINDOW_SIZE + 10):
    """Fetch recent NIFTY50 data for signal generation."""
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=int(days_back * 1.5))  # extra buffer for holidays

    print(f"  [FETCH] {TICKER} {start} -> {end}...")
    df = yf.download(TICKER, start=str(start), end=str(end), interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    if len(df) < WINDOW_SIZE + 10:
        print(f"[ERROR] Not enough data. Got {len(df)} bars, need at least {WINDOW_SIZE + 10}")
        sys.exit(1)

    # Apply AlphaFactory features
    df = AlphaFactory.apply_all(df)
    df.dropna(inplace=True)

    print(f"  [OK] {len(df)} trading days loaded ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def get_ai_signal(brain, df, target_idx=None, prev_position=1):
    """Get the AI's trading signal."""
    if target_idx is not None:
        if target_idx < WINDOW_SIZE:
            return None, 0, {}
        end_idx = target_idx + 1
    else:
        end_idx = len(df)

    if end_idx < WINDOW_SIZE + 1:
        return None, 0, {}

    try:
        feature_cols = [c for c in df.columns if c not in
                        ['Open', 'High', 'Low', 'Close', 'Volume', 'Date',
                         'Adj Close', 'Adj_Close']]

        window_start = end_idx - WINDOW_SIZE
        window_data = df.iloc[window_start:end_idx]

        obs_features = window_data[feature_cols].values[:, :9]
        if obs_features.shape != (WINDOW_SIZE, 9):
            padded = np.zeros((WINDOW_SIZE, 9), dtype=np.float32)
            n_feat = min(9, obs_features.shape[1])
            padded[:, :n_feat] = obs_features[:, :n_feat]
            obs_features = padded

        pos_val = float(prev_position - 1)
        position_channel = np.full((WINDOW_SIZE, 1), pos_val, dtype=np.float32)
        full_obs = np.hstack([obs_features.astype(np.float32), position_channel])

        state = full_obs.flatten()
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        action = brain.get_action(state, 0)

        latest = df.iloc[end_idx - 1]
        features = {
            "close": round(float(latest["Close"]), 2),
        }
        for col in ["Log_Ret", "ADX", "MACD_Hist", "BB_Pct", "ATR_Pct"]:
            if col in df.columns:
                features[col.lower()] = round(float(latest[col]), 4)

        return action, 0.7, features

    except Exception as e:
        print(f"  [WARN] Signal generation error: {e}")
        return None, 0, {}


def execute_paper_trade(log, action, price, date):
    """Execute a paper trade if position changed."""
    current = log.data["current_position"]

    if action == current:
        return None

    trade_value = log.data["current_capital"]
    cost = trade_value * (FEE_PCT + SLIPPAGE_PCT)

    log.data["current_capital"] -= cost
    trade = log.add_trade(date, current, action, price, cost)

    return trade


def calculate_daily_pnl(log, df):
    """Recalculate capital based on daily price movements."""
    signals = sorted(log.data["daily_signals"], key=lambda x: x["date"])
    if len(signals) < 2:
        return

    capital = log.data["initial_capital"]
    position = 1

    for i in range(1, len(signals)):
        prev = signals[i - 1]
        curr = signals[i]

        if curr["action"] != position:
            cost = capital * (FEE_PCT + SLIPPAGE_PCT)
            capital -= cost
            position = curr["action"]

        pos_map = position - 1
        if prev["price"] > 0:
            daily_ret = (curr["price"] - prev["price"]) / prev["price"]
            capital *= (1 + pos_map * daily_ret)

    log.data["current_capital"] = round(capital, 2)
    log.data["current_position"] = position


def cmd_signal(brain, log):
    """Get today's trading signal."""
    print("\n" + "=" * 70)
    print("  NIFTY50 AI - TODAY'S SIGNAL")
    print("=" * 70)

    df = fetch_recent_data()
    latest_date = df.index[-1].date()
    latest_price = float(df.iloc[-1]["Close"])

    prev_pos = log.data["current_position"]
    
    # --- SAFETY PROTOCOL: Stop Loss for the Brain ---
    current_cap = log.data["current_capital"]
    initial_cap = log.data["initial_capital"]
    drawdown = (initial_cap - current_cap) / initial_cap
    
    if drawdown > 0.10:
        print(f"  [SAFETY] Portfolio drawdown ({drawdown:.1%}) exceeds safety limit.")
        print(f"  [SAFETY] Overriding AI signal to NEUTRAL (Capital Preservation Mode)")
        action = 1
        conf = 1.0
        features = {"note": "Safety Override Active"}
    else:
        action, conf, features = get_ai_signal(brain, df, prev_position=prev_pos)

    if action is None:
        print("  [ERROR] Could not generate signal. Check data.")
        return

    signal = log.add_signal(latest_date, action, latest_price, conf, features)
    trade = execute_paper_trade(log, action, latest_price, latest_date)

    calculate_daily_pnl(log, df)
    log.update_equity(latest_date, latest_price)
    log.save()

    # --- AUTO-RECOVERY: Trigger watchdog ---
    if drawdown > 0.05:
        print(f"\n  [WATCHDOG] Performance issues detected. Running Recovery Autopilot...")
        subprocess.run([sys.executable, os.path.join(ROOT_DIR, "recovery_autopilot.py")])

    # Display
    print(f"\n  Date:       {latest_date}")
    print(f"  NIFTY50:    {latest_price:,.2f}")
    print(f"\n  +-----------------------------------------+")
    print(f"  |  AI SIGNAL:  {ACTION_EMOJI[action]:>8s}                    |")
    print(f"  |  Action:     {ACTION_NAMES[action]:<10s}                  |")
    print(f"  +-----------------------------------------+")

    if trade:
        print(f"\n  >> TRADE EXECUTED: {trade['from']} -> {trade['to']}")
    else:
        print(f"\n  >> No trade (holding {ACTION_NAMES[action]})")

    print(f"\n  --- Portfolio ---")
    print(f"  Capital:    Rs {log.data['current_capital']:>12,.2f}")
    print(f"  P&L:        {log.current_pnl_pct:>+8.2f}%")
    print(f"  Position:   {ACTION_NAMES[log.data['current_position']]}")

    print(f"\n  Trade log saved to: {TRADE_LOG_FILE}")
    print("=" * 70)


def cmd_backfill(brain, log, days):
    """Simulate trading over the last N days."""
    print("\n" + "=" * 70)
    print(f"  NIFTY50 AI - BACKFILL SIMULATION ({days} DAYS)")
    print("=" * 70)

    log.data = {
        "initial_capital": INITIAL_CAPITAL,
        "current_capital": INITIAL_CAPITAL,
        "current_position": 1,
        "trades": [],
        "daily_signals": [],
        "equity_curve": [],
    }

    df = fetch_recent_data(days_back=days + LOOKBACK_DAYS + 30)
    start_idx = max(WINDOW_SIZE + 5, len(df) - days)
    trade_dates = df.index[start_idx:]

    for i, date in enumerate(trade_dates):
        date_str = date.date()
        price = float(df.loc[date, "Close"])
        iloc_idx = df.index.get_loc(date)
        prev_pos = log.data["current_position"]

        action, conf, features = get_ai_signal(brain, df, target_idx=iloc_idx, prev_position=prev_pos)
        if action is None: continue

        log.add_signal(date_str, action, price, conf, features)
        execute_paper_trade(log, action, price, date_str)
        calculate_daily_pnl(log, df)
        log.update_equity(date_str, price)

    log.save()
    print(f"\n  Backfill complete. Final P&L: {log.current_pnl_pct:>+8.2f}%")


def cmd_dashboard(log):
    """Display portfolio dashboard."""
    print("\n" + "=" * 70)
    print("  NIFTY50 AI - PAPER TRADING DASHBOARD")
    print("=" * 70)
    print(f"\n  Current Capital: Rs {log.data['current_capital']:>12,.2f}")
    print(f"  Total P&L:       {log.current_pnl_pct:>+8.2f}%")
    print(f"  Position:        {ACTION_NAMES[log.data['current_position']]}")
    print("=" * 70)


def cmd_schedule(brain, log):
    """Daily scheduler."""
    print("  Scheduler active. Press Ctrl+C to stop.")
    while True:
        now = datetime.datetime.now()
        if now.hour == 15 and now.minute == 45:
            if now.weekday() < 5:
                cmd_signal(brain, log)
            time.sleep(60)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="NIFTY50 AI Live Paper Trader")
    parser.add_argument("--backfill", type=int, default=0)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--brain", default=BRAIN_FILE)
    args = parser.parse_args()

    log = TradeLog()
    if args.reset:
        if os.path.exists(TRADE_LOG_FILE): os.remove(TRADE_LOG_FILE)
        log = TradeLog()

    if not os.path.exists(args.brain):
        print(f"[ERROR] Brain not found: {args.brain}")
        sys.exit(1)

    with open(args.brain, "rb") as f:
        brain = pickle.load(f)

    if args.dashboard: cmd_dashboard(log)
    elif args.backfill > 0: cmd_backfill(brain, log, args.backfill)
    elif args.schedule: cmd_schedule(brain, log)
    else: cmd_signal(brain, log)

if __name__ == "__main__":
    main()
