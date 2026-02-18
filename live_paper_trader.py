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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pickle
import json
import argparse
import time
import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
    """Get the AI's trading signal by constructing the state directly.

    This is the FAST path -- builds the observation vector from the
    last WINDOW_SIZE rows of features, matching what FinancialRegimeEnv
    does internally, but without stepping through the entire episode.

    Args:
        brain: Loaded brain object
        df: Processed DataFrame with AlphaFactory features
        target_idx: Integer index into df (if None, uses last row)
        prev_position: Previous position (0=Short, 1=Neutral, 2=Long)

    Returns: (action, confidence, features_dict)
    """
    # Determine the target row
    if target_idx is not None:
        if target_idx < WINDOW_SIZE:
            return None, 0, {}
        end_idx = target_idx + 1
    else:
        end_idx = len(df)

    if end_idx < WINDOW_SIZE + 1:
        return None, 0, {}

    try:
        # Extract the 9 feature columns that the environment uses
        # These match FinancialRegimeEnv.signal_features
        feature_cols = [c for c in df.columns if c not in
                        ['Open', 'High', 'Low', 'Close', 'Volume', 'Date',
                         'Adj Close', 'Adj_Close']]

        # Get the window of data
        window_start = end_idx - WINDOW_SIZE
        window_data = df.iloc[window_start:end_idx]

        # Build the 9-feature observation (same as env._process_data)
        obs_features = window_data[feature_cols].values[:, :9]  # Take first 9 features
        if obs_features.shape != (WINDOW_SIZE, 9):
            # Pad or truncate to exactly 9 features
            padded = np.zeros((WINDOW_SIZE, 9), dtype=np.float32)
            n_feat = min(9, obs_features.shape[1])
            padded[:, :n_feat] = obs_features[:, :n_feat]
            obs_features = padded

        # Add position channel (10th feature)
        pos_val = float(prev_position - 1)  # 0->-1, 1->0, 2->1
        position_channel = np.full((WINDOW_SIZE, 1), pos_val, dtype=np.float32)
        full_obs = np.hstack([obs_features.astype(np.float32), position_channel])

        # Flatten to 200-dim state vector
        state = full_obs.flatten()

        # Replace NaN/Inf with 0
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        # Get brain's action
        action = brain.get_action(state, 0)

        # Extract key features for logging
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
        import traceback; traceback.print_exc()
        return None, 0, {}


def execute_paper_trade(log, action, price, date):
    """Execute a paper trade if position changed."""
    current = log.data["current_position"]

    if action == current:
        return None  # No change

    # Calculate cost
    trade_value = log.data["current_capital"]
    cost = trade_value * (FEE_PCT + SLIPPAGE_PCT)

    # Calculate P&L from position change
    # When switching positions, we close the old and open the new
    log.data["current_capital"] -= cost

    trade = log.add_trade(date, current, action, price, cost)

    return trade


def calculate_daily_pnl(log, df):
    """Recalculate capital based on daily price movements and current position."""
    signals = sorted(log.data["daily_signals"], key=lambda x: x["date"])
    if len(signals) < 2:
        return

    capital = log.data["initial_capital"]
    position = 1  # Start neutral

    for i in range(1, len(signals)):
        prev = signals[i - 1]
        curr = signals[i]

        # If position changed, apply cost
        if curr["action"] != position:
            cost = capital * (FEE_PCT + SLIPPAGE_PCT)
            capital -= cost
            position = curr["action"]

        # Apply daily P&L based on position
        pos_map = position - 1  # -1, 0, 1
        if prev["price"] > 0:
            daily_ret = (curr["price"] - prev["price"]) / prev["price"]
            capital *= (1 + pos_map * daily_ret)

    log.data["current_capital"] = round(capital, 2)
    log.data["current_position"] = position


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_signal(brain, log):
    """Get today's trading signal."""
    print("\n" + "=" * 70)
    print("  NIFTY50 AI - TODAY'S SIGNAL")
    print("=" * 70)

    df = fetch_recent_data()
    latest_date = df.index[-1].date()
    latest_price = float(df.iloc[-1]["Close"])

    prev_pos = log.data["current_position"]
    action, conf, features = get_ai_signal(brain, df, prev_position=prev_pos)

    if action is None:
        print("  [ERROR] Could not generate signal. Check data.")
        return

    signal = log.add_signal(latest_date, action, latest_price, conf, features)

    # Execute paper trade if position changed
    trade = execute_paper_trade(log, action, latest_price, latest_date)

    # Recalculate P&L
    calculate_daily_pnl(log, df)
    log.update_equity(latest_date, latest_price)
    log.save()

    # Display
    print(f"\n  Date:       {latest_date}")
    print(f"  NIFTY50:    {latest_price:,.2f}")
    print(f"\n  +-----------------------------------------+")
    print(f"  |  AI SIGNAL:  {ACTION_EMOJI[action]:>8s}                    |")
    print(f"  |  Action:     {ACTION_NAMES[action]:<10s}                  |")
    print(f"  +-----------------------------------------+")

    if trade:
        print(f"\n  >> TRADE EXECUTED: {trade['from']} -> {trade['to']}")
        print(f"     Cost: Rs {trade['cost']:,.2f}")
    else:
        print(f"\n  >> No trade (holding {ACTION_NAMES[action]})")

    print(f"\n  --- Portfolio ---")
    print(f"  Capital:    Rs {log.data['current_capital']:>12,.2f}")
    print(f"  P&L:        {log.current_pnl_pct:>+8.2f}%")
    print(f"  Position:   {ACTION_NAMES[log.data['current_position']]}")
    print(f"  Trades:     {log.total_trades}")
    print(f"  Signals:    {log.total_signals}")

    print(f"\n  --- Key Indicators ---")
    for k, v in features.items():
        print(f"  {k:>10s}: {v}")

    print(f"\n  Trade log saved to: {TRADE_LOG_FILE}")
    print("=" * 70)


def cmd_backfill(brain, log, days):
    """Simulate trading over the last N days."""
    print("\n" + "=" * 70)
    print(f"  NIFTY50 AI - BACKFILL SIMULATION ({days} DAYS)")
    print("=" * 70)

    # Reset log for clean backfill
    log.data = {
        "initial_capital": INITIAL_CAPITAL,
        "current_capital": INITIAL_CAPITAL,
        "current_position": 1,
        "trades": [],
        "daily_signals": [],
        "equity_curve": [],
    }

    df = fetch_recent_data(days_back=days + LOOKBACK_DAYS + 30)

    if len(df) < days:
        print(f"  [WARN] Only {len(df)} days available, using all")
        days = len(df) - WINDOW_SIZE - 5

    start_idx = max(WINDOW_SIZE + 5, len(df) - days)
    trade_dates = df.index[start_idx:]

    print(f"  Simulating {len(trade_dates)} trading days...")
    print(f"  Period: {trade_dates[0].date()} to {trade_dates[-1].date()}")
    print(f"  Starting Capital: Rs {INITIAL_CAPITAL:,.2f}")
    print()
    print(f"  {'Date':>12s}  {'NIFTY':>10s}  {'Signal':>8s}  {'Trade':>15s}  {'Capital':>14s}  {'P&L':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*15}  {'-'*14}  {'-'*8}")

    for i, date in enumerate(trade_dates):
        date_str = date.date()
        price = float(df.loc[date, "Close"])

        # Get integer index into the dataframe
        iloc_idx = df.index.get_loc(date)
        prev_pos = log.data["current_position"]

        # Get signal using fast direct-state approach
        action, conf, features = get_ai_signal(brain, df, target_idx=iloc_idx, prev_position=prev_pos)

        if action is None:
            continue

        # Record signal
        log.add_signal(date_str, action, price, conf, features)

        # Execute trade
        trade = execute_paper_trade(log, action, price, date_str)

        # Update equity
        calculate_daily_pnl(log, df)
        log.update_equity(date_str, price)

        trade_str = ""
        if trade:
            trade_str = f"{trade['from']}->{trade['to']}"

        pnl = log.current_pnl_pct
        print(f"  {str(date_str):>12s}  {price:>10,.2f}  {ACTION_NAMES[action]:>8s}  {trade_str:>15s}  "
              f"Rs {log.data['current_capital']:>11,.2f}  {pnl:>+7.2f}%")

    log.save()

    # Summary
    print(f"\n  {'=' * 60}")
    print(f"  BACKFILL SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  Period:         {trade_dates[0].date()} to {trade_dates[-1].date()}")
    print(f"  Trading Days:   {len(trade_dates)}")
    print(f"  Initial Capital: Rs {INITIAL_CAPITAL:>12,.2f}")
    print(f"  Final Capital:   Rs {log.data['current_capital']:>12,.2f}")
    print(f"  Total P&L:       {log.current_pnl_pct:>+8.2f}%")
    print(f"  Total Trades:    {log.total_trades}")
    print(f"  Total Signals:   {log.total_signals}")

    # Buy and hold comparison
    first_price = float(df.loc[trade_dates[0], "Close"])
    last_price = float(df.loc[trade_dates[-1], "Close"])
    bah_return = ((last_price / first_price) - 1) * 100
    print(f"\n  NIFTY50 B&H:     {bah_return:>+8.2f}%")
    print(f"  AI Alpha:        {log.current_pnl_pct - bah_return:>+8.2f}%")
    print(f"\n  Trade log saved: {TRADE_LOG_FILE}")


def cmd_dashboard(log):
    """Display full portfolio dashboard."""
    print("\n" + "=" * 70)
    print("  NIFTY50 AI - PAPER TRADING DASHBOARD")
    print("=" * 70)

    print(f"\n  --- Portfolio ---")
    print(f"  Initial Capital: Rs {log.data['initial_capital']:>12,.2f}")
    print(f"  Current Capital: Rs {log.data['current_capital']:>12,.2f}")
    print(f"  Total P&L:       {log.current_pnl_pct:>+8.2f}%")
    print(f"  Position:        {ACTION_NAMES[log.data['current_position']]}")
    print(f"  Total Trades:    {log.total_trades}")
    print(f"  Total Signals:   {log.total_signals}")

    # Recent signals
    if log.data["daily_signals"]:
        print(f"\n  --- Last 10 Signals ---")
        print(f"  {'Date':>12s}  {'Price':>10s}  {'Signal':>8s}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*8}")
        for s in log.data["daily_signals"][-10:]:
            print(f"  {s['date']:>12s}  {s['price']:>10,.2f}  {s['action_name']:>8s}")

    # Recent trades
    if log.data["trades"]:
        print(f"\n  --- Trade History ---")
        print(f"  {'Date':>12s}  {'From':>8s}  {'To':>8s}  {'Price':>10s}  {'Cost':>10s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
        for t in log.data["trades"][-15:]:
            print(f"  {t['date']:>12s}  {t['from']:>8s}  {t['to']:>8s}  "
                  f"{t['price']:>10,.2f}  Rs {t['cost']:>7,.2f}")

    # Equity curve
    if log.data["equity_curve"]:
        print(f"\n  --- Equity Curve ---")
        ec = log.data["equity_curve"]
        if len(ec) > 10:
            # Show first 3, ..., last 5
            for e in ec[:3]:
                print(f"  {e['date']:>12s}  Rs {e['equity']:>12,.2f}")
            print(f"  {'...':>12s}")
            for e in ec[-5:]:
                print(f"  {e['date']:>12s}  Rs {e['equity']:>12,.2f}")
        else:
            for e in ec:
                print(f"  {e['date']:>12s}  Rs {e['equity']:>12,.2f}")

    print(f"\n  Log file: {TRADE_LOG_FILE}")
    print("=" * 70)


def cmd_schedule(brain, log):
    """Run signal generation daily at 3:45 PM IST (after market close)."""
    print("\n" + "=" * 70)
    print("  NIFTY50 AI - DAILY SCHEDULER")
    print("  Running signal at 3:45 PM IST (after NSE market close)")
    print("  Press Ctrl+C to stop")
    print("=" * 70)

    TARGET_HOUR = 15
    TARGET_MINUTE = 45

    while True:
        now = datetime.datetime.now()
        target = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0)

        if now > target:
            # Already past today's target, schedule for tomorrow
            target += datetime.timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        print(f"\n  Next signal at: {target.strftime('%Y-%m-%d %H:%M')} "
              f"(in {wait_seconds/3600:.1f} hours)")

        # Wait until target time
        while datetime.datetime.now() < target:
            time.sleep(30)

        # Skip weekends
        if target.weekday() >= 5:
            print(f"  [SKIP] Weekend - no market today")
            continue

        print(f"\n  [RUN] Generating signal at {datetime.datetime.now().strftime('%H:%M:%S')}")
        try:
            cmd_signal(brain, log)
        except Exception as e:
            print(f"  [ERROR] {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIFTY50 AI Live Paper Trader")
    parser.add_argument("--backfill", type=int, default=0,
                        help="Simulate last N trading days")
    parser.add_argument("--dashboard", action="store_true",
                        help="Show portfolio dashboard")
    parser.add_argument("--schedule", action="store_true",
                        help="Auto-run daily at 3:45 PM IST")
    parser.add_argument("--reset", action="store_true",
                        help="Reset trade log")
    parser.add_argument("--brain", default=BRAIN_FILE,
                        help="Path to brain pickle file")
    args = parser.parse_args()

    brain_path = args.brain
    if not os.path.isabs(brain_path):
        brain_path = os.path.join(ROOT_DIR, brain_path)

    if not os.path.exists(brain_path):
        print(f"[ERROR] Brain file not found: {brain_path}")
        sys.exit(1)

    with open(brain_path, "rb") as f:
        brain = pickle.load(f)

    log = TradeLog()

    if args.reset:
        if os.path.exists(TRADE_LOG_FILE):
            os.remove(TRADE_LOG_FILE)
            print("[OK] Trade log reset.")
        log = TradeLog()

    if args.dashboard:
        cmd_dashboard(log)
    elif args.backfill > 0:
        cmd_backfill(brain, log, args.backfill)
    elif args.schedule:
        cmd_schedule(brain, log)
    else:
        cmd_signal(brain, log)


if __name__ == "__main__":
    main()
