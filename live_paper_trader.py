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
LOOKBACK_DAYS = 60
FEE_PCT = 0.0007
SLIPPAGE_PCT = 0.0003
INITIAL_CAPITAL = 1000000

ACTION_NAMES = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
ACTION_EMOJI = {0: "[SHORT]", 1: "[CASH]", 2: "[LONG]"}

class TradeLog:
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
            "current_position": 1,
            "trades": [],
            "daily_signals": [],
            "equity_curve": [{"date": datetime.date.today().isoformat(), "equity": INITIAL_CAPITAL}],
        }

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_signal(self, date, action, price, confidence, features_summary):
        signal = {
            "date": str(date),
            "action": int(action),
            "action_name": ACTION_NAMES[action],
            "price": float(price),
            "confidence": float(confidence),
            "features": features_summary,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        existing_dates = {s["date"] for s in self.data["daily_signals"]}
        if str(date) not in existing_dates:
            self.data["daily_signals"].append(signal)
        return signal

    def add_trade(self, date, from_pos, to_pos, price, cost):
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
        if len(self.data["equity_curve"]) == 0: return
        last = self.data["equity_curve"][-1]
        if str(date) == last.get("date"): return
        self.data["equity_curve"].append({
            "date": str(date),
            "equity": float(self.data["current_capital"]),
            "position": ACTION_NAMES[self.data["current_position"]],
            "price": float(price),
        })

    @property
    def current_pnl_pct(self):
        return ((self.data["current_capital"] / self.data["initial_capital"]) - 1) * 100

def fetch_recent_data(days_back=LOOKBACK_DAYS + WINDOW_SIZE + 10):
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=int(days_back * 1.5))
    df = yf.download(TICKER, start=str(start), end=str(end), interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None: df.index = df.index.tz_convert(None)
    df = AlphaFactory.apply_all(df)
    df.dropna(inplace=True)
    return df

def get_ai_signal(brain, df, target_idx=None, prev_position=1):
    if target_idx is not None:
        if target_idx < WINDOW_SIZE: return None, 0, {}
        end_idx = target_idx + 1
    else: end_idx = len(df)

    if end_idx < WINDOW_SIZE + 1: return None, 0, {}

    try:
        feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Adj Close', 'Adj_Close']]
        window_start = end_idx - WINDOW_SIZE
        window_data = df.iloc[window_start:end_idx]
        obs_features = window_data[feature_cols].values[:, :9]
        if obs_features.shape != (WINDOW_SIZE, 9):
            padded = np.zeros((WINDOW_SIZE, 9), dtype=np.float32)
            padded[:, :min(9, obs_features.shape[1])] = obs_features[:, :min(9, obs_features.shape[1])]
            obs_features = padded
        pos_val = float(prev_position - 1)
        position_channel = np.full((WINDOW_SIZE, 1), pos_val, dtype=np.float32)
        full_obs = np.hstack([obs_features.astype(np.float32), position_channel])
        state = np.nan_to_num(full_obs.flatten())
        action = brain.get_action(state, 0)
        
        latest = df.iloc[end_idx - 1]
        features = {"close": round(float(latest["Close"]), 2)}
        
        # --- STRATEGY LAYER: v2.0 Decision Matrix ---
        vix = latest.get("VIX_Level", 0.15) * 100
        dte = latest.get("DTE_Norm", 0.5)
        strategy, regime = "CASH", "Normal"
        if action == 2:
            strategy, regime = ("LONG CALL / FUTURES", "Low Vol Trend") if vix < 18 else ("BULL PUT SPREAD", "High Vol Trend (VRP)")
        elif action == 0:
            strategy, regime = ("LONG PUT / FUTURES", "Low Vol Trend") if vix < 18 else ("BEAR CALL SPREAD", "High Vol Trend (VRP)")
        elif action == 1:
            strategy, regime = ("IRON FLY / STRADDLE", "High Vol Mean-Reversion") if vix > 20 else ("CASH", "Low Vol Chop")
        
        features.update({"strategy": strategy, "regime": regime, "vix": round(vix, 2), "dte_pct": round(dte * 100, 1)})
        return action, 0.7, features
    except Exception as e:
        print(f"  [WARN] Signal generation error: {e}"); return None, 0, {}

def execute_paper_trade(log, action, price, date):
    current = log.data["current_position"]
    if action == current: return None
    cost = log.data["current_capital"] * (FEE_PCT + SLIPPAGE_PCT)
    log.data["current_capital"] -= cost
    return log.add_trade(date, current, action, price, cost)

def calculate_daily_pnl(log, df):
    signals = sorted(log.data["daily_signals"], key=lambda x: x["date"])
    if len(signals) < 2: return
    cap, pos = log.data["initial_capital"], 1
    for i in range(1, len(signals)):
        prev, curr = signals[i-1], signals[i]
        if curr["action"] != pos:
            cap -= cap * (FEE_PCT + SLIPPAGE_PCT)
            pos = curr["action"]
        if prev["price"] > 0: cap *= (1 + (pos - 1) * (curr["price"] - prev["price"]) / prev["price"])
    log.data["current_capital"] = round(cap, 2)
    log.data["current_position"] = pos

def cmd_signal(brain, log):
    print("\n" + "=" * 70 + "\n  NIFTY50 AI - TODAY'S SIGNAL\n" + "=" * 70)
    df = fetch_recent_data()
    latest_date, latest_price = df.index[-1].date(), float(df.iloc[-1]["Close"])
    drawdown = (log.data["initial_capital"] - log.data["current_capital"]) / log.data["initial_capital"]
    if drawdown > 0.10:
        action, conf, features = 1, 1.0, {"note": "Safety Override", "strategy": "CASH"}
    else:
        action, conf, features = get_ai_signal(brain, df, prev_position=log.data["current_position"])
    if action is None: return
    log.add_signal(latest_date, action, latest_price, conf, features)
    trade = execute_paper_trade(log, action, latest_price, latest_date)
    calculate_daily_pnl(log, df)
    log.update_equity(latest_date, latest_price)
    log.save()
    if drawdown > 0.05: subprocess.run([sys.executable, os.path.join(ROOT_DIR, "recovery_autopilot.py")])
    
    strat, regime, vix = features.get("strategy", "CASH"), features.get("regime", "Unknown"), features.get("vix", 0)
    print(f"\n  Date: {latest_date} | NIFTY: {latest_price:,.2f} | VIX: {vix:.2f}\n  Regime: {regime}\n  +-----------------------------------------+\n  | AI SIGNAL: {ACTION_EMOJI[action]:>8s} |\n  | Strategy: {strat:<20s} |\n  +-----------------------------------------+")
    if trade: print(f"\n  >> TRADE: {trade['from']} -> {trade['to']}")
    print(f"\n  Capital: Rs {log.data['current_capital']:,.2f} | P&L: {log.current_pnl_pct:>+8.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=int, default=0)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    log = TradeLog()
    if args.reset and os.path.exists(TRADE_LOG_FILE): os.remove(TRADE_LOG_FILE); log = TradeLog()
    if not os.path.exists(BRAIN_FILE): print("Brain missing"); sys.exit(1)
    with open(BRAIN_FILE, "rb") as f: brain = pickle.load(f)
    if args.dashboard: print(f"Capital: Rs {log.data['current_capital']:,.2f}")
    elif args.schedule:
        while True:
            if datetime.datetime.now().hour == 15 and datetime.datetime.now().minute == 45: cmd_signal(brain, log); time.sleep(60)
            time.sleep(30)
    else: cmd_signal(brain, log)

if __name__ == "__main__": main()
