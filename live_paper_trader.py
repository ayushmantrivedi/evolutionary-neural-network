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
FEE_PCT = 0.0005 # 0.05% Exchange + Brokerage
SLIPPAGE_PCT = 0.0045 # 0.45% Slippage (Options realism)
# TOTAL COST: 0.5% per trade (Realistic India Context)
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
    before_drop = len(df)
    df.dropna(inplace=True)
    if len(df) < before_drop:
        print(f"  [WARN] Dropped {before_drop - len(df)} rows due to NaNs in indicators.")
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
        
        # --- DIRECTIONAL CONFIRMATION LAYER (v2.1) ---
        vix = latest.get("VIX_Level", 0.15) * 100
        dte = latest.get("DTE_Norm", 0.5)
        # --- DIRECTIONAL & STRUCTURAL CONFIRMATION (v2.2 ALPHA-COMPLETE) ---
        vix = latest.get("VIX_Level", 0.15) * 100
        vix_rank = latest.get("VIX_Rank", 0.5)
        vrp = latest.get("VRP", 0)
        atr_slope = latest.get("ATR_Slope", 0)
        
        sma_slope = latest.get("SMA20_Slope", 0)
        dist_sma = latest.get("Dist_SMA20", 0)
        sma5 = latest.get("SMA5", latest["Close"])
        low_5d = latest.get("Low_5d", latest["Close"])
        
        # 1. Structural Trend Detection (Multi-Anchor)
        # BULL: Price > SMA5 AND Price > Low_5d
        is_bull_regime = (latest["Close"] > sma5) and (latest["Close"] > low_5d)
        
        # BEAR: Price < Low_5d (Breakdown) OR (Price < SMA5 AND Slope < 0)
        is_bear_regime = (latest["Close"] < low_5d) or (latest["Close"] < sma5 and sma_slope < 0)
        
        # 2. Volatility Edge Extraction (VRP)
        has_vrp_edge = vrp > 0.02 # IV is overpriced compared to Realized Vol
        
        # 3. Active Bias Determination & Monetization
        final_action = action
        
        # ACTIVE SHORTING / MONETIZATION: 
        # If structural floor is broken (Price < Low_5d) -> FORCE SHORT (0) to monetize downside
        if latest["Close"] < low_5d:
            final_action = 0 
            regime = "Bearish Breakdown"
        # If in Bear Regime but not a full breakdown yet -> Neutralize Longs
        elif is_bear_regime and action == 2:
            final_action = 1 
            regime = "Bearish Transition"
        # BULLISH: Only allow Long if price structure is healthy
        elif is_bull_regime and action == 2:
            final_action = 2
            
        # 4. Strategy Router (Dynamic)
        # Decision: Fast Move (ATR Spike) -> Directional (Gamma) | Slow Move -> Structural (Theta/VRP)
        is_fast_move = atr_slope > 0.05 or abs(latest["Log_Ret"]) > 0.015
        strategy, regime = "CASH", "Neutral"
        
        if final_action == 2: # LONG
            if is_fast_move:
                strategy, regime = "LONG CALL / FUTURES", "Momentum Long"
            else:
                strategy, regime = "BULL PUT SPREAD", "Structural Long"
        
        elif final_action == 0: # SHORT
            if is_fast_move:
                strategy, regime = "LONG PUT / FUTURES", "Momentum Short"
            else:
                strategy, regime = "BEAR CALL SPREAD", "Structural Short"
        
        # 5. Position Sizing Engine
        # size = base_capital * regime_score * (1 / vol)
        # Normalized Vol: (VIX / 20)
        vol_scalar = 1.0 / (vix / 18.0) # Reduce size as VIX rises
        regime_score = 1.0 if (is_bull_regime or is_bear_regime) else 0.5
        pos_size_pct = min(1.0, 0.5 * regime_score * vol_scalar) # Max 50% exposure per trade
        
        # 6. PREDICTIVE KILL-SWITCH (Risk Expansion)
        vix_prev = df.iloc[end_idx - 2].get("VIX_Level", 0.15) * 100
        vix_spike = (vix - vix_prev) / vix_prev if vix_prev > 0 else 0
        
        if vix > 30 or (vix_spike > 0.12 and atr_slope > 0.05):
            final_action = 1
            strategy, regime = "CASH (SAFETY)", "Extreme Stress Expansion"

        features.update({
            "strategy": strategy, 
            "regime": regime, 
            "vix": round(vix, 2),
            "vix_rank": round(vix_rank * 100, 1),
            "vrp_edge": round(vrp * 100, 2),
            "pos_size_pct": round(pos_size_pct * 100, 1),
            "trend": "BULL" if is_bull_regime else ("BEAR" if is_bear_regime else "CHOP")
        })
        return final_action, pos_size_pct, features
    except Exception as e:
        print(f"  [WARN] Signal generation error: {e}"); return None, 0, {}

def execute_paper_trade(log, action, price, date):
    current = log.data["current_position"]
    if action == current: return None
    cost = log.data["current_capital"] * (FEE_PCT + SLIPPAGE_PCT)
    log.data["current_capital"] -= cost
    # We don't store pos_size in trade log yet, it's inferred from signals
    return log.add_trade(date, current, action, price, cost)

def calculate_daily_pnl(log, df):
    signals = sorted(log.data["daily_signals"], key=lambda x: x["date"])
    if len(signals) < 2: return
    cap, pos = log.data["initial_capital"], 1
    for i in range(1, len(signals)):
        prev, curr = signals[i-1], signals[i]
        size = curr.get("confidence", 1.0) # Use confidence as pos_size
        if curr["action"] != pos:
            cap -= cap * (FEE_PCT + SLIPPAGE_PCT)
            pos = curr["action"]
        if prev["price"] > 0: 
            ret = (pos - 1) * (curr["price"] - prev["price"]) / prev["price"]
            cap *= (1 + ret * size)
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

def cmd_backfill(brain, log, days):
    """Simulate trading over the last N days with v2.0 Structural Logic."""
    print("\n" + "=" * 80)
    print(f"  NIFTY50 AI v2.0 - BACKFILL SIMULATION ({days} DAYS)")
    print("=" * 80)

    # Reset for simulation
    log.data.update({"current_capital": INITIAL_CAPITAL, "current_position": 1, "trades": [], "daily_signals": [], "equity_curve": []})
    
    df = fetch_recent_data(days_back=days + WINDOW_SIZE + 40)
    print(f"  [DEBUG] Fetched {len(df)} total days for simulation.")
    start_idx = max(WINDOW_SIZE + 5, len(df) - days)
    trade_dates = df.index[start_idx:]
    print(f"  [DEBUG] Simulating from {trade_dates[0].date()} to {trade_dates[-1].date()} ({len(trade_dates)} days)")

    print(f"\n  {'Date':<12} | {'NIFTY':<10} | {'VIX':<6} | {'Signal':<8} | {'Strategy':<22} | {'P&L':<8}")
    print(f"  {'-'*12} | {'-'*10} | {'-'*6} | {'-'*8} | {'-'*22} | {'-'*8}")

    for date in trade_dates:
        date_str = date.date()
        price = float(df.loc[date, "Close"])
        iloc_idx = df.index.get_loc(date)
        prev_pos = log.data["current_position"]

        action, conf, feat = get_ai_signal(brain, df, target_idx=iloc_idx, prev_position=prev_pos)
        if action is None: continue
        
        # Debug Trend
        trend_label = feat.get("trend", "SIDE")
        vix = feat.get("vix", 0)
        slope = df.iloc[iloc_idx].get("SMA20_Slope", 0)
        dist = df.iloc[iloc_idx].get("Dist_SMA20", 0)
        print(f"  [DEBUG] {date_str} | Trend: {trend_label} | Slope: {slope:.5f} | Dist: {dist:.5f}")

        log.add_signal(date_str, action, price, conf, feat)
        execute_paper_trade(log, action, price, date_str)
        calculate_daily_pnl(log, df)
        
        strat = feat.get("strategy", "CASH")
        vix = feat.get("vix", 0)
        pnl = log.current_pnl_pct
        
        print(f"  {str(date_str):<12} | {price:10,.2f} | {vix:6.2f} | {ACTION_NAMES[action]:<8} | {strat:<22} | {pnl:>+7.2f}%")

    log.save()
    print("\n" + "=" * 80)
    print(f"  Simulation Complete. Final Capital: Rs {log.data['current_capital']:,.2f} ({log.current_pnl_pct:>+8.2f}%)")
    print("=" * 80)

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
    elif args.backfill > 0: cmd_backfill(brain, log, args.backfill)
    elif args.schedule:
        while True:
            if datetime.datetime.now().hour == 15 and datetime.datetime.now().minute == 45: cmd_signal(brain, log); time.sleep(60)
            time.sleep(30)
    else: cmd_signal(brain, log)

if __name__ == "__main__": main()
