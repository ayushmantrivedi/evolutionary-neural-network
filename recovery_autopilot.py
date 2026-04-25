"""
+============================================================================+
|                     EVOTRADER AI - RECOVERY AUTOPILOT                      |
|                     Self-Healing & Robustness Watchdog                     |
+============================================================================+
|                                                                            |
|  Monitors the paper trading portfolio for drawdowns. If a failure is       |
|  detected, it triggers a targeted neuroevolutionary retuning phase        |
|  specifically weighted to fix the failure modes observed.                  |
|                                                                            |
+============================================================================+
"""

import os
import sys
import json
import time
import datetime
import subprocess
import pandas as pd
import numpy as np

# Path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_TRADES_FILE = os.path.join(ROOT_DIR, "paper_trades.json")
RETUNE_SCRIPT = os.path.join(ROOT_DIR, "retune_brain.py")

# Configuration
DRAWDOWN_THRESHOLD = 0.05  # 5% drawdown triggers recovery
RECOVERY_GENS = 100        # More generations for recovery
PAIN_THRESHOLD = -0.005    # Days with < -0.5% return are marked as 'pain'

def load_portfolio():
    if not os.path.exists(PAPER_TRADES_FILE):
        print(f"[ERROR] Paper trades file not found: {PAPER_TRADES_FILE}")
        return None
    with open(PAPER_TRADES_FILE, "r") as f:
        return json.load(f)

def detect_failure(data):
    """Checks if the AI is in a significant drawdown or losing streak."""
    if not data or "equity_curve" not in data or not data["equity_curve"]:
        return False, 0, []

    equity = [e["equity"] for e in data["equity_curve"]]
    if not equity:
        return False, 0, []

    peak = max(equity)
    current = equity[-1]
    drawdown = (peak - current) / peak

    print(f"  [MONITOR] Peak Equity:    Rs {peak:,.2f}")
    print(f"  [MONITOR] Current Equity: Rs {current:,.2f}")
    print(f"  [MONITOR] Drawdown:       {drawdown:.2%}")

    # Identify pain points (specific losing days)
    pain_dates = []
    signals = data.get("daily_signals", [])
    
    # Calculate daily returns from signals
    for i in range(1, len(signals)):
        prev = signals[i-1]
        curr = signals[i]
        
        pos_map = curr["action"] - 1 # -1, 0, 1
        daily_ret = pos_map * ((curr["price"] - prev["price"]) / prev["price"])
        
        if daily_ret < PAIN_THRESHOLD:
            pain_dates.append(curr["date"])

    is_failing = drawdown >= DRAWDOWN_THRESHOLD
    if is_failing:
        print(f"  [ALERT] Drawdown threshold exceeded ({DRAWDOWN_THRESHOLD:.1%})!")
        print(f"  [ALERT] Found {len(pain_dates)} critical failure points.")
    
    return is_failing, drawdown, pain_dates

def trigger_recovery(pain_dates):
    """Runs the retuning engine with targeted focus."""
    print("\n" + "!"*70)
    print("  🚀 TRIGGERING AI RECOVERY PROTOCOL")
    print("  Regime-Aware Targeted Retuning commencing...")
    print("!"*70 + "\n")

    # We use subprocess to run retune_brain.py
    # Since we've updated retune_brain.py to automatically load feedback from paper_trades.json,
    # we just need to run it with enough generations.
    
    cmd = [
        sys.executable,
        RETUNE_SCRIPT,
        "--gens", str(RECOVERY_GENS),
        "--months", "6",  # Focus on recent 6 months
        "--mutation", "0.08" # Higher mutation to break out of local minima
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(f"    {line.strip()}")
        process.wait()
        
        if process.returncode == 0:
            print("\n  [SUCCESS] AI Recovery complete. New brain promoted.")
            print("  [INFO] Extracting weights for cloud deployment...")
            subprocess.run([sys.executable, os.path.join(ROOT_DIR, "extract_brain.py")])
            return True
        else:
            print(f"\n  [FAILURE] Recovery script exited with code {process.returncode}")
            return False
    except Exception as e:
        print(f"  [ERROR] Could not run recovery: {e}")
        return False

def main():
    print("="*70)
    print("  EVOTRADER AI - RECOVERY WATCHDOG")
    print("="*70)
    
    data = load_portfolio()
    if not data:
        return

    is_failing, dd, pain = detect_failure(data)
    
    if is_failing:
        success = trigger_recovery(pain)
        if success:
            print("\n  [INFO] System status: RECOVERED. Monitor next signal.")
        else:
            print("\n  [WARN] System status: STALLED. Manual intervention may be needed.")
    else:
        print("\n  [OK] System health within normal parameters. No recovery needed.")

if __name__ == "__main__":
    main()
