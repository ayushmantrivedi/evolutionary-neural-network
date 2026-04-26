import os
import sys
import pickle
import pandas as pd
import numpy as np
import datetime
from evonet.trader.alpha_factory import AlphaFactory
from live_paper_trader import get_ai_signal, WINDOW_SIZE, INITIAL_CAPITAL, FEE_PCT, SLIPPAGE_PCT

def run_wfo(brain_path, lookback_days=100, step_days=10):
    print("\n" + "="*80)
    print(f"  INSTITUTIONAL WALK-FORWARD VALIDATION (WFO)")
    print("="*80)
    
    # 1. Load Data
    import yfinance as yf
    ticker = "^NSEI"
    end = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days + 100)
    df = yf.download(ticker, start=str(start), end=str(end), interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None: df.index = df.index.tz_convert(None)
    
    df = AlphaFactory.apply_all(df)
    df.dropna(inplace=True)
    
    # 2. Load Brain
    with open(brain_path, "rb") as f:
        brain = pickle.load(f)
        
    # 3. Sliding Window Loop
    total_days = len(df)
    start_idx = total_days - lookback_days
    
    capital = INITIAL_CAPITAL
    equity_curve = []
    position = 1 # Neutral
    
    print(f"\n  {'Window Start':<15} | {'Window End':<15} | {'P&L %':<10} | {'Max DD %':<10}")
    print(f"  {'-'*15} | {'-'*15} | {'-'*10} | {'-'*10}")
    
    for i in range(start_idx, total_days, step_days):
        window_end = min(i + step_days, total_days)
        window_df = df.iloc[:window_end]
        
        window_pnl = 0
        window_equity = []
        
        for j in range(i, window_end):
            date = df.index[j]
            price = float(df.iloc[j]["Close"])
            prev_price = float(df.iloc[j-1]["Close"])
            
            # Get Signal with Position Sizing
            action, pos_size, feat = get_ai_signal(brain, window_df, target_idx=j, prev_position=position)
            if action is None: continue
            
            # Cost of switching
            if action != position:
                capital -= capital * (FEE_PCT + SLIPPAGE_PCT)
                position = action
                
            # Daily P&L calculation (sized)
            # return = (pos - 1) * price_change * pos_size
            daily_ret = (position - 1) * (price - prev_price) / prev_price
            capital *= (1 + daily_ret * pos_size)
            
            equity_curve.append(capital)
            window_equity.append(capital)
            
        # Calculate Window Stats
        w_pnl = (window_equity[-1] / window_equity[0] - 1) * 100
        # Simple Max Drawdown for window
        peak = np.maximum.accumulate(window_equity)
        dd = (peak - window_equity) / peak
        max_dd = np.max(dd) * 100
        
        print(f"  {str(df.index[i].date()):<15} | {str(df.index[window_end-1].date()):<15} | {w_pnl:>+8.2f}% | {max_dd:>8.2f}%")
        
    final_pnl = (capital / INITIAL_CAPITAL - 1) * 100
    print("\n" + "="*80)
    print(f"  WFO COMPLETE. Final Alpha: {final_pnl:>+8.2f}%")
    print("="*80)

if __name__ == "__main__":
    brain_file = "nifty50_brain_validated.pkl"
    if os.path.exists(brain_file):
        run_wfo(brain_file)
    else:
        print("Brain not found.")
