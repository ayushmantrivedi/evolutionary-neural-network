import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_evotrader import make_env, TICKER, WINDOW_SIZE
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv


def determine_regime(row):
    """
    Real-time Regime Classifier based on Weighted Feature Voting.
    
    Rank / Weights:
    1. SMA_200 (Long Term Trend) - Critical
    2. SMA_50 (Medium Term Trend) - Major
    3. ADX (Trend Strength) - Filter
    4. Volatility (ATR) - Panic Detector
    """
    price = row['Close']
    sma50 = row['SMA_50']
    sma200 = row['SMA_200']
    adx = row['ADX']
    macd = row['MACD_Hist']
    
    # 1. Panic/Crash Check (Priority Override)
    # If price is far below SMA50 (>10% drop), it's a crash
    if pd.notna(sma50) and price < sma50 * 0.90:
         return "bear"

    # 2. Trend Voting
    score = 0
    
    # SMA 50 (Medium Trend) - Primary Direction (Weight: 3)
    if pd.notna(sma50):
        if price > sma50: score += 3
        else: score -= 3
        
    # SMA 200 (Long Trend) - Confirmation (Weight: 2)
    if pd.notna(sma200):
        if price > sma200: score += 2
        else: score -= 2
        
    # Momentum (Weight: 1)
    if pd.notna(macd):
        if macd > 0: score += 1
        else: score -= 1
        
    # Decision
    # Max Score: 6 (Strong Bull)
    # Min Score: -6 (Strong Bear)
    # Conflicted: -1 to 1 (Chop)
    
    if score >= 2:
        return "bull"
    elif score <= -2:
        return "bear"
    else:
        return "chop"

def run_backtest():
    print("🎬 STARTING DYNAMIC REGIME BACKTEST (2020-2023)")
    print("   Mode: Real-Time Detection (No Hindsight)")
    
    # 1. Load Brain
    try:
        with open("evotrader_brain.pkl", "rb") as f:
            pilot = pickle.load(f)
        print("🧠 Artificial Brain Loaded Successfully.")
    except FileNotFoundError:
        print("❌ Brain not found! Run train_evotrader.py first.")
        return

    # 2. Load Full History (H1 Mode)
    # WARNING: yfinance limits 1h data to last 730 days.
    # We will test on 2024-2025 (The recent cycle).
    start_date = "2024-02-01"
    end_date = "2025-12-30" 
    
    print(f"   Settings: Interval=1h, Range={start_date} to {end_date}")
    
    fetcher = DataFetcher(TICKER, start_date=start_date, end_date=end_date, interval="1h")
    df = fetcher.fetch_data()
    df = fetcher.add_advanced_features()
    
    # 3. Setup Environment (Single Continuous Run)
    # We use the whole dataframe
    env = make_env(df)
    state, info = env.reset()
    
    equity = 1.0
    equity_curve = [equity]
    
    # Benchmark
    initial_price = df['Close'].iloc[WINDOW_SIZE] # Env starts at Window
    benchmark_curve = [1.0]
    
    steps = 0
    regime_history = []
    
    current_regime = "chop" # Default start
    pilot.recover_memory(current_regime) # load default
    
    terminated = False
    truncated = False
    
    print("\n👉 STARTING SIMULATION...")
    
    while not (terminated or truncated):
        # A. Detect Regime (The "Cortex" Step)
        # We look at the data available at this step
        # Env step aligns with dataframe index: env._current_tick
        current_tick = env._current_tick
        
        if current_tick < len(df):
            row = df.iloc[current_tick]
            detected_regime = determine_regime(row)
            
            # Switch if needed
            if detected_regime != current_regime:
                # print(f"   Tick {current_tick}: Switch {current_regime} -> {detected_regime}")
                pilot.recover_memory(detected_regime)
                current_regime = detected_regime
                
            regime_history.append(current_regime)
        
        # B. Get Action from Specialist
        # 0=Best Agent
        action = pilot.get_action(state, 0)
        
        # C. Execute
        state, r, terminated, truncated, _ = env.step(action)
        
        # D. Track Metrics
        equity = equity * np.exp(r)
        equity_curve.append(equity)
        
        # Benchmark (Approximate for speed, or strict price ratio)
        # current_price = df['Close'].iloc[current_tick] / initial_price # slightly off due to step adv
        # Let's just append placeholder, we'll fix axes later
        
        steps += 1
        if steps % 100 == 0:
            print(f"   Step {steps} | Equity: {equity:.2f} | Regime: {current_regime.upper()}")

    print(f"\n📊 DYNAMIC SIMULATION COMPLETE")
    print(f"   Final Equity: ${10000 * equity:.2f} (+{(equity-1)*100:.1f}%)")
    
    # Plotting
    # Realign dates
    # Env consumes [WINDOW_SIZE : End]
    traded_df = df.iloc[WINDOW_SIZE : WINDOW_SIZE + len(equity_curve)]
    
    # Benchmark
    btc_buy_hold = (traded_df['Close'] / traded_df['Close'].iloc[0]).values
    
    plt.figure(figsize=(12, 8))
    
    # Top Panel: Equity
    plt.subplot(2, 1, 1)
    plt.plot(btc_buy_hold, label='Bitcoin (Hold)', color='gray', alpha=0.5)
    plt.plot(equity_curve, label='EvoTrader (Dynamic)', color='lime', linewidth=1.5)
    plt.title("EvoTrader Dynamic Regime Test (2020-2023)")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom Panel: Regime Strip
    plt.subplot(2, 1, 2)
    # Convert regimes to numeric for plotting
    regime_map_int = {'bull': 1, 'chop': 0, 'bear': -1}
    regime_vals = [regime_map_int[r] for r in regime_history]
    # Pad to match equity curve length if needed
    if len(regime_vals) < len(equity_curve):
        regime_vals = [0]*(len(equity_curve)-len(regime_vals)) + regime_vals
        
    plt.step(range(len(regime_vals)), regime_vals, where='post', color='orange')
    plt.yticks([-1, 0, 1], ['Bear', 'Chop', 'Bull'])
    plt.title("Active Specialist Regime")
    plt.xlabel("Days")
    
    plt.tight_layout()
    plt.savefig("evotrader_dynamic_test.png")
    print("📸 Chart saved to 'evotrader_dynamic_test.png'")


if __name__ == "__main__":
    run_backtest()
