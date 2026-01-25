
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evonet.trader.environment import FinancialRegimeEnv
from evonet.trader.data_loader import DataFetcher
from evonet.trader.oms import OrderManagementSystem, Order, OrderType, OrderState
from evonet.trader.risk import RiskGuard

# --- H1 CONFIGURATION ---
WINDOW_SIZE = 48 # Matches train_h1_specialists.py
TICKER = "BTC-USD"

def make_env_pro(df):
    return FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, len(df)), window_size=WINDOW_SIZE, fee=0.001)

def run_pro_simulation():
    print("💎 STARTING PROFESSIONAL 'REAL WORLD' SIMULATION (H1 RETRAINED)")
    print("   Integration: Brain(H1) + OMS (Slippage) + RiskGuard (Kelly Sizing)")
    
    # 1. Load Brain
    brain_file = "evotrader_brain_h1.pkl"
    try:
        with open(brain_file, "rb") as f:
            pilot = pickle.load(f)
        print(f"🧠 Artificial Brain Loaded: {brain_file}")
    except FileNotFoundError:
        print(f"❌ Brain {brain_file} not found!")
        return

    # 2. Load Data (H1)
    # Using the cached range from previous step
    start_date = "2024-02-01"
    end_date = "2025-12-30" 
    fetcher = DataFetcher(TICKER, start_date=start_date, end_date=end_date, interval="1h")
    df = fetcher.fetch_data()
    df = fetcher.add_advanced_features()
    
    # 3. Initialize Systems
    oms = OrderManagementSystem(slippage_model="ATR")
    risk = RiskGuard(max_risk_per_trade=0.10, max_drawdown_limit=0.20)
    
    # 4. Simulation State
    initial_capital = 10000.0
    cash = initial_capital
    position_qty = 0.0 # BTC amount
    
    equity_curve = [initial_capital]
    drawdown_curve = [0.0]
    
    # Env for getting observations (Use H1 Env)
    env = make_env_pro(df)
    state, _ = env.reset()
    
    # For Regime Logic
    current_regime = "chop"
    pilot.recover_memory(current_regime)
    
    steps = 0
    trades = 0
    
    # Loop
    print("\n👉 STARTING EXECUTION LOOP...")
    terminated = False
    
    while not terminated:
        current_tick = env._current_tick
        if current_tick >= len(df) - 1: break
            
        row = df.iloc[current_tick]
        current_price = row['Close']
        current_atr = row['ATR']
        
        # --- A. Update OMS (Fill Orders) ---
        fills = oms.match_orders(row)
        for fill in fills:
            # Update Portfolio
            if fill.side == "BUY":
                cost = fill.fill_price * fill.qty
                cash -= cost
                position_qty += fill.qty
                # print(f"   🟢 BUY FILLED: {fill.qty:.4f} @ {fill.fill_price:.2f}")
            elif fill.side == "SELL":
                revenue = fill.fill_price * fill.qty
                cash += revenue
                position_qty -= fill.qty
                # print(f"   🔴 SELL FILLED: {fill.qty:.4f} @ {fill.fill_price:.2f}")
            
            trades += 1
            # Update Risk Stats
            # (Simplification: We need realized PnL for Risk Stats, complex to track here accurately per trade)
            risk.update_stats(True) # Dummy update for now
            
        # --- B. Calculate Equity ---
        equity = cash + (position_qty * current_price)
        equity_curve.append(equity)
        
        # Drawdown Check
        peak = max(equity_curve)
        dd = (peak - equity) / peak
        drawdown_curve.append(dd)
        
        if dd > risk.max_drawdown_limit:
            print(f"💀 HARD STOP TRIGGERED! Drawdown {dd*100:.1f}% > Limit {risk.max_drawdown_limit*100:.1f}%")
            break
            
        # --- C. Brain Decision ---
        # 1. Regime Check (Matching Training Logic)
        sma50 = row['SMA_50']
        adx = row['ADX']
        
        new_regime = current_regime
        
        # Priority 1: Chop (Low ADX)
        if pd.notna(adx) and adx < 0.20:
            new_regime = "chop"
        # Priority 2: Directional (Bull/Bear)
        elif pd.notna(sma50):
            if current_price > sma50:
                new_regime = "bull"
            elif current_price <= sma50:
                new_regime = "bear"
                
        # Switch Context if changed
        if new_regime != current_regime:
            current_regime = new_regime
            pilot.recover_memory(current_regime)
            # print(f"   Context Switch: {current_regime}")
        
        # 2. Get AI Action
        # Action: 0=Short, 1=Neutral, 2=Long
        action_idx = pilot.get_action(state, 0)
        
        # --- IRON HAND FILTER ---
        # If in Chop Regime (Low ADX), FORCE NEUTRAL.
        # The AI is too hyperactive in noise. We save fees by doing nothing.
        if current_regime == "chop":
            action_idx = 1 # Force Neutral
            
        # --- D. Risk Guard Approval ---
        is_safe, reason = risk.check_market_conditions(current_atr, current_price)
        
        if not is_safe:
            # Force Neutral/Close if dangerous? 
            # For now, just block new entries
            pass
        else:
            # --- E. Execute Strategy ---
            # Logic: If Signal != Current Position, Flip.
            
            # Current Position Direction
            current_side = 0 # Neutral
            if position_qty > 0.0001: current_side = 1 # Long
            elif position_qty < -0.0001: current_side = -1 # Short
            
            target_side = 0
            if action_idx == 2: target_side = 1 # Long
            elif action_idx == 0: target_side = -1 # Short
            
            if target_side != current_side:
                # We need to trade.
                # 1. Close existing
                if current_side != 0:
                    # Close all
                    order = Order(TICKER, "SELL" if current_side == 1 else "BUY", abs(position_qty), OrderType.MARKET)
                    oms.submit_order(order)
                    
                # 2. Open new (if target is not neutral)
                if target_side != 0:
                    # Size?
                    # Use Kelly Sizing on *Available Equity*
                    size_usd = risk.calculate_position_size(equity, 0.6, current_atr/current_price)
                    qty = size_usd / current_price
                    
                    if qty * current_price > 10.0: # Min trade $10
                        side = "BUY" if target_side == 1 else "SELL"
                        order = Order(TICKER, side, qty, OrderType.MARKET)
                        oms.submit_order(order)
        
        # Step Env
        state, r, terminated, truncated, _ = env.step(action_idx)
        steps += 1
        
        if steps % 1000 == 0:
            print(f"   Step {steps} | Equity: ${equity:.2f} | DD: {dd*100:.1f}% | Reg: {current_regime}")

    # End
    print(f"\n🏆 SIMULATION COMPLETE")
    print(f"   Final Equity: ${equity:.2f}")
    print(f"   Total Trades: {trades}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label='Pro Equity (Net)')
    plt.title(f"Professional Simulation (H1, Kelly, Slippage)\nFinal: ${equity:.0f}")
    plt.savefig("evotrader_pro_result.png")
    print("📸 Saved chart to 'evotrader_pro_result.png'")

if __name__ == "__main__":
    run_pro_simulation()
