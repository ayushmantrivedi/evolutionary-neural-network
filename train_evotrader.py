
import numpy as np
import os
import sys
import logging
import gymnasium as gym
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot, run_episode

# Configure
TICKER = "BTC-USD"
WINDOW_SIZE = 20 # Increased context for Alpha Signals
POP_SIZE = 30    # Larger population for harder problem
GENS = 50

# --- DOMAIN EXPERT METRICS ---
def calculate_downside_deviation(returns, target_return=0.0):
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return 1e-6
    return np.std(downside_returns) + 1e-6

def calculate_sortino(returns, target_return=0.0):
    avg_return = np.mean(returns)
    down_dev = calculate_downside_deviation(returns, target_return)
    return avg_return / down_dev

def calculate_max_drawdown(equity_curve):
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd

def run_pro_episode(env, pilot, pilot_index):
    """
    Runs an episode collecting Professional Metrics (Equity, Drawdown).
    """
    state, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    equity = 1.0 # Starting Equity Unit
    equity_curve = [equity]
    returns_log = []
    
    steps = 0
    # Increase steps to see long-term stability
    while not (terminated or truncated) and steps < 400:
        action = pilot.get_action(state, pilot_index)
        state, r, terminated, truncated, _ = env.step(action)
        
        # 'r' here is log-return adjusted by fee
        # Update Equity: Equity_t = Equity_{t-1} * exp(r)
        equity = equity * np.exp(r)
        equity_curve.append(equity)
        returns_log.append(r)
        
        total_reward += r
        steps += 1
        
    returns_log = np.array(returns_log)
    
    # --- RISK ADJUSTED FITNESS ---
    # Fitness = Returns * log(1 + Sortino) - Penalty * MaxDD
    
    # 1. Sortino
    if np.std(returns_log) < 1e-9: # No volatility (Sat in cash)
        sortino = 0.0
    else:
        sortino = calculate_sortino(returns_log)
        
    # 2. Max Drawdown
    mdd = calculate_max_drawdown(equity_curve)
    
    # 3. Activity Constraint (Don't just sit in cash)
    n_trades = env._current_tick if hasattr(env, '_current_tick') else steps # Approx
    
    # Heuristic Fitness
    # Sortino > 0.05 per step is godlike. 
    # Use simpler Annualized-ish logic
    
    # Total Return %
    total_ret_pct = (equity - 1.0) * 100
    
    # Fitness Score: Reward Stability & Safety
    # Base: Total Return
    # Multiplier: Sortino (Reward efficiency)
    # Penalty: Drawdown^2 (Severe penalty for crashing)
    
    fitness = total_ret_pct * (1.0 + sortino) - (mdd * 100 * 2.0)
    
    # Sanity check: If lost money, fitness should be bad
    if total_ret_pct < 0:
        fitness = total_ret_pct * (1.0 + mdd) # Compounding penalty on loss
        
    return fitness, total_ret_pct, sortino, mdd, equity_curve

def make_env(df):
    """Creates the Financial Environment from a DataFrame slice."""
    # Pass Friction Params here
    return FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, len(df)), window_size=WINDOW_SIZE, fee=0.001)

def train_specialist(regime_name, df_slice):
    print(f"\n🪙 TRAINING EXPERT: {regime_name.upper()}")
    env = make_env(df_slice)
    
    pilot = MemoryEvoPilot()
    # Override input dim: Window * 7 features (Added ADX, etc)
    pilot.input_dim = WINDOW_SIZE * 7
    # Override output dim: 3 (Short, Neutral, Long)
    pilot.output_dim = 3
    
    # Reinit network
    from evonet.core.network import MultiClassEvoNet
    pilot.net = MultiClassEvoNet(pilot.input_dim, pilot.output_dim)
    pilot.flat_init = pilot.get_flat_weights(0)
    pilot.memory.theta_init = pilot.flat_init
    
    # --- DEBUG: Run Baseline (Buy & Hold) ---
    print("   🔎 DEBUG: Running 'Buy & Hold' Baseline...")
    bh_env = make_env(df_slice)
    s, _ = bh_env.reset()
    bh_equity = 1.0
    steps = 0
    while steps < 400:
        s, r, t, tr, _ = bh_env.step(2) # 2 = Long
        bh_equity *= np.exp(r)
        steps += 1
        if t or tr: break
    print(f"   🔎 Buy/Hold Final Equity: {bh_equity:.4f} ({ (bh_equity-1)*100:.2f}%)")
    bh_env.close()
    
    best_fitness_global = -float('inf')
    
    for gen in range(1, GENS + 1):
        fitness_scores = []
        stats_log = [] # (Ret, Sortino, MDD)
        
        for i in range(pilot.pop_size):
            fit, ret, sort, mdd, eq = run_pro_episode(env, pilot, i)
            fitness_scores.append(fit)
            stats_log.append((ret, sort, mdd))
            
        best_idx = np.argmax(fitness_scores)
        best_fit = fitness_scores[best_idx]
        best_stat = stats_log[best_idx]
        
        if best_fit > best_fitness_global:
            best_fitness_global = best_fit
        
        pilot.update_hall_of_fame(fitness_scores)
        pilot.evolve(fitness_scores)
        
        if gen % 5 == 0:
            print(f"   Gen {gen}: Best Fitness={best_fit:.2f} | Ret: {best_stat[0]:.1f}% | Sortino: {best_stat[1]:.2f} | MaxDD: {best_stat[2]*100:.1f}%")
            
    # Store skill
    pilot.store_memory(regime_name, best_idx)
    env.close()
    
    return pilot

def run_simulation():
    print("🚀 LAUNCHING HEDGE FUND SIMULATION (3-CYCLE MASTERY)")
    
    # 1. Load Data
    fetcher = DataFetcher(TICKER)
    df = fetcher.fetch_data()
    # Use NEW Alpha features
    df = fetcher.add_advanced_features()
    
    # 2. Split Regimes
    bull_df = fetcher.split_by_regime("bull_2020")
    bear_df = fetcher.split_by_regime("bear_2022")
    chop_df = fetcher.split_by_regime("chop_2023")
    
    # 3. Train Experts Independently (To avoid Bias)
    
    # --- A. BULL EXPERT (Momentum) ---
    bull_pilot = train_specialist("bull", bull_df)
    
    # --- B. BEAR EXPERT (Short Seller) ---
    # We do NOT fine-tune from Bull. We start fresh to learn "Shorting" natively.
    bear_pilot = train_specialist("bear", bear_df)
    
    # --- C. NEUTRAL EXPERT (Sniper/Cash) ---
    # Training using 2023 Chop data
    chop_pilot = train_specialist("chop", chop_df)
    
    print("\n✅ TRAINING COMPLETE. 3 Experts Created.")
    print(f"   Bull Memory: {bull_pilot.stored_tasks}")
    print(f"   Bear Memory: {bear_pilot.stored_tasks}")
    print(f"   Chop Memory: {chop_pilot.stored_tasks}")
    
    # 4. Consolidate into Master Brain (The EvoTrader)
    print("\n🧠 CONSOLIDATING INTO MASTER BRAIN...")
    master_pilot = MemoryEvoPilot()
    master_pilot.input_dim = WINDOW_SIZE * 7
    master_pilot.output_dim = 3
    from evonet.core.network import MultiClassEvoNet
    master_pilot.net = MultiClassEvoNet(master_pilot.input_dim, master_pilot.output_dim)
    
    # CRITICAL FIX: Re-initialize memory structure to match new network size
    master_pilot.flat_init = master_pilot.get_flat_weights(0)
    from evonet.core.memory import DirectionalMemory
    master_pilot.memory = DirectionalMemory(master_pilot.flat_init)
    
    # Inject Memories
    # We transfer the learned vectors from the specialists to the master's memory
    # In a real app, we'd save these to disk. Here we just copy.
    
    # Transfer Bull
    # Note: accessing internal 'memory' object directly for demo
    if "bull" in bull_pilot.memory.task_directions:
        master_pilot.memory.store_task("bull", bull_pilot.memory.task_directions["bull"] + master_pilot.memory.theta_init)
    if "bear" in bear_pilot.memory.task_directions:
        master_pilot.memory.store_task("bear", bear_pilot.memory.task_directions["bear"] + master_pilot.memory.theta_init)
    if "chop" in chop_pilot.memory.task_directions:
        master_pilot.memory.store_task("chop", chop_pilot.memory.task_directions["chop"] + master_pilot.memory.theta_init)
        
    master_pilot.stored_tasks = {"bull", "bear", "chop"}
    
    print("🎉 EvoTrader 'Full Cycle' Expert Ready.")
    
    # 5. Save the Brain
    import pickle
    brain_path = "evotrader_brain.pkl"
    with open(brain_path, "wb") as f:
        pickle.dump(master_pilot, f)
    print(f"💾 BRAIN SAVED: {brain_path}")
    print("   -> Contains: Weights, Memory Vectors (Bull/Bear/Chop)")
    
    # 6. Recommendation
    print("\nNext Step: Run 'python run_backtest.py' to see it trade 2020-2023!")

if __name__ == "__main__":
    run_simulation()
