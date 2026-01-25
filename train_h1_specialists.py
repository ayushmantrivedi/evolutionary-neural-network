
import numpy as np
import pandas as pd
import pickle
import os
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot

import torch
import random
import numpy as np

# --- H1 CONFIGURATION ---
TICKER = "BTC-USD"
WINDOW_SIZE = 48 # 48 Hours context (2 Days) to reduce noise
POP_SIZE = 50    # PRODUCTION MODE: Standard population
GENS = 100       # PRODUCTION MODE: Deep evolution for optimal convergence
SEED = 42        # Critical for Base Network Synchronization

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_env_h1(df):
    return FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, len(df)), window_size=WINDOW_SIZE, fee=0.001)

def train_specialist(name, df_subset):
    print(f"\n🪙 TRAINING H1 SPECIALIST: {name.upper()} ({len(df_subset)} hours)")
    if len(df_subset) < 500:
        print("   ⚠️ Not enough data for this regime! Skipping.")
        return None
        
    env = make_env_h1(df_subset)
    
    # CRITICAL: Force Seed for Consistent Base Network
    set_seed(SEED)
    
    pilot = MemoryEvoPilot()
    pilot.input_dim = WINDOW_SIZE * 7
    pilot.output_dim = 3
    
    # Reinit Net
    from evonet.core.network import MultiClassEvoNet
    pilot.net = MultiClassEvoNet(pilot.input_dim, pilot.output_dim)
    pilot.flat_init = pilot.get_flat_weights(0)
    pilot.memory.theta_init = pilot.flat_init
    
    best_fitness_global = -float('inf')
    best_idx_global = 0
    
    # Evolution Loop
    for gen in range(1, GENS + 1):
        fitness_scores = []
        for i in range(pilot.pop_size):
            # Evaluate using internal 'run_episode' logic or similar
            # Since we can't import run_pro_episode easily if it's local in train_evotrader
            # We copy specific logic here inline for safety
            
            # --- EVAL FUNCTION ---
            fit, _, _, _, _ = run_h1_eval(env, pilot, i)
            fitness_scores.append(fit)
            
        best_idx = np.argmax(fitness_scores)
        best_fit = fitness_scores[best_idx]
        
        if best_fit > best_fitness_global:
            best_fitness_global = best_fit
            best_idx_global = best_idx
            
        pilot.update_hall_of_fame(fitness_scores)
        pilot.evolve(fitness_scores)
        
        if gen % 5 == 0:
            print(f"   Gen {gen}: Best Fit={best_fit:.2f}")
            
    pilot.store_memory(name, best_idx_global)
    env.close()
    return pilot

def run_h1_eval(env, pilot, pilot_idx):
    # Simplified evaluation for speed
    state, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    steps = 0
    
    # Limit steps per episode to prevent infinite loops in broken agents
    max_steps = 500 
    
    equity = 1.0
    
    while not (terminated or truncated) and steps < max_steps:
        action = pilot.get_action(state, pilot_idx)
        state, r, terminated, truncated, _ = env.step(action)
        total_reward += r
        equity *= np.exp(r)
        steps += 1
        
    # Simple Fitness: Equity Return
    fitness = (equity - 1.0) * 100
    
    # Penalize inactivity
    if steps < 10: fitness = -100
    
    return fitness, 0, 0, 0, []

def main():
    print("🚀 STARTING H1 FREQUENCY RETRAINING")
    
    # 1. Fetch H1 Data
    # 2024-2025
    fetcher = DataFetcher(TICKER, start_date="2024-02-01", end_date="2025-12-30", interval="1h")
    df = fetcher.fetch_data()
    df = fetcher.add_advanced_features()
    
    # 2. Auto-Split Regimes (The "Smart" Splitter)
    # We create masks
    print("\n🔪 AUTO-SPLITTING REGIMES...")
    
    # BULL: Price > SMA50 (Primary Trend)
    # BEAR: Price < SMA50
    # CHOP: ADX < 20 (Weak Trend) - Overrides Trend
    
    mask_chop = (df['ADX'] < 0.20)
    mask_bull = (df['Close'] > df['SMA_50']) & (~mask_chop)
    mask_bear = (df['Close'] <= df['SMA_50']) & (~mask_chop)
    # 2. Auto-Split Regimes (The "Smart" Splitter)
    # We create masks
    print("\n🔪 AUTO-SPLITTING REGIMES...")
    
    # BULL: Price > SMA50 (Primary Trend)
    # BEAR: Price < SMA50
    # CHOP: ADX < 20 (Weak Trend) - Overrides Trend
    
    mask_chop = (df['ADX'] < 0.20)
    mask_bull = (df['Close'] > df['SMA_50']) & (~mask_chop)
    mask_bear = (df['Close'] <= df['SMA_50']) & (~mask_chop)
    
    df_chop = df[mask_chop].copy()
    df_bull = df[mask_bull].copy()
    df_bear = df[mask_bear].copy()
    
    print(f"   Subset Sizes: Bull={len(df_bull)}, Bear={len(df_bear)}, Chop={len(df_chop)}")
    
    # 3. Train
    bull_pilot = train_specialist("bull", df_bull)
    bear_pilot = train_specialist("bear", df_bear)
    chop_pilot = train_specialist("chop", df_chop)
    
    # 4. Consolidate
    print("\n🧠 CONSOLIDATING H1 BRAIN...")
    
    # CRITICAL: Reset Seed to ensure Master Network matches Specialist Networks
    set_seed(SEED)
    
    master_pilot = MemoryEvoPilot()
    master_pilot.input_dim = WINDOW_SIZE * 7
    master_pilot.output_dim = 3
    
    from evonet.core.network import MultiClassEvoNet
    master_pilot.net = MultiClassEvoNet(master_pilot.input_dim, master_pilot.output_dim)
    
    # Reinit Memory
    master_pilot.flat_init = master_pilot.get_flat_weights(0)
    from evonet.core.memory import DirectionalMemory
    master_pilot.memory = DirectionalMemory(master_pilot.flat_init)
    
    # Store
    if bull_pilot:
        master_pilot.memory.store_task("bull", bull_pilot.memory.task_directions["bull"] + master_pilot.memory.theta_init)
    if bear_pilot:
        master_pilot.memory.store_task("bear", bear_pilot.memory.task_directions["bear"] + master_pilot.memory.theta_init)
    if chop_pilot:
        master_pilot.memory.store_task("chop", chop_pilot.memory.task_directions["chop"] + master_pilot.memory.theta_init)
        
    master_pilot.stored_tasks = {"bull", "bear", "chop"}
    
    # Save
    with open("evotrader_brain_h1.pkl", "wb") as f:
        pickle.dump(master_pilot, f)
        
    print("💾 H1 BRAIN SAVED: evotrader_brain_h1.pkl")
    print("   -> Next: Run run_pro_simulation.py (Make sure to update it to load _h1.pkl)")

if __name__ == "__main__":
    main()
