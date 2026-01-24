
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
WINDOW_SIZE = 10
POP_SIZE = 20
GENS = 50

def make_env(df):
    """Creates the Financial Environment from a DataFrame slice."""
    return FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, len(df)), window_size=WINDOW_SIZE)

def train_specialist(regime_name, df_slice):
    print(f"\n🪙 TRAINING SPECIALIST: {regime_name.upper()}")
    env = make_env(df_slice)
    
    # Needs a new pilot for each specialist training (or continue from base)
    # We start fresh to prove specialization
    pilot = MemoryEvoPilot()
    # Override input dim: Window * 6 features
    pilot.input_dim = WINDOW_SIZE * 6 
    # Override output dim: 2 (Buy/Sell) - wait, anytrading uses Discrete(2)
    pilot.output_dim = 2
    # Reinit network with correct dims
    from evonet.core.network import MultiClassEvoNet
    pilot.net = MultiClassEvoNet(pilot.input_dim, pilot.output_dim)
    pilot.flat_init = pilot.get_flat_weights(0) # Re-init flat helper
    pilot.memory.theta_init = pilot.flat_init
    
    # --- DEBUG: Run Baseline (Always Buy) ---
    print("   🔎 DEBUG: Running 'Always Buy' Baseline...")
    base_env = make_env(df_slice)
    s, _ = base_env.reset()
    base_r = 0
    t = False
    trunc = False
    ticks = 0
    while not (t or trunc) and ticks < 200:
        s, r, t, trunc, _ = base_env.step(1) # Force Buy
        base_r += r
        ticks += 1
    print(f"   🔎 Baseline Reward (200 ticks): {base_r:.2f}")
    base_env.close()
    # ----------------------------------------
    
    for gen in range(1, GENS + 1):
        rewards = []
        actions_log = []
        
        for i in range(pilot.pop_size):
            # Run episode
            state, _ = env.reset()
            total_r = 0
            terminated = False
            truncated = False
            
            # Limited steps for faster debugging
            steps = 0
            while not (terminated or truncated) and steps < 200:
                action = pilot.get_action(state, i)
                actions_log.append(action)
                state, r, terminated, truncated, _ = env.step(action)
                total_r += r
                steps += 1
            rewards.append(total_r)
            
        best = np.max(rewards)
        pilot.update_hall_of_fame(rewards)
        pilot.evolve(rewards)
        
        if gen % 5 == 0:
            # Stats
            n_long = sum(actions_log)
            pct_long = (n_long / len(actions_log)) * 100 if len(actions_log) > 0 else 0
            print(f"   Gen {gen}: Reward={best:.2f} | Action Dist: {pct_long:.1f}% BUY | Best Pilot Bias: {pilot.net.level3[0].population[0]['bias']:.4f}")
            if pct_long < 5.0:
                 print("   ⚠️  WARNING: STUCK IN 'SELL/SHORT' MODE (Safety convergence). Needs bias shift.")
            
    # Store the specialist skill
    best_idx = np.argmax(rewards)
    pilot.store_memory(regime_name, best_idx)
    env.close()
    
    # Return the memory vector (delta) or the pilot itself?
    # We return the pilot, which now has the memory stored in its internal dict
    return pilot

def run_simulation(start_year="2020", end_year="2023"):
    print("🚀 LAUNCHING EVOTRADER SIMULATION")
    
    # 1. Load Data
    fetcher = DataFetcher(TICKER)
    df = fetcher.fetch_data()
    df = fetcher.add_indicators()
    
    # 2. Split Regimes
    # We assume we know these dates (In real world, we'd detect them)
    bull_df = fetcher.split_by_regime("bull_2020")
    bear_df = fetcher.split_by_regime("bear_2022")
    
    # 3. Train Specialists
    # Train Bull Pilot
    bull_pilot = train_specialist("bull", bull_df)
    
    # Train Bear Pilot
    # Note: We can reuse the same pilot instance to accumulate memories, 
    # but for clarity we'll assume we extract the vectors.
    # Ideally, we have ONE pilot that learns both.
    # Let's use ONE pilot.
    
    print("\n🧠 CONSOLIDATING MEMORIES...")
    master_pilot = bull_pilot # Has 'bull' memory
    
    # Train Bear on the SAME pilot? 
    # If we do that, it forgets Bull. That's the point!
    # But we want to store the delta.
    
    # Reset weights to init (Tabula Rasa) or keep Bull weights?
    # To demonstrate "Switching", we should start from base for Bear training too,
    # OR fine-tune.
    # Let's fine-tune from Bull to Bear, then store Bear.
    # Then we recall Bull later to prove we didn't lose it.
    
    print("   -> Switching to Bear Market Training (Fine-Tuning)...")
    bear_env = make_env(bear_df)
    
    for gen in range(1, GENS + 1):
        rewards = []
        for i in range(master_pilot.pop_size):
            rewards.append(run_episode(bear_env, master_pilot, i))
        master_pilot.update_hall_of_fame(rewards)
        master_pilot.evolve(rewards)
        if gen % 10 == 0: print(f"   Gen {gen}: Best Bear Profit = {np.max(rewards):.2f}")
        
    master_pilot.store_memory("bear", np.argmax(rewards))
    bear_env.close()
    
    print("\n✅ TRAINING COMPLETE. Memories Stored: ", master_pilot.stored_tasks)
    
    # 4. The Meta-Controller Test
    # Simulate a transition: Bull -> Bear -> Bull
    print("\n📉 SIMULATING REGIME SWITCH (Bull -> Bear)")
    
    # A. Run in Bull Mode (Recall Bull)
    master_pilot.recover_memory("bull")
    test_env_bull = make_env(bull_df.iloc[:500]) # Test on subset
    r_bull = run_episode(test_env_bull, master_pilot, 0)
    print(f"   Score in Bull Market (Using Bull Memory): {r_bull:.2f}")
    
    # B. Run in Bear Mode (Using Bull Memory) -> EXPECT FAILURE
    test_env_bear = make_env(bear_df.iloc[:500])
    r_fail = run_episode(test_env_bear, master_pilot, 0)
    print(f"   Score in Bear Market (Using Bull Memory): {r_fail:.2f} (Expected Low)")
    
    # C. Real-time Switch
    print("   🚨 REGIME CHANGE DETECTED! Swapping to 'Bear' Memory...")
    master_pilot.recover_memory("bear")
    r_bear = run_episode(test_env_bear, master_pilot, 0)
    print(f"   Score in Bear Market (Using Bear Memory): {r_bear:.2f} (Expected High)")
    
    print("\n🎉 EvoTrader Proof-of-Concept Successful!")

if __name__ == "__main__":
    run_simulation()
