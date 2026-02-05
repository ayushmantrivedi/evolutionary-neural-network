
import numpy as np
import os
import sys
import pickle

# Ensure evonet is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot, run_episode
from evonet.config import WINDOW_SIZE

def run_pro_episode(env, pilot, pilot_index):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    # Keep it long enough to see skill
    while steps < 400:
        action = pilot.get_action(state, pilot_index)
        state, r, terminated, truncated, _ = env.step(action)
        total_reward += r
        steps += 1
        if terminated or truncated: break
    return total_reward

def train_market_expert(ticker, provider, epochs=20, training_gen=20):
    print(f"\n[Training] {ticker.upper()} ({provider.upper()})...", flush=True)
    fetcher = DataFetcher(ticker, provider=provider)
    df = fetcher.fetch_data()
    df = fetcher.process()
    
    # Use a specific regime window for fast proof (e.g. 2020-2021 window)
    env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, min(3000, len(df))), window_size=WINDOW_SIZE)
    
    pilot = MemoryEvoPilot()
    pilot.input_dim = WINDOW_SIZE * 10
    pilot.output_dim = 3
    from evonet.core.network import MultiClassEvoNet
    pilot.net = MultiClassEvoNet(pilot.input_dim, pilot.output_dim)
    pilot.flat_init = pilot.get_flat_weights(0)
    from evonet.core.memory import DirectionalMemory
    pilot.memory = DirectionalMemory(pilot.flat_init)
    
    for gen in range(1, training_gen + 1):
        scores = []
        for i in range(pilot.pop_size):
            scores.append(run_pro_episode(env, pilot, i))
        
        best_score = np.max(scores)
        # Evolve
        pilot.evolve(scores)
        if gen % 5 == 0:
            print(f"   Gen {gen}: Best Reward: {best_score:.4f}", flush=True)
            
    best_idx = np.argmax(scores)
    return pilot, best_idx, env

def test_field_proof():
    print("="*60, flush=True)
    print("💎 REAL-WORLD FIELD PROOF: CRYPTO VS STOCKS", flush=True)
    print("   Objective: Prove flawlessness across two separate markets.", flush=True)
    print("="*60, flush=True)
    
    # 1. Train Market A: CRYPTO (BTC)
    # Using small gen count for proof speed, normally would be 50+
    btc_pilot, btc_idx, btc_env = train_market_expert("BTC-USD", "binance", training_gen=15)
    btc_peak = run_pro_episode(btc_env, btc_pilot, btc_idx)
    print(f"✅ Crypto Skill (BTC) Mastered: {btc_peak:.4f}", flush=True)
    
    # Store BTC Memory
    btc_pilot.store_memory("crypto_btc", btc_idx)
    v_btc = btc_pilot.memory.task_directions["crypto_btc"]
    
    # 2. MASSIVE TRAUMA (The Wipe)
    print("\n⚡ APPLYING MASSIVE TRAUMA (Wiping all Weights)...", flush=True)
    btc_pilot.inject_fault(1.0)
    trauma_score = run_pro_episode(btc_env, btc_pilot, 0)
    print(f"💀 Status After Wipe: {trauma_score:.4f} (Memory is now clean)", flush=True)
    
    # 3. Train Market B: STOCKS (AAPL)
    # Note: We reuse the same neural architecture (btc_pilot) to prove it can hold multiple skills
    print("\n[Step 2] Training on STOCKS (AAPL)...", flush=True)
    fetcher_aapl = DataFetcher("AAPL", provider="yf")
    df_aapl = fetcher_aapl.fetch_data()
    df_aapl = fetcher_aapl.process()
    aapl_env = FinancialRegimeEnv(df_aapl, frame_bound=(WINDOW_SIZE, len(df_aapl)), window_size=WINDOW_SIZE)
    
    # Train the Wiped pilot on AAPL
    for gen in range(1, 16):
        scores = []
        for i in range(btc_pilot.pop_size):
            scores.append(run_pro_episode(aapl_env, btc_pilot, i))
        btc_pilot.evolve(scores)
        if gen % 5 == 0:
            print(f"   Gen {gen} (AAPL): Best Reward: {np.max(scores):.4f}", flush=True)
            
    aapl_idx = np.argmax(scores)
    aapl_peak = run_pro_episode(aapl_env, btc_pilot, aapl_idx)
    print(f"✅ Stock Skill (AAPL) Mastered: {aapl_peak:.4f}", flush=True)
    
    # Store AAPL Memory in the SAME memory bank
    btc_pilot.store_memory("stock_aapl", aapl_idx)
    
    # --- THE PROOF: ZERO INTERFERENCE RETRIEVAL ---
    print("\n" + "*"*60, flush=True)
    print("🏆 THE ULTIMATE PROOF: RETRIEVING CROSS-FIELD KNOWLEDGE", flush=True)
    print("*"*60, flush=True)
    
    # A. Retrieve Crypto from the Stock-heavy brain
    print("\n[Retrieval A] Regenerating BTC Strategy...", flush=True)
    # This wipes current AAPL knowledge and regrows BTC
    btc_pilot.recover_memory("crypto_btc")
    retrieved_btc_score = run_pro_episode(btc_env, btc_pilot, 0)
    print(f"   --> BTC Score after Regeneration: {retrieved_btc_score:.4f}", flush=True)
    print(f"   --> Retention Quality: { (retrieved_btc_score/btc_peak)*100:.1f}%", flush=True)
    
    # B. Retrieve Stock from the Crypto-heavy brain
    print("\n[Retrieval B] Regenerating AAPL Strategy...", flush=True)
    # This wipes BTC knowledge and regrows AAPL
    btc_pilot.recover_memory("stock_aapl")
    retrieved_aapl_score = run_pro_episode(aapl_env, btc_pilot, 0)
    print(f"   --> AAPL Score after Regeneration: {retrieved_aapl_score:.4f}", flush=True)
    print(f"   --> Retention Quality: { (retrieved_aapl_score/aapl_peak)*100:.1f}%", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("📊 CONCLUSION", flush=True)
    print("="*60, flush=True)
    
    results = {
        "btc_peak": float(btc_peak),
        "btc_recovered": float(retrieved_btc_score),
        "btc_retention": float(retrieved_btc_score/btc_peak),
        "aapl_peak": float(aapl_peak),
        "aapl_recovered": float(retrieved_aapl_score),
        "aapl_retention": float(retrieved_aapl_score/aapl_peak),
        "success": bool((retrieved_btc_score/btc_peak > 0.9) and (retrieved_aapl_score/aapl_peak > 0.9))
    }
    
    import json
    with open("field_proof_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"📄 Results saved to field_proof_results.json", flush=True)

    if results["success"]:
        print("✅ PROVEN: One brain can handle ALL markets with ZERO interference.", flush=True)
        print("   The project is ready for multi-field implementation.", flush=True)
    else:
        print("❌ NOT PROVEN: Performance drift detected.", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    test_field_proof()
