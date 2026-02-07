"""
COLAB SIMPLE TRAINER
====================
Simplified version that uses your existing MemoryEvoPilot architecture
Copy-paste this ENTIRE file into a Colab cell and run!

INSTRUCTIONS:
1. Upload 'evonet' folder to Colab
2. Paste this entire script into a cell
3. Run it
4. Download ultimate_brain_colab.pkl after ~2-3 hours
"""

import sys
import os
import subprocess

# Install dependencies
print("📦 Installing Dependencies...")
subprocess.run(['pip', 'install', 'gymnasium', 'gym-anytrading', 'yfinance', 
                'pandas', 'numpy', 'pandas-ta', '-q'], check=True)

# Find evonet module
print("🔍 Locating evonet module...")
possible_paths = ['/content', '/content/evolutionary-neural-network']
for path in possible_paths:
    if os.path.exists(os.path.join(path, 'evonet')):
        sys.path.insert(0, path)
        print(f"✅ Found evonet in: {path}")
        break

# Imports
import numpy as np
import pickle
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot  # Use existing implementation!

# Configuration
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20
GENERATIONS = 200

print("\n" + "="*70)
print("🚀 ULTIMATE BRAIN TRAINING - SIMPLIFIED")
print("="*70)
print(f"📊 Dataset: {TICKER} ({START_DATE} to {END_DATE})")
print(f"🔄 Generations: {GENERATIONS}")
print("="*70 + "\n")

# Fetch Data
print("📥 Fetching Data...")
fetcher = DataFetcher(TICKER, start_date=START_DATE, end_date=END_DATE, provider="yf")
df = fetcher.fetch_data()
df = fetcher.process()
print(f"✅ Loaded {len(df)} data points\n")

# Create Environment
print("🌍 Creating Environment...")
safe_end = len(df) - (WINDOW_SIZE * 3)
env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, safe_end), window_size=WINDOW_SIZE)
print(f"✅ Environment Ready\n")

# Initialize Pilot (uses your existing code!)
print("🧠 Initializing Pilot...")
pilot = MemoryEvoPilot()
pilot.input_dim = WINDOW_SIZE * 10
pilot.output_dim = 3
print(f"✅ Pilot Ready (Population: {pilot.net.pop_size})\n")

# Training Function
def evaluate_genome(genome_idx):
    state, _ = env.reset()
    equity = 1.0
    returns = []
    terminated = False
    steps = 0
    
    while not terminated and steps < 1000:
        action = pilot.get_action(state, genome_idx)
        state, r, terminated, _, _ = env.step(action)
        equity *= np.exp(r)
        returns.append(r)
        steps += 1
        
    returns = np.array(returns)
    total_ret = (equity - 1.0)
    
    # Sortino
    downside = returns[returns < 0]
    down_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (np.mean(returns) / (down_std + 1e-9)) * np.sqrt(252)
    
    # Max DD
    peak = 1.0
    max_dd = 0.0
    equity_curve = [1.0]
    for r in returns:
        equity_curve.append(equity_curve[-1] * np.exp(r))
    for val in equity_curve:
        if val > peak: peak = val
        dd = (peak - val) / peak
        if dd > max_dd: max_dd = dd
        
    # Fitness
    dd_penalty = max(0, (max_dd - 0.15) * 50.0)
    fitness = (total_ret * 0.3) + (sortino * 0.5) - dd_penalty
    
    return fitness, sortino, max_dd

# Training Loop
print("🔥 Starting Training...\n")
best_ever_fitness = -999
best_ever_idx = 0

for gen in range(1, GENERATIONS + 1):
    # Evaluate all genomes
    scores = []
    stats = []
    
    for i in range(pilot.net.pop_size):
        fit, sortino, max_dd = evaluate_genome(i)
        scores.append(fit)
        stats.append((fit, sortino, max_dd))
        
    # Find best
    best_idx = np.argmax(scores)
    best_fit, best_sortino, best_dd = stats[best_idx]
    
    if best_fit > best_ever_fitness:
        best_ever_fitness = best_fit
        best_ever_idx = best_idx
        
    # Log
    if gen % 10 == 0 or gen == 1:
        print(f"Gen {gen:3d}/{GENERATIONS} | Fit: {best_fit:7.3f} | "
              f"Sortino: {best_sortino:6.2f} | MaxDD: {best_dd:5.1%}")
              
    # Checkpoint every 50 gens
    if gen % 50 == 0:
        with open(f"checkpoint_gen_{gen}.pkl", 'wb') as f:
            pickle.dump(pilot, f)
        print(f"    💾 Saved checkpoint_gen_{gen}.pkl")
        
    # Evolve
    pilot.evolve(scores)
    
env.close()

# Save final brain
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"✅ Best Fitness: {best_ever_fitness:.3f}")

# Copy best genome to index 0
best_weights = pilot.get_flat_weights(best_ever_idx)
pilot.set_flat_weights(best_weights, 0)

with open("ultimate_brain_colab.pkl", 'wb') as f:
    pickle.dump(pilot, f)
    
print("\n✅ Brain saved: ultimate_brain_colab.pkl")
print("📥 Download it from Colab Files panel!")
print("="*70)
