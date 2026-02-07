"""
🚀 COLAB BRAIN TRAINER - 2-CELL SOLUTION
=========================================

CELL 1: RUN THIS FIRST (Installs packages & restarts runtime)
CELL 2: RUN AFTER RESTART (Trains the model)

This avoids all NumPy/scipy/sklearn conflicts!
"""

# ==============================================================================
# CELL 1: SETUP & INSTALL (Copy-paste this into FIRST cell)
# ==============================================================================
"""
import subprocess
import os

print("📦 Installing dependencies with compatible versions...")

# Install specific compatible versions to avoid conflicts
subprocess.run(['pip', 'install', '-q',
    'numpy==1.26.4',
    'scipy==1.11.4', 
    'scikit-learn==1.3.2',
    'pandas==2.1.4',
    'yfinance',
    'gymnasium',
    'gym-anytrading',
    'pandas-ta'
], check=True)

print("✅ Dependencies installed")
print("⚠️  Now restart runtime: Runtime → Restart runtime")
print("   Then run CELL 2!")
"""

# ==============================================================================  
# CELL 2: TRAINING (Copy-paste this into SECOND cell, run AFTER restart)
# ==============================================================================
"""
import subprocess
import sys
import os
import numpy as np
import pickle
from typing import Tuple

# Clone repo
print("📥 Cloning repository...")
if not os.path.exists('/content/evolutionary-neural-network'):
    subprocess.run([
        'git', 'clone',
        'https://github.com/ayushmantrivedi/evolutionary-neural-network.git'
    ], check=True)

os.chdir('/content/evolutionary-neural-network')
sys.path.insert(0, '/content/evolutionary-neural-network')

# Import modules (after runtime restart, no conflicts!)
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot

# Configuration
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20
GENERATIONS = 200

print("="*80)
print("🚀 ULTIMATE BRAIN TRAINING")
print("="*80)

# Fetch data
print("\n📥 Fetching data...")
fetcher = DataFetcher(TICKER, start_date=START_DATE, end_date=END_DATE, provider="yf")
df = fetcher.fetch_data()
df = fetcher.process()
print(f"✅ Loaded {len(df)} data points")

# Setup environment
safe_end = len(df) - (WINDOW_SIZE * 3)
env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, safe_end), window_size=WINDOW_SIZE)

# Initialize pilot
pilot = MemoryEvoPilot()
pilot.input_dim = WINDOW_SIZE * 10
pilot.output_dim = 3
print(f"✅ Pilot ready (Pop: {pilot.net.pop_size})")

# Evaluation function
def evaluate_genome(genome_idx: int) -> Tuple[float, float, float]:
    state, _ = env.reset()
    equity = 1.0
    equity_curve = [1.0]
    returns = []
    terminated = False
    steps = 0
    
    while not terminated and steps < 1000:
        action = pilot.get_action(state, genome_idx)
        state, reward, terminated, _, _ = env.step(action)
        equity *= np.exp(reward)
        equity_curve.append(equity)
        returns.append(reward)
        steps += 1
    
    returns = np.array(returns)
    
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (np.mean(returns) / (downside_std + 1e-9)) * np.sqrt(252)
    
    peak = 1.0
    max_dd = 0.0
    for val in equity_curve:
        if val > peak: peak = val
        dd = (peak - val) / peak
        if dd > max_dd: max_dd = dd
    
    dd_penalty = max(0, (max_dd - 0.15) * 50.0)
    fitness = (sortino * 0.6) + ((equity - 1.0) * 0.01) - dd_penalty
    
    return fitness, sortino, max_dd

# Training loop
print("\n🔥 Starting training...\n")
best_ever_fitness = -999.0
best_ever_idx = 0

for gen in range(1, GENERATIONS + 1):
    scores = []
    stats = []
    
    for i in range(pilot.net.pop_size):
        fit, sortino, max_dd = evaluate_genome(i)
        scores.append(fit)
        stats.append((fit, sortino, max_dd))
    
    best_idx = np.argmax(scores)
    best_fit, best_sortino, best_dd = stats[best_idx]
    
    if best_fit > best_ever_fitness:
        best_ever_fitness = best_fit
        best_ever_idx = best_idx
    
    if gen % 10 == 0 or gen == 1:
        print(f"Gen {gen:3d}/{GENERATIONS} | Fit: {best_fit:7.3f} | "
              f"Sortino: {best_sortino:6.2f} | MaxDD: {best_dd:5.1%}")
    
    if gen % 50 == 0:
        with open(f"checkpoint_{gen}.pkl", 'wb') as f:
            pickle.dump(pilot, f)
        print(f"    💾 Checkpoint saved")
    
    pilot.evolve(scores)

env.close()

# Save final brain
best_weights = pilot.get_flat_weights(best_ever_idx)
pilot.set_flat_weights(best_weights, 0)

with open("ultimate_brain_colab.pkl", 'wb') as f:
    pickle.dump(pilot, f)

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print(f"✅ Best Fitness: {best_ever_fitness:.3f}")
print("📥 Download: ultimate_brain_colab.pkl")
print("="*80)
"""
