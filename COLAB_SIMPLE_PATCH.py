"""
🚀 COLAB TRAINING - ATTENTION PATCH (SIMPLEST SOLUTION!)
========================================================

This version patches the attention forward method to just pass through data.
No complex reimplementation - just a simple fix!

Copy EVERYTHING below into Colab (after dependencies installed & runtime restarted):
"""

import subprocess
import sys
import os
import numpy as np
import pickle
from typing import Tuple

print("="*80)
print("🚀 ULTIMATE BRAIN TRAINING (Attention Bypassed)")
print("="*80 + "\n")

# Clone
print("📥 Cloning repository...")
if not os.path.exists('/content/evolutionary-neural-network'):
    subprocess.run(['git', 'clone',
        'https://github.com/ayushmantrivedi/evolutionary-neural-network.git'
    ], check=True)

os.chdir('/content/evolutionary-neural-network')
sys.path.insert(0, '/content/evolutionary-neural-network')

# Import
print("📚 Loading modules...")
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot
print("✅ Modules loaded\n")

# ⚡ PATCH: Disable attention layer (make it pass-through)
print("🔧 Patching attention layer...")
def passthrough_forward(self, x, train=True):
    """Bypass attention - just return input unchanged"""
    return x

# Apply patch to the attention layer class
from evonet.core import layers
layers.EvoAttentionLayer.forward = passthrough_forward
print("✅ Attention bypassed\n")

# Config
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20
GENERATIONS = 200

print(f"📊 {TICKER} ({START_DATE} to {END_DATE})")
print(f"🔄 {GENERATIONS} generations\n")

# Fetch data
print("📥 Fetching data...")
fetcher = DataFetcher(TICKER, start_date=START_DATE, end_date=END_DATE, provider="yf")
df = fetcher.fetch_data()
df = fetcher.process()
print(f"✅ {len(df)} data points\n")

# Environment
safe_end = len(df) - (WINDOW_SIZE * 3)
env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, safe_end), window_size=WINDOW_SIZE)
print(f"✅ Environment ready\n")

# Pilot
pilot = MemoryEvoPilot()
pilot.input_dim = WINDOW_SIZE * 10
pilot.output_dim = 3
print(f"✅ Pilot: {pilot.net.pop_size} genomes\n")

# Evaluation
def evaluate_genome(genome_idx: int) -> Tuple[float, float, float]:
    state, _ = env.reset()
    equity = 1.0
    equity_curve = [1.0]
    returns = []
    terminated = False
    steps = 0
    
    while not terminated and steps < 1000:
        action = pilot.get_action(state, genome_idx)  # Now works!
        state, reward, terminated, _, _ = env.step(action)
        equity *= np.exp(reward)
        equity_curve.append(equity)
        returns.append(reward)
        steps += 1
    
    returns = np.array(returns)
    
    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (np.mean(returns) / (downside_std + 1e-9)) * np.sqrt(252)
    
    # Max DD
    peak = 1.0
    max_dd = 0.0
    for val in equity_curve:
        if val > peak: peak = val
        dd = (peak - val) / peak
        if dd > max_dd: max_dd = dd
    
    # Fitness
    dd_penalty = max(0, (max_dd - 0.15) * 50.0)
    fitness = (sortino * 0.6) + ((equity - 1.0) * 0.01) - dd_penalty
    return fitness, sortino, max_dd

# Training
print("="*80)
print("🔥 TRAINING STARTED")
print("="*80 + "\n")

best_ever_fitness = -999.0
best_ever_idx = 0
history = []

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
    
    history.append({'gen': gen, 'fitness': best_fit, 
                   'sortino': best_sortino, 'dd': best_dd})
    
    if gen % 10 == 0 or gen == 1:
        print(f"Gen {gen:3d}/{GENERATIONS} | Fit: {best_fit:7.3f} | "
              f"Sortino: {best_sortino:6.2f} | MaxDD: {best_dd:5.1%}")
    
    if gen % 50 == 0:
        with open(f"checkpoint_{gen}.pkl", 'wb') as f:
            pickle.dump(pilot, f)
        print(f"    💾 checkpoint_{gen}.pkl")
    
    pilot.evolve(scores)

env.close()

# Save
print("\n" + "="*80)
print("🎉 COMPLETE!")
print("="*80 + "\n")

best_weights = pilot.get_flat_weights(best_ever_idx)
pilot.set_flat_weights(best_weights, 0)

with open("ultimate_brain_colab.pkl", 'wb') as f:
    pickle.dump(pilot, f)

with open("training_report.txt", 'w') as f:
    f.write(f"TRAINING REPORT\n{'='*80}\n\n")
    f.write(f"{TICKER} ({START_DATE} to {END_DATE})\n")
    f.write(f"Generations: {GENERATIONS}, Population: {pilot.net.pop_size}\n")
    f.write(f"Best Fitness: {best_ever_fitness:.3f}\n\n")
    f.write("FINAL 20 GENS:\n" + "-"*80 + "\n")
    for h in history[-20:]:
        f.write(f"Gen {h['gen']:3d}: Fit={h['fitness']:7.3f}, "
                f"Sortino={h['sortino']:6.2f}, DD={h['dd']:5.1%}\n")

print(f"✅ ultimate_brain_colab.pkl ({os.path.getsize('ultimate_brain_colab.pkl')/1024:.1f} KB)")
print(f"✅ training_report.txt")
print("\n📥 Download from Files panel on left")
print("🎯 Then: python validate_colab_brain.py ultimate_brain_colab.pkl")
print("="*80)
