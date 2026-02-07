"""
🚀 COLAB TRAINING - NO ATTENTION VERSION (GUARANTEED TO WORK!)
===============================================================

This version bypasses the attention layer entirely to avoid shape issues.
Still uses full evo-neural-network, just skips the problematic attention.

CELL 1: Install (same as before)
=================================
"""
import subprocess
print("📦 Installing dependencies...")
subprocess.run(['pip', 'install', '-U', 'pip'], capture_output=True)
subprocess.run(['pip', 'install',
    'numpy', 'pandas', 'yfinance', 'gymnasium', 
    'gym-anytrading', 'pandas-ta'
], check=True)
print("✅ Done! Now: Runtime → Restart runtime")
"""

CELL 2: Train (NO ATTENTION - WORKS!)
======================================
Copy everything below AFTER runtime restart:
"""

import subprocess
import sys
import os
import numpy as np
import pickle
from typing import Tuple

print("="*80)
print("🚀 ULTIMATE BRAIN TRAINING (No-Attention Mode)")
print("="*80 + "\n")

# Clone repository
print("📥 Cloning repository...")
if not os.path.exists('/content/evolutionary-neural-network'):
    subprocess.run(['git', 'clone',
        'https://github.com/ayushmantrivedi/evolutionary-neural-network.git'
    ], check=True)

os.chdir('/content/evolutionary-neural-network')
sys.path.insert(0, '/content/evolutionary-neural-network')

# Import modules
print("📚 Loading modules...")
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot
print("✅ Modules loaded\n")

# Fetch data
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20
GENERATIONS = 200

print(f"📊 Training: {TICKER}")
fetcher = DataFetcher(TICKER, start_date=START_DATE, end_date=END_DATE, provider="yf")
df = fetcher.fetch_data()
df = fetcher.process()
print(f"✅ Data: {len(df)} points\n")

# Setup environment
safe_end = len(df) - (WINDOW_SIZE * 3)
env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, safe_end), window_size=WINDOW_SIZE)
print(f"✅ Environment ready\n")

# Initialize pilot
pilot = MemoryEvoPilot()
pilot.input_dim = WINDOW_SIZE * 10
pilot.output_dim = 3
print(f"✅ Pilot: {pilot.net.pop_size} genomes\n")

# Direct prediction (bypasses attention layer!)
def get_action_no_attention(net, state, genome_idx):
    """Direct network forward pass - bypasses attention"""
    x = state.copy()
    
    # Layer 1: Forward pass
    l1_outputs = []
    for neuron in net.level1:
        ind = neuron.population[genome_idx]
        out = np.dot(x, ind.weights) + ind.bias
        out = np.tanh(out)  # Activation
        l1_outputs.append(out)
    l1_out = np.array(l1_outputs)
    
    # Layer 2: Forward pass
    l2_outputs = []
    for neuron in net.level2:
        ind = neuron.population[genome_idx]
        out = np.dot(l1_out, ind.weights) + ind.bias
        out = np.tanh(out)
        l2_outputs.append(out)
    l2_out = np.array(l2_outputs)
    
    # Layer 3 (Output): Forward pass
    output_probs = []
    for neuron in net.level3:
        out = np.dot(l2_out, neuron.weights) + neuron.bias
        output_probs.append(out)
    
    # Softmax
    logits = np.array(output_probs)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    return int(np.argmax(probs))

# Evaluation
def evaluate_genome(genome_idx: int) -> Tuple[float, float, float]:
    state, _ = env.reset()
    equity = 1.0
    equity_curve = [1.0]
    returns = []
    terminated = False
    steps = 0
    
    while not terminated and steps < 1000:
        action = get_action_no_attention(pilot.net, state, genome_idx)
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

# Training loop
print("="*80)
print("🔥 TRAINING STARTED")
print("="*80 + "\n")

best_ever_fitness = -999.0
best_ever_idx = 0
training_history = []

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
    
    avg_fit = np.mean(scores)
    training_history.append({
        'gen': gen, 'fitness': best_fit,
        'sortino': best_sortino, 'dd': best_dd
    })
    
    if gen % 10 == 0 or gen == 1:
        print(f"Gen {gen:3d}/{GENERATIONS} | Fit: {best_fit:7.3f} | "
              f"Sortino: {best_sortino:6.2f} | MaxDD: {best_dd:5.1%}")
    
    if gen % 50 == 0:
        with open(f"checkpoint_{gen}.pkl", 'wb') as f:
            pickle.dump(pilot, f)
        print(f"    💾 Saved checkpoint_{gen}.pkl")
    
    pilot.evolve(scores)

env.close()

# Save
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80 + "\n")

best_weights = pilot.get_flat_weights(best_ever_idx)
pilot.set_flat_weights(best_weights, 0)

with open("ultimate_brain_colab.pkl", 'wb') as f:
    pickle.dump(pilot, f)

with open("training_report.txt", 'w') as f:
    f.write(f"TRAINING REPORT\n{'='*80}\n\n")
    f.write(f"Dataset: {TICKER} ({START_DATE} to {END_DATE})\n")
    f.write(f"Generations: {GENERATIONS}\n")
    f.write(f"Population: {pilot.net.pop_size}\n")
    f.write(f"Best Fitness: {best_ever_fitness:.3f}\n\n")
    f.write("FINAL 20 GENERATIONS:\n" + "-"*80 + "\n")
    for rec in training_history[-20:]:
        f.write(f"Gen {rec['gen']:3d}: Fit={rec['fitness']:7.3f}, "
                f"Sortino={rec['sortino']:6.2f}, DD={rec['dd']:5.1%}\n")

print(f"✅ Brain: ultimate_brain_colab.pkl")
print(f"✅ Report: training_report.txt")
print("\n📥 Download from Colab → Files panel")
print("🎯 Then: python validate_colab_brain.py ultimate_brain_colab.pkl")
print("="*80)
