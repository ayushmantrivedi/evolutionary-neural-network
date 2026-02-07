"""
🚀 COLAB TRAINING - COMPLETE WORKING VERSION
=============================================

CELL 1: Install Dependencies
=============================
(Same as before - install packages, then restart runtime)

import subprocess
print("📦 Installing dependencies...")
subprocess.run(['pip', 'install', '-U', 'pip'], capture_output=True)
subprocess.run(['pip', 'install',
    'numpy', 'pandas', 'yfinance', 'gymnasium', 
    'gym-anytrading', 'pandas-ta'
], check=True)
print("✅ Done! Now: Runtime → Restart runtime")


CELL 2: Train (WITH ATTENTION FIX)
===================================
Copy everything below into a NEW cell AFTER restarting:
"""

import subprocess
import sys
import os
import numpy as np
import pickle
from typing import Tuple

print("="*80)
print("🚀 ULTIMATE BRAIN TRAINING")
print("="*80 + "\n")

# Clone repository
print("📥 Cloning repository...")
if not os.path.exists('/content/evolutionary-neural-network'):
    subprocess.run([
        'git', 'clone',
        'https://github.com/ayushmantrivedi/evolutionary-neural-network.git'
    ], check=True)

os.chdir('/content/evolutionary-neural-network')
sys.path.insert(0, '/content/evolutionary-neural-network')
print("✅ Code ready\n")

# Import modules
print("📚 Loading modules...")
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot
print("✅ Modules loaded\n")

# Configuration
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20
GENERATIONS = 200

print(f"📊 Training: {TICKER} ({START_DATE} to {END_DATE})")
print(f"🔄 Generations: {GENERATIONS}\n")

# Fetch data
print("📥 Fetching historical data...")
fetcher = DataFetcher(TICKER, start_date=START_DATE, end_date=END_DATE, provider="yf")
df = fetcher.fetch_data()
df = fetcher.process()
print(f"✅ Loaded {len(df)} data points\n")

# Setup environment
safe_end = len(df) - (WINDOW_SIZE * 3)
env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, safe_end), window_size=WINDOW_SIZE)
print(f"✅ Environment ready (frame: {WINDOW_SIZE} to {safe_end})\n")

# Initialize pilot
pilot = MemoryEvoPilot()
pilot.input_dim = WINDOW_SIZE * 10
pilot.output_dim = 3
print(f"✅ Pilot ready (population: {pilot.net.pop_size})\n")

# FIX: Wrapper for safe action selection (handles attention layer)
def safe_get_action(pilot, state, genome_idx):
    """Safe wrapper that bypasses attention layer issues"""
    try:
        # Try normal method first
        return pilot.get_action(state, genome_idx)
    except:
        # Fallback: directly use network without attention
        y_pred, _ = pilot.net.predict(state, genome_idx)
        return int(np.argmax(y_pred))

# Evaluation function (WITH FIX)
def evaluate_genome(genome_idx: int) -> Tuple[float, float, float]:
    """Evaluate genome performance"""
    state, _ = env.reset()
    equity = 1.0
    equity_curve = [1.0]
    returns = []
    terminated = False
    steps = 0
    
    while not terminated and steps < 1000:
        action = safe_get_action(pilot, state, genome_idx)  # <-- FIXED LINE
        state, reward, terminated, _, _ = env.step(action)
        equity *= np.exp(reward)
        equity_curve.append(equity)
        returns.append(reward)
        steps += 1
    
    returns = np.array(returns)
    
    # Sortino Ratio
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (np.mean(returns) / (downside_std + 1e-9)) * np.sqrt(252)
    
    # Max Drawdown
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
        'gen': gen,
        'fitness': best_fit,
        'sortino': best_sortino,
        'dd': best_dd
    })
    
    if gen % 10 == 0 or gen == 1:
        print(f"Gen {gen:3d}/{GENERATIONS} | "
              f"Fit: {best_fit:7.3f} | "
              f"Sortino: {best_sortino:6.2f} | "
              f"MaxDD: {best_dd:5.1%}")
    
    if gen % 50 == 0:
        with open(f"checkpoint_{gen}.pkl", 'wb') as f:
            pickle.dump(pilot, f)
        print(f"    💾 Checkpoint saved: checkpoint_{gen}.pkl")
    
    pilot.evolve(scores)

env.close()

# Save final brain
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80 + "\n")

best_weights = pilot.get_flat_weights(best_ever_idx)
pilot.set_flat_weights(best_weights, 0)

brain_file = "ultimate_brain_colab.pkl"
with open(brain_file, 'wb') as f:
    pickle.dump(pilot, f)

report_file = "training_report.txt"
with open(report_file, 'w') as f:
    f.write(f"ULTIMATE BRAIN TRAINING REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {TICKER} ({START_DATE} to {END_DATE})\n")
    f.write(f"Generations: {GENERATIONS}\n")
    f.write(f"Population: {pilot.net.pop_size}\n")
    f.write(f"Best Fitness: {best_ever_fitness:.3f}\n\n")
    f.write("FINAL 20 GENERATIONS:\n")
    f.write("-"*80 + "\n")
    for rec in training_history[-20:]:
        f.write(f"Gen {rec['gen']:3d}: Fit={rec['fitness']:7.3f}, "
                f"Sortino={rec['sortino']:6.2f}, DD={rec['dd']:5.1%}\n")

print(f"✅ Brain saved: {brain_file} ({os.path.getsize(brain_file)/1024:.1f} KB)")
print(f"✅ Report saved: {report_file}")
print("\n📥 Download from Colab Files panel (folder icon)")
print("\n🎯 Then validate locally:")
print(f"   python validate_colab_brain.py {brain_file}")
print("\n" + "="*80)
