"""
🚀 ULTIMATE BRAIN TRAINER - GOOGLE COLAB EDITION
=================================================

COPY-PASTE THIS ENTIRE CELL INTO GOOGLE COLAB AND RUN!

This is a COMPLETE, STANDALONE script that:
✅ Clones your GitHub repo automatically
✅ Uses your existing MemoryEvoPilot code (no recreation issues!)
✅ Trains for 200 generations (~2-3 hours on T4 GPU)
✅ Saves ultimate_brain_colab.pkl for download

NO FILES TO UPLOAD - JUST COPY AND RUN!
"""

# ==============================================================================
# STEP 1: SETUP & DEPENDENCIES
# ==============================================================================

import subprocess
import sys
import os

print("="*80)
print("🚀 ULTIMATE BRAIN TRAINING - STARTING SETUP")
print("="*80 + "\n")

# Install dependencies with proper NumPy version
print("📦 Installing dependencies...")

# First, upgrade NumPy to latest stable version
subprocess.run(['pip', 'install', '--upgrade', 'numpy'], check=True)

# Then install other packages (they'll use the correct NumPy)
subprocess.run([
    'pip', 'install', '-q',
    'gymnasium', 'gym-anytrading', 'yfinance', 'pandas'
], check=True)

# Install pandas-ta separately (can be picky about versions)
subprocess.run(['pip', 'install', '-q', 'pandas-ta'], check=False)  # Don't fail if it has issues

print("✅ Dependencies installed\n")

# Clone repo (if not already cloned)
print("📥 Fetching code from GitHub...")
if os.path.exists('/content/evolutionary-neural-network'):
    print("   ⚠️  Repo already exists. Pulling latest changes...")
    os.chdir('/content/evolutionary-neural-network')
    subprocess.run(['git', 'pull'], check=False)  # Don't fail if pull doesn't work
else:
    subprocess.run([
        'git', 'clone',
        'https://github.com/ayushmantrivedi/evolutionary-neural-network.git',
        '/content/evolutionary-neural-network'
    ], check=True)
    
os.chdir('/content/evolutionary-neural-network')
sys.path.insert(0, '/content/evolutionary-neural-network')
print("✅ Code ready at: /content/evolutionary-neural-network\n")

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
except:
    print("⚠️  GPU check skipped (PyTorch not installed)\n")

# ==============================================================================
# STEP 2: IMPORTS
# ==============================================================================

print("📚 Loading modules...")
import numpy as np
import pickle
import pandas as pd
from typing import Tuple

from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot
print("✅ All modules loaded\n")

# ==============================================================================
# STEP 3: CONFIGURATION
# ==============================================================================

TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20
TOTAL_GENERATIONS = 200
CHECKPOINT_EVERY = 50

print("="*80)
print("🧠 TRAINING CONFIGURATION")
print("="*80)
print(f"📊 Ticker:       {TICKER}")
print(f"📅 Date Range:   {START_DATE} to {END_DATE}")
print(f"🔄 Generations:  {TOTAL_GENERATIONS}")
print(f"💾 Checkpoints:  Every {CHECKPOINT_EVERY} generations")
print("="*80 + "\n")

# ==============================================================================
# STEP 4: DATA PREPARATION
# ==============================================================================

print("📥 Fetching historical data...")
fetcher = DataFetcher(TICKER, start_date=START_DATE, end_date=END_DATE, provider="yf")
df = fetcher.fetch_data()
df = fetcher.process()
print(f"✅ Loaded {len(df)} data points")
print(f"   Date range: {df.index[0]} to {df.index[-1]}\n")

# ==============================================================================
# STEP 5: ENVIRONMENT SETUP
# ==============================================================================

print("🌍 Initializing trading environment...")
safe_end = len(df) - (WINDOW_SIZE * 3)
if safe_end <= WINDOW_SIZE:
    raise ValueError(f"Dataset too small! Need at least {WINDOW_SIZE * 4} rows, got {len(df)}")
    
env = FinancialRegimeEnv(
    df, 
    frame_bound=(WINDOW_SIZE, safe_end),
    window_size=WINDOW_SIZE,
    fee=0.001,
    slippage_std=0.0001
)
print(f"✅ Environment ready")
print(f"   Frame bound: {WINDOW_SIZE} to {safe_end}")
print(f"   Episode length: ~{safe_end - WINDOW_SIZE} steps\n")

# ==============================================================================
# STEP 6: PILOT INITIALIZATION
# ==============================================================================

print("🧠 Initializing MemoryEvoPilot...")
pilot = MemoryEvoPilot()
pilot.input_dim = WINDOW_SIZE * 10  # 10 features per timestep
pilot.output_dim = 3  # Short, Neutral, Long

print(f"✅ Pilot initialized")
print(f"   Population size: {pilot.net.pop_size}")
print(f"   Input dimension: {pilot.input_dim}")
print(f"   Output classes: {pilot.output_dim}\n")

# ==============================================================================
# STEP 7: EVALUATION FUNCTION
# ==============================================================================

def evaluate_genome(genome_idx: int, max_steps: int = 1000) -> Tuple[float, float, float]:
    """
    Evaluate a single genome's trading performance.
    
    Returns:
        (fitness, sortino_ratio, max_drawdown)
    """
    state, _ = env.reset()
    equity = 1.0
    equity_curve = [1.0]
    returns = []
    terminated = False
    steps = 0
    
    while not terminated and steps < max_steps:
        # Get action from this genome
        action = pilot.get_action(state, genome_idx)
        
        # Execute in environment
        state, reward, terminated, _, _ = env.step(action)
        
        # Track performance
        equity *= np.exp(reward)
        equity_curve.append(equity)
        returns.append(reward)
        steps += 1
    
    returns = np.array(returns)
    
    # Calculate metrics
    total_return = (equity - 1.0) * 100  # Percentage
    
    # Sortino Ratio (annualized, downside deviation only)
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (np.mean(returns) / (downside_std + 1e-9)) * np.sqrt(252)
    
    # Maximum Drawdown
    peak = 1.0
    max_dd = 0.0
    for equity_val in equity_curve:
        if equity_val > peak:
            peak = equity_val
        dd = (peak - equity_val) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Fitness Function: Sortino-focused with drawdown penalty
    dd_penalty = 0.0
    if max_dd > 0.15:  # Penalize drawdowns > 15%
        dd_penalty = (max_dd - 0.15) * 50.0
    
    fitness = (sortino * 0.6) + (total_return * 0.01) - dd_penalty
    
    return fitness, sortino, max_dd

# ==============================================================================
# STEP 8: TRAINING LOOP
# ==============================================================================

print("="*80)
print("🔥 STARTING TRAINING")
print("="*80 + "\n")

best_ever_fitness = -999.0
best_ever_genome_idx = 0
training_history = []

for generation in range(1, TOTAL_GENERATIONS + 1):
    # Evaluate all genomes
    fitness_scores = []
    generation_stats = []
    
    for genome_idx in range(pilot.net.pop_size):
        fit, sortino, max_dd = evaluate_genome(genome_idx)
        fitness_scores.append(fit)
        generation_stats.append((fit, sortino, max_dd))
    
    # Find best genome of this generation
    best_idx_this_gen = np.argmax(fitness_scores)
    best_fit, best_sortino, best_dd = generation_stats[best_idx_this_gen]
    
    # Track all-time best
    if best_fit > best_ever_fitness:
        best_ever_fitness = best_fit
        best_ever_genome_idx = best_idx_this_gen
    
    # Log progress
    avg_fitness = np.mean(fitness_scores)
    training_history.append({
        'generation': generation,
        'best_fitness': best_fit,
        'avg_fitness': avg_fitness,
        'best_sortino': best_sortino,
        'best_drawdown': best_dd
    })
    
    # Print every 10 generations
    if generation % 10 == 0 or generation == 1:
        print(f"Gen {generation:3d}/{TOTAL_GENERATIONS} | "
              f"Best Fit: {best_fit:7.3f} | "
              f"Avg Fit: {avg_fitness:7.3f} | "
              f"Sortino: {best_sortino:6.2f} | "
              f"MaxDD: {best_dd:5.1%}")
    
    # Save checkpoint
    if generation % CHECKPOINT_EVERY == 0:
        checkpoint_path = f"checkpoint_gen_{generation}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(pilot, f)
        print(f"    💾 Checkpoint saved: {checkpoint_path}")
    
    # Evolve population for next generation
    pilot.evolve(fitness_scores)

env.close()

# ==============================================================================
# STEP 9: SAVE FINAL BRAIN
# ==============================================================================

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80 + "\n")

print(f"🏆 Best Fitness Ever: {best_ever_fitness:.3f}")
print(f"   Achieved in Generation: {training_history[best_ever_genome_idx]['generation']}")

# Copy best genome weights to index 0 for easy loading
print("\n📦 Preparing final brain for export...")
best_weights = pilot.get_flat_weights(best_ever_genome_idx)
pilot.set_flat_weights(best_weights, 0)

# Save brain file
brain_filename = "ultimate_brain_colab.pkl"
with open(brain_filename, 'wb') as f:
    pickle.dump(pilot, f)

print(f"✅ Brain saved: {brain_filename}")
print(f"   File size: {os.path.getsize(brain_filename) / 1024:.1f} KB")

# Save training report
report_filename = "training_report.txt"
with open(report_filename, 'w') as f:
    f.write("ULTIMATE BRAIN TRAINING REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {TICKER} ({START_DATE} to {END_DATE})\n")
    f.write(f"Total Generations: {TOTAL_GENERATIONS}\n")
    f.write(f"Population Size: {pilot.net.pop_size}\n")
    f.write(f"Best Fitness: {best_ever_fitness:.3f}\n\n")
    f.write("FINAL 20 GENERATIONS:\n")
    f.write("-"*80 + "\n")
    for record in training_history[-20:]:
        f.write(f"Gen {record['generation']:3d}: "
                f"Fit={record['best_fitness']:7.3f}, "
                f"Sortino={record['best_sortino']:6.2f}, "
                f"DD={record['best_drawdown']:5.1%}\n")

print(f"✅ Report saved: {report_filename}\n")

print("="*80)
print("📥 DOWNLOAD YOUR FILES")
print("="*80)
print("1. Click the folder icon 📁 on the left sidebar")
print(f"2. Find and download: {brain_filename}")
print(f"3. Also download: {report_filename}")
print("\n🎯 NEXT STEPS:")
print("   • Upload the brain to your local project")
print("   • Run: python validate_colab_brain.py ultimate_brain_colab.pkl")
print("   • Use the metrics in your investor pitch!")
print("="*80)
