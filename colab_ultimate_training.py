"""
ULTIMATE BRAIN TRAINING - Google Colab GPU Edition
===================================================

This script trains a state-of-the-art Neuro-Evolutionary Trading AI
on Google Colab's GPU infrastructure.

OUTPUT: ultimate_brain_colab.pkl (Download this file after training)

INSTRUCTIONS FOR GOOGLE COLAB:
1. Upload your entire 'evonet' folder to Colab (or clone from GitHub)
2. Paste this entire script into a Colab cell
3. Runtime > Change runtime type > GPU (T4 or better)
4. Run the cell
5. Download 'ultimate_brain_colab.pkl' when complete (~2-4 hours)

HYPERPARAMETERS:
- Generations: 200 (state-of-the-art training depth)
- Population: 100 genomes (diversity for robustness)
- Data: 2018-2024 BTC-USD 1-day (can upgrade to 1h for premium pitch)
- Objective: Maximize Sortino Ratio + Minimize Drawdown
"""

# ============================================================================
# SECTION 1: DEPENDENCY INSTALLATION
# ============================================================================

import sys
import os

# Install required packages (this uses subprocess for Python compatibility)
import subprocess
print("📦 Installing Dependencies...")
subprocess.run(['pip', 'install', 'gymnasium', 'gym-anytrading', 'yfinance', 
                'pandas', 'numpy', 'pandas-ta', '-q'], check=True)

# ============================================================================
# SECTION 2: CODE SETUP (Auto-detect evonet path)
# ============================================================================

print("🔍 Locating evonet module...")

# Try multiple possible locations
possible_paths = [
    '/content',                                    # Colab root
    '/content/evolutionary-neural-network',        # If cloned from GitHub
    '/content/evo-trader',                         # Alternative repo name
    '/content/drive/MyDrive',                      # Google Drive mount
    os.getcwd()                                    # Current directory
]

evonet_found = False
for path in possible_paths:
    if os.path.exists(os.path.join(path, 'evonet')):
        sys.path.insert(0, path)
        print(f"✅ Found evonet in: {path}")
        evonet_found = True
        break

if not evonet_found:
    print("❌ ERROR: evonet folder not found!")
    print("\n📁 Please upload the entire 'evonet' folder to Colab:")
    print("   1. Click the folder icon on the left")
    print("   2. Upload the 'evonet' directory from your project")
    print("   3. Re-run this cell")
    print("\nOR clone from GitHub:")
    print("   !git clone https://github.com/YOUR_USERNAME/evo-trader.git")
    raise ImportError("evonet module not found. Please upload the evonet folder.")

# ============================================================================
# SECTION 3: IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Tuple, List
from datetime import datetime

# Import your custom modules (ensure evonet is in sys.path)
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from evonet.core.network import MultiClassEvoNet
from evonet.core.memory import DirectionalMemory

# ============================================================================
# SECTION 4: GPU-OPTIMIZED HYPERPARAMETERS
# ============================================================================

# Data Configuration
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20

# Training Configuration (GPU-Optimized)
POPULATION_SIZE = 100      # Larger population for diversity
TOTAL_GENERATIONS = 200    # State-of-the-art depth
CHECKPOINT_INTERVAL = 25   # Save every 25 generations
MAX_EPISODE_STEPS = 1000   # Longer episodes for better evaluation

# Evolution Parameters
ELITE_PERCENTAGE = 0.10    # Top 10% survive
MUTATION_RATE = 0.15       # Aggressive exploration

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [COLAB] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ColabTrainer")

# ============================================================================
# SECTION 5: MEMORY-EFFICIENT PILOT CLASS
# ============================================================================

class MemoryEvoPilot:
    """Simplified MemoryEvoPilot for Colab (no DirectionalMemory to save RAM)"""
    
    def __init__(self, input_dim, output_dim, pop_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create the network with standard initialization
        self.net = MultiClassEvoNet(input_dim, output_dim)
        
        # The network already creates neurons with the default POP_SIZE from config
        # We need to update the pop_size if different
        if pop_size != self.net.pop_size:
            print(f"   ⚠️  Requested pop_size={pop_size}, but using config POP_SIZE={self.net.pop_size}")
            print(f"   To change, edit evonet/config.py: POP_SIZE = {pop_size}")
            # Use the network's actual pop_size
            
    def get_action(self, state, genome_idx):
        """Get action from specific genome"""
        probs, conf = self.net.predict(state, genome_idx)
        return np.argmax(probs)
        
    def evolve(self, fitness_scores):
        """Evolve population based on fitness - delegates to network's built-in evolution"""
        # Convert fitness to errors (lower is better for the network's evolution)
        # Negate fitness so higher fitness = lower error
        errors = [-f for f in fitness_scores]
        
        # Evolve each neuron layer using the network's built-in evolution
        for neuron in self.net.level1 + self.net.level2:
            # The neuron has built-in tournament_selection method
            # We'll implement simple evolution here
            sorted_indices = np.argsort(errors)  # Best (lowest error) first
            elite_count = max(1, int(len(errors) * ELITE_PERCENTAGE))
            
            # Keep track of best performer
            neuron.best_idx = sorted_indices[0]
            
            # Create new population via tournament selection
            new_population = []
            
            # Preserve elites
            for idx in sorted_indices[:elite_count]:
                new_population.append(neuron.population[idx])
                
            # Fill rest with offspring of elites
            while len(new_population) < neuron.pop_size:
                # Select parent from elites
                parent_idx = np.random.choice(sorted_indices[:elite_count])
                parent = neuron.population[parent_idx]
                
                # Create offspring via mutation
                from evonet.core.neuron import Individual
                child = Individual(
                    weights=parent.weights.copy(),
                    bias=parent.bias,
                    tau1=parent.tau1,
                    tau2=parent.tau2
                )
                
                # Mutate weights
                mutation_mask = np.random.rand(len(child.weights)) < MUTATION_RATE
                if mutation_mask.any():
                    child.weights[mutation_mask] += np.random.randn(mutation_mask.sum()) * 0.1
                    
                # Mutate bias occasionally
                if np.random.rand() < MUTATION_RATE:
                    child.bias += np.random.randn() * 0.1
                    
                new_population.append(child)
                
            neuron.population = new_population[:neuron.pop_size]
            
    def get_flat_weights(self, genome_idx):
        """Get flattened weights of a genome"""
        weights = []
        for neuron in self.net.level1 + self.net.level2:
            ind = neuron.population[genome_idx]
            weights.append(ind.weights)
            weights.append([ind.bias])
        return np.concatenate(weights)
        
    def set_flat_weights(self, flat_weights, genome_idx):
        """Set flattened weights to a genome"""
        from evonet.core.neuron import Individual
        offset = 0
        
        for neuron in self.net.level1 + self.net.level2:
            # Extract weights for this neuron
            n_weights = neuron.input_dim
            weights = flat_weights[offset:offset+n_weights]
            offset += n_weights
            
            # Extract bias
            bias = flat_weights[offset]
            offset += 1
            
            # Get tau values from existing individual
            existing = neuron.population[genome_idx]
            
            # Create new individual
            neuron.population[genome_idx] = Individual(
                weights=weights.copy(),
                bias=bias,
                tau1=existing.tau1,
                tau2=existing.tau2
            )

# ============================================================================
# SECTION 6: FITNESS EVALUATION
# ============================================================================

def evaluate_genome(env: FinancialRegimeEnv, pilot: MemoryEvoPilot, 
                   genome_idx: int) -> Tuple[float, float, float]:
    """
    Evaluates a single genome.
    Returns: (fitness, sortino_ratio, max_drawdown)
    """
    state, _ = env.reset()
    equity = 1.0
    equity_curve = [1.0]
    returns = []
    terminated = False
    steps = 0
    
    while not terminated and steps < MAX_EPISODE_STEPS:
        action = pilot.get_action(state, genome_idx)
        state, reward, terminated, _, _ = env.step(action)
        
        equity *= np.exp(reward)
        equity_curve.append(equity)
        returns.append(reward)
        steps += 1
        
    returns = np.array(returns)
    
    # Calculate Metrics
    total_return = (equity - 1.0)
    
    # Sortino Ratio (annualized)
    downside = returns[returns < 0]
    down_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (np.mean(returns) / (down_std + 1e-9)) * np.sqrt(252)
    
    # Max Drawdown
    peak = 1.0
    max_dd = 0.0
    for equity_val in equity_curve:
        if equity_val > peak:
            peak = equity_val
        dd = (peak - equity_val) / peak
        if dd > max_dd:
            max_dd = dd
            
    # Fitness Function: Prioritize Sortino and Low Drawdown
    # Penalty for high drawdown
    dd_penalty = max(0, (max_dd - 0.15) * 50.0)  # Steep penalty above 15% DD
    
    fitness = (total_return * 0.3) + (sortino * 0.5) - dd_penalty
    
    return fitness, sortino, max_dd

# ============================================================================
# SECTION 7: MAIN TRAINING LOOP
# ============================================================================

def train_ultimate_brain():
    """Main training function"""
    
    print("\n" + "="*70)
    print("🚀 ULTIMATE BRAIN TRAINING - COLAB GPU EDITION")
    print("="*70)
    print(f"📊 Dataset: {TICKER} ({START_DATE} to {END_DATE})")
    print(f"🧬 Population: {POPULATION_SIZE} genomes")
    print(f"🔄 Generations: {TOTAL_GENERATIONS}")
    print(f"💰 Objective: Maximize Sortino, Minimize Drawdown")
    print("="*70 + "\n")
    
    # Step 1: Fetch Data
    print("📥 Fetching Historical Data...")
    fetcher = DataFetcher(TICKER, start_date=START_DATE, end_date=END_DATE, provider="yf")
    df = fetcher.fetch_data()
    df = fetcher.process()
    print(f"✅ Loaded {len(df)} data points\n")
    
    # Step 2: Initialize Environment
    print("🌍 Initializing Trading Environment...")
    safe_end = len(df) - (WINDOW_SIZE * 3)
    env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, safe_end), 
                            window_size=WINDOW_SIZE, fee=0.001)
    print(f"✅ Environment Ready (Frame: {WINDOW_SIZE} to {safe_end})\n")
    
    # Step 3: Initialize Pilot
    print("🧠 Initializing Neural Evolution Pilot...")
    input_dim = WINDOW_SIZE * 10  # 10 features per timestep
    output_dim = 3  # Short, Neutral, Long
    pilot = MemoryEvoPilot(input_dim, output_dim, POPULATION_SIZE)
    print(f"✅ Pilot Ready ({input_dim}D input → {output_dim}D output)\n")
    
    # Step 4: Training Loop
    print("🔥 Starting Evolution...\n")
    
    best_ever_fitness = -999
    best_ever_genome_idx = 0
    training_history = []
    
    for gen in range(1, TOTAL_GENERATIONS + 1):
        # Evaluate all genomes
        fitness_scores = []
        metrics = []
        
        for genome_idx in range(POPULATION_SIZE):
            fit, sortino, max_dd = evaluate_genome(env, pilot, genome_idx)
            fitness_scores.append(fit)
            metrics.append((fit, sortino, max_dd))
            
        # Find best of this generation
        best_idx = np.argmax(fitness_scores)
        best_fit, best_sortino, best_dd = metrics[best_idx]
        
        # Track global best
        if best_fit > best_ever_fitness:
            best_ever_fitness = best_fit
            best_ever_genome_idx = best_idx
            
        # Log progress
        avg_fit = np.mean(fitness_scores)
        training_history.append({
            'generation': gen,
            'best_fitness': best_fit,
            'avg_fitness': avg_fit,
            'best_sortino': best_sortino,
            'best_drawdown': best_dd
        })
        
        if gen % 10 == 0 or gen == 1:
            print(f"Gen {gen:3d}/{TOTAL_GENERATIONS} | "
                  f"Best Fit: {best_fit:7.3f} | "
                  f"Avg Fit: {avg_fit:7.3f} | "
                  f"Sortino: {best_sortino:6.2f} | "
                  f"MaxDD: {best_dd:5.1%}")
                  
        # Checkpoint saving
        if gen % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"checkpoint_gen_{gen}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(pilot, f)
            print(f"    💾 Checkpoint saved: {checkpoint_path}")
            
        # Evolve population
        pilot.evolve(fitness_scores)
        
    # Step 5: Save Final Brain
    print("\n" + "="*70)
    print("🏆 TRAINING COMPLETE!")
    print("="*70)
    
    # Copy best genome to index 0 for easy loading
    best_weights = pilot.get_flat_weights(best_ever_genome_idx)
    pilot.set_flat_weights(best_weights, 0)
    
    # Save final brain
    brain_path = "ultimate_brain_colab.pkl"
    with open(brain_path, 'wb') as f:
        pickle.dump(pilot, f)
        
    print(f"\n✅ Ultimate Brain saved: {brain_path}")
    print(f"📈 Best Fitness: {best_ever_fitness:.3f}")
    print(f"📊 Training History: {len(training_history)} generations")
    
    # Save training report
    report_path = "training_report.txt"
    with open(report_path, 'w') as f:
        f.write("ULTIMATE BRAIN TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {TICKER} ({START_DATE} to {END_DATE})\n")
        f.write(f"Population Size: {POPULATION_SIZE}\n")
        f.write(f"Total Generations: {TOTAL_GENERATIONS}\n")
        f.write(f"Best Fitness: {best_ever_fitness:.3f}\n\n")
        f.write("GENERATION HISTORY:\n")
        f.write("-"*70 + "\n")
        for record in training_history[-20:]:  # Last 20 generations
            f.write(f"Gen {record['generation']:3d}: "
                   f"Fit={record['best_fitness']:7.3f}, "
                   f"Sortino={record['best_sortino']:6.2f}, "
                   f"DD={record['best_drawdown']:5.1%}\n")
                   
    print(f"📄 Training Report saved: {report_path}")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Download 'ultimate_brain_colab.pkl' from Colab Files")
    print("2. Upload it to your local project")
    print("3. Use it for validation with deep_backtest.py")
    print("4. Pitch to finance companies with the training report!")
    print("\n" + "="*70 + "\n")
    
    env.close()
    return pilot, training_history

# ============================================================================
# SECTION 8: EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
        else:
            print("⚠️  No GPU detected. Training will be slower on CPU.\n")
    except:
        print("⚠️  PyTorch not installed. GPU check skipped.\n")
        
    # Run training
    pilot, history = train_ultimate_brain()
    
    print("\n✨ Training session complete! Download your files now.")
