
import numpy as np
import os
import sys
import logging
import pickle
import pandas as pd
from typing import List, Tuple, Dict

# Core Imports
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv
from train_memory_autopilot import MemoryEvoPilot
from evonet.core.network import MultiClassEvoNet
from evonet.core.memory import DirectionalMemory

# Configuration for The Ultimate Run
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20
POP_SIZE = 50
GENS_PER_STAGE = 20 # 20 gens per curriculum stage

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | [ULTIMATE] %(message)s')
logger = logging.getLogger("UltimateTrainer")

class RegimeCurriculum:
    """
    Manages the 'Lesson Plan' for the AI.
    Splits history into difficulty levels.
    """
    def __init__(self, full_df: pd.DataFrame):
        self.full_df = full_df
        
    def get_stage_data(self, stage_name: str) -> pd.DataFrame:
        if stage_name == "Stage1_Foundations":
            # Strong Trends (2017 style, here using 2020-2021)
            # Goal: Learn to capture massive moves (Greed)
            mask = (self.full_df.index >= "2020-10-01") & (self.full_df.index <= "2021-04-01")
            return self.full_df.loc[mask]
            
        elif stage_name == "Stage2_Complexity":
            # Choppy / Range Bound (2018-2019, 2023)
            # Goal: Learn to sit tight and filter noise (Patience)
            mask = (self.full_df.index >= "2023-01-01") & (self.full_df.index <= "2023-10-01")
            return self.full_df.loc[mask]
            
        elif stage_name == "Stage3_Survival":
            # Crashes / Bear Markets (2022)
            # Goal: Learn to Short and Protect Capital (Fear)
            mask = (self.full_df.index >= "2022-01-01") & (self.full_df.index <= "2022-12-31")
            return self.full_df.loc[mask]
            
        return self.full_df

class UltimateTrainer:
    def __init__(self):
        self.pilot = self._init_pilot()
        
    def _init_pilot(self):
        pilot = MemoryEvoPilot()
        pilot.input_dim = WINDOW_SIZE * 10
        pilot.output_dim = 3
        pilot.net = MultiClassEvoNet(pilot.input_dim, pilot.output_dim) # Removed invalid arg
        
        # Init Memory
        pilot.flat_init = pilot.get_flat_weights(0)
        pilot.memory = DirectionalMemory(pilot.flat_init)
        return pilot

    def evaluate_genome(self, env: FinancialRegimeEnv, genome_idx: int) -> Tuple[float, float, float]:
        """Runs one episode and returns (Fitness, Sortino, MaxDD)"""
        state, _ = env.reset()
        equity = 1.0
        equity_curve = [1.0]
        returns = []
        terminated = False
        steps = 0
        
        while not terminated and steps < 500:
            action = self.pilot.get_action(state, genome_idx)
            state, r, terminated, _, _ = env.step(action)
            equity *= np.exp(r)
            equity_curve.append(equity)
            returns.append(r)
            steps += 1
            
        returns = np.array(returns)
        
        # Metrics
        total_ret = (equity - 1.0)
        
        # Sortino
        downside = returns[returns < 0]
        down_dev = np.std(downside) if len(downside) > 0 else 1e-6
        sortino = np.mean(returns) / (down_dev + 1e-9) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # MaxDD
        peak = 1.0
        max_dd = 0.0
        for val in equity_curve:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
            
        # Fitness: Mix of Profit and Safety
        # Heavily penalize DD > 20%
        penalty = 0
        if max_dd > 0.20:
             penalty = (max_dd - 0.20) * 10.0 # Steep penalty
        
        fitness = total_ret + (sortino * 0.1) - penalty
        return fitness, sortino, max_dd

    def train_stage(self, stage_name: str, curriculum: RegimeCurriculum):
        print(f"\n[CURRICULUM] Entering {stage_name}...")
        df = curriculum.get_stage_data(stage_name)
        
        if len(df) < 120:
            print(f"   [SKIP] Stage {stage_name} has only {len(df)} rows. Skipping.")
            return

        # Fix IndexError: Reduce bound MAX (80)
        safe_end = len(df) - 80
        if safe_end <= WINDOW_SIZE:
             safe_end = len(df) - 1 
             
        print(f"   [DEBUG] Stage Data Len: {len(df)} | FrameBound: ({WINDOW_SIZE}, {safe_end})")
        sys.stdout.flush()
        env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, safe_end), window_size=WINDOW_SIZE, fee=0.001)
        
        best_gen_fitness = -999
        best_genome_idx = 0
        
        for gen in range(1, GENS_PER_STAGE + 1):
            scores = []
            stats = []
            
            # Use dynamic pop_size from network to be safe
            for i in range(self.pilot.net.pop_size):
                fit, sort, mdd = self.evaluate_genome(env, i)
                scores.append(fit)
                stats.append((fit, sort, mdd))
            
            # Evolution Step
            self.pilot.evolve(scores)
            
            # Stats
            best_idx = np.argmax(scores)
            best_fit, best_sort, best_mdd = stats[best_idx]
            
            if best_fit > best_gen_fitness:
                best_gen_fitness = best_fit
                best_genome_idx = best_idx
                
            if gen % 5 == 0:
                print(f"   Gen {gen}: Fit={best_fit:.3f} | Sortino={best_sort:.2f} | MaxDD={best_mdd:.1%}")
                
        # Store "Lesson Learned" in Long Term Memory
        print(f"[MEMORY] Consolidating {stage_name} experience...")
        
        # Explicitly set the 0-th individual to the Best Weights for saving/inference
        best_ind = self.pilot.net.level1[0].population[best_genome_idx]
        # We need to copy best weights to index 0 for ALL neurons in ALL layers
        # Helper function in MemoryEvoPilot would be better, but doing it manually here:
        self.pilot.set_flat_weights(self.pilot.get_flat_weights(best_genome_idx), 0)
        
        env.close()

    def run(self):
        print("STARTING ULTIMATE TRAINING RUN")
        print(f"   Data: {START_DATE} to {END_DATE}")
        
        # 1. Fetch Deep Data
        fetcher = DataFetcher(TICKER, start_date=START_DATE, end_date=END_DATE, provider="yf")
        df = fetcher.fetch_data()
        df = fetcher.process()
        
        # 2. SIMPLIFIED TRAINING: Use FULL Dataset
        # (Curriculum stages were too small for gym-anytrading)
        print("\\n[TRAINING] Using Full 2018-2024 Dataset (Simplified Strategy)")
        
        safe_end = len(df) - (WINDOW_SIZE * 3)
        if safe_end <= WINDOW_SIZE:
            print("[ERROR] Dataset too small.")
            return
            
        env = FinancialRegimeEnv(df, frame_bound=(WINDOW_SIZE, safe_end), window_size=WINDOW_SIZE, fee=0.001)
        
        best_gen_fitness = -999
        best_genome_idx = 0
        
        for gen in range(1, GENS_PER_STAGE + 1):
            scores = []
            stats = []
            
            for i in range(self.pilot.net.pop_size):
                fit, sort, mdd = self.evaluate_genome(env, i)
                scores.append(fit)
                stats.append((fit, sort, mdd))
            
            self.pilot.evolve(scores)
            
            best_idx = np.argmax(scores)
            best_fit, best_sort, best_mdd = stats[best_idx]
            
            if best_fit > best_gen_fitness:
                best_gen_fitness = best_fit
                best_genome_idx = best_idx
                
            if gen % 5 == 0:
                print(f"   Gen {gen}: Fit={best_fit:.3f} | Sortino={best_sort:.2f} | MaxDD={best_mdd:.1%}")
                
        self.pilot.set_flat_weights(self.pilot.get_flat_weights(best_genome_idx), 0)
        print(f"\\n[BEST] Fitness: {best_gen_fitness:.3f}")
        env.close()
        
        # 3. Save Ultimate Brain
        with open("ultimate_brain.pkl", "wb") as f:
            pickle.dump(self.pilot, f)
            
        print("\n\nTRAINING COMPLETE. 'ultimate_brain.pkl' Saved.")
        print("   The model has graduated Black Belt.")

if __name__ == "__main__":
    trainer = UltimateTrainer()
    trainer.run()
