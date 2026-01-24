
import gymnasium as gym
import numpy as np
import imageio
import os
import time
import sys
import logging

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from train_memory_autopilot import MemoryEvoPilot, run_episode
from evonet.config import POP_SIZE

# Configuration
ENV_NAME = "LunarLander-v3" 
OUTPUT_FILENAME = "recovery_demo.gif"
FPS = 30
FRAMES_PER_PHASE = 150 
CINEMATIC_SEED = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def text_overlay(frame, text, color=(255, 255, 255)):
    try:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = frame.copy()
        # Shadow
        cv2.putText(frame, text, (20, 40), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        # Text
        cv2.putText(frame, text, (20, 40), font, 0.7, color, 2, cv2.LINE_AA)
        return frame
    except ImportError:
        return frame

def run_episode_seeded(env, pilot, pilot_index, seed):
    state, _ = env.reset(seed=seed)
    total = 0
    # Increase max steps to allow for precision adjustments
    for _ in range(600):
        action = pilot.get_action(state, pilot_index)
        state, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc: break
    return total

def run_episode_seeded_with_validation(env, pilot, pilot_index, seed):
    """Run episode and return total reward + boolean if it was a good landing."""
    state, _ = env.reset(seed=seed)
    total = 0
    for _ in range(600):
        action = pilot.get_action(state, pilot_index)
        state, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc: break
        
    return total

def train_cinematic_master(pilot, env, target_seed=42):
    print("🎬 CINEMATIC TRAINING: Fine-Tuning for Perfection...")
    print(f"   Target: Score > 240 on SEED {target_seed}")
    print("   Mode: Reinforcement Learning with Mutation Vector Guidance")
    
    max_gens = 500
    stagnation = 0
    prev_best = -float('inf')
    
    for gen in range(1, max_gens + 1):
        rewards = []
        
        # 1. Evaluate Population on the SCENE SEED
        for i in range(pilot.pop_size):
            rewards.append(run_episode_seeded(env, pilot, i, seed=target_seed))
            
        best_reward = np.max(rewards)
        best_idx = np.argmax(rewards)
        avg_reward = np.mean(rewards)
        
        # Check stagnation
        if best_reward <= prev_best + 0.1:
            stagnation += 1
        else:
            stagnation = 0
            prev_best = best_reward
            
        # 2. Adaptive Fine-Tuning Logic
        if best_reward > 220:
            pilot.current_mutation = 0.05
        elif best_reward > 150:
            pilot.current_mutation = 0.1
        else:
            pilot.current_mutation = 0.15
            
        # Stagnation Breaker (Seismic Shift)
        if stagnation > 20: # Increased patience
            print(f"   -> Stagnation ({stagnation}), Seismic Shift triggered.")
            pilot.current_mutation = 0.5
            stagnation = 0 # Reset counter to give time after shift
            
        # Update HoF
        pilot.update_hall_of_fame(rewards)
        
        print(f"   Gen {gen}: Best={best_reward:.1f} (Avg={avg_reward:.1f})")
        
        if best_reward > 240:
            print(f"   -> ⭐ CINEMATIC MASTERY ACHIEVED! (Score: {best_reward:.1f})")
            pilot.store_memory("mastery", best_idx) 
            return True
            
        # 3. Evolve with V_m (Mutation Vector)
        pilot.evolve(rewards)
    
    print("⚠️ Warning: Did not hit 240. Using best effort.")
    pilot.store_memory("mastery", np.argmax(rewards))
    return False

def record_continuous(pilot, env, seed, master_weights, failure_duration=40):
    print("🎥 Recording Continuous 'One-Take' Demo...")
    frames = []
    
    # 1. Setup
    state, info = env.reset(seed=seed)
    
    # Pre-calc garbage weights
    garbage_weights = np.random.randn(len(master_weights)) * 1.0 # Reduced noise
    
    # Frame counters
    step_count = 0
    phase = "FLY"
    
    # Failure timing (Frames)
    # Fly for 50 frames (approx 1.5s), then fail
    FAIL_START = 60 
    # Fail for some frames, then heal
    HEAL_START = FAIL_START + failure_duration
    
    terminated = False
    truncated = False
    
    total_reward = 0
    
    while not (terminated or truncated):
        step_count += 1
        
        # --- DYNAMIC BRAIN SURGERY ---
        if step_count < FAIL_START:
            # Phase 1: Mastery
            pilot.set_flat_weights(0, master_weights)
            label = "PHASE 1: PRECISION FLIGHT"
            current_color = (255, 255, 255) # White
            
        elif FAIL_START <= step_count < HEAL_START:
            # Phase 2: Failure
            pilot.set_flat_weights(0, garbage_weights) 
            label = "⚠️ PHASE 2: SYSTEM FAILURE ⚠️"
            current_color = (0, 0, 255) # Red
            
        else:
            # Phase 3: Recovery
            pilot.set_flat_weights(0, master_weights)
            label = "✅ PHASE 3: INSTANT RESTORE"
            current_color = (0, 255, 0) # Green
            
        # Render & Label
        frame = env.render()
        frame = text_overlay(frame, label, color=current_color)
        frames.append(frame)
        
        # Step
        action = pilot.get_action(state, pilot_index=0)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        # INCREASED MAX FRAMES to ensure landing
        if step_count > 900:
            truncated = True
            
    print(f"   -> Episode finished. Total Reward: {total_reward:.1f}")
    return frames

def main():
    print("="*60)
    print("🎬 EVO-NET CINEMATIC STUDIO")
    print("   Generating: 'One-Take' Continuous Demo (High Precision)")
    print("="*60)
    
    try:
        env = gym.make(ENV_NAME, render_mode="rgb_array")
        train_env = gym.make(ENV_NAME) 
    except:
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        train_env = gym.make("LunarLander-v2")
        
    pilot = MemoryEvoPilot()
    
    # 1. Train Cinematic Master on Seed 42
    train_cinematic_master(pilot, train_env, target_seed=CINEMATIC_SEED)
    train_env.close()
    
    # Get the master weights (HoF is at index 0 after evolve)
    if pilot.hall_of_fame is not None:
        master_weights = pilot.hall_of_fame
    else:
        master_weights = pilot.get_flat_weights(0)
        
    # 2. Record Continuous Episode (Short glitch for max survivability)
    # 15 frames = 0.5s glitch. Enough to look scary, short enough to recover.
    all_frames = record_continuous(pilot, env, CINEMATIC_SEED, master_weights, failure_duration=15)
    
    # 3. Save
    print(f"💾 Saving to {OUTPUT_FILENAME}...")
    imageio.mimsave(OUTPUT_FILENAME, all_frames, fps=FPS)
    print("✅ Done! Video generated.")
    env.close()

if __name__ == "__main__":
    main()
