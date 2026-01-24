
import gymnasium as gym
import numpy as np
import imageio
import os
import time
import sys

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from train_memory_autopilot import MemoryEvoPilot, run_episode
from evonet.config import POP_SIZE

# Configuration
ENV_NAME = "LunarLander-v3" # or v2
OUTPUT_FILENAME = "recovery_demo.gif"
FPS = 30
FRAMES_PER_PHASE = 150 

def record_continuous(pilot, env, seed, master_weights, failure_duration=40):
    print("🎥 Recording Continuous 'One-Take' Demo...")
    frames = []
    
    # 1. Setup
    state, info = env.reset(seed=seed)
    
    # Pre-calc garbage weights
    garbage_weights = np.random.randn(len(master_weights)) * 1.0
    
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
            label = "PHASE 1: NORMAL FLIGHT"
            current_color = (255, 255, 255) # White
            
        elif FAIL_START <= step_count < HEAL_START:
            # Phase 2: Failure
            pilot.set_flat_weights(0, garbage_weights) 
            label = "⚠️ PHASE 2: SYSTEM FAILURE ⚠️"
            current_color = (0, 0, 255) # Red
            
        else:
            # Phase 3: Recovery
            pilot.set_flat_weights(0, master_weights)
            label = "✅ PHASE 3: SYSTEM RESTORED"
            current_color = (0, 255, 0) # Green
            
        # Render & Label
        frame = env.render()
        frame = text_overlay(frame, label, color=current_color)
        frames.append(frame)
        
        # Step
        action = pilot.get_action(state, pilot_index=0)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        # Stop if too long
        if step_count > 600:
            truncated = True
            
    print(f"   -> Episode finished. Total Reward: {total_reward:.1f}")
    return frames

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

def main():
    print("="*60)
    print("🎬 EVO-NET VISUALIZATION STUDIO (PRO MODE)")
    print("   Generating: 'One-Take' Continuous Demo")
    print("="*60)
    
    try:
        env = gym.make(ENV_NAME, render_mode="rgb_array")
        train_env = gym.make(ENV_NAME) 
    except:
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        train_env = gym.make("LunarLander-v2")
        
    pilot = MemoryEvoPilot()
    
    # 1. Find a seed where we hover high enough to survive a mid-air failure
    # We want a relatively stable start so we don't crash immediately upon failure
    # Seed 42 is usually calm.
    DEMO_SEED = 42
    
    # 2. Train Master
    # We train specifically on this seed to ensure precision landing (Between Flags)
    train_until_mastery(pilot, train_env, target_seed=DEMO_SEED)
    train_env.close()
    
    # Get the master weights (HoF is at index 0 after evolve)
    if pilot.hall_of_fame is not None:
        master_weights = pilot.hall_of_fame
    else:
        master_weights = pilot.get_flat_weights(0)
        
    # 3. Record Continuous Episode
    # Reduced failure duration to 15 frames (0.5s) to ensure recoverability
    all_frames = record_continuous(pilot, env, DEMO_SEED, master_weights, failure_duration=15)
    
    # 5. Save
    print(f"💾 Saving to {OUTPUT_FILENAME}...")
    imageio.mimsave(OUTPUT_FILENAME, all_frames, fps=FPS)
    print("✅ Done! Video generated.")
    env.close()

def train_until_mastery(pilot, env, target_seed=42):
    print("🎓 Training Master Pilot (Cinematic Mode - Overfitting to Scene)...")
    print(f"   Target: Score > 260 on SCENE SEED ({target_seed})")
    
    max_gens = 300
    prev_best = -float('inf')
    stagnation = 0
    
    for gen in range(1, max_gens + 1):
        rewards = []
        # 1. FORCE TRAIN ON THE DEMO SEED
        for i in range(pilot.pop_size):
            rewards.append(run_episode_seeded(env, pilot, i, seed=target_seed))
            
        best_reward = np.max(rewards)
        best_idx = np.argmax(rewards)
        
        # Check Stagnation
        if best_reward <= prev_best + 0.1:
            stagnation += 1
        else:
            stagnation = 0
            prev_best = best_reward
            
        # Stagnation Breaker
        if stagnation > 5:
            # Force high mutation for one gen
            pilot.current_mutation = 0.5 
            print(f"   -> Stagnation detected ({stagnation}), BOOSTING MUTATION to 0.5")
        
        pilot.update_hall_of_fame(rewards)
        
        print(f"   Gen {gen}: Best={best_reward:.1f} (Stagnation: {stagnation})")
        
        if best_reward > 260:
            print(f"   -> Mastery Achieved (Perfect Scene Rehearsed)!")
            pilot.store_memory("mastery", best_idx) 
            return True
            
        pilot.evolve(rewards)
        
        # Reset mutation after boost if we broke out (adaptive logic in evolve handles it, but let's reset base)
        if stagnation > 5:
             pilot.current_mutation = 0.1 # Reset
    
    print(f"⚠️ Warning: Did not hit 260. Using best effort.")
    pilot.store_memory("mastery", np.argmax(rewards))
    return False

def run_episode_seeded(env, pilot, pilot_index, seed):
    state, _ = env.reset(seed=seed)
    total = 0
    for _ in range(400):
        action = pilot.get_action(state, pilot_index)
        state, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc: break
    return total


if __name__ == "__main__":
    main()
