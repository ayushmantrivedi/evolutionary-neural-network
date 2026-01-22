
import gymnasium as gym
import numpy as np
import time
import logging
from typing import List, Dict, Any, Union

from evonet.core.network import MultiClassEvoNet
from evonet.config import POP_SIZE
from evonet.core.memory import DirectionalMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MultiTaskPilot:
    """
    A Pilot capable of learning multiple tasks with different input/output dimensions
    by using a 'Max-Input' architecture and Zero-Padding.
    """
    def __init__(self):
        # Architecture defined by the LARGEST task
        # LunarLander: In 8, Out 4
        # CartPole:    In 4, Out 2
        self.max_input_dim = 8
        self.max_output_dim = 4
        
        self.net = MultiClassEvoNet(self.max_input_dim, self.max_output_dim)
        self.pop_size = POP_SIZE
        
        self.flat_init = self.get_flat_weights(pilot_index=0)
        self.memory = DirectionalMemory(self.flat_init)
        self.stored_tasks = set()
        
        # Evolution State
        self.hall_of_fame = None
        self.hof_score = -float('inf')
        self.stagnation_counter = 0
        self.current_mutation = 0.1
        
        self.current_task = "lunar_lander"

    def get_flat_weights(self, pilot_index: int) -> np.ndarray:
        weights = []
        layers = [self.net.level1, self.net.level2, self.net.level3]
        for layer in layers:
            for neuron in layer:
                ind = neuron.population[pilot_index]
                weights.append(ind['weights'].flatten())
                weights.append(np.array([ind['bias']]))
        return np.concatenate(weights)

    def set_flat_weights(self, pilot_index: int, flat_vector: np.ndarray):
        offset = 0
        layers = [self.net.level1, self.net.level2, self.net.level3]
        for layer in layers:
            for neuron in layer:
                w_shape = neuron.population[pilot_index]['weights'].shape
                w_size = np.prod(w_shape)
                w_flat = flat_vector[offset : offset + w_size]
                neuron.population[pilot_index]['weights'] = w_flat.reshape(w_shape)
                offset += w_size
                neuron.population[pilot_index]['bias'] = float(flat_vector[offset])
                offset += 1

    def get_action(self, state: np.ndarray, pilot_index: int) -> int:
        # 1. Input Adaptation (Padding)
        if state.shape[0] < self.max_input_dim:
            padded_state = np.zeros(self.max_input_dim)
            padded_state[:state.shape[0]] = state
            state = padded_state
            
        # 2. Forward Pass
        l1_out = []
        for neuron in self.net.level1:
            ind = neuron.population[pilot_index]
            val = np.dot(state, ind['weights']) + ind['bias']
            l1_out.append(np.maximum(0, val))
        l1_out = np.array(l1_out)
        
        l2_out = []
        for neuron in self.net.level2:
            ind = neuron.population[pilot_index]
            val = np.dot(l1_out, ind['weights']) + ind['bias']
            l2_out.append(np.maximum(0, val))
        l2_out = np.array(l2_out)
        
        # Skip Connections
        w_shape = self.net.level3[0].population[pilot_index]['weights'].shape[0]
        if w_shape == len(l2_out) + len(l1_out):
            l3_in = np.concatenate([l2_out, l1_out])
        else:
            l3_in = l2_out
            
        final_out = []
        for neuron in self.net.level3:
            ind = neuron.population[pilot_index]
            val = np.dot(l3_in, ind['weights']) + ind['bias']
            final_out.append(val)
            
        # 3. Output Adaptation (Masking)
        # If CartPole (2 actions), only look at first 2 outputs
        if self.current_task == "cartpole":
            return int(np.argmax(final_out[:2]))
        else:
            return int(np.argmax(final_out))

    def update_hall_of_fame(self, rewards: List[float]):
        current_best_idx = np.argmax(rewards)
        current_best_score = rewards[current_best_idx]
        
        if current_best_score > self.hof_score:
            self.hof_score = current_best_score
            self.hall_of_fame = self.get_flat_weights(current_best_idx)
            self.stagnation_counter = 0
            self.current_mutation = max(0.02, self.current_mutation * 0.9)
        else:
            self.stagnation_counter += 1
            import math
            phase = (self.stagnation_counter % 10) / 10.0
            factor = 0.5 * (1 - math.cos(2 * math.pi * phase))
            self.current_mutation = 0.05 + (factor * 0.15)

    def evolve(self, fitness_scores: List[float]):
        layers = [self.net.level1, self.net.level2, self.net.level3]
        for layer in layers:
            for neuron in layer:
                neuron.evolve_rl(fitness_scores, mutation_strength=self.current_mutation)
        if self.hall_of_fame is not None:
             self.set_flat_weights(0, self.hall_of_fame)

    def store_memory(self, task_name: str, best_pilot_idx: int):
        if self.hall_of_fame is not None:
            theta_star = self.hall_of_fame
        else:
            theta_star = self.get_flat_weights(best_pilot_idx)
        self.memory.store_task(task_name, theta_star)
        self.stored_tasks.add(task_name)
        print(f"💾 MEMORY STORED: '{task_name}'")

    def recover_memory(self, task_name: str):
        if task_name not in self.stored_tasks:
            return
        print(f"🧠 MEMORY RECALL: '{task_name}'")
        for i in range(self.pop_size):
            theta_current = self.get_flat_weights(i)
            # Use Regrowth (from_origin=True) because switching tasks is a drastic change
            theta_recovered = self.memory.recover(theta_current, task_name, alpha=1.0, steps=1, from_origin=True)
            self.set_flat_weights(i, theta_recovered)
        
        # Reset meta-learning state for the "new" (restored) task
        self.hof_score = -float('inf') 
        self.stagnation_counter = 0
        self.current_mutation = 0.05

def run_episode(env, pilot, pilot_index):
    state, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 400:
        action = pilot.get_action(state, pilot_index)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward

def train_phase(pilot, env_name, task_name, max_epochs=50, solve_threshold=100):
    print(f"\n--- STARTING PHASE: {task_name.upper()} ({env_name}) ---")
    try:
        env = gym.make(env_name)
    except:
        print(f"Error: Could not load {env_name}")
        return False

    pilot.current_task = task_name
    pilot.hof_score = -float('inf') # Reset HoF for new task
    pilot.hall_of_fame = None
    pilot.stagnation_counter = 0
    pilot.current_mutation = 0.2 # Force exploration for new task
    
    solved = False
    for epoch in range(1, max_epochs + 1):
        rewards = [run_episode(env, pilot, i) for i in range(pilot.pop_size)]
        
        best_reward = np.max(rewards)
        avg_reward = np.mean(rewards)
        best_idx = np.argmax(rewards)
        
        pilot.update_hall_of_fame(rewards)
        
        if epoch % 5 == 0 or best_reward > solve_threshold:
            print(f"[{task_name}] Epoch {epoch}: Best={best_reward:.1f}, Avg={avg_reward:.1f}")
            
        if best_reward > solve_threshold:
            print(f"✨ {task_name} SOLVED at Epoch {epoch}!")
            pilot.store_memory(task_name, best_idx)
            solved = True
            break
            
        pilot.evolve(rewards)
    
    env.close()
    if not solved:
        print(f"⚠️ Failed to fully solve {task_name} in time. Storing best effort.")
        # Force store best even if not perfect, to test recall mechanics
        pilot.store_memory(task_name, np.argmax(rewards))
        
    return solved

def main():
    print("="*60)
    print("🧠 MULTI-TASK CONTINUAL LEARNING DEMO")
    print("   Objective: Lander -> CartPole -> Recal Lander")
    print("="*60)
    
    pilot = MultiTaskPilot()
    
    # 1. Train Lander
    train_phase(pilot, "LunarLander-v3", "lunar_lander", max_epochs=50, solve_threshold=100)
    
    # 2. Train CartPole (Catastrophic Forgetting of Lander happens here)
    # CartPole is easier, solves at 195. We use 150 as "good enough" threshold
    train_phase(pilot, "CartPole-v1", "cartpole", max_epochs=30, solve_threshold=150)
    
    # 3. Recall Lander
    print("\n--- TEST: BACKWARD TRANSFER (RECALL) ---")
    print("Switching environment back to LunarLander-v3...")
    try:
        env = gym.make("LunarLander-v3")
    except:
        env = gym.make("LunarLander-v2")
        
    pilot.current_task = "lunar_lander"
    
    # Verify Forgetting (Should be terrible because we optimized for CartPole)
    print("Running Zero-Shot Eval (Before Recall)...")
    rewards_before = [run_episode(env, pilot, i) for i in range(pilot.pop_size)]
    print(f"❌ Performance WITHOUT Memory: Mean={np.mean(rewards_before):.1f}, Best={np.max(rewards_before):.1f}")
    
    # Invoke Memory
    pilot.recover_memory("lunar_lander")
    
    # Verify Recall
    print("Running Zero-Shot Eval (AFTER Recall)...")
    rewards_after = [run_episode(env, pilot, i) for i in range(pilot.pop_size)]
    msg_after = f"Performance WITH Memory:    Mean={np.mean(rewards_after):.1f}, Best={np.max(rewards_after):.1f}"
    print(msg_after)
    
    with open("final_report.txt", "w", encoding="utf-8") as f:
        f.write("MULTI-TASK CONTINUAL LEARNING RESULTS\n")
        f.write("======================================\n")
        f.write(f"Performance WITHOUT Memory: Mean={np.mean(rewards_before):.1f}, Best={np.max(rewards_before):.1f}\n")
        f.write(f"{msg_after}\n")
        f.write("======================================\n")
    
    env.close()

if __name__ == "__main__":
    main()
