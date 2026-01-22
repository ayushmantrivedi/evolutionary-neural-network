
import gymnasium as gym
import numpy as np
import time
import logging
from typing import List, Dict, Any

from evonet.core.network import MultiClassEvoNet
from evonet.config import POP_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class EvoPilot:
    """
    RL Adapter for EvoNet.
    Treats the neuron population indices as 'Parallel Universes' (Pilots).
    Pilot[i] uses the i-th weight variant of every neuron in the network.
    """
    def __init__(self):
        # LunarLander-v2: 8 continuous inputs, 4 discrete actions
        self.input_dim = 8
        self.output_dim = 4
        # Reusing MultiClassEvoNet structure but manually controlling forward pass
        self.net = MultiClassEvoNet(self.input_dim, self.output_dim)
        
        # Ensure consistent population size
        self.pop_size = POP_SIZE
        
    def get_action(self, state: np.ndarray, pilot_index: int) -> int:
        """
        Forward pass for a specific pilot index.
        """
        # Layer 1
        l1_out = []
        for neuron in self.net.level1:
            ind = neuron.population[pilot_index]
            # Simple Linear + ReLU (using _activate default which is identity? No, usually check neuron type)
            # EvoNeuron._activate is identity by default.
            # Let's add ReLU manually for hidden layers to allow non-linearity
            val = np.dot(state, ind['weights']) + ind['bias']
            l1_out.append(np.maximum(0, val)) # ReLU
            
        l1_out = np.array(l1_out)
        
        # Layer 2
        l2_out = []
        for neuron in self.net.level2:
            ind = neuron.population[pilot_index]
            val = np.dot(l1_out, ind['weights']) + ind['bias']
            l2_out.append(np.maximum(0, val)) # ReLU
            
        l2_out = np.array(l2_out)
        
        # Output Layer
        # Handle Skip Connections dynamically
        # Check expected input dim from First Output Neuron
        w_shape = self.net.level3[0].population[pilot_index]['weights'].shape[0]
        
        if w_shape == len(l2_out) + len(l1_out):
            # Skip connections active: Concat [L2, L1]
            l3_in = np.concatenate([l2_out, l1_out])
        else:
            # Standard sequential
            l3_in = l2_out
            
        final_out = []
        for neuron in self.net.level3:
            ind = neuron.population[pilot_index]
            val = np.dot(l3_in, ind['weights']) + ind['bias']
            final_out.append(val)
            
        # Argmax for discrete action
        return int(np.argmax(final_out))

    def evolve(self, fitness_scores: List[float]):
        """
        Evolve all neurons based on the fitness scores of the pilots.
        """
        # Normalize fitness slightly to prevent extreme outliers?
        # Tournament selection handles scale well, so raw scores are fine.
        
        mutation_strength = 0.1 # Fixed for RL stability
        
        layers = [self.net.level1, self.net.level2, self.net.level3]
        for layer in layers:
            for neuron in layer:
                neuron.evolve_rl(fitness_scores, mutation_strength)

    def inject_fault(self, percentage: float):
        """
        Simulate hardware failure by zeroing out weights.
        """
        total_zeroed = 0
        layers = [self.net.level1, self.net.level2, self.net.level3]
        for layer in layers:
            for neuron in layer:
                n_kill = int(self.pop_size * percentage)
                kill_indices = np.random.choice(self.pop_size, n_kill, replace=False)
                for idx in kill_indices:
                    neuron.population[idx]['weights'] = np.zeros_like(neuron.population[idx]['weights'])
                    neuron.population[idx]['bias'] = 0.0
                    total_zeroed += 1
        return total_zeroed

def run_episode(env, pilot, pilot_index):
    state, _ = env.reset()
    total_reward = 0
    truncated = False
    terminated = False
    step_count = 0
    
    while not (terminated or truncated) and step_count < 400: # Limit steps
        action = pilot.get_action(state, pilot_index)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1
        
    return total_reward

def train_autopilot():
    print("="*60)
    print("🚀 FLIGHT SIMULATOR: FAULT-TOLERANT AUTOPILOT TRAINING")
    print("   Environment: LunarLander-v2")
    print(f"   Population: {POP_SIZE} Pilots")
    print("="*60)
    
    # Try v3 first (latest), then v2
    try:
        env = gym.make("LunarLander-v3")
        print("   Loaded: LunarLander-v3")
    except:
        try:
            env = gym.make("LunarLander-v2") 
            print("   Loaded: LunarLander-v2")
        except Exception as e:
            print(f"   ERROR: Could not load LunarLander. {e}")
            return
    
    pilot = EvoPilot()
    
    history_best = []
    
    # 100 Epochs total. Fault at Epoch 20.
    TOTAL_EPOCHS = 100
    FAULT_EPOCH = 20
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        
        # 1. Evaluate all pilots
        rewards = []
        # Run sequentially (slow but simple)
        # For a demo, running population 50 is okay.
        # To speed up, we can use a subset or shorter episodes?
        # LunarLander is fast.
        
        for i in range(pilot.pop_size):
            r = run_episode(env, pilot, i)
            rewards.append(r)
            
        avg_reward = np.mean(rewards)
        best_reward = np.max(rewards)
        history_best.append(best_reward)
        
        print(f"Epoch {epoch:02d}: Best Reward: {best_reward:6.1f} | Avg: {avg_reward:6.1f} | Time: {time.time()-epoch_start:.1f}s")
        
        # 2. Check for Fault Injection
        if epoch == FAULT_EPOCH:
            print("\n⚡ WARNING: MID-FLIGHT CATASTROPHIC FAILURE DETECTED ⚡")
            print("   (Simulating radiation strike on 30% of memory banks...)")
            n_zeroed = pilot.inject_fault(0.30)
            print(f"   DAMAGE REPORT: {n_zeroed} Neural Units Destroyed.")
            
            # Post-damage check
            check_rewards = []
            for i in range(pilot.pop_size):
                 check_rewards.append(run_episode(env, pilot, i))
            print(f"   STATUS AFTER IMPACT: Best Reward: {np.max(check_rewards):.1f} | Avg: {np.mean(check_rewards):.1f}")
            print("   ... Initiating Self-Healing Protocols ...\n")
            
            # Use post-damage rewards for evolution so it routes around damage immediately
            rewards = check_rewards
            
        # 3. Evolve
        pilot.evolve(rewards)
        
        # Success criteria for LunarLander is > 200
        if best_reward > 200 and epoch < FAULT_EPOCH:
            print("✨ Optimal Flight Path Achieved (Pre-Fault).")
            
    print("\n="*60)
    print("RESULT SUMMARY")
    print(f"Pre-Fault Peak: {max(history_best[:FAULT_EPOCH]):.1f}")
    print(f"Post-Fault Peak: {max(history_best[FAULT_EPOCH:]):.1f}")
    
    env.close()

if __name__ == "__main__":
    train_autopilot()
