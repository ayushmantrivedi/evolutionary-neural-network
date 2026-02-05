
import numpy as np
import os
import sys

# Ensure evonet is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.core.network import MultiClassEvoNet
from evonet.core.memory import DirectionalMemory

# --- MOCK ENVIRONMENTS ---
class MockMarketField:
    """Mock market with a predictable but field-specific pattern."""
    def __init__(self, field_type="crypto"):
        self.field_type = field_type
        # Field A (Crypto): High Volatility, Trend Following
        # Field B (Stocks): Low Volatility, Mean Reverting
        
    def get_reward(self, state, action):
        # State is just a dummy range
        # Field A: Reward for high-conviction momentum
        if self.field_type == "crypto":
            # Action 2 (Long) is good if state is positive
            return (action - 1) * state if state > 0.5 else -(action - 1)
        # Field B: Reward for mean reversion
        else:
            # Action 1 (Neutral/Close) is good if state is high (overbought)
            return 2.0 if (state > 0.7 and action == 1) else -1.0

def run_evaluation(net, field, samples=100):
    rewards = []
    pilot_idx = 0 # Testing with the first individual
    for _ in range(samples):
        state = np.random.randn(10) # 10 features
        # dummy forward pass using network.predict
        y_pred, _ = net.predict(state, pilot_idx)
        action = np.argmax(y_pred)
        rewards.append(field.get_reward(state[0], action))
    return np.mean(rewards)

def train_on_field(net, field, epochs=20):
    """Simplified evolution loop for the test."""
    pilot_idx = 0
    best_reward = -float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        # Evaluate population (pop_size is from config)
        scores = []
        for i in range(net.pop_size):
            r = run_evaluation(net, field, samples=20)
            scores.append(r)
        
        # Evolve
        # Since we use neuron.evolve_rl, we pass the scores
        mut_strength = 0.1
        for layer in [net.level1, net.level2, net.level3]:
            for neuron in layer:
                neuron.evolve_rl(scores, mutation_strength=mut_strength)
        
        current_best = np.max(scores)
        if current_best > best_reward:
            best_reward = current_best
            # Capture best weights for memory storage
            # (In a real pilot we'd use HoF, here we just get pop[argmax])
            # get_flat_weights from MemoryEvoPilot is better, but here we just prove the concept
    return best_reward

def get_flat_weights(net, pilot_index=0):
    weights = []
    layers = [net.level1, net.level2, net.level3]
    for layer in layers:
        for neuron in layer:
            ind = neuron.population[pilot_index]
            weights.append(ind['weights'].flatten())
            weights.append(np.array([ind['bias']]))
    return np.concatenate(weights)

def set_flat_weights(net, flat_vector, pilot_index=0):
    offset = 0
    layers = [net.level1, net.level2, net.level3]
    for layer in layers:
        for neuron in layer:
            # Weights
            w_shape = neuron.population[pilot_index]['weights'].shape
            w_size = np.prod(w_shape)
            w_flat = flat_vector[offset : offset + w_size]
            neuron.population[pilot_index]['weights'] = w_flat.reshape(w_shape)
            offset += w_size
            # Bias
            neuron.population[pilot_index]['bias'] = float(flat_vector[offset])
            offset += 1

def perform_massive_trauma(net):
    """Wipe all neural weights to prove regeneration."""
    for layer in [net.level1, net.level2, net.level3]:
        for neuron in layer:
            for i in range(net.pop_size):
                neuron.population[i]['weights'] *= 0
                neuron.population[i]['bias'] = 0

def test_directional_regeneration():
    print("="*60, flush=True)
    print("🔬 MULTI-FIELD KNOWLEDGE VALIDATION", flush=True)
    print("   Testing retrieval of Crypto vs Stock strategies.", flush=True)
    print("="*60, flush=True)
    
    input_dim = 10
    num_classes = 3
    net = MultiClassEvoNet(input_dim, num_classes)
    
    # Initialize Memory with Theta_Init
    theta_init = get_flat_weights(net, 0)
    memory = DirectionalMemory(theta_init)
    
    # --- TASK 1: CRYPTO FIELD ---
    crypto_field = MockMarketField("crypto")
    print("\n[Step 1] Training on Field A: CRYPTO...", flush=True)
    train_on_field(net, crypto_field)
    peak_crypto = run_evaluation(net, crypto_field, samples=200)
    print(f"   Peak Crypto Score: {peak_crypto:.4f}", flush=True)
    
    # Store Displacement Vector
    theta_crypto = get_flat_weights(net, 0)
    memory.store_task("crypto", theta_crypto)
    
    # --- STEP 2: MASSIVE TRAUMA ---
    print("\n[Step 2] Applying MASSIVE TRAUMA (Brain Wipe)...", flush=True)
    perform_massive_trauma(net)
    trauma_score = run_evaluation(net, crypto_field, samples=200)
    print(f"   Score After Trauma: {trauma_score:.4f} (Should be near zero/random)", flush=True)
    
    # --- TASK 2: STOCK FIELD ---
    stock_field = MockMarketField("stocks")
    print("\n[Step 3] Training on Field B: STOCKS...", flush=True)
    train_on_field(net, stock_field)
    peak_stocks = run_evaluation(net, stock_field, samples=200)
    print(f"   Peak Stocks Score: {peak_stocks:.4f}", flush=True)
    
    # Store Displacement Vector
    theta_stocks = get_flat_weights(net, 0)
    memory.store_task("stocks", theta_stocks)
    
    # --- STEP 4: DIRECTIONAL REGENERATION ---
    print("\n[Step 4] Attempting Knowledge Retrieval (Regeneration)...", flush=True)
    
    # A. Recover Crypto
    print("   --> Regrowing 'CRYPTO' Skill...", flush=True)
    theta_rec_crypto = memory.recover(None, "crypto", alpha=1.0, from_origin=True)
    set_flat_weights(net, theta_rec_crypto, 0)
    rec_crypto_score = run_evaluation(net, crypto_field, samples=200)
    
    # B. Recover Stocks
    print("   --> Regrowing 'STOCKS' Skill...", flush=True)
    theta_rec_stocks = memory.recover(None, "stocks", alpha=1.0, from_origin=True)
    set_flat_weights(net, theta_rec_stocks, 0)
    rec_stocks_score = run_evaluation(net, stock_field, samples=200)
    
    print("\n" + "="*60, flush=True)
    print("📊 VALIDATION RESULTS", flush=True)
    print("="*60, flush=True)
    print(f"Task A (Crypto) Retention: { (rec_crypto_score/peak_crypto)*100:.1f}%", flush=True)
    print(f"Task B (Stocks) Retention: { (rec_stocks_score/peak_stocks)*100:.1f}%", flush=True)
    
    success = (rec_crypto_score/peak_crypto > 0.9) and (rec_stocks_score/peak_stocks > 0.9)
    if success:
        print("\n✅ SUCCESS: Multi-Field Regeneration works with ZERO interference!", flush=True)
    else:
        print("\n❌ FAILURE: Information lost during multi-field transition.", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    test_directional_regeneration()
