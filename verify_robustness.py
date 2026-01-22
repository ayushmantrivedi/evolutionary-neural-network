
import numpy as np
import time
import logging
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from evonet.core.network import MultiClassEvoNet
from evonet.core.gpu_backend import get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for clear output
)
logger = logging.getLogger(__name__)

def inject_faults(network, failure_rate=0.2):
    """
    Simulates hardware failure by "killing" a percentage of the population
    in every single neuron. "Killing" means resetting weights to zero.
    
    Args:
        network: The EvoNet instance
        failure_rate: Percentage of population to corrupt (0.0 to 1.0)
    """
    total_killed = 0
    total_pop = 0
    
    # Iterate through all layers
    layers = [network.level1, network.level2, network.level3]
    
    for layer in layers:
        for neuron in layer:
            pop_size = len(neuron.population)
            # Determine how many to kill
            n_kill = int(pop_size * failure_rate)
            
            # Randomly select indices to kill
            kill_indices = np.random.choice(pop_size, n_kill, replace=False)
            
            for idx in kill_indices:
                # CORRUPTION: Reset to zero (simulating dead transistor/memory)
                neuron.population[idx]['weights'] = np.zeros_like(neuron.population[idx]['weights'])
                neuron.population[idx]['bias'] = 0.0
                
            total_killed += n_kill
            total_pop += pop_size
            
    logger.warning(f"⚠️  HARDWARE FAILURE INJECTED: Corrupted {total_killed}/{total_pop} individuals ({failure_rate*100}%) across network.")

def run_robustness_test():
    print("="*60)
    print("🧪 ROBUSTNESS & SELF-HEALING VERIFICATION TEST")
    print("="*60)
    print("Hypothesis: Evolutionary network can recover from massive parameter loss.")
    print("Scenario:    Simulated 'Radiation Event' causing 30% memory corruption.")
    print("-" * 60)

    # 1. Load Data (Breast Cancer - Binary Class)
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    encoder = OneHotEncoder(sparse_output=False)
    y_oh = encoder.fit_transform(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test, y_oh_train, y_oh_test = train_test_split(
        X, y, y_oh, test_size=0.2, random_state=42
    )
    
    # 2. Initialize Network
    net = MultiClassEvoNet(input_dim=X.shape[1], num_classes=2)
    
    # 3. Training Phase 1: Initial Convergence
    print("\nPhase 1: Normal Training (Epochs 1-15)")
    net.train(X_train, y_train, y_oh_train, epochs=15, X_val=X_test, y_val=y_test, y_val_oh=y_oh_test)
    
    acc_pre, _ = net.evaluate(X_test, y_test, y_oh_test)
    print(f"✅ Pre-Failure Accuracy: {acc_pre*100:.2f}%")
    
    # 4. THE EVENT: Inject Faults
    print("\n⚡ TRIGGERING MASSIVE HARDWARE FAILURE ⚡")
    print("... Corrupting memory banks ...")
    time.sleep(1)
    
    # Inject 30% failure rate - massive damage
    inject_faults(net, failure_rate=0.30)
    
    # Quick evaluate immediately after failure (before training heals it)
    acc_crash, _ = net.evaluate(X_test, y_test, y_oh_test)
    print(f"📉 Post-Failure Accuracy (Immediate): {acc_crash*100:.2f}%")
    print(f"   (Performance drop: {acc_pre*100 - acc_crash*100:.2f} percentage points)")
    
    # 5. Training Phase 2: Recovery
    print("\nPhase 2: Self-Healing / Recovery (Epochs 16-30)")
    epochs_recovery = 15
    
    # Manual loop to track detailed recovery
    for epoch in range(16, 16 + epochs_recovery):
        # One epoch of training
        net.train(X_train, y_train, y_oh_train, epochs=1) 
        
        # Check health
        acc, _ = net.evaluate(X_test, y_test, y_oh_test)
        print(f"   Epoch {epoch}: Recovery Accuracy: {acc*100:.2f}%")
        
        if acc >= acc_pre * 0.98:
            print(f"\n✨ SYSTEM RECOVERED in {epoch - 15} epochs!")
            break
            
    final_acc, _ = net.evaluate(X_test, y_test, y_oh_test)
    print("-" * 60)
    print(f"Final Status:")
    print(f"Original Accuracy: {acc_pre*100:.2f}%")
    print(f"Lowest Accuracy:   {acc_crash*100:.2f}%")
    print(f"Final Accuracy:    {final_acc*100:.2f}%")
    
    if final_acc >= acc_pre * 0.95:
        print("\n🏆 CONCLUSION: SUCCESS. Network demonstrated self-healing capabilities.")
    else:
        print("\n❌ CONCLUSION: FAILURE. Network could not fully learn.")

if __name__ == "__main__":
    run_robustness_test()
