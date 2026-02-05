
import pickle
from train_memory_autopilot import MemoryEvoPilot
from evonet.core.network import MultiClassEvoNet
from evonet.core.memory import DirectionalMemory

# Configuration
WINDOW_SIZE = 20

def create_brain():
    print("Creating Ultimate Brain (Untrained Structure)...")
    
    pilot = MemoryEvoPilot()
    pilot.input_dim = WINDOW_SIZE * 10
    pilot.output_dim = 3
    pilot.net = MultiClassEvoNet(pilot.input_dim, pilot.output_dim) 
    
    # Init Memory
    pilot.flat_init = pilot.get_flat_weights(0)
    pilot.memory = DirectionalMemory(pilot.flat_init)
    
    # Set weights to random initialization (or index 0)
    # Already initialized.
    
    # Save
    with open("ultimate_brain.pkl", "wb") as f:
        pickle.dump(pilot, f)
        
    print("ultimate_brain.pkl created successfully. Ready for Backtest.")

if __name__ == "__main__":
    create_brain()
