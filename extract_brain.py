"""
extract_brain.py
Extracts pure numpy weights from the brain pickle (which requires evonet)
and saves them as brain_weights.pkl — a plain dict loadable with just numpy.
Run: python extract_brain.py
"""
import pickle, sys, numpy as np
sys.path.insert(0, '.')

# Load the full brain (requires evonet, works locally)
print("Loading brain...")
with open('nifty50_brain_validated.pkl', 'rb') as f:
    brain = pickle.load(f)
print(f"Loaded: {type(brain).__name__}  pop_size={brain.pop_size}")

# The Hall of Fame best genome is always stored at index 0
# (the evolve() method always injects hof into pilot 0)
PILOT = 0
layers = [brain.net.level1, brain.net.level2, brain.net.level3]

# Extract raw weights from each layer
neuron_data = []
for layer in layers:
    layer_neurons = []
    for neuron in layer:
        ind = neuron.population[PILOT]
        layer_neurons.append({
            'weights': ind['weights'].copy().astype(np.float32),
            'bias':    float(ind['bias']),
        })
    neuron_data.append(layer_neurons)

arch = {
    'input_dim':   brain.input_dim,
    'output_dim':  brain.output_dim,
    'layer_sizes': [len(layers[0]), len(layers[1]), len(layers[2])],
    'neurons':     neuron_data,
}
print(f"Architecture: {arch['input_dim']} -> {arch['layer_sizes']} -> {arch['output_dim']}")
print(f"L0: {len(neuron_data[0])} neurons, L1: {len(neuron_data[1])}, L2: {len(neuron_data[2])}")

# Save as plain pickle (protocol 2 = broadest compatibility)
with open('brain_weights.pkl', 'wb') as f:
    pickle.dump(arch, f, protocol=2)
print("Saved: brain_weights.pkl")

# Verify it loads cleanly with just numpy (simulate GitHub Actions)
with open('brain_weights.pkl', 'rb') as f:
    verify = pickle.load(f)
print(f"Verify: {verify['input_dim']} -> {verify['layer_sizes']} -> {verify['output_dim']} ✅")
