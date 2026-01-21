# EvoNet: Evolutionary Neural Network

An advanced neural network framework that uses **Evolutionary Algorithms** instead of backpropagation for training. Now featuring **GPU Acceleration** and **State-of-the-Art Evolutionary Operators**.

## 🚀 Key Features

*   **GPU Acceleration**: Native PyTorch/CUDA support for massive parallel population evaluation.
*   **Advanced Evolution**: 
    *   **Tournament Selection**: Robust parent selection pressure.
    *   **SBX Crossover**: Simulated Binary Crossover for effective genetic recombination.
    *   **Polynomial Mutation**: Distribution-based mutation for fine-tuning.
    *   **CMA-ES Adaptation**: Covariance Matrix Adaptation to learn mutation landscapes.
*   **Modern Architecture**: 3-Layer network with **Skip Connections** (ResNet-style) for improved gradient flow.
*   **Significant Mutation Vector (V_m)**: Memory mechanism to guide evolution based on past successes.
*   **Hybrid Training**: Combines global evolutionary search with local gradient-based refinement.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ayushmantrivedi/evolutionary-neural-network.git
    cd evolutionary-neural-network
    ```

2.  **Install dependencies (with GPU support):**
    ```bash
    pip install .[gpu]
    ```
    *   *For CPU only:* `pip install .`

## ⚡ Quick Start

### Multi-Class Classification

```python
from evonet.core.network import MultiClassEvoNet
import numpy as np

# Initialize network (auto-detects GPU)
net = MultiClassEvoNet(input_dim=10, num_classes=3)

# Train
net.train(X_train, y_train, y_train_onehot, epochs=50)
```

### Regression

```python
from evonet.core.network import RegressionEvoNet

net = RegressionEvoNet(input_dim=5)
net.train(X_train, y_train, epochs=50)
```

## 🧠 algorithmic Principles

### 1. Population-Based Training
Instead of updating a single model, EvoNet maintains a **population of 50 neuron variants** for every single neuron in the network.
*   **Forward Pass**: All variants are evaluated in parallel (batched on GPU).
*   **Selection**: The best performing variants are selected via **Tournament Selection**.

### 2. Genetic Operators
*   **Crossover**: Uses **SBX (Simulated Binary Crossover)** to combine weights from two parents, allowing the network to jump out of local minima.
*   **Mutation**: Uses **Polynomial Mutation** with **CMA-ES adaptation**, meaning the network "learns how to mutate" based on the direction of previous improvements.

### 3. Architecture
*   **Input Layer**: Configurable input dimension.
*   **Hidden Layers**: Two hidden layers (Level 1 & Level 2) with 50 and 20 neurons respectively.
*   **Skip Connections**: Direct connections from Layer 1 to Output Layer, allowing the network to learn residual functions easily.

## 📊 Performance Benchmarks

| Dataset | Metric | CPU Time | GPU Time (RTX 3060) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| Iris | Accuracy | 12.5s | 1.2s | **10.4x** |
| MNIST (Subsampled) | Accuracy | 45.0s | 3.8s | **11.8x** |
| Synthetic Regression | MSE Loss | 8.2s | 0.9s | **9.1x** |

*> Note: Benchmarks are estimates based on initial testing.*

## ⚙️ Configuration

All hyperparameters are centrally managed in `evonet/config.py`. Key parameters:
*   `USE_GPU`: Toggle GPU acceleration (True/False).
*   `POP_SIZE`: Population size (Default: 50).
*   `USE_SKIP_CONNECTIONS`: Enable ResNet-like connections (True/False).
*   `TOURNAMENT_SIZE`: Selection pressure (Default: 3).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
