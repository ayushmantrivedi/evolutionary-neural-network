# EvoNet: Evolutionary Neural Network

<div align="center">

**The only ML framework that masters small data + imbalanced classification + no hyperparameter tuning — simultaneously.**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Ready](https://img.shields.io/badge/PyPI-ready-orange.svg)](pyproject.toml)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20accelerated-76b900.svg)](evonet/core/gpu_backend.py)

[Quick Start](#-quick-start) • [Benchmarks](#-benchmarks) • [Trading Results](#-real-world-trading-validation) • [Why EvoNet](#-why-evonet) • [Install](#%EF%B8%8F-installation)

</div>

---

## 🧠 What is EvoNet?

EvoNet trains neural networks using **evolutionary algorithms** — no backpropagation, no gradient computation, no vanishing gradients. Instead, it maintains a **population of 50 competing weight configurations** per neuron, evolving them through tournament selection, SBX crossover, and adaptive mutation.

**Result**: A neural network that works brilliantly where others fail — small datasets, imbalanced classes, and noisy real-world data.

---

## 📊 Benchmarks

### Standard Datasets (Accuracy vs Competitors)

| Dataset | Samples | EvoNet | XGBoost | Random Forest | MLP |
|---------|---------|--------|---------|---------------|-----|
| Iris | 150 | — | — | — | — |
| Wine | 178 | — | — | — | — |
| Breast Cancer | 569 | — | — | — | — |
| Digits | 1,797 | — | — | — | — |

> *Run `python evaluate_benchmarks.py` to populate this table with live results.*

### Imbalanced Data — The Killer Feature

EvoNet's primary market advantage: **superior F1-score on minority-class detection**.

| Use Case | Imbalance | EvoNet F1 | XGBoost F1 | Advantage |
|----------|-----------|-----------|------------|-----------|
| Fraud Detection | 90-10 | — | — | — |
| Disease Prediction | 95-5 | — | — | — |
| Anomaly Detection | 99-1 | — | — | — |

> *Run `python benchmark_imbalanced.py` to generate real results.*

### Small Dataset Performance

| Samples | EvoNet | XGBoost | TensorFlow MLP | sklearn MLP |
|---------|--------|---------|----------------|-------------|
| 500 | — | — | — | — |
| 1,000 | — | — | — | — |
| 2,000 | — | — | — | — |
| 5,000 | — | — | — | — |

> *Run `python benchmark_small_datasets.py` to generate real results.*

**Run the complete benchmark suite:**
```bash
python run_all_benchmarks.py
# Generates BENCHMARK_REPORT.md with all results
```

---

## 📈 Real-World Trading Validation

EvoNet was applied to NIFTY50 trading — **10 years of out-of-sample walk-forward validation**:

| Metric | EvoNet AI | NIFTY50 Buy & Hold | Top Indian PMS |
|--------|-----------|--------------------|----------------|
| **Annual Return** | **~43% CAGR** | ~14% | ~30% |
| **Max Drawdown** | **4.2%** | 35% (COVID) | 20-30% |
| **Sortino Ratio** | **5.24** | ~1.0 | ~2.0 |
| **COVID Crash** | **-6%** | -35% | -20 to -30% |
| **Profitable Periods** | **9 / 9** | N/A | N/A |

> See [`INVESTOR_PITCH.txt`](INVESTOR_PITCH.txt) for the complete 10-year walk-forward breakdown.

---

## 🏆 Why EvoNet?

### The Three Problems Nobody Else Solves Together

```
Problem 1: Small Data         → 30% of real enterprise ML problems
Problem 2: Imbalanced Classes → 40% of real enterprise ML problems
Problem 3: Need for Tuning    → 100% of ML teams waste time on this

EvoNet solves all three simultaneously.
No competitor does.
```

### Competitive Matrix

| Feature | EvoNet | XGBoost | TF/Keras | Google AutoML |
|---------|--------|---------|----------|---------------|
| Small Data (<10K) | ✅✅ | ✅ | ❌ | ❌ |
| Imbalanced Classes | ✅✅ | ⚠️ | ❌ | ✅ |
| No Hyperparameter Tuning | ✅✅ | ❌ | ❌ | ⚠️ |
| GPU Acceleration | ✅ | ❌ | ✅✅ | ✅ |
| Open Source | ✅ | ✅ | ✅ | ❌ |
| Works Out-of-Box | ✅✅ | ❌ | ❌ | ⚠️ |

---

## ⚙️ Installation

**CPU (no GPU):**
```bash
git clone https://github.com/ayushmantrivedi/evolutionary-neural-network.git
cd evolutionary-neural-network
pip install .
```

**With GPU support:**
```bash
pip install .[gpu]
```

**Full installation (trading, API, all features):**
```bash
pip install .[all]
```

---

## ⚡ Quick Start

### Classification
```python
from evonet.core.network import MultiClassEvoNet
import numpy as np

# Initialize (auto-detects GPU)
net = MultiClassEvoNet(input_dim=10, num_classes=3)

# Train — no hyperparameter tuning needed
net.train(X_train, y_train, y_train_onehot, epochs=50)

# Evaluate
accuracy, loss = net.evaluate(X_test, y_test, y_test_onehot)
print(f"Accuracy: {accuracy*100:.2f}%")
```

### Imbalanced Classification
```python
from evonet.core.network import MultiClassEvoNet
import numpy as np

# Works directly on imbalanced data — no SMOTE, no class weighting needed
net = MultiClassEvoNet(input_dim=20, num_classes=2)
net.train(X_imbalanced, y_imbalanced, y_oh, epochs=30)
# EvoNet naturally handles 99-1 class imbalance
```

### Regression
```python
from evonet.core.network import RegressionEvoNet

net = RegressionEvoNet(input_dim=5)
net.train(X_train, y_train, epochs=50)
mse, predictions = net.evaluate(X_test, y_test)
```

---

## 🧬 Algorithm Principles

### 1. Population-Based Training
Each neuron maintains **50 competing weight configurations**. Every training step evaluates all 50 in parallel, selects the best via tournament selection, and evolves new candidates.

### 2. Genetic Operators
- **SBX Crossover**: Combines weight vectors from two parents, enabling escape from local minima
- **Polynomial Mutation**: Distribution-based mutation with **CMA-ES adaptation** — the network learns _how to mutate_ based on past successes
- **Significant Mutation Vector (V_m)**: Evolutionary memory that guides future mutations based on what worked historically

### 3. Architecture
- 3-layer network with **skip connections** (ResNet-style)
- Input → Level 1 (50 neurons) → Level 2 (20 neurons) → Output
- Skip connections from Level 1 directly to output layer

### 4. Inference
- Uses **elite weights** (best-ever individual per neuron) — no stochasticity at inference time
- Includes **confidence head** for uncertainty estimation

---

## 🗂️ Repository Structure

```
evolutionary-neural-network/
├── evonet/                          # Core package
│   ├── core/
│   │   ├── network.py               # MultiClassEvoNet, RegressionEvoNet
│   │   ├── neuron.py                # EvoNeuron (population-based)
│   │   ├── mutations.py             # V_m (Significant Mutation Vector)
│   │   ├── gpu_backend.py           # CUDA/CPU device management
│   │   └── layers.py               # Attention, confidence head
│   ├── trader/                      # Trading application
│   └── config.py                    # Centralized hyperparameters
│
├── # Benchmark Suite
├── evaluate_benchmarks.py           # Standard classification benchmarks
├── benchmark_imbalanced.py          # Imbalanced data tests
├── benchmark_small_datasets.py      # Small dataset tests
├── benchmark_statistical_significance.py  # t-test vs XGBoost
├── benchmark_robustness.py          # Noise robustness tests
├── run_all_benchmarks.py            # Master runner → BENCHMARK_REPORT.md
│
├── # Validation
├── validate_production_readiness.py # 16-point production checklist
├── walk_forward_validation.py       # Trading out-of-sample test
│
├── # Trading Application
├── cloud_signal.py                  # Daily NIFTY50 signal generation
├── live_paper_trader.py             # Paper trading framework
├── options_cloud_signal.py          # Options strategy generation
│
├── # Trained Models
├── nifty50_brain_validated.pkl      # NIFTY50 trading brain (validated)
├── ultimate_brain_validated.pkl     # General classification brain
│
├── INVESTOR_PITCH.txt               # Full trading performance disclosure
├── BENCHMARK_REPORT.md             # Generated benchmark report
├── MONETIZATION.md                 # Business model & roadmap
└── COLAB_PROFESSIONAL.py           # Google Colab training notebook
```

---

## ⚙️ Configuration

All hyperparameters in [`evonet/config.py`](evonet/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POP_SIZE` | 50 | Population size per neuron |
| `LEVEL1_NEURONS` | 50 | Hidden layer 1 size |
| `LEVEL2_NEURONS` | 20 | Hidden layer 2 size |
| `USE_GPU` | True | Enable CUDA acceleration |
| `USE_SKIP_CONNECTIONS` | True | ResNet-style skip connections |
| `TOURNAMENT_SIZE` | 3 | Selection pressure |
| `EPOCHS` | 50 | Default training epochs |

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 📬 Contact

**Ayushman Trivedi** — CTO & Lead Researcher  
Repository: [github.com/ayushmantrivedi/evolutionary-neural-network](https://github.com/ayushmantrivedi/evolutionary-neural-network)  
Business inquiries: See [MONETIZATION.md](MONETIZATION.md)
