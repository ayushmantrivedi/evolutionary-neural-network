# EvoNet - Evolutionary Neural Network Package
"""
EvoNet: A neural network implementation using evolutionary algorithms
instead of backpropagation for training.

This package provides:
- Population-based neuron evolution
- Significant Mutation Vector (V_m) for preserving successful patterns
- Adaptive mutation strategies
- Support for classification (binary/multi-class) and regression
"""

__version__ = "1.0.0"
__author__ = "Ayushman Trivedi"

# Core components
from evonet.core.neuron import EvoNeuron, OutputNeuron, RegressionOutputNeuron
from evonet.core.network import MultiClassEvoNet, RegressionEvoNet
from evonet.core.mutations import SignificantMutationVector
from evonet.core.losses import mse_loss, ce_loss, ce_loss_with_confidence, softmax

# Data handling
from evonet.data.loaders import (
    load_housing_dataset,
    load_cancer_dataset,
    load_iris_dataset,
    load_wine_dataset,
    load_digits_dataset,
    load_custom_csv_dataset,
    load_telemetry_dataset
)
from evonet.data.preprocessing import preprocess_dataset, clean_dataset, detect_problem_type

# Training
from evonet.training.trainers import (
    mini_batch_evolution_training,
    early_stopping_training
)
from evonet.training.evolutionary_search import (
    evolutionary_search_parallel,
    parallel_neuron_predict
)

# Configuration
from evonet.config import EvoNetConfig

__all__ = [
    # Core
    "EvoNeuron",
    "OutputNeuron", 
    "RegressionOutputNeuron",
    "MultiClassEvoNet",
    "RegressionEvoNet",
    "SignificantMutationVector",
    # Losses
    "mse_loss",
    "ce_loss",
    "ce_loss_with_confidence",
    "softmax",
    # Data
    "load_housing_dataset",
    "load_cancer_dataset",
    "load_iris_dataset",
    "load_wine_dataset",
    "load_digits_dataset",
    "load_custom_csv_dataset",
    "load_telemetry_dataset",
    "preprocess_dataset",
    "clean_dataset",
    "detect_problem_type",
    # Training
    "mini_batch_evolution_training",
    "early_stopping_training",
    "evolutionary_search_parallel",
    "parallel_neuron_predict",
    # Config
    "EvoNetConfig",
]
