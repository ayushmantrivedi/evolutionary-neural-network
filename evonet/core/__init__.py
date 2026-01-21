# Core neural network components
from evonet.core.neuron import EvoNeuron, OutputNeuron, RegressionOutputNeuron
from evonet.core.network import MultiClassEvoNet, RegressionEvoNet
from evonet.core.mutations import SignificantMutationVector
from evonet.core.losses import mse_loss, ce_loss, ce_loss_with_confidence, softmax

# GPU acceleration and advanced evolution
from evonet.core.gpu_backend import (
    GPUPopulation, 
    GPUTensorWrapper,
    get_device, 
    is_gpu_available,
    TORCH_AVAILABLE,
    CUDA_AVAILABLE
)
from evonet.core.evolution import (
    tournament_selection,
    sbx_crossover,
    polynomial_mutation,
    CMAESAdapter,
    generate_offspring,
    diversity_preservation
)

__all__ = [
    # Core classes
    "EvoNeuron",
    "OutputNeuron",
    "RegressionOutputNeuron",
    "MultiClassEvoNet",
    "RegressionEvoNet",
    "SignificantMutationVector",
    # Loss functions
    "mse_loss",
    "ce_loss",
    "ce_loss_with_confidence",
    "softmax",
    # GPU backend
    "GPUPopulation",
    "GPUTensorWrapper",
    "get_device",
    "is_gpu_available",
    "TORCH_AVAILABLE",
    "CUDA_AVAILABLE",
    # Evolution operators
    "tournament_selection",
    "sbx_crossover",
    "polynomial_mutation",
    "CMAESAdapter",
    "generate_offspring",
    "diversity_preservation",
]

