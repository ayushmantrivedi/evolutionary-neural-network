"""
EvoNet Configuration Module

Centralized hyperparameters and configuration for the evolutionary neural network.
All hyperparameters are defined here with type hints for production use.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class EvoNetConfig:
    """
    Configuration class for EvoNet hyperparameters.
    
    Attributes:
        level1_neurons: Number of neurons in first layer
        level2_neurons: Number of neurons in second layer
        pop_size: Population size for each neuron's evolution
        vm_history: Maximum history length for Significant Mutation Vector
        vm_influence_prob: Probability of V_m influencing new mutations
        vm_improvement_thresh: Threshold for V_m improvement
        tau1: Error threshold for layer 1
        tau2: Error threshold for layer 2
        mut_strength_base: Base mutation strength
        epochs: Number of training epochs
        print_interval: Interval for printing progress
        thought_interval: Interval for printing internal state
        min_mut_strength: Minimum mutation strength
        local_gd_enabled: Enable local gradient descent refinement
        local_gd_lr: Learning rate for local GD
        local_linesearch_alphas: Alpha values for line search
        local_min_improvement: Minimum improvement threshold
        use_gpu: Enable GPU acceleration if available
        tournament_size: Size of tournament for selection
        crossover_prob: Probability of applying crossover
        eta_crossover: SBX crossover distribution index
        eta_mutation: Polynomial mutation distribution index
        use_cma_adaptation: Enable CMA-ES style covariance adaptation
        elite_count: Number of elite individuals to preserve
        use_skip_connections: Enable skip connections in network
    """
    # Network architecture
    level1_neurons: int = 50
    level2_neurons: int = 20
    
    # Population parameters (increased from 20 for better diversity)
    pop_size: int = 50
    
    # Significant Mutation Vector parameters
    vm_history: int = 20
    vm_influence_prob: float = 0.2
    vm_improvement_thresh: float = 0.15
    
    # Error thresholds
    tau1: float = 0.15
    tau2: float = 0.10
    
    # Mutation parameters
    mut_strength_base: float = 0.1
    min_mut_strength: float = 0.01
    
    # Training parameters
    epochs: int = 50
    print_interval: int = 5
    thought_interval: int = 10
    
    # Local refinement parameters
    local_gd_enabled: bool = True
    local_gd_lr: float = 0.05
    local_linesearch_alphas: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    )
    local_min_improvement: float = 1e-8
    
    # GPU Configuration
    use_gpu: bool = True  # Enable GPU if available (auto-fallback to CPU)
    
    # Advanced Evolution Configuration
    tournament_size: int = 3  # Tournament selection size
    crossover_prob: float = 0.9  # SBX crossover probability
    eta_crossover: float = 20.0  # SBX distribution index
    eta_mutation: float = 20.0  # Polynomial mutation distribution index
    use_cma_adaptation: bool = True  # Enable CMA-ES style adaptation
    elite_count: int = 3  # Number of elite individuals to preserve
    
    # Network Architecture Enhancements
    use_skip_connections: bool = True  # Skip connections between layers
    
    # Output directory
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Set default output directory if not provided."""
        if self.output_dir is None:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'output'
            )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_output_directory(self) -> str:
        """Get the output directory for saving plots and results."""
        output_dir = os.environ.get('EVO_OUTPUT_DIR', self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


# Global default configuration instance
DEFAULT_CONFIG = EvoNetConfig()

# Expose individual hyperparameters for backward compatibility
LEVEL1_NEURONS = DEFAULT_CONFIG.level1_neurons
LEVEL2_NEURONS = DEFAULT_CONFIG.level2_neurons
POP_SIZE = DEFAULT_CONFIG.pop_size
VM_HISTORY = DEFAULT_CONFIG.vm_history
VM_INFLUENCE_PROB = DEFAULT_CONFIG.vm_influence_prob
VM_IMPROVEMENT_THRESH = DEFAULT_CONFIG.vm_improvement_thresh
TAU1 = DEFAULT_CONFIG.tau1
TAU2 = DEFAULT_CONFIG.tau2
MUT_STRENGTH_BASE = DEFAULT_CONFIG.mut_strength_base
EPOCHS = DEFAULT_CONFIG.epochs
PRINT_INTERVAL = DEFAULT_CONFIG.print_interval
THOUGHT_INTERVAL = DEFAULT_CONFIG.thought_interval
MIN_MUT_STRENGTH = DEFAULT_CONFIG.min_mut_strength
LOCAL_GD_ENABLED = DEFAULT_CONFIG.local_gd_enabled
LOCAL_GD_LR = DEFAULT_CONFIG.local_gd_lr
LOCAL_LINESEARCH_ALPHAS = DEFAULT_CONFIG.local_linesearch_alphas
LOCAL_MIN_IMPROVEMENT = DEFAULT_CONFIG.local_min_improvement

# New GPU and evolution configuration exports
USE_GPU = DEFAULT_CONFIG.use_gpu
TOURNAMENT_SIZE = DEFAULT_CONFIG.tournament_size
CROSSOVER_PROB = DEFAULT_CONFIG.crossover_prob
ETA_CROSSOVER = DEFAULT_CONFIG.eta_crossover
ETA_MUTATION = DEFAULT_CONFIG.eta_mutation
USE_CMA_ADAPTATION = DEFAULT_CONFIG.use_cma_adaptation
ELITE_COUNT = DEFAULT_CONFIG.elite_count
USE_SKIP_CONNECTIONS = DEFAULT_CONFIG.use_skip_connections


def get_output_directory() -> str:
    """Get the output directory for saving plots and results."""
    return DEFAULT_CONFIG.get_output_directory()
