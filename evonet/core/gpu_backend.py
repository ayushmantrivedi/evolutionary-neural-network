"""
GPU Backend Module

This module provides GPU acceleration using PyTorch for the evolutionary neural network.
Supports automatic device detection with CPU fallback for systems without CUDA.
"""

import numpy as np
from typing import Optional, Union, Tuple, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Try to import PyTorch
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info(f"✅ PyTorch with CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("✅ PyTorch available (CPU mode)")
except ImportError:
    logger.warning("⚠️ PyTorch not installed. GPU acceleration unavailable.")
    logger.warning("   Install with: pip install torch")


def get_device(use_gpu: bool = True) -> str:
    """
    Get the appropriate device for computation.
    
    Args:
        use_gpu: Whether to use GPU if available
        
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if use_gpu and TORCH_AVAILABLE and CUDA_AVAILABLE:
        return 'cuda'
    return 'cpu'


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return TORCH_AVAILABLE and CUDA_AVAILABLE


class GPUTensorWrapper:
    """
    Wrapper for seamless NumPy/PyTorch tensor conversion.
    
    Provides unified interface for operations that work on both CPU and GPU.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the wrapper.
        
        Args:
            device: Target device ('cuda' or 'cpu')
        """
        self.device = device
        self._torch_available = TORCH_AVAILABLE
    
    def to_tensor(self, array: np.ndarray) -> Union['torch.Tensor', np.ndarray]:
        """Convert numpy array to tensor on device."""
        if self._torch_available:
            import torch
            return torch.from_numpy(array.astype(np.float32)).to(self.device)
        return array.astype(np.float32)
    
    def to_numpy(self, tensor: Union['torch.Tensor', np.ndarray]) -> np.ndarray:
        """Convert tensor back to numpy array."""
        if self._torch_available and hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        return np.asarray(tensor)


class GPUPopulation:
    """
    GPU-accelerated population for evolutionary neurons.
    
    Maintains population weights on GPU for fast parallel evaluation.
    All operations are batched for maximum throughput.
    
    Example:
        >>> pop = GPUPopulation(input_dim=10, pop_size=50, device='cuda')
        >>> outputs = pop.forward_batch(inputs)
        >>> pop.update_from_selection(survivor_indices)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        pop_size: int = 50,
        device: str = 'cpu'
    ):
        """
        Initialize GPU population.
        
        Args:
            input_dim: Dimension of input features
            pop_size: Population size
            device: Target device ('cuda' or 'cpu')
        """
        self.input_dim = input_dim
        self.pop_size = pop_size
        self.device = device
        self._use_torch = TORCH_AVAILABLE
        
        # Initialize weights and biases
        if self._use_torch:
            import torch
            self.weights = torch.randn(pop_size, input_dim, device=device, dtype=torch.float32)
            self.biases = torch.randn(pop_size, device=device, dtype=torch.float32)
        else:
            self.weights = np.random.randn(pop_size, input_dim).astype(np.float32)
            self.biases = np.random.randn(pop_size).astype(np.float32)
        
        # Elite tracking
        self.elite_weights: Optional[Union['torch.Tensor', np.ndarray]] = None
        self.elite_bias: Optional[float] = None
        self.elite_fitness: float = float('inf')
    
    def forward_batch(
        self, 
        x: Union[np.ndarray, 'torch.Tensor']
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Compute forward pass for all individuals in population.
        
        Args:
            x: Input features (1D array of shape [input_dim])
            
        Returns:
            Outputs for all individuals (shape [pop_size])
        """
        if self._use_torch:
            import torch
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x.astype(np.float32)).to(self.device)
            # Batched matrix-vector product: [pop_size, input_dim] @ [input_dim] + [pop_size]
            return torch.mv(self.weights, x) + self.biases
        else:
            # NumPy fallback
            return np.dot(self.weights, x) + self.biases
    
    def evaluate_fitness(
        self, 
        outputs: Union[np.ndarray, 'torch.Tensor'],
        target: float
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Compute fitness (MSE error) for all individuals.
        
        Args:
            outputs: Population outputs from forward_batch
            target: Target value
            
        Returns:
            Fitness values for all individuals (lower is better)
        """
        if self._use_torch:
            import torch
            return (outputs - target) ** 2
        else:
            return (outputs - target) ** 2
    
    def get_best_individual(
        self
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], float, int]:
        """
        Get the weights and bias of the best individual.
        
        Returns:
            Tuple of (weights, bias, index)
        """
        return self.weights[0].clone() if self._use_torch else self.weights[0].copy(), \
               float(self.biases[0]), 0
    
    def tournament_select(
        self, 
        fitness: Union[np.ndarray, 'torch.Tensor'],
        tournament_size: int = 3,
        num_parents: int = 2
    ) -> List[int]:
        """
        Tournament selection for parent selection.
        
        Args:
            fitness: Fitness values for all individuals
            tournament_size: Number of individuals in each tournament
            num_parents: Number of parents to select
            
        Returns:
            Indices of selected parents
        """
        if self._use_torch:
            import torch
            fitness_np = fitness.cpu().numpy()
        else:
            fitness_np = fitness
        
        selected = []
        for _ in range(num_parents):
            # Random tournament
            candidates = np.random.choice(self.pop_size, tournament_size, replace=False)
            # Winner is the one with lowest fitness (error)
            winner = candidates[np.argmin(fitness_np[candidates])]
            selected.append(winner)
        
        return selected
    
    def sbx_crossover(
        self, 
        parent1_idx: int, 
        parent2_idx: int,
        eta: float = 20.0
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], float]:
        """
        Simulated Binary Crossover (SBX) for continuous optimization.
        
        Args:
            parent1_idx: Index of first parent
            parent2_idx: Index of second parent
            eta: Distribution index (higher = children closer to parents)
            
        Returns:
            Tuple of (child_weights, child_bias)
        """
        if self._use_torch:
            import torch
            p1_w = self.weights[parent1_idx]
            p2_w = self.weights[parent2_idx]
            p1_b = self.biases[parent1_idx]
            p2_b = self.biases[parent2_idx]
            
            # SBX for weights
            u = torch.rand(self.input_dim, device=self.device)
            beta = torch.where(
                u <= 0.5,
                (2 * u) ** (1 / (eta + 1)),
                (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            )
            
            child_w = 0.5 * ((1 + beta) * p1_w + (1 - beta) * p2_w)
            
            # SBX for bias
            u_b = np.random.random()
            if u_b <= 0.5:
                beta_b = (2 * u_b) ** (1 / (eta + 1))
            else:
                beta_b = (1 / (2 * (1 - u_b))) ** (1 / (eta + 1))
            child_b = 0.5 * ((1 + beta_b) * float(p1_b) + (1 - beta_b) * float(p2_b))
            
            return child_w, child_b
        else:
            # NumPy implementation
            p1_w = self.weights[parent1_idx]
            p2_w = self.weights[parent2_idx]
            p1_b = self.biases[parent1_idx]
            p2_b = self.biases[parent2_idx]
            
            u = np.random.random(self.input_dim)
            beta = np.where(
                u <= 0.5,
                (2 * u) ** (1 / (eta + 1)),
                (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            )
            
            child_w = 0.5 * ((1 + beta) * p1_w + (1 - beta) * p2_w)
            
            u_b = np.random.random()
            if u_b <= 0.5:
                beta_b = (2 * u_b) ** (1 / (eta + 1))
            else:
                beta_b = (1 / (2 * (1 - u_b))) ** (1 / (eta + 1))
            child_b = 0.5 * ((1 + beta_b) * p1_b + (1 - beta_b) * p2_b)
            
            return child_w.astype(np.float32), float(child_b)
    
    def polynomial_mutation(
        self,
        weights: Union[np.ndarray, 'torch.Tensor'],
        bias: float,
        mutation_rate: float = 0.1,
        eta: float = 20.0
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], float]:
        """
        Polynomial mutation for continuous variables.
        
        Args:
            weights: Weight vector to mutate
            bias: Bias value to mutate
            mutation_rate: Probability of mutation per gene
            eta: Distribution index
            
        Returns:
            Tuple of (mutated_weights, mutated_bias)
        """
        if self._use_torch:
            import torch
            mask = torch.rand(self.input_dim, device=self.device) < mutation_rate
            u = torch.rand(self.input_dim, device=self.device)
            
            delta = torch.where(
                u < 0.5,
                (2 * u) ** (1 / (eta + 1)) - 1,
                1 - (2 * (1 - u)) ** (1 / (eta + 1))
            )
            
            mutated_w = weights.clone()
            mutated_w[mask] = weights[mask] + delta[mask]
            
            # Mutate bias
            if np.random.random() < mutation_rate:
                u_b = np.random.random()
                if u_b < 0.5:
                    delta_b = (2 * u_b) ** (1 / (eta + 1)) - 1
                else:
                    delta_b = 1 - (2 * (1 - u_b)) ** (1 / (eta + 1))
                mutated_b = bias + delta_b
            else:
                mutated_b = bias
            
            return mutated_w, mutated_b
        else:
            # NumPy implementation
            mask = np.random.random(self.input_dim) < mutation_rate
            u = np.random.random(self.input_dim)
            
            delta = np.where(
                u < 0.5,
                (2 * u) ** (1 / (eta + 1)) - 1,
                1 - (2 * (1 - u)) ** (1 / (eta + 1))
            )
            
            mutated_w = weights.copy()
            mutated_w[mask] = weights[mask] + delta[mask]
            
            if np.random.random() < mutation_rate:
                u_b = np.random.random()
                if u_b < 0.5:
                    delta_b = (2 * u_b) ** (1 / (eta + 1)) - 1
                else:
                    delta_b = 1 - (2 * (1 - u_b)) ** (1 / (eta + 1))
                mutated_b = bias + delta_b
            else:
                mutated_b = bias
            
            return mutated_w.astype(np.float32), float(mutated_b)
    
    def evolve_population(
        self,
        fitness: Union[np.ndarray, 'torch.Tensor'],
        mutation_strength: float = 0.1,
        tournament_size: int = 3,
        elite_count: int = 2,
        crossover_prob: float = 0.9,
        V_m: Optional[List] = None
    ) -> None:
        """
        Evolve the population using tournament selection, SBX crossover, and mutation.
        
        Args:
            fitness: Fitness values for current population
            mutation_strength: Base mutation rate
            tournament_size: Tournament selection size
            elite_count: Number of elite individuals to preserve
            crossover_prob: Probability of applying crossover
            V_m: Significant mutation vector for guided evolution
        """
        if self._use_torch:
            import torch
            fitness_np = fitness.cpu().numpy()
            sorted_idx = np.argsort(fitness_np)
        else:
            sorted_idx = np.argsort(fitness)
        
        # Track elite
        best_fitness = float(fitness_np[sorted_idx[0]] if self._use_torch else fitness[sorted_idx[0]])
        if best_fitness < self.elite_fitness:
            self.elite_fitness = best_fitness
            if self._use_torch:
                self.elite_weights = self.weights[sorted_idx[0]].clone()
            else:
                self.elite_weights = self.weights[sorted_idx[0]].copy()
            self.elite_bias = float(self.biases[sorted_idx[0]])
        
        # Create new population
        new_weights = []
        new_biases = []
        
        # Keep elites
        for i in range(min(elite_count, self.pop_size)):
            if self._use_torch:
                new_weights.append(self.weights[sorted_idx[i]].clone())
            else:
                new_weights.append(self.weights[sorted_idx[i]].copy())
            new_biases.append(float(self.biases[sorted_idx[i]]))
        
        # Include global elite if exists
        if self.elite_weights is not None:
            if self._use_torch:
                new_weights.append(self.elite_weights.clone())
            else:
                new_weights.append(self.elite_weights.copy())
            new_biases.append(self.elite_bias)
        
        # Generate offspring
        while len(new_weights) < self.pop_size:
            # Tournament selection
            if self._use_torch:
                parents = self.tournament_select(fitness, tournament_size, 2)
            else:
                parents = self.tournament_select(fitness, tournament_size, 2)
            
            # Crossover
            if np.random.random() < crossover_prob:
                child_w, child_b = self.sbx_crossover(parents[0], parents[1])
            else:
                # Clone a parent
                if self._use_torch:
                    child_w = self.weights[parents[0]].clone()
                else:
                    child_w = self.weights[parents[0]].copy()
                child_b = float(self.biases[parents[0]])
            
            # Mutation
            child_w, child_b = self.polynomial_mutation(
                child_w, child_b, mutation_rate=mutation_strength
            )
            
            # V_m influence (20% chance)
            if V_m and len(V_m) > 0 and np.random.random() < 0.2:
                v = np.random.choice(V_m) if isinstance(V_m, list) else V_m[np.random.randint(len(V_m))]
                if isinstance(v, dict) and 'weights' in v:
                    v_weights = v['weights']
                    if v_weights.shape == (self.input_dim,):
                        if self._use_torch:
                            import torch
                            v_tensor = torch.from_numpy(v_weights.astype(np.float32)).to(self.device)
                            child_w = child_w + 0.3 * v_tensor
                        else:
                            child_w = child_w + 0.3 * v_weights
                        child_b = child_b + 0.3 * v.get('bias', 0)
            
            new_weights.append(child_w)
            new_biases.append(child_b)
        
        # Update population
        if self._use_torch:
            import torch
            self.weights = torch.stack(new_weights[:self.pop_size])
            self.biases = torch.tensor(new_biases[:self.pop_size], device=self.device, dtype=torch.float32)
        else:
            self.weights = np.array(new_weights[:self.pop_size], dtype=np.float32)
            self.biases = np.array(new_biases[:self.pop_size], dtype=np.float32)
    
    def to_dict_list(self) -> List[dict]:
        """Convert population to list of dictionaries (for compatibility)."""
        result = []
        for i in range(self.pop_size):
            if self._use_torch:
                result.append({
                    'weights': self.weights[i].cpu().numpy(),
                    'bias': float(self.biases[i])
                })
            else:
                result.append({
                    'weights': self.weights[i].copy(),
                    'bias': float(self.biases[i])
                })
        return result
    
    def from_dict_list(self, pop_list: List[dict]) -> None:
        """Load population from list of dictionaries."""
        weights_list = [p['weights'] for p in pop_list]
        biases_list = [p['bias'] for p in pop_list]
        
        if self._use_torch:
            import torch
            self.weights = torch.tensor(np.array(weights_list), device=self.device, dtype=torch.float32)
            self.biases = torch.tensor(np.array(biases_list), device=self.device, dtype=torch.float32)
        else:
            self.weights = np.array(weights_list, dtype=np.float32)
            self.biases = np.array(biases_list, dtype=np.float32)


def gpu_mse_loss_batch(
    predictions: Union[np.ndarray, 'torch.Tensor'],
    targets: Union[np.ndarray, 'torch.Tensor', float]
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Batched MSE loss computation.
    
    Args:
        predictions: Predicted values (can be batched)
        targets: Target values
        
    Returns:
        MSE loss values
    """
    if TORCH_AVAILABLE:
        import torch
        if isinstance(predictions, torch.Tensor):
            if isinstance(targets, (int, float)):
                targets = torch.tensor(targets, device=predictions.device)
            return (predictions - targets) ** 2
    return (predictions - targets) ** 2


def gpu_softmax(
    x: Union[np.ndarray, 'torch.Tensor']
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    GPU-accelerated softmax.
    
    Args:
        x: Input logits
        
    Returns:
        Softmax probabilities
    """
    if TORCH_AVAILABLE:
        import torch
        if isinstance(x, torch.Tensor):
            return torch.softmax(x, dim=-1)
    # NumPy fallback
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
