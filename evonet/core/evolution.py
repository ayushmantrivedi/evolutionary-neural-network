"""
Evolution Strategies Module

Advanced evolutionary operators for improving optimization performance:
- Tournament Selection: Configurable selection pressure
- SBX Crossover: Simulated Binary Crossover for continuous optimization  
- CMA-ES Inspired Adaptation: Covariance matrix adaptation for mutation
- Diversity Preservation: Mechanisms to prevent premature convergence
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def tournament_selection(
    population: List[Dict[str, Any]],
    fitness: np.ndarray,
    tournament_size: int = 3,
    num_select: int = 2
) -> List[int]:
    """
    Tournament selection for parent selection.
    
    Selects individuals by running tournaments where random subsets
    compete, and the best individual in each tournament wins.
    
    Args:
        population: List of individuals
        fitness: Fitness values (lower is better)
        tournament_size: Number of competitors per tournament
        num_select: Number of individuals to select
        
    Returns:
        List of selected individual indices
    """
    pop_size = len(population)
    selected = []
    
    for _ in range(num_select):
        # Random tournament
        candidates = np.random.choice(pop_size, min(tournament_size, pop_size), replace=False)
        # Winner has lowest fitness (error)
        winner = candidates[np.argmin(fitness[candidates])]
        selected.append(winner)
    
    return selected


def sbx_crossover(
    parent1: Dict[str, Any],
    parent2: Dict[str, Any],
    eta: float = 20.0,
    crossover_prob: float = 0.9
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simulated Binary Crossover (SBX) for real-coded genetic algorithms.
    
    Creates two children from two parents using a probability distribution
    that mimics single-point crossover for binary chromosomes.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        eta: Distribution index (higher = children closer to parents)
        crossover_prob: Probability of crossover per gene
        
    Returns:
        Tuple of two child individuals
    """
    w1 = parent1['weights']
    w2 = parent2['weights']
    b1 = parent1['bias']
    b2 = parent2['bias']
    
    input_dim = w1.shape[0]
    
    # Apply crossover with probability
    if np.random.random() > crossover_prob:
        return (
            {'weights': w1.copy(), 'bias': b1},
            {'weights': w2.copy(), 'bias': b2}
        )
    
    # SBX for weights
    child1_w = np.empty_like(w1)
    child2_w = np.empty_like(w2)
    
    for i in range(input_dim):
        if np.random.random() <= 0.5:
            # Apply SBX
            if abs(w1[i] - w2[i]) > 1e-14:
                if w1[i] < w2[i]:
                    y1, y2 = w1[i], w2[i]
                else:
                    y1, y2 = w2[i], w1[i]
                
                u = np.random.random()
                
                # Calculate beta
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                
                child1_w[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                child2_w[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
            else:
                child1_w[i] = w1[i]
                child2_w[i] = w2[i]
        else:
            child1_w[i] = w1[i]
            child2_w[i] = w2[i]
    
    # SBX for bias
    u = np.random.random()
    if u <= 0.5:
        beta = (2 * u) ** (1 / (eta + 1))
    else:
        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    
    child1_b = 0.5 * ((b1 + b2) - beta * abs(b2 - b1))
    child2_b = 0.5 * ((b1 + b2) + beta * abs(b2 - b1))
    
    return (
        {'weights': child1_w.astype(np.float32), 'bias': np.float32(child1_b)},
        {'weights': child2_w.astype(np.float32), 'bias': np.float32(child2_b)}
    )


def polynomial_mutation(
    individual: Dict[str, Any],
    mutation_rate: float = 0.1,
    eta: float = 20.0,
    bounds: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Polynomial mutation for real-coded genetic algorithms.
    
    Applies a polynomial distribution to mutate genes,
    producing small changes most of the time with occasional large jumps.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutation per gene
        eta: Distribution index (higher = smaller mutations)
        bounds: Optional (lower, upper) bounds for values
        
    Returns:
        Mutated individual
    """
    weights = individual['weights'].copy()
    bias = individual['bias']
    input_dim = weights.shape[0]
    
    for i in range(input_dim):
        if np.random.random() < mutation_rate:
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            
            weights[i] += delta
            
            # Apply bounds if specified
            if bounds:
                weights[i] = np.clip(weights[i], bounds[0], bounds[1])
    
    # Mutate bias
    if np.random.random() < mutation_rate:
        u = np.random.random()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        bias += delta
    
    return {'weights': weights.astype(np.float32), 'bias': np.float32(bias)}


class CMAESAdapter:
    """
    CMA-ES Inspired Covariance Adaptation for mutation.
    
    Tracks the covariance structure of successful mutations and uses it
    to guide future mutations. Simplified version of full CMA-ES.
    
    This enables the algorithm to learn the shape of the fitness landscape
    and adapt mutation distribution accordingly.
    """
    
    def __init__(self, input_dim: int, learning_rate: float = 0.3):
        """
        Initialize CMA-ES adapter.
        
        Args:
            input_dim: Dimension of weight vectors
            learning_rate: Rate of covariance adaptation
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        
        # Start with identity covariance (isotropic)
        self.C = np.eye(input_dim, dtype=np.float32)
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.sigma = 1.0  # Step size
        
        # Evolution path for cumulation
        self.p_c = np.zeros(input_dim, dtype=np.float32)
        self.p_sigma = np.zeros(input_dim, dtype=np.float32)
        
        # History of successful steps
        self.success_history = []
        self.max_history = 20
    
    def update(self, successful_weights: List[np.ndarray], old_mean: np.ndarray) -> None:
        """
        Update covariance based on successful mutations.
        
        Args:
            successful_weights: List of weight vectors that performed well
            old_mean: Previous mean vector
        """
        if len(successful_weights) == 0:
            return
        
        # Compute new mean from successful individuals
        new_mean = np.mean(successful_weights, axis=0).astype(np.float32)
        
        # Evolution step
        step = new_mean - old_mean
        
        # Update evolution path
        c_c = 4 / (self.input_dim + 4)
        self.p_c = (1 - c_c) * self.p_c + np.sqrt(c_c * (2 - c_c)) * step / self.sigma
        
        # Update covariance matrix (rank-one update simplified)
        c_cov = 2 / (self.input_dim ** 2)
        
        # Outer product for covariance update
        step_outer = np.outer(self.p_c, self.p_c)
        
        # Exponential moving average of covariance
        self.C = (1 - c_cov) * self.C + c_cov * step_outer
        
        # Ensure positive definite (regularization)
        self.C = (self.C + self.C.T) / 2  # Symmetrize
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-6:
            self.C += (1e-6 - min_eig) * np.eye(self.input_dim)
        
        # Update mean
        self.mean = new_mean
        
        # Store in history
        self.success_history.append(step)
        if len(self.success_history) > self.max_history:
            self.success_history.pop(0)
    
    def sample(self, base_weights: np.ndarray, mutation_strength: float) -> np.ndarray:
        """
        Sample mutation from adapted distribution.
        
        Args:
            base_weights: Starting weight vector
            mutation_strength: Scale of mutation
            
        Returns:
            Mutated weight vector
        """
        try:
            # Sample from multivariate normal with adapted covariance
            L = np.linalg.cholesky(self.C)
            z = np.random.randn(self.input_dim)
            sample = base_weights + mutation_strength * self.sigma * (L @ z)
            return sample.astype(np.float32)
        except np.linalg.LinAlgError:
            # Fallback to isotropic if Cholesky fails
            return (base_weights + mutation_strength * np.random.randn(self.input_dim)).astype(np.float32)
    
    def reset(self) -> None:
        """Reset adapter to initial state."""
        self.C = np.eye(self.input_dim, dtype=np.float32)
        self.mean = np.zeros(self.input_dim, dtype=np.float32)
        self.p_c = np.zeros(self.input_dim, dtype=np.float32)
        self.success_history = []


def diversity_preservation(
    population: List[Dict[str, Any]],
    fitness: np.ndarray,
    diversity_threshold: float = 0.1
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Preserve diversity by penalizing similar individuals.
    
    Adds a crowding-based penalty to fitness to maintain population diversity
    and prevent premature convergence.
    
    Args:
        population: Current population
        fitness: Original fitness values
        diversity_threshold: Minimum distance between individuals
        
    Returns:
        Tuple of (population, adjusted_fitness)
    """
    pop_size = len(population)
    adjusted_fitness = fitness.copy()
    
    # Compute pairwise distances
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist = np.linalg.norm(
                population[i]['weights'] - population[j]['weights']
            )
            
            if dist < diversity_threshold:
                # Penalize both individuals proportionally
                penalty = (diversity_threshold - dist) / diversity_threshold
                adjusted_fitness[i] += penalty * 0.1
                adjusted_fitness[j] += penalty * 0.1
    
    return population, adjusted_fitness


def generate_offspring(
    parents: List[Dict[str, Any]],
    num_offspring: int,
    mutation_strength: float = 0.1,
    eta_crossover: float = 20.0,
    eta_mutation: float = 20.0,
    crossover_prob: float = 0.9,
    cma_adapter: Optional[CMAESAdapter] = None,
    V_m: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Generate offspring population using advanced operators.
    
    Combines SBX crossover, polynomial mutation, CMA-ES adaptation,
    and V_m influence to create diverse, high-quality offspring.
    
    Args:
        parents: List of parent individuals
        num_offspring: Number of offspring to generate
        mutation_strength: Base mutation rate
        eta_crossover: SBX distribution index
        eta_mutation: Polynomial mutation distribution index
        crossover_prob: Probability of crossover
        cma_adapter: Optional CMA-ES adapter for guided mutation
        V_m: Optional significant mutation vector
        
    Returns:
        List of offspring individuals
    """
    offspring = []
    num_parents = len(parents)
    
    while len(offspring) < num_offspring:
        # Select two random parents
        p1_idx = np.random.randint(num_parents)
        p2_idx = np.random.randint(num_parents)
        while p2_idx == p1_idx and num_parents > 1:
            p2_idx = np.random.randint(num_parents)
        
        # Crossover
        child1, child2 = sbx_crossover(
            parents[p1_idx], 
            parents[p2_idx],
            eta=eta_crossover,
            crossover_prob=crossover_prob
        )
        
        # Mutation
        for child in [child1, child2]:
            if len(offspring) >= num_offspring:
                break
            
            # Apply CMA-ES guided mutation if available
            if cma_adapter is not None and np.random.random() < 0.5:
                child['weights'] = cma_adapter.sample(
                    child['weights'], 
                    mutation_strength
                )
            else:
                child = polynomial_mutation(
                    child,
                    mutation_rate=mutation_strength,
                    eta=eta_mutation
                )
            
            # V_m influence (20% chance)
            if V_m and len(V_m) > 0 and np.random.random() < 0.2:
                v = V_m[np.random.randint(len(V_m))]
                if v['weights'].shape == child['weights'].shape:
                    child['weights'] = child['weights'] + 0.3 * v['weights']
                    child['bias'] = child['bias'] + 0.3 * v['bias']
            
            offspring.append(child)
    
    return offspring[:num_offspring]


def rank_selection(
    population: List[Dict[str, Any]],
    fitness: np.ndarray,
    selection_pressure: float = 2.0
) -> List[int]:
    """
    Rank-based selection with configurable pressure.
    
    Args:
        population: Current population
        fitness: Fitness values
        selection_pressure: Higher = more pressure toward best individuals
        
    Returns:
        List of selected indices
    """
    pop_size = len(population)
    ranks = np.argsort(np.argsort(fitness))  # Lower fitness = lower rank
    
    # Linear ranking probabilities
    probs = (2 - selection_pressure + 2 * (selection_pressure - 1) * 
             (pop_size - 1 - ranks) / (pop_size - 1)) / pop_size
    probs = np.maximum(probs, 0)  # Ensure non-negative
    probs /= probs.sum()  # Normalize
    
    selected = np.random.choice(pop_size, size=pop_size // 2, p=probs, replace=False)
    return selected.tolist()
