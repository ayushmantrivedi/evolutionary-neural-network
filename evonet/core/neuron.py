"""
Evolutionary Neuron Module

This module contains the core EvoNeuron class and its variants.
Each neuron maintains a population of weight/bias combinations that
evolve through selection and mutation.

Enhanced with GPU acceleration and advanced evolution operators.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import logging

from evonet.config import (
    POP_SIZE, TAU1, TAU2, VM_INFLUENCE_PROB,
    LOCAL_GD_ENABLED, LOCAL_GD_LR, LOCAL_LINESEARCH_ALPHAS, LOCAL_MIN_IMPROVEMENT,
    TOURNAMENT_SIZE, CROSSOVER_PROB, ETA_CROSSOVER, ETA_MUTATION, 
    USE_CMA_ADAPTATION, ELITE_COUNT, USE_GPU
)
from evonet.core.losses import mse_loss

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
Individual = Dict[str, Union[np.ndarray, float]]
ErrorFunction = Callable[[Any, Any], float]


def population_forward(
    x: np.ndarray, 
    weights: np.ndarray, 
    biases: np.ndarray
) -> np.ndarray:
    """
    Vectorized forward pass for entire population.
    
    Args:
        x: Input features (1D array)
        weights: Population weights (pop_size, input_dim)
        biases: Population biases (pop_size,)
        
    Returns:
        np.ndarray: Output for each individual in population
    """
    return np.dot(weights.astype(np.float32), x.astype(np.float32)) + biases.astype(np.float32)


class EvoNeuron:
    """
    Evolutionary Neuron - A neuron that evolves its weights through selection and mutation.
    
    This is the core building block of the evolutionary neural network. Each neuron
    maintains a population of weight configurations that compete and evolve.
    
    Attributes:
        input_dim: Dimension of input features
        pop_size: Size of the weight population
        population: List of individuals (weight/bias configurations)
        elite: Best individual found so far
        elite_error: Error of the elite individual
        
    Example:
        >>> neuron = EvoNeuron(input_dim=10, pop_size=20)
        >>> output, error = neuron.forward(x, y_true, mse_loss, 0.1)
    """
    
    def __init__(self, input_dim: int, pop_size: int = POP_SIZE):
        """
        Initialize the evolutionary neuron.
        
        Args:
            input_dim: Dimension of input features
            pop_size: Population size for evolution
        """
        self.input_dim = input_dim
        self.pop_size = pop_size
        self.population: List[Individual] = [
            self._random_individual() for _ in range(pop_size)
        ]
        self.best_idx: int = 0
        self.last_error: Optional[float] = None
        self.last_output: Optional[float] = None
        self.last_weights: Optional[np.ndarray] = None
        self.last_bias: Optional[float] = None
        self.elite: Optional[Individual] = None
        self.elite_error: float = float('inf')
    
    def _random_individual(self) -> Individual:
        """Create a random individual with weights and bias."""
        return {
            'weights': np.random.randn(self.input_dim).astype(np.float32),
            'bias': np.float32(np.random.randn()),
        }
    
    def _check_population_shapes(self) -> None:
        """Verify and fix population shapes if needed."""
        for ind in self.population:
            if ind['weights'].shape != (self.input_dim,):
                logger.debug(
                    f"Population shape mismatch detected. "
                    f"Reinitializing population for input_dim={self.input_dim}."
                )
                self.population = [
                    self._random_individual() for _ in range(self.pop_size)
                ]
                break
    
    def _activate(self, x: float) -> float:
        """Activation function. Override in subclasses."""
        return x
    
    def forward(
        self,
        x: np.ndarray,
        y_true: Union[np.ndarray, float],
        error_fn: ErrorFunction,
        mutation_strength: float,
        V_m: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, float]:
        """
        Forward pass through the neuron population.
        
        Args:
            x: Input features
            y_true: Target value
            error_fn: Error function to evaluate fitness
            mutation_strength: Current mutation strength
            V_m: Significant mutation vector (optional)
            
        Returns:
            Tuple of (best_output, best_error)
        """
        self._check_population_shapes()
        
        # Vectorized forward pass
        weights = np.array([ind['weights'] for ind in self.population])
        biases = np.array([ind['bias'] for ind in self.population])
        outs = population_forward(x, weights, biases)
        
        # Evaluate fitness
        errors = np.array([error_fn(out, y_true) for out in outs])
        best_idx = np.argmin(errors)
        
        # Store results
        self.best_idx = best_idx
        self.last_error = errors[best_idx]
        self.last_output = outs[best_idx]
        self.last_weights = self.population[best_idx]['weights'].copy()
        self.last_bias = self.population[best_idx]['bias']
        
        # Update elite if better
        if self.last_error < self.elite_error:
            self.elite_error = self.last_error
            self.elite = {
                'weights': self.last_weights.copy(),
                'bias': self.last_bias
            }
        
        return self.last_output, self.last_error
    
    def evolve(
        self,
        x: np.ndarray,
        y_true: Union[np.ndarray, float],
        error_fn: ErrorFunction,
        mutation_strength: float,
        V_m: Optional[List[Dict[str, Any]]] = None,
        tau: Optional[float] = None
    ) -> None:
        """
        Evolve the neuron population through tournament selection and SBX crossover.
        
        Uses advanced evolutionary operators:
        - Tournament selection for parent selection
        - Simulated Binary Crossover (SBX) for recombination
        - Polynomial mutation for exploration
        
        Args:
            x: Input features
            y_true: Target value
            error_fn: Error function for fitness evaluation
            mutation_strength: Current mutation strength
            V_m: Significant mutation vector (optional)
            tau: Error threshold for local refinement (optional)
        """
        self._check_population_shapes()
        
        # Evaluate all individuals
        errors = np.array([
            error_fn(self._activate(np.dot(x, ind['weights']) + ind['bias']), y_true)
            for ind in self.population
        ])
        
        # Track elite
        best_idx = np.argmin(errors)
        if errors[best_idx] < self.elite_error:
            self.elite_error = errors[best_idx]
            self.elite = {
                'weights': self.population[best_idx]['weights'].copy(),
                'bias': self.population[best_idx]['bias']
            }
        
        # Tournament selection: select better survivors with pressure
        sorted_idx = np.argsort(errors)
        num_survivors = max(ELITE_COUNT, 3)
        survivors = [
            {'weights': self.population[i]['weights'].copy(), 'bias': self.population[i]['bias']}
            for i in sorted_idx[:num_survivors]
        ]
        
        # Add global elite
        if self.elite is not None:
            survivors.append({
                'weights': self.elite['weights'].copy(),
                'bias': self.elite['bias']
            })
        
        # Local gradient descent refinement on best
        if LOCAL_GD_ENABLED and len(survivors) > 0:
            self._local_refinement(survivors[0], x, y_true, error_fn, tau, V_m)
        
        # Create new population with SBX crossover and polynomial mutation
        new_pop = survivors.copy()
        
        while len(new_pop) < self.pop_size:
            # Tournament selection for parents
            tournament = np.random.choice(len(survivors), size=min(TOURNAMENT_SIZE, len(survivors)), replace=False)
            parent1_idx = tournament[np.argmin([errors[sorted_idx[i]] for i in tournament if i < len(sorted_idx)])] if len(tournament) > 0 else 0
            
            tournament2 = np.random.choice(len(survivors), size=min(TOURNAMENT_SIZE, len(survivors)), replace=False)
            parent2_idx = tournament2[0] if tournament2[0] != parent1_idx else (tournament2[1] if len(tournament2) > 1 else 0)
            
            parent1 = survivors[min(parent1_idx, len(survivors)-1)]
            parent2 = survivors[min(parent2_idx, len(survivors)-1)]
            
            # SBX Crossover
            if random.random() < CROSSOVER_PROB:
                # Simulated Binary Crossover
                u = np.random.random(self.input_dim)
                beta = np.where(
                    u <= 0.5,
                    (2 * u) ** (1 / (ETA_CROSSOVER + 1)),
                    (1 / (2 * (1 - u + 1e-10))) ** (1 / (ETA_CROSSOVER + 1))
                )
                child_w = 0.5 * ((1 + beta) * parent1['weights'] + (1 - beta) * parent2['weights'])
                child_b = 0.5 * (parent1['bias'] + parent2['bias'])
            else:
                child_w = parent1['weights'].copy()
                child_b = parent1['bias']
            
            # Polynomial mutation
            mutation_mask = np.random.random(self.input_dim) < mutation_strength
            u = np.random.random(self.input_dim)
            delta = np.where(
                u < 0.5,
                (2 * u) ** (1 / (ETA_MUTATION + 1)) - 1,
                1 - (2 * (1 - u)) ** (1 / (ETA_MUTATION + 1))
            )
            child_w[mutation_mask] += delta[mutation_mask] * mutation_strength
            
            if random.random() < mutation_strength:
                u_b = random.random()
                delta_b = ((2 * u_b) ** (1 / (ETA_MUTATION + 1)) - 1) if u_b < 0.5 else (1 - (2 * (1 - u_b)) ** (1 / (ETA_MUTATION + 1)))
                child_b += delta_b * mutation_strength
            
            # V_m influence (memory-guided evolution)
            if V_m and len(V_m) > 0 and random.random() < VM_INFLUENCE_PROB:
                v = random.choice(V_m)
                if v['weights'].shape == (self.input_dim,):
                    child_w += v['weights'] * 0.3
                child_b += v['bias'] * 0.3
            
            new_pop.append({
                'weights': child_w.astype(np.float32),
                'bias': np.float32(child_b)
            })
        
        self.population = new_pop[:self.pop_size]
    
    def _local_refinement(
        self,
        best: Individual,
        x: np.ndarray,
        y_true: Union[np.ndarray, float],
        error_fn: ErrorFunction,
        tau: Optional[float],
        V_m: Optional[List[Dict[str, Any]]]
    ) -> None:
        """
        Apply local gradient descent refinement aligned with V_m direction.
        
        Args:
            best: Best individual to refine
            x: Input features
            y_true: Target value
            error_fn: Error function
            tau: Error threshold
            V_m: Significant mutation vector
        """
        w = best['weights']
        b = best['bias']
        out = self._activate(np.dot(x, w) + b)
        err_current = error_fn(out, y_true)
        
        trigger = (tau is None) or (err_current >= tau)
        if not trigger:
            return
        
        # Compute direction from V_m mean
        d_hat_w = None
        d_hat_b = None
        
        if V_m and len(V_m) > 0:
            vm_filtered = [
                v for v in V_m 
                if isinstance(v.get('weights', None), np.ndarray) 
                and v['weights'].shape == w.shape
            ]
            if len(vm_filtered) > 0:
                w_bar = np.mean([v['weights'] for v in vm_filtered], axis=0)
                b_bar = float(np.mean([v['bias'] for v in vm_filtered]))
                d_w = (w_bar - w).astype(np.float32)
                d_b = np.float32(b_bar - b)
                denom = np.sqrt(np.dot(d_w, d_w) + float(d_b) * float(d_b)) + 1e-12
                d_hat_w = d_w / denom
                d_hat_b = d_b / denom
        
        # Compute gradient signal
        if isinstance(y_true, np.ndarray):
            y_bar = float(np.mean(y_true))
        else:
            y_bar = float(y_true)
        
        grad_signal = 2.0 * (float(out) - y_bar)
        s_w = (-LOCAL_GD_LR * grad_signal) * x.astype(np.float32)
        s_b = np.float32(-LOCAL_GD_LR * grad_signal)
        
        # Project to align with V_m direction
        if d_hat_w is not None:
            dot_sb = float(np.dot(s_w, d_hat_w) + float(s_b) * float(d_hat_b))
            if dot_sb < 0.0:
                s_w = s_w - dot_sb * d_hat_w
                s_b = np.float32(float(s_b) - dot_sb * float(d_hat_b))
        
        # Try gradient step
        w_prop = (w + s_w).astype(np.float32)
        b_prop = np.float32(b + s_b)
        out_prop = self._activate(np.dot(x, w_prop) + b_prop)
        err_prop = error_fn(out_prop, y_true)
        improved = (err_current - err_prop) > LOCAL_MIN_IMPROVEMENT
        
        # Line search if gradient step didn't improve
        if (not improved) and (d_hat_w is not None):
            best_alpha = 0.0
            best_err = err_current
            for alpha in LOCAL_LINESEARCH_ALPHAS:
                if alpha <= 0.0:
                    continue
                w_ls = (w + alpha * d_hat_w).astype(np.float32)
                b_ls = np.float32(b + alpha * float(d_hat_b))
                out_ls = self._activate(np.dot(x, w_ls) + b_ls)
                err_ls = error_fn(out_ls, y_true)
                if err_ls + 1e-12 < best_err:
                    best_err = err_ls
                    best_alpha = alpha
            
            if best_alpha > 0.0:
                w_prop = (w + best_alpha * d_hat_w).astype(np.float32)
                b_prop = np.float32(b + best_alpha * float(d_hat_b))
                err_prop = best_err
                improved = True
        
        # Apply improvement
        if improved:
            best['weights'] = w_prop
            best['bias'] = b_prop


class OutputNeuron(EvoNeuron):
    """Output neuron with linear activation (for classification logits)."""
    
    def _activate(self, x: float) -> float:
        """Linear activation."""
        return x


class RegressionOutputNeuron(EvoNeuron):
    """Output neuron for regression tasks with linear activation."""
    
    def _activate(self, x: float) -> float:
        """Linear activation for regression."""
        return x
