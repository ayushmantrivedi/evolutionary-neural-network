import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional
from evonet.core.neuron import EvoNeuron, population_forward
from evonet.config import POP_SIZE

class EvoAttentionLayer:
    """
    Revolutionary Attention mechanism for Neuroevolution.
    Instead of Gradient Descent, it evolves 'Context Masks' to highlight 
    relevant features during different market regimes.
    """
    def __init__(self, input_dim: int, pop_size: int = POP_SIZE):
        self.input_dim = input_dim
        self.pop_size = pop_size
        # Population of attention masks (weights that scale the input features)
        self.population = [
            {'weights': np.random.uniform(0, 1, input_dim).astype(np.float32)} 
            for _ in range(pop_size)
        ]
        self.best_idx = 0
        self.last_scores = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        """
        Applies attention masks to input X.
        Handles both single samples (1D) and mini-batches (2D).
        Returns the weighted feature vector from the best mask.
        """
        best_mask = self.population[self.best_idx]['weights']
        
        # Sigmoid-style gating with clipping to prevent overflow
        clipped_mask = np.clip(best_mask, -100.0, 100.0)
        attention_scores = 1.0 / (1.0 + np.exp(-clipped_mask))
        self.last_scores = attention_scores
        
        # Works for both [D] (single) and [B, D] (batch) inputs
        return x * attention_scores


    def evolve(self, fitness_scores: List[float], mutation_strength: float):
        """Standard neuroevolution logic for attention weights."""
        fitness = np.array(fitness_scores)
        self.best_idx = np.argmax(fitness)
        
        # Tournament-based evolution
        new_pop = []
        for _ in range(self.pop_size):
            # Select parent
            candidates = np.random.choice(self.pop_size, 3, replace=False)
            parent = self.population[candidates[np.argmax(fitness[candidates])]]
            
            # Mutate
            child_w = parent['weights'].copy()
            mutation_mask = np.random.random(self.input_dim) < mutation_strength
            child_w[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * mutation_strength
            
            new_pop.append({'weights': child_w.astype(np.float32)})
            
        self.population = new_pop

class ConfidenceHead:
    """
    Secondary output that predicts 'Model Confidence'.
    Used to inform the user when the model is unsure of its prediction.
    """
    def __init__(self, input_dim: int):
        self.neuron = EvoNeuron(input_dim)
    
    def get_confidence(self, x: np.ndarray) -> Union[float, np.ndarray]:
        # Map raw output to 0-1 range
        w = self.neuron.elite['weights'] if self.neuron.elite else self.neuron.population[0]['weights']
        b = self.neuron.elite['bias'] if self.neuron.elite else self.neuron.population[0]['bias']
        
        is_single = (x.ndim == 1)
        if is_single:
            raw = float(np.dot(w.astype(np.float32), x.astype(np.float32)) + float(b))
            raw_clipped = np.clip(raw, -100.0, 100.0)
            with np.errstate(over='ignore'):
                return float(1.0 / (1.0 + np.exp(-raw_clipped)))
        else:
            raw = np.dot(x.astype(np.float32), w.astype(np.float32).T) + float(b)
            raw_clipped = np.clip(raw, -100.0, 100.0)
            with np.errstate(over='ignore'):
                return 1.0 / (1.0 + np.exp(-raw_clipped))

