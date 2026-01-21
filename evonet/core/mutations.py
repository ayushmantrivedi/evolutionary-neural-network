"""
Significant Mutation Vector (V_m) Module

This module implements the Significant Mutation Vector, a key novel component
of the evolutionary neural network. V_m preserves successful evolutionary 
patterns to guide future mutations.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MutationRecord:
    """
    Record of a significant mutation.
    
    Attributes:
        weights: Weight vector that produced good results
        bias: Bias value that produced good results
    """
    weights: np.ndarray
    bias: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias
        }


class SignificantMutationVector:
    """
    Significant Mutation Vector (V_m) for preserving successful mutations.
    
    This is a key novel component of the evolutionary neural network.
    It maintains a history of successful mutations and uses them to
    guide future evolution, acting as a form of evolutionary memory.
    
    Attributes:
        maxlen: Maximum number of mutations to remember
        deque: Internal storage for mutation records
    
    Example:
        >>> vm = SignificantMutationVector(maxlen=20)
        >>> vm.add(np.array([0.1, 0.2, 0.3]), -0.5)
        >>> mutations = vm.get()
        >>> len(mutations)
        1
    """
    
    def __init__(self, maxlen: int = 20):
        """
        Initialize the Significant Mutation Vector.
        
        Args:
            maxlen: Maximum history length (default: 20)
        """
        self.maxlen = maxlen
        self._deque: deque = deque(maxlen=maxlen)
    
    def add(self, weights: np.ndarray, bias: float) -> None:
        """
        Add a significant mutation to the vector.
        
        Args:
            weights: Weight vector from successful mutation
            bias: Bias value from successful mutation
        """
        self._deque.append({
            'weights': weights.copy(),
            'bias': bias
        })
    
    def get(self) -> List[Dict[str, Any]]:
        """
        Get all stored mutations.
        
        Returns:
            List of mutation dictionaries with 'weights' and 'bias' keys
        """
        return list(self._deque)
    
    def __len__(self) -> int:
        """Return number of stored mutations."""
        return len(self._deque)
    
    def clear(self) -> None:
        """Clear all stored mutations."""
        self._deque.clear()
    
    @property
    def is_empty(self) -> bool:
        """Check if the vector is empty."""
        return len(self._deque) == 0
