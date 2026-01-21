"""
Evolutionary Search Module

Parallel evolutionary search for optimal neuron weights.
"""

import numpy as np
from typing import Tuple, Optional


def parallel_neuron_predict(
    X: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    n_classes: int = 2
) -> np.ndarray:
    """
    Parallel prediction using multiple neurons.
    
    Args:
        X: Input features (n_samples, n_features)
        weights: Neuron weights (n_neurons, n_features)
        biases: Neuron biases (n_neurons,)
        n_classes: Number of classes for classification
        
    Returns:
        np.ndarray: Predictions (n_samples,)
    """
    if n_classes == 2:
        # Binary classification: voting
        neuron_outputs = (np.dot(X, weights.T) + biases) > 0
        votes = np.sum(neuron_outputs, axis=1)
        return (votes >= (weights.shape[0] / 2)).astype(int)
    else:
        # Multi-class: argmax
        neuron_outputs = np.dot(X, weights.T) + biases
        return np.argmax(neuron_outputs, axis=1)


def evolutionary_search_parallel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_neurons: int = 200,
    n_iter: int = 100,
    pop_size: int = 30,
    early_stop_thresh: float = 0.005,
    patience: int = 10,
    n_classes: int = 2
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Evolutionary search for optimal parallel neuron weights.
    
    Uses random search with early stopping to find the best
    weight configuration for a layer of parallel neurons.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_neurons: Number of neurons
        n_iter: Number of iterations
        pop_size: Population size per iteration
        early_stop_thresh: Improvement threshold for early stopping
        patience: Patience for early stopping
        n_classes: Number of classes
        
    Returns:
        Tuple of (best_weights, best_biases, best_accuracy)
    """
    n_features = X_train.shape[1]
    best_acc = 0.0
    best_weights: Optional[np.ndarray] = None
    best_biases: Optional[np.ndarray] = None
    rng = np.random.default_rng(42)
    no_improve_count = 0
    
    for iter_idx in range(n_iter):
        improved = False
        
        for _ in range(pop_size):
            weights = rng.normal(0, 1, (n_neurons, n_features))
            biases = rng.normal(0, 1, n_neurons)
            
            val_pred = parallel_neuron_predict(X_val, weights, biases, n_classes)
            val_acc = np.mean(val_pred.ravel() == y_val.ravel())
            
            if val_acc > best_acc + early_stop_thresh:
                best_acc = val_acc
                best_weights = weights.copy()
                best_biases = biases.copy()
                improved = True
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        if no_improve_count >= patience:
            print(f"Early stopping at iteration {iter_idx+1} due to no improvement > "
                  f"{early_stop_thresh} for {patience} iterations.")
            break
    
    return best_weights, best_biases, best_acc
