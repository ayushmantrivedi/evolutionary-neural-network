"""
Training Methods Module

Alternative training methods for the evolutionary neural network.
Includes mini-batch evolution and early stopping training.
"""

import numpy as np
import logging
from typing import Optional

from evonet.core.losses import mse_loss

# Configure logging
logger = logging.getLogger(__name__)


def mini_batch_evolution_training(
    model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    iterations: int = 1000,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> float:
    """
    Fast training using mini-batch evolution.
    
    Instead of full epochs, samples mini-batches and evolves on them.
    Much faster for large datasets.
    
    Args:
        model: RegressionEvoNet instance
        X: Training features
        y: Training targets
        batch_size: Mini-batch size
        iterations: Number of evolution iterations
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        
    Returns:
        float: Average loss over iterations
    """
    print(f"Starting mini-batch evolution training...")
    print(f"Batch size: {batch_size}, Iterations: {iterations}")
    
    n_samples = X.shape[0]
    total_loss = 0.0
    
    for iteration in range(iterations):
        # Sample mini-batch
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        batch_loss = 0.0
        for i in range(batch_size):
            x = X_batch[i]
            y_true = y_batch[i]
            y_pred, l1_errs, l2_errs = model.forward(x, y_true, train=True)
            loss = mse_loss(y_pred, y_true)
            batch_loss += loss
            
            # Update V_m
            l1_bench_idx = np.argmin(l1_errs)
            l1_bench = model.level1[l1_bench_idx]
            if l1_bench.last_error is not None and l1_bench.last_error < model.tau1:
                model.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
            
            l2_bench_idx = np.argmin(l2_errs)
            l2_bench = model.level2[l2_bench_idx]
            if l2_bench.last_error is not None and l2_bench.last_error < model.tau2:
                model.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
        
        avg_batch_loss = batch_loss / batch_size
        total_loss += avg_batch_loss
        model.global_error = avg_batch_loss
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Batch MSE Loss: {avg_batch_loss:.4f}")
        
        if iteration % 500 == 0:
            print(f"[Thought] Iteration {iteration}: Global error is {model.global_error:.4f}, "
                  f"mutation strength now {model.get_mutation_strength():.4f}")
        
        if X_val is not None and iteration % 200 == 0:
            val_loss, _ = model.evaluate(X_val, y_val)
            print(f"[Validation] MSE Loss: {val_loss:.4f}")
    
    return total_loss / iterations


def early_stopping_training(
    model,
    X: np.ndarray,
    y: np.ndarray,
    patience: int = 20,
    min_delta: float = 0.001,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> float:
    """
    Training with early stopping to avoid overfitting.
    
    Stops training when validation loss stops improving.
    
    Args:
        model: RegressionEvoNet instance
        X: Training features
        y: Training targets
        patience: Number of iterations without improvement before stopping
        min_delta: Minimum improvement to count as progress
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        
    Returns:
        float: Final average loss
    """
    print(f"Starting early stopping training...")
    print(f"Patience: {patience}, Min delta: {min_delta}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    iteration = 0
    avg_loss = 0.0
    
    while patience_counter < patience:
        # Sample batch
        batch_indices = np.random.choice(len(X), min(64, len(X)), replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        batch_loss = 0.0
        for i in range(len(X_batch)):
            x = X_batch[i]
            y_true = y_batch[i]
            y_pred, l1_errs, l2_errs = model.forward(x, y_true, train=True)
            loss = mse_loss(y_pred, y_true)
            batch_loss += loss
            
            # Update V_m
            l1_bench_idx = np.argmin(l1_errs)
            l1_bench = model.level1[l1_bench_idx]
            if l1_bench.last_error is not None and l1_bench.last_error < model.tau1:
                model.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
            
            l2_bench_idx = np.argmin(l2_errs)
            l2_bench = model.level2[l2_bench_idx]
            if l2_bench.last_error is not None and l2_bench.last_error < model.tau2:
                model.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
        
        avg_loss = batch_loss / len(X_batch)
        model.global_error = avg_loss
        
        # Early stopping check
        if X_val is not None:
            val_loss, _ = model.evaluate(X_val, y_val)
            
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: Train MSE: {avg_loss:.4f}, "
                      f"Val MSE: {val_loss:.4f}, Patience: {patience_counter}")
        else:
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: MSE Loss: {avg_loss:.4f}")
        
        iteration += 1
        
        if iteration > 2000:  # Max iterations
            break
    
    print(f"Training stopped after {iteration} iterations")
    return avg_loss
