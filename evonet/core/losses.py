"""
Loss Functions and Activation Functions

This module contains all loss functions and activation functions
used in the evolutionary neural network.
"""

import numpy as np
from typing import Union

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, float]


def mse_loss(pred: ArrayLike, true: ArrayLike) -> float:
    """
    Mean Squared Error loss function.
    
    Args:
        pred: Predicted values
        true: True/target values
        
    Returns:
        float: Mean squared error between predictions and targets
    """
    return float(np.mean((pred - true) ** 2))


def ce_loss(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Cross-Entropy loss function for classification.
    
    Args:
        pred: Predicted probabilities (after softmax)
        true: One-hot encoded true labels
        
    Returns:
        float: Cross-entropy loss
    """
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    return float(-np.sum(true * np.log(pred)))


def ce_loss_with_confidence(
    pred: np.ndarray, 
    true: np.ndarray, 
    reward_weight: float = 0.1
) -> float:
    """
    Cross-Entropy loss with confidence reward.
    
    This loss function rewards high confidence in correct predictions
    by adding a margin-based bonus to the loss.
    
    Args:
        pred: Predicted probabilities (after softmax)
        true: One-hot encoded true labels
        reward_weight: Weight for the confidence reward term
        
    Returns:
        float: Modified cross-entropy loss with confidence reward
    """
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    
    # Standard cross-entropy
    ce = -np.sum(true * np.log(pred))
    
    # Confidence margin: correct class probability - max incorrect probability
    correct_prob = np.sum(pred * true)
    max_other_prob = np.max(pred * (1 - true))
    confidence_margin = correct_prob - max_other_prob
    
    # Reward for high confidence
    reward = reward_weight * confidence_margin
    
    return float(ce - reward)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.
    
    Converts raw logits to probability distribution.
    Uses numerical stability trick (subtract max).
    
    Args:
        x: Input array of logits
        
    Returns:
        np.ndarray: Probability distribution that sums to 1
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
