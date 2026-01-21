"""
Feature Selection Module

Functions for selecting the best features for training.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


def select_best_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    min_accuracy: float = 0.0
) -> Tuple[np.ndarray, List[int], List[str], Dict[str, float]]:
    """
    Select features based on individual feature accuracy.
    
    For each feature, evaluate its individual accuracy using a threshold rule.
    Keep only features whose accuracy is above min_accuracy.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array
        feature_names: Optional list of feature names
        min_accuracy: Minimum accuracy threshold
        
    Returns:
        Tuple of:
            - X_selected: Filtered feature matrix
            - selected_indices: Indices of selected features
            - selected_names: Names of selected features
            - feature_scores: Dict mapping feature names to accuracy scores
    """
    n_samples, n_features = X.shape
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    selected_indices = []
    feature_scores = {}
    dropped_features = []
    
    for i in range(n_features):
        feature = X[:, i]
        best_acc = 0
        best_thresh = feature.mean()
        best_dir = 1
        
        # Try thresholds between min and max
        for thresh in np.linspace(feature.min(), feature.max(), 100):
            for direction in [1, -1]:
                if direction == 1:
                    pred = (feature > thresh).astype(int)
                else:
                    pred = (feature < thresh).astype(int)
                
                acc = np.mean(pred == y.ravel())
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
                    best_dir = direction
        
        feature_scores[feature_names[i]] = best_acc
        
        if best_acc >= min_accuracy:
            selected_indices.append(i)
        else:
            dropped_features.append((feature_names[i], best_acc))
    
    print("\n--- Feature Selection Report ---")
    print(f"Minimum accuracy threshold: {min_accuracy}")
    print(f"Selected features ({len(selected_indices)}/{n_features}):")
    
    for i in selected_indices:
        print(f"  {feature_names[i]}: accuracy={feature_scores[feature_names[i]]:.3f}")
    
    if dropped_features:
        print(f"Dropped features ({len(dropped_features)}):")
        for name, acc in dropped_features:
            print(f"  {name}: accuracy={acc:.3f}")
    
    print("-------------------------------\n")
    
    X_selected = X[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]
    
    return X_selected, selected_indices, selected_names, feature_scores
