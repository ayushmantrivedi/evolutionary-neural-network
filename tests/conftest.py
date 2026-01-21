"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)
    
    return X, y


@pytest.fixture
def sample_multiclass_data():
    """Generate sample multi-class classification data."""
    np.random.seed(42)
    n_samples = 150
    n_features = 4
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.repeat(np.arange(n_classes), n_samples // n_classes).reshape(-1, 1)
    
    return X, y, n_classes


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    
    return X, y


@pytest.fixture
def small_input():
    """Small input for quick tests."""
    np.random.seed(42)
    return np.random.randn(5)


@pytest.fixture
def small_target():
    """Small target for quick tests."""
    return 1.0
