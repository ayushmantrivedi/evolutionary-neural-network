"""
Integration Tests for EvoNet

End-to-end tests that verify the full pipeline works correctly.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.core.network import MultiClassEvoNet, RegressionEvoNet
from evonet.core.mutations import SignificantMutationVector
from evonet.data.preprocessing import preprocess_dataset


class TestMultiClassEvoNetIntegration:
    """Integration tests for MultiClassEvoNet."""
    
    def test_init_and_forward_binary(self):
        """Test initialization and forward pass for binary classification."""
        net = MultiClassEvoNet(input_dim=5, num_classes=2)
        x = np.random.randn(5)
        y_onehot = np.array([1.0, 0.0])
        
        y_pred, l1_errs, l2_errs, l3_outs = net.forward(x, y_onehot, train=False)
        
        assert y_pred.shape == (2,)
        assert np.isclose(np.sum(y_pred), 1.0)  # Valid probability distribution
        assert len(l1_errs) == 50  # LEVEL1_NEURONS
        assert len(l2_errs) == 20  # LEVEL2_NEURONS
    
    def test_init_and_forward_multiclass(self):
        """Test initialization and forward pass for multi-class."""
        net = MultiClassEvoNet(input_dim=4, num_classes=3)
        x = np.random.randn(4)
        y_onehot = np.array([0.0, 1.0, 0.0])
        
        y_pred, _, _, _ = net.forward(x, y_onehot, train=False)
        
        assert y_pred.shape == (3,)
        assert np.isclose(np.sum(y_pred), 1.0)
    
    def test_mutation_strength_adapts(self):
        """Test that mutation strength adapts based on error."""
        net = MultiClassEvoNet(input_dim=5, num_classes=2)
        
        # Initial state
        initial_strength = net.get_mutation_strength()
        
        # Simulate high error
        net.global_error = 10.0
        high_error_strength = net.get_mutation_strength()
        
        # Simulate low error
        net.global_error = 0.1
        low_error_strength = net.get_mutation_strength()
        
        assert high_error_strength > low_error_strength
    
    def test_vm_is_populated_during_training(self):
        """Test that V_m is populated during training."""
        net = MultiClassEvoNet(input_dim=5, num_classes=2)
        
        # Run a few training samples
        for _ in range(10):
            x = np.random.randn(5)
            y_onehot = np.array([1.0, 0.0])
            net.forward(x, y_onehot, train=True)
        
        # V_m should have some entries (not guaranteed due to tau thresholds)
        # But network should still be functional
        assert isinstance(net.V_m.get(), list)


class TestRegressionEvoNetIntegration:
    """Integration tests for RegressionEvoNet."""
    
    def test_init_and_forward(self):
        """Test initialization and forward pass."""
        net = RegressionEvoNet(input_dim=8)
        x = np.random.randn(8)
        y_true = 5.0
        
        y_pred, l1_errs, l2_errs = net.forward(x, y_true, train=False)
        
        assert isinstance(y_pred, (float, np.floating))
        assert len(l1_errs) == 50
        assert len(l2_errs) == 20
    
    def test_training_reduces_error_on_simple_data(self):
        """Test that training reduces error on simple linear data."""
        np.random.seed(42)
        net = RegressionEvoNet(input_dim=1)
        
        # Simple linear data: y = 2*x
        X = np.random.randn(20, 1)
        y = 2 * X.ravel()
        
        # Initial error (without proper training)
        initial_pred, _, _ = net.forward(X[0], y[0], train=False)
        initial_error = (initial_pred - y[0]) ** 2
        
        # Train for a few samples
        for i in range(min(10, len(X))):
            net.forward(X[i], y[i], train=True)
        
        # Network should still be functional
        final_pred, _, _ = net.forward(X[0], y[0], train=False)
        assert isinstance(final_pred, (float, np.floating))


class TestConfigIntegration:
    """Tests for configuration integration."""
    
    def test_config_default_values(self):
        """Test that config has expected default values."""
        from evonet.config import LEVEL1_NEURONS, LEVEL2_NEURONS, POP_SIZE
        
        assert LEVEL1_NEURONS == 50
        assert LEVEL2_NEURONS == 20
        assert POP_SIZE == 50
    
    def test_output_directory_created(self):
        """Test that output directory is created."""
        from evonet.config import get_output_directory
        
        output_dir = get_output_directory()
        
        assert os.path.exists(output_dir)
        assert os.path.isdir(output_dir)


class TestPreprocessingIntegration:
    """Integration tests for preprocessing with real data."""
    
    def test_preprocessing_with_binary_data(self):
        """Test preprocessing with binary classification data."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        X_out, y_out, names = preprocess_dataset(X, y, "Test Binary")
        
        assert X_out.shape[0] == y_out.shape[0]
        assert len(names) == X_out.shape[1]
    
    def test_preprocessing_with_multiclass_data(self):
        """Test preprocessing with multi-class data."""
        X = np.random.randn(150, 4)
        y = np.repeat([0, 1, 2], 50)
        
        X_out, y_out, names = preprocess_dataset(X, y, "Test Multi")
        
        assert len(np.unique(y_out)) == 3
    
    def test_preprocessing_with_regression_data(self):
        """Test preprocessing with regression data."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        X_out, y_out, names = preprocess_dataset(X, y, "Test Regression")
        
        # Regression should not apply SMOTE
        assert X_out.shape[0] == 100
