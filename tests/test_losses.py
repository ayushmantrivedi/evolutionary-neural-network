"""
Tests for loss functions and activation functions.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.core.losses import mse_loss, ce_loss, ce_loss_with_confidence, softmax


class TestMSELoss:
    """Tests for MSE loss function."""
    
    def test_mse_loss_zero_for_perfect_prediction(self):
        """Test MSE is 0 for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        
        loss = mse_loss(pred, true)
        
        assert loss == 0.0
    
    def test_mse_loss_positive_for_errors(self):
        """Test MSE is positive for imperfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([2.0, 3.0, 4.0])
        
        loss = mse_loss(pred, true)
        
        assert loss > 0.0
        assert loss == 1.0  # Each error is 1, squared is 1, mean is 1
    
    def test_mse_loss_scalars(self):
        """Test MSE works with scalar inputs."""
        pred = 5.0
        true = 3.0
        
        loss = mse_loss(pred, true)
        
        assert loss == 4.0  # (5-3)^2 = 4


class TestCELoss:
    """Tests for Cross-Entropy loss function."""
    
    def test_ce_loss_low_for_confident_correct(self):
        """Test CE is low for confident correct predictions."""
        pred = np.array([0.9, 0.05, 0.05])
        true = np.array([1.0, 0.0, 0.0])
        
        loss = ce_loss(pred, true)
        
        assert loss < 0.2  # Low loss for confident correct
    
    def test_ce_loss_high_for_wrong_prediction(self):
        """Test CE is high for wrong predictions."""
        pred = np.array([0.1, 0.8, 0.1])
        true = np.array([1.0, 0.0, 0.0])
        
        loss = ce_loss(pred, true)
        
        assert loss > 2.0  # High loss for incorrect
    
    def test_ce_loss_clips_probabilities(self):
        """Test CE clips probabilities to avoid log(0)."""
        pred = np.array([0.0, 1.0, 0.0])  # Contains 0
        true = np.array([1.0, 0.0, 0.0])
        
        # Should not raise error
        loss = ce_loss(pred, true)
        
        assert np.isfinite(loss)


class TestCELossWithConfidence:
    """Tests for CE loss with confidence reward."""
    
    def test_confidence_reward_lowers_loss(self):
        """Test that high confidence reduces loss."""
        pred = np.array([0.9, 0.05, 0.05])
        true = np.array([1.0, 0.0, 0.0])
        
        loss_without_confidence = ce_loss(pred, true)
        loss_with_confidence = ce_loss_with_confidence(pred, true, reward_weight=0.1)
        
        # With confidence reward, loss should be slightly lower
        assert loss_with_confidence < loss_without_confidence
    
    def test_returns_finite_value(self):
        """Test always returns finite value."""
        pred = np.array([0.5, 0.3, 0.2])
        true = np.array([0.0, 1.0, 0.0])
        
        loss = ce_loss_with_confidence(pred, true)
        
        assert np.isfinite(loss)


class TestSoftmax:
    """Tests for softmax activation function."""
    
    def test_softmax_sums_to_one(self):
        """Test softmax output sums to 1."""
        x = np.array([1.0, 2.0, 3.0])
        
        result = softmax(x)
        
        assert np.isclose(np.sum(result), 1.0)
    
    def test_softmax_all_positive(self):
        """Test softmax output is all positive."""
        x = np.array([-5.0, 0.0, 5.0])
        
        result = softmax(x)
        
        assert np.all(result > 0)
    
    def test_softmax_largest_input_has_largest_output(self):
        """Test largest input gives largest output probability."""
        x = np.array([1.0, 5.0, 2.0])
        
        result = softmax(x)
        
        assert np.argmax(result) == 1  # index of 5.0
    
    def test_softmax_numerical_stability(self):
        """Test softmax is numerically stable for large inputs."""
        x = np.array([1000.0, 1001.0, 1002.0])
        
        result = softmax(x)
        
        assert np.all(np.isfinite(result))
        assert np.isclose(np.sum(result), 1.0)
