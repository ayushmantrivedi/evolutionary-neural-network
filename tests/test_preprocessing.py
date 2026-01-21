"""
Tests for data preprocessing functions.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.data.preprocessing import detect_problem_type, preprocess_dataset


class TestDetectProblemType:
    """Tests for problem type detection."""
    
    def test_binary_classification(self):
        """Test detection of binary classification."""
        y = np.array([0, 1, 0, 1, 0, 1])
        
        result = detect_problem_type(y)
        
        assert result == 'binary_classification'
    
    def test_multi_class_classification(self):
        """Test detection of multi-class classification."""
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        
        result = detect_problem_type(y)
        
        assert result == 'multi_class_classification'
    
    def test_regression(self):
        """Test detection of regression (many unique values)."""
        y = np.random.randn(100)  # Continuous values
        
        result = detect_problem_type(y)
        
        assert result == 'regression'
    
    def test_edge_case_10_classes(self):
        """Test edge case with exactly 10 unique values."""
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        result = detect_problem_type(y)
        
        assert result == 'multi_class_classification'
    
    def test_edge_case_11_classes(self):
        """Test edge case with 11 unique values (regression)."""
        y = np.arange(11)
        
        result = detect_problem_type(y)
        
        assert result == 'regression'


class TestPreprocessDataset:
    """Tests for preprocess_dataset function."""
    
    def test_returns_correct_tuple(self, sample_binary_data):
        """Test returns (X, y, feature_names) tuple."""
        X, y = sample_binary_data
        
        X_out, y_out, names = preprocess_dataset(X, y, "Test")
        
        assert isinstance(X_out, np.ndarray)
        assert isinstance(y_out, np.ndarray)
        assert isinstance(names, list)
    
    def test_X_is_scaled(self, sample_binary_data):
        """Test that X is standardized."""
        X, y = sample_binary_data
        
        X_out, _, _ = preprocess_dataset(X, y, "Test")
        
        # Standardized data should have mean ~0 and std ~1
        assert np.abs(np.mean(X_out)) < 0.1
        assert np.abs(np.std(X_out) - 1.0) < 0.2
    
    def test_y_is_flattened(self, sample_binary_data):
        """Test that y is flattened to 1D."""
        X, y = sample_binary_data
        
        _, y_out, _ = preprocess_dataset(X, y, "Test")
        
        assert y_out.ndim == 1
    
    def test_feature_names_count_matches(self, sample_binary_data):
        """Test feature names count matches features."""
        X, y = sample_binary_data
        
        X_out, _, names = preprocess_dataset(X, y, "Test")
        
        assert len(names) == X_out.shape[1]
