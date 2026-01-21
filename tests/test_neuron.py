"""
Tests for EvoNeuron and related neuron classes.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.core.neuron import EvoNeuron, OutputNeuron, RegressionOutputNeuron, population_forward
from evonet.core.losses import mse_loss


class TestPopulationForward:
    """Tests for the population_forward function."""
    
    def test_population_forward_shape(self):
        """Test output shape of population_forward."""
        x = np.random.randn(10).astype(np.float32)
        weights = np.random.randn(20, 10).astype(np.float32)
        biases = np.random.randn(20).astype(np.float32)
        
        result = population_forward(x, weights, biases)
        
        assert result.shape == (20,)
    
    def test_population_forward_computation(self):
        """Test correctness of population_forward computation."""
        x = np.array([1.0, 2.0, 3.0]).astype(np.float32)
        weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).astype(np.float32)
        biases = np.array([0.0, 1.0]).astype(np.float32)
        
        result = population_forward(x, weights, biases)
        
        expected = np.array([1.0, 3.0])  # [1*1 + 0, 1*2 + 1]
        np.testing.assert_array_almost_equal(result, expected)


class TestEvoNeuron:
    """Tests for the EvoNeuron class."""
    
    def test_initialization(self):
        """Test EvoNeuron initialization."""
        neuron = EvoNeuron(input_dim=10, pop_size=20)
        
        assert neuron.input_dim == 10
        assert neuron.pop_size == 20
        assert len(neuron.population) == 20
        assert neuron.elite is None
        assert neuron.elite_error == float('inf')
    
    def test_population_individual_shape(self):
        """Test that each individual has correct weight shape."""
        neuron = EvoNeuron(input_dim=15, pop_size=10)
        
        for ind in neuron.population:
            assert ind['weights'].shape == (15,)
            assert isinstance(ind['bias'], (float, np.floating))
    
    def test_forward_returns_correct_types(self, random_seed):
        """Test forward pass returns correct types."""
        neuron = EvoNeuron(input_dim=5)
        x = np.random.randn(5)
        y_true = 1.0
        
        output, error = neuron.forward(x, y_true, mse_loss, 0.1)
        
        assert isinstance(output, (float, np.floating))
        assert isinstance(error, (float, np.floating))
    
    def test_forward_updates_last_values(self, random_seed):
        """Test that forward updates last_* attributes."""
        neuron = EvoNeuron(input_dim=5)
        x = np.random.randn(5)
        y_true = 1.0
        
        neuron.forward(x, y_true, mse_loss, 0.1)
        
        assert neuron.last_error is not None
        assert neuron.last_output is not None
        assert neuron.last_weights is not None
        assert neuron.last_bias is not None
    
    def test_forward_updates_elite_on_improvement(self, random_seed):
        """Test that elite is updated when error improves."""
        neuron = EvoNeuron(input_dim=5)
        x = np.random.randn(5)
        y_true = 0.0  # Easy target
        
        # Force a very low error scenario
        neuron.population[0]['weights'] = np.zeros(5).astype(np.float32)
        neuron.population[0]['bias'] = np.float32(0.0)
        
        neuron.forward(x, y_true, mse_loss, 0.1)
        
        # Elite should be updated since any error < inf
        assert neuron.elite is not None
    
    def test_evolve_maintains_population_size(self, random_seed):
        """Test that evolve maintains population size."""
        neuron = EvoNeuron(input_dim=5, pop_size=15)
        x = np.random.randn(5)
        y_true = 1.0
        
        neuron.evolve(x, y_true, mse_loss, 0.1)
        
        assert len(neuron.population) == 15
    
    def test_evolve_with_vm_influence(self, random_seed):
        """Test evolve with V_m influence."""
        neuron = EvoNeuron(input_dim=5, pop_size=10)
        x = np.random.randn(5)
        y_true = 1.0
        
        V_m = [
            {'weights': np.random.randn(5).astype(np.float32), 'bias': 0.5}
        ]
        
        # Run multiple evolutions to increase chance of V_m influence
        for _ in range(10):
            neuron.evolve(x, y_true, mse_loss, 0.1, V_m=V_m)
        
        # Should still have correct population size
        assert len(neuron.population) == 10


class TestOutputNeuron:
    """Tests for OutputNeuron class."""
    
    def test_inherits_from_evoneuron(self):
        """Test that OutputNeuron inherits from EvoNeuron."""
        neuron = OutputNeuron(input_dim=10)
        
        assert isinstance(neuron, EvoNeuron)
    
    def test_activation_is_linear(self):
        """Test that activation is linear (identity)."""
        neuron = OutputNeuron(input_dim=5)
        
        assert neuron._activate(5.0) == 5.0
        assert neuron._activate(-3.0) == -3.0
        assert neuron._activate(0.0) == 0.0


class TestRegressionOutputNeuron:
    """Tests for RegressionOutputNeuron class."""
    
    def test_inherits_from_evoneuron(self):
        """Test that RegressionOutputNeuron inherits from EvoNeuron."""
        neuron = RegressionOutputNeuron(input_dim=10)
        
        assert isinstance(neuron, EvoNeuron)
    
    def test_activation_is_linear(self):
        """Test that activation is linear (identity)."""
        neuron = RegressionOutputNeuron(input_dim=5)
        
        assert neuron._activate(10.5) == 10.5
        assert neuron._activate(-7.2) == -7.2
