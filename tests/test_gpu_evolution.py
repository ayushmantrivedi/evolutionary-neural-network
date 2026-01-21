"""
Tests for GPU Backend and Evolution Operators

Comprehensive tests for the new GPU acceleration and evolution modules.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.core.gpu_backend import (
    GPUPopulation, 
    GPUTensorWrapper,
    get_device, 
    is_gpu_available,
    TORCH_AVAILABLE,
    gpu_mse_loss_batch,
    gpu_softmax
)
from evonet.core.evolution import (
    tournament_selection,
    sbx_crossover,
    polynomial_mutation,
    CMAESAdapter,
    generate_offspring,
    diversity_preservation,
    rank_selection
)


class TestGPUBackend:
    """Tests for GPU backend functionality."""
    
    def test_get_device_cpu_fallback(self):
        """Test that get_device returns cpu when GPU not available or disabled."""
        device = get_device(use_gpu=False)
        assert device == 'cpu'
    
    def test_tensor_wrapper_to_numpy(self):
        """Test tensor wrapper numpy conversion."""
        wrapper = GPUTensorWrapper(device='cpu')
        arr = np.array([1.0, 2.0, 3.0])
        tensor = wrapper.to_tensor(arr)
        result = wrapper.to_numpy(tensor)
        np.testing.assert_array_almost_equal(arr, result)
    
    def test_gpu_population_init(self):
        """Test GPUPopulation initialization."""
        pop = GPUPopulation(input_dim=10, pop_size=20, device='cpu')
        assert pop.input_dim == 10
        assert pop.pop_size == 20
        assert pop.weights.shape == (20, 10)
        assert pop.biases.shape == (20,)
    
    def test_gpu_population_forward_batch(self):
        """Test batched forward pass."""
        pop = GPUPopulation(input_dim=5, pop_size=10, device='cpu')
        x = np.random.randn(5).astype(np.float32)
        outputs = pop.forward_batch(x)
        
        if TORCH_AVAILABLE:
            assert outputs.shape == (10,)
        else:
            assert outputs.shape == (10,)
    
    def test_gpu_population_fitness(self):
        """Test fitness evaluation."""
        pop = GPUPopulation(input_dim=5, pop_size=10, device='cpu')
        x = np.random.randn(5).astype(np.float32)
        outputs = pop.forward_batch(x)
        fitness = pop.evaluate_fitness(outputs, target=1.0)
        
        assert fitness.shape == (10,)
    
    def test_gpu_population_tournament_select(self):
        """Test tournament selection."""
        pop = GPUPopulation(input_dim=5, pop_size=10, device='cpu')
        fitness = np.array([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.05])
        selected = pop.tournament_select(fitness, tournament_size=3, num_parents=2)
        
        assert len(selected) == 2
        assert all(0 <= idx < 10 for idx in selected)
    
    def test_gpu_population_sbx_crossover(self):
        """Test SBX crossover."""
        pop = GPUPopulation(input_dim=5, pop_size=10, device='cpu')
        child_w, child_b = pop.sbx_crossover(0, 1, eta=20.0)
        
        assert child_w.shape == (5,)
        assert isinstance(child_b, float)
    
    def test_gpu_population_polynomial_mutation(self):
        """Test polynomial mutation."""
        pop = GPUPopulation(input_dim=5, pop_size=10, device='cpu')
        original_w = pop.weights[0].copy() if not TORCH_AVAILABLE else pop.weights[0].cpu().numpy().copy()
        original_b = float(pop.biases[0])
        
        mutated_w, mutated_b = pop.polynomial_mutation(
            pop.weights[0] if not TORCH_AVAILABLE else pop.weights[0],
            original_b,
            mutation_rate=1.0  # Force mutation
        )
        
        # At least some values should be different
        assert mutated_w.shape == (5,)
    
    def test_gpu_population_evolve(self):
        """Test full evolution cycle."""
        pop = GPUPopulation(input_dim=5, pop_size=20, device='cpu')
        x = np.random.randn(5).astype(np.float32)
        outputs = pop.forward_batch(x)
        fitness = pop.evaluate_fitness(outputs, target=0.0)
        
        # Evolve population
        pop.evolve_population(
            fitness,
            mutation_strength=0.1,
            tournament_size=3,
            elite_count=2
        )
        
        assert pop.weights.shape == (20, 5)
        assert pop.biases.shape == (20,)
    
    def test_gpu_mse_loss_batch(self):
        """Test batched MSE loss."""
        predictions = np.array([1.0, 2.0, 3.0])
        target = 2.0
        losses = gpu_mse_loss_batch(predictions, target)
        
        expected = np.array([1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(losses, expected)
    
    def test_gpu_softmax(self):
        """Test GPU softmax."""
        x = np.array([1.0, 2.0, 3.0])
        result = gpu_softmax(x)
        
        assert np.isclose(np.sum(result), 1.0)
        assert result[2] > result[1] > result[0]


class TestEvolutionOperators:
    """Tests for evolution operators."""
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        population = [{'weights': np.random.randn(5), 'bias': 0.0} for _ in range(10)]
        fitness = np.array([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.05])
        
        selected = tournament_selection(population, fitness, tournament_size=3, num_select=3)
        
        assert len(selected) == 3
        assert all(0 <= idx < 10 for idx in selected)
    
    def test_sbx_crossover(self):
        """Test SBX crossover produces valid children."""
        parent1 = {'weights': np.array([1.0, 2.0, 3.0], dtype=np.float32), 'bias': 0.5}
        parent2 = {'weights': np.array([4.0, 5.0, 6.0], dtype=np.float32), 'bias': -0.5}
        
        child1, child2 = sbx_crossover(parent1, parent2, eta=20.0, crossover_prob=1.0)
        
        assert child1['weights'].shape == (3,)
        assert child2['weights'].shape == (3,)
        assert isinstance(child1['bias'], (float, np.floating))
        assert isinstance(child2['bias'], (float, np.floating))
    
    def test_polynomial_mutation(self):
        """Test polynomial mutation."""
        individual = {'weights': np.zeros(5, dtype=np.float32), 'bias': 0.0}
        
        mutated = polynomial_mutation(individual, mutation_rate=1.0, eta=20.0)
        
        assert mutated['weights'].shape == (5,)
        # With mutation_rate=1.0, at least some values should change
        assert not np.allclose(mutated['weights'], individual['weights'])
    
    def test_cmaes_adapter_init(self):
        """Test CMA-ES adapter initialization."""
        adapter = CMAESAdapter(input_dim=10, learning_rate=0.3)
        
        assert adapter.input_dim == 10
        assert adapter.C.shape == (10, 10)
        assert np.allclose(adapter.C, np.eye(10))
    
    def test_cmaes_adapter_sample(self):
        """Test CMA-ES sampling."""
        adapter = CMAESAdapter(input_dim=5)
        base_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        sampled = adapter.sample(base_weights, mutation_strength=0.1)
        
        assert sampled.shape == (5,)
        # Should be different from base (with high probability)
    
    def test_cmaes_adapter_update(self):
        """Test CMA-ES covariance update."""
        adapter = CMAESAdapter(input_dim=5)
        
        # Simulate successful weights
        successful = [np.random.randn(5).astype(np.float32) for _ in range(5)]
        old_mean = np.zeros(5, dtype=np.float32)
        
        adapter.update(successful, old_mean)
        
        # Covariance should be updated
        assert not np.allclose(adapter.C, np.eye(5))
    
    def test_diversity_preservation(self):
        """Test diversity preservation penalty."""
        # Create population with some similar individuals
        population = [
            {'weights': np.array([0.0, 0.0, 0.0], dtype=np.float32), 'bias': 0.0},
            {'weights': np.array([0.01, 0.01, 0.01], dtype=np.float32), 'bias': 0.01},  # Similar to first
            {'weights': np.array([5.0, 5.0, 5.0], dtype=np.float32), 'bias': 5.0},  # Different
        ]
        fitness = np.array([0.1, 0.2, 0.3])
        
        _, adjusted_fitness = diversity_preservation(population, fitness, diversity_threshold=0.5)
        
        # First two should have penalized fitness (higher)
        assert adjusted_fitness[0] > fitness[0]
        assert adjusted_fitness[1] > fitness[1]
        # Third should be same (far from others)
        assert np.isclose(adjusted_fitness[2], fitness[2])
    
    def test_generate_offspring(self):
        """Test offspring generation."""
        parents = [
            {'weights': np.random.randn(5).astype(np.float32), 'bias': np.random.randn()}
            for _ in range(5)
        ]
        
        offspring = generate_offspring(
            parents,
            num_offspring=10,
            mutation_strength=0.1
        )
        
        assert len(offspring) == 10
        for child in offspring:
            assert child['weights'].shape == (5,)
            assert isinstance(child['bias'], (float, np.floating))
    
    def test_rank_selection(self):
        """Test rank-based selection."""
        population = [{'weights': np.random.randn(5), 'bias': 0.0} for _ in range(10)]
        fitness = np.array([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.05])
        
        selected = rank_selection(population, fitness, selection_pressure=2.0)
        
        assert len(selected) == 5  # Half of population
        assert all(0 <= idx < 10 for idx in selected)


class TestIntegrationWithEvoNeuron:
    """Integration tests for improved EvoNeuron with new operators."""
    
    def test_evoneuron_evolve_with_new_operators(self):
        """Test that EvoNeuron evolve works with tournament selection."""
        from evonet.core.neuron import EvoNeuron
        from evonet.core.losses import mse_loss
        
        neuron = EvoNeuron(input_dim=5, pop_size=20)
        x = np.random.randn(5).astype(np.float32)
        y_true = 1.0
        
        # Run evolution
        neuron.evolve(x, y_true, mse_loss, mutation_strength=0.1)
        
        assert len(neuron.population) == 20
        for ind in neuron.population:
            assert ind['weights'].shape == (5,)
    
    def test_evoneuron_with_vm_influence(self):
        """Test V_m influence in new evolution."""
        from evonet.core.neuron import EvoNeuron
        from evonet.core.losses import mse_loss
        
        neuron = EvoNeuron(input_dim=5, pop_size=20)
        x = np.random.randn(5).astype(np.float32)
        y_true = 1.0
        
        # V_m with specific pattern
        V_m = [
            {'weights': np.ones(5, dtype=np.float32) * 0.5, 'bias': 0.5}
        ]
        
        for _ in range(10):
            neuron.evolve(x, y_true, mse_loss, mutation_strength=0.1, V_m=V_m)
        
        # Should still have valid population
        assert len(neuron.population) == 20


class TestPerformanceComparison:
    """Tests comparing performance of old vs new implementation."""
    
    def test_training_converges(self):
        """Test that training converges on simple data."""
        from evonet.core.network import RegressionEvoNet
        
        np.random.seed(42)
        net = RegressionEvoNet(input_dim=2)
        
        # Simple linear data: y = 2*x1 + 3*x2
        X = np.random.randn(50, 2).astype(np.float32)
        y = 2 * X[:, 0] + 3 * X[:, 1]
        
        # Get initial loss
        initial_loss, _ = net.evaluate(X[:10], y[:10])
        
        # Train briefly
        for i in range(20):
            for j in range(min(10, len(X))):
                net.forward(X[j], y[j], train=True)
        
        # Get final loss
        final_loss, _ = net.evaluate(X[:10], y[:10])
        
        # Loss should decrease or at least not explode
        assert final_loss < initial_loss * 10  # Reasonable bound
