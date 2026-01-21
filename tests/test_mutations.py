"""
Tests for SignificantMutationVector (V_m) class.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.core.mutations import SignificantMutationVector


class TestSignificantMutationVector:
    """Tests for SignificantMutationVector class."""
    
    def test_initialization_empty(self):
        """Test V_m initializes empty."""
        vm = SignificantMutationVector(maxlen=20)
        
        assert len(vm) == 0
        assert vm.is_empty
    
    def test_add_single_mutation(self):
        """Test adding a single mutation."""
        vm = SignificantMutationVector(maxlen=20)
        weights = np.array([1.0, 2.0, 3.0])
        bias = 0.5
        
        vm.add(weights, bias)
        
        assert len(vm) == 1
        assert not vm.is_empty
    
    def test_get_returns_list(self):
        """Test get() returns list of dicts."""
        vm = SignificantMutationVector(maxlen=20)
        vm.add(np.array([1.0, 2.0]), 0.5)
        
        result = vm.get()
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert 'weights' in result[0]
        assert 'bias' in result[0]
    
    def test_maxlen_enforced(self):
        """Test maxlen is enforced."""
        vm = SignificantMutationVector(maxlen=5)
        
        for i in range(10):
            vm.add(np.array([float(i)]), float(i))
        
        assert len(vm) == 5
    
    def test_fifo_order(self):
        """Test FIFO order (oldest removed first)."""
        vm = SignificantMutationVector(maxlen=3)
        
        vm.add(np.array([1.0]), 1.0)
        vm.add(np.array([2.0]), 2.0)
        vm.add(np.array([3.0]), 3.0)
        vm.add(np.array([4.0]), 4.0)  # Should remove first
        
        result = vm.get()
        
        # First element should be 2.0 (1.0 removed)
        assert result[0]['bias'] == 2.0
        assert result[-1]['bias'] == 4.0
    
    def test_weights_are_copied(self):
        """Test weights are copied, not referenced."""
        vm = SignificantMutationVector(maxlen=20)
        weights = np.array([1.0, 2.0, 3.0])
        
        vm.add(weights, 0.5)
        
        # Modify original
        weights[0] = 999.0
        
        # V_m should still have original value
        result = vm.get()
        assert result[0]['weights'][0] == 1.0
    
    def test_clear(self):
        """Test clear() empties the vector."""
        vm = SignificantMutationVector(maxlen=20)
        vm.add(np.array([1.0, 2.0]), 0.5)
        vm.add(np.array([3.0, 4.0]), 0.7)
        
        vm.clear()
        
        assert len(vm) == 0
        assert vm.is_empty
    
    def test_default_maxlen(self):
        """Test default maxlen is 20."""
        vm = SignificantMutationVector()
        
        assert vm.maxlen == 20
