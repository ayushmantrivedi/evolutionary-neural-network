"""
Neural Network Module

This module contains the main neural network classes:
- MultiClassEvoNet: For classification tasks
- RegressionEvoNet: For regression tasks

Both networks use evolutionary neurons organized in a 3-layer architecture.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any

from evonet.config import (
    LEVEL1_NEURONS, LEVEL2_NEURONS, POP_SIZE, TAU1, TAU2, 
    MUT_STRENGTH_BASE, MIN_MUT_STRENGTH, EPOCHS, 
    PRINT_INTERVAL, THOUGHT_INTERVAL, USE_SKIP_CONNECTIONS,
    USE_GPU
)
from evonet.core.neuron import EvoNeuron, OutputNeuron, RegressionOutputNeuron
from evonet.core.mutations import SignificantMutationVector
from evonet.core.losses import mse_loss, ce_loss, ce_loss_with_confidence, softmax
from evonet.core.gpu_backend import get_device, is_gpu_available, GPUTensorWrapper
from evonet.core.layers import EvoAttentionLayer, ConfidenceHead

# Configure logging
logger = logging.getLogger(__name__)


class MultiClassEvoNet:
    """
    Multi-Class Evolutionary Neural Network.
    
    A three-layer neural network using evolutionary neurons for 
    multi-class classification. Uses evolutionary algorithms instead
    of backpropagation for training.
    
    Architecture:
        - Layer 1: LEVEL1_NEURONS (50) evolutionary neurons
        - Layer 2: LEVEL2_NEURONS (20) evolutionary neurons  
        - Layer 3: num_classes output neurons
    
    Attributes:
        level1: List of first layer neurons
        level2: List of second layer neurons
        level3: List of output neurons
        num_classes: Number of output classes
        V_m: Significant Mutation Vector for evolutionary memory
        global_error: Current global error for mutation scaling
        
    Example:
        >>> net = MultiClassEvoNet(input_dim=10, num_classes=3)
        >>> net.train(X_train, y_train, y_train_onehot, epochs=50)
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize the multi-class evolutionary network.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
        """
        self.level1: List[EvoNeuron] = [
            EvoNeuron(input_dim) for _ in range(LEVEL1_NEURONS)
        ]
        self.level2: List[EvoNeuron] = [
            EvoNeuron(LEVEL1_NEURONS) for _ in range(LEVEL2_NEURONS)
        ]
        self.level3: List[OutputNeuron] = [
            OutputNeuron(LEVEL2_NEURONS + (LEVEL1_NEURONS if USE_SKIP_CONNECTIONS else 0)) for _ in range(num_classes)
        ]
        self.num_classes = num_classes
        
        # --- NEW PRO FEATURES ---
        self.attention = EvoAttentionLayer(input_dim)
        self.confidence_head = ConfidenceHead(LEVEL2_NEURONS)
        
        self.V_m = SignificantMutationVector()
        self.tau1 = TAU1
        self.tau2 = TAU2
        self.pop_size = POP_SIZE
        self.mut_strength_base = MUT_STRENGTH_BASE
        self.global_error: float = 1.0
        self.device = get_device(use_gpu=USE_GPU)
        self.tensor_wrapper = GPUTensorWrapper(self.device)
    
    def get_mutation_strength(self) -> float:
        """
        Calculate adaptive mutation strength based on global error.
        
        Returns:
            float: Current mutation strength
        """
        mut_strength = self.mut_strength_base * (self.global_error ** 2)
        return max(mut_strength, MIN_MUT_STRENGTH)
    
    def forward(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Forward pass through the network with GPU acceleration updates.
        
        Args:
            x: Input features
            y_true: One-hot encoded target
            train: Whether to evolve neurons
        """
        mut_strength = self.get_mutation_strength()
        vm_list = self.V_m.get()
        
        # --- NEW: Apply Attention ---
        x = self.attention.forward(x, train=train)

        # Layer 1
        l1_outputs = []
        l1_errors = []
        l1_marks = []
        
        for neuron in self.level1:
            out, err = neuron.forward(x, y_true, mse_loss, mut_strength, vm_list)
            l1_outputs.append(out)
            l1_errors.append(err)
            if err < self.tau1:
                l1_marks.append(out)
            else:
                l1_marks.append(('*', out))  # Mark failed neuron
        
        l1_outputs = np.array(l1_outputs, dtype=object)
        l1_errors = np.array(l1_errors)
        
        # Layer 2
        l2_inputs = l1_marks
        assert len(l2_inputs) == LEVEL1_NEURONS
        
        l2_outputs = []
        l2_errors = []
        l2_marks = []
        
        for neuron, inp in zip(self.level2, l2_inputs):
            inp_val = inp[1] if (isinstance(inp, tuple) and inp[0] == '*') else inp
            
            out, err = neuron.forward(
                np.full(LEVEL1_NEURONS, inp_val), 
                y_true, 
                mse_loss, 
                mut_strength, 
                vm_list
            )
            l2_outputs.append(out)
            l2_errors.append(err)
            
            if err < self.tau2:
                l2_marks.append(out)
            else:
                l2_marks.append(('*', out))
        
        l2_outputs = np.array(l2_outputs, dtype=object)
        l2_errors = np.array(l2_errors)
        
        # Layer 3 (output)
        l3_inputs = l2_marks
        
        # Prepare inputs for Layer 3 (incorporating skip connections if enabled)
        if USE_SKIP_CONNECTIONS:
            # Flatten L1 outputs (marks) to use as skip connection
            l1_vals = [m[1] if isinstance(m, tuple) and m[0] == '*' else m for m in l1_marks]
            l2_vals = [m[1] if isinstance(m, tuple) and m[0] == '*' else m for m in l2_marks]
            l3_input_vector = np.concatenate([l2_vals, l1_vals])
            input_dim_l3 = LEVEL2_NEURONS + LEVEL1_NEURONS
        else:
            l3_input_vector = np.array([m[1] if isinstance(m, tuple) and m[0] == '*' else m for m in l2_marks])
            input_dim_l3 = LEVEL2_NEURONS

        l3_outputs = []
        for i, neuron in enumerate(self.level3):
            out, _ = neuron.forward(
                l3_input_vector, # Use full vector for all output neurons
                y_true[i],
                mse_loss,
                mut_strength,
                vm_list
            )
            l3_outputs.append(out)
        
        l3_outputs = np.array(l3_outputs)
        y_pred = softmax(l3_outputs)
        
        # Calculate confidence based on hidden layer representation
        confidence = self.confidence_head.get_confidence(l3_input_vector[:LEVEL2_NEURONS])
        
        return y_pred, l1_errors, l2_errors, l3_outputs, confidence
    
    def predict(self, x: np.ndarray, pilot_index: int) -> Tuple[np.ndarray, float]:
        """
        Fast prediction for a specific individual in the population.
        Used for RL episodes.
        """
        # Apply Attention from the best individual
        x = self.attention.forward(x, train=False)
        
        # Layer 1
        l1_outs = []
        for neuron in self.level1:
            ind = neuron.population[pilot_index]
            l1_outs.append(np.maximum(0, np.dot(x, ind['weights']) + ind['bias']))
        l1_outs = np.array(l1_outs)
        
        # Layer 2
        l2_outs = []
        for neuron in self.level2:
            ind = neuron.population[pilot_index]
            l2_outs.append(np.maximum(0, np.dot(l1_outs, ind['weights']) + ind['bias']))
        l2_outs = np.array(l2_outs)
        
        # Skip Connections
        if USE_SKIP_CONNECTIONS:
            l3_in = np.concatenate([l2_outs, l1_outs])
        else:
            l3_in = l2_outs
            
        # Layer 3
        l3_outs = []
        for neuron in self.level3:
            ind = neuron.population[pilot_index]
            l3_outs.append(np.dot(l3_in, ind['weights']) + ind['bias'])
        
        y_pred = softmax(np.array(l3_outs))
        confidence = self.confidence_head.get_confidence(l3_in[:LEVEL2_NEURONS])
        
        return y_pred, confidence

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_oh: np.ndarray,
        epochs: int = EPOCHS,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        y_val_oh: Optional[np.ndarray] = None
    ) -> None:
        """
        Train the network for specified epochs.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            y_oh: One-hot encoded training labels
            epochs: Number of training epochs
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            y_val_oh: One-hot encoded validation labels (optional)
        """
        for epoch in range(1, epochs + 1):
            logger.info(f"Starting epoch {epoch}")
            correct = 0
            total_loss = 0.0
            
            for i in range(X.shape[0]):
                x = X[i]
                y_true = y_oh[i]
                y_label = y[i]
                
                y_pred, l1_errs, l2_errs, l3_outs = self.forward(x, y_true, train=True)
                pred_label = np.argmax(y_pred)
                
                if pred_label == y_label:
                    correct += 1
                
                loss = ce_loss_with_confidence(y_pred, y_true)
                total_loss += loss
                
                # Update V_m with best neurons
                l1_bench_idx = np.argmin(l1_errs)
                l1_bench = self.level1[l1_bench_idx]
                if l1_bench.last_error is not None and l1_bench.last_error < self.tau1:
                    self.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
                
                l2_bench_idx = np.argmin(l2_errs)
                l2_bench = self.level2[l2_bench_idx]
                if l2_bench.last_error is not None and l2_bench.last_error < self.tau2:
                    self.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
            
            acc = correct / X.shape[0]
            avg_loss = total_loss / X.shape[0]
            self.global_error = avg_loss
            
            logger.info(f"Finished epoch {epoch}")
            
            if epoch % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch}: Global Accuracy: {acc*100:.2f}%, Loss: {avg_loss:.4f}")
            
            if epoch % THOUGHT_INTERVAL == 0:
                print(f"[Thought] Epoch {epoch}: Global error is {self.global_error:.4f}, "
                      f"mutation strength now {self.get_mutation_strength():.4f}")
            
            if X_val is not None and epoch % PRINT_INTERVAL == 0:
                val_acc, val_loss = self.evaluate(X_val, y_val, y_val_oh)
                print(f"[Validation] Accuracy: {val_acc*100:.2f}%, Loss: {val_loss:.4f}")
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_oh: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate the network on a dataset.
        
        Args:
            X: Features
            y: Labels
            y_oh: One-hot encoded labels
            
        Returns:
            Tuple of (accuracy, average_loss)
        """
        correct = 0
        total_loss = 0.0
        
        for i in range(X.shape[0]):
            x = X[i]
            y_true = y_oh[i]
            y_label = y[i]
            
            y_pred, _, _, _, confidence = self.forward(x, y_true, train=False)
            pred_label = np.argmax(y_pred)
            
            if pred_label == y_label:
                correct += 1
            
            loss = ce_loss(y_pred, y_true)
            total_loss += loss
        
        acc = correct / X.shape[0]
        avg_loss = total_loss / X.shape[0]
        
        return acc, avg_loss


class RegressionEvoNet:
    """
    Regression Evolutionary Neural Network.
    
    A three-layer neural network using evolutionary neurons for
    regression tasks. Similar architecture to MultiClassEvoNet but
    with a single output neuron and MSE loss.
    
    Architecture:
        - Layer 1: LEVEL1_NEURONS (50) evolutionary neurons
        - Layer 2: LEVEL2_NEURONS (20) evolutionary neurons
        - Output: Single regression output neuron
        
    Example:
        >>> net = RegressionEvoNet(input_dim=8)
        >>> net.train(X_train, y_train, epochs=50)
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize the regression evolutionary network.
        
        Args:
            input_dim: Number of input features
        """
        self.level1: List[EvoNeuron] = [
            EvoNeuron(input_dim) for _ in range(LEVEL1_NEURONS)
        ]
        self.level2: List[EvoNeuron] = [
            EvoNeuron(LEVEL1_NEURONS) for _ in range(LEVEL2_NEURONS)
        ]
        self.output_neuron = RegressionOutputNeuron(LEVEL2_NEURONS + (LEVEL1_NEURONS if USE_SKIP_CONNECTIONS else 0))
        
        self.V_m = SignificantMutationVector()
        self.tau1 = TAU1
        self.tau2 = TAU2
        self.mut_strength_base = MUT_STRENGTH_BASE
        self.global_error: float = 1.0
        self.device = get_device(use_gpu=USE_GPU)
        self.tensor_wrapper = GPUTensorWrapper(self.device)
    
    def get_mutation_strength(self) -> float:
        """Calculate adaptive mutation strength."""
        mut_strength = self.mut_strength_base * (self.global_error ** 2)
        return max(mut_strength, MIN_MUT_STRENGTH)
    
    def forward(
        self,
        x: np.ndarray,
        y_true: float,
        train: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Forward pass through the network.
        
        Args:
            x: Input features
            y_true: Target value
            train: Whether to evolve neurons
            
        Returns:
            Tuple of (prediction, l1_errors, l2_errors)
        """
        mut_strength = self.get_mutation_strength()
        vm_list = self.V_m.get()
        
        # Layer 1
        l1_outputs = []
        l1_errors = []
        l1_marks = []
        
        for neuron in self.level1:
            out, err = neuron.forward(x, y_true, mse_loss, mut_strength, vm_list)
            l1_outputs.append(out)
            l1_errors.append(err)
            if err < self.tau1:
                l1_marks.append(out)
            else:
                l1_marks.append(('*', out))
        
        l1_outputs = np.array(l1_outputs, dtype=object)
        l1_errors = np.array(l1_errors)
        
        # Layer 2
        l2_inputs = l1_marks
        assert len(l2_inputs) == LEVEL1_NEURONS
        
        l2_outputs = []
        l2_errors = []
        l2_marks = []
        
        for neuron, inp in zip(self.level2, l2_inputs):
            if isinstance(inp, tuple) and inp[0] == '*':
                inp_val = inp[1]
            else:
                inp_val = inp
            
            out, err = neuron.forward(
                np.full(LEVEL1_NEURONS, inp_val),
                y_true,
                mse_loss,
                mut_strength,
                vm_list
            )
            l2_outputs.append(out)
            l2_errors.append(err)
            
            if err < self.tau2:
                l2_marks.append(out)
            else:
                l2_marks.append(('*', out))
        
        l2_outputs = np.array(l2_outputs, dtype=object)
        l2_errors = np.array(l2_errors)
        
        # Output
        l3_inputs = l2_marks
        
        # Prepare inputs for Output Layer (incorporating skip connections if enabled)
        if USE_SKIP_CONNECTIONS:
            # Flatten L1 outputs (marks) to use as skip connection
            l1_vals = [m[1] if isinstance(m, tuple) and m[0] == '*' else m for m in l1_marks]
            l2_vals = [m[1] if isinstance(m, tuple) and m[0] == '*' else m for m in l2_marks]
            l3_input_vector = np.concatenate([l2_vals, l1_vals])
        else:
            l3_input_vector = np.array([m[1] if isinstance(m, tuple) and m[0] == '*' else m for m in l2_marks])
            
        out, _ = self.output_neuron.forward(
            l3_input_vector,
            y_true,
            mse_loss,
            mut_strength,
            vm_list
        )
        
        y_pred = out
        
        if train:
            self.output_neuron.evolve(
                l3_input_vector,
                y_true,
                mse_loss,
                mut_strength,
                vm_list,
                tau=self.tau2
            )
        
        return y_pred, l1_errors, l2_errors
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = EPOCHS,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Train the regression network.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of epochs
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        for epoch in range(1, epochs + 1):
            logger.info(f"Starting epoch {epoch}")
            total_loss = 0.0
            
            for i in range(X.shape[0]):
                x = X[i]
                y_true = y[i]
                y_pred, l1_errs, l2_errs = self.forward(x, y_true, train=True)
                
                loss = mse_loss(y_pred, y_true)
                total_loss += loss
                
                # Update V_m
                l1_bench_idx = np.argmin(l1_errs)
                l1_bench = self.level1[l1_bench_idx]
                if l1_bench.last_error is not None and l1_bench.last_error < self.tau1:
                    self.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
                
                l2_bench_idx = np.argmin(l2_errs)
                l2_bench = self.level2[l2_bench_idx]
                if l2_bench.last_error is not None and l2_bench.last_error < self.tau2:
                    self.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
            
            avg_loss = total_loss / X.shape[0]
            self.global_error = avg_loss
            logger.info(f"Finished epoch {epoch}")
            
            if epoch % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch}: MSE Loss: {avg_loss:.4f}")
            
            if epoch % THOUGHT_INTERVAL == 0:
                print(f"[Thought] Epoch {epoch}: Global error is {self.global_error:.4f}, "
                      f"mutation strength now {self.get_mutation_strength():.4f}")
            
            if X_val is not None and epoch % PRINT_INTERVAL == 0:
                val_loss, _ = self.evaluate(X_val, y_val)
                print(f"[Validation] MSE Loss: {val_loss:.4f}")
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate the network.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Tuple of (average_loss, predictions)
        """
        total_loss = 0.0
        predictions = []
        
        for i in range(X.shape[0]):
            x = X[i]
            y_true = y[i]
            y_pred, _, _ = self.forward(x, y_true, train=False)
            
            loss = mse_loss(y_pred, y_true)
            total_loss += loss
            predictions.append(y_pred)
        
        avg_loss = total_loss / X.shape[0]
        return avg_loss, np.array(predictions)
