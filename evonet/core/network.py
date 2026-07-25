"""
Neural Network Module

This module contains the main neural network classes:
- MultiClassEvoNet: For classification tasks
- RegressionEvoNet: For regression tasks

Both networks use evolutionary neurons organized in a 3-layer architecture.

KEY DESIGN: Proper Neuroevolution
- Each neuron maintains a population of POP_SIZE weight vectors
- For training: we run a full network forward pass using population[k] for each k,
  compute the network-level loss, and evolve each neuron's population toward
  configurations that minimize the actual output loss
- For inference: we use each neuron's stored elite (best-ever individual)
- The V_m (Significant Mutation Vector) stores successful weight directions to
  guide future mutations - acting as long-term evolutionary memory
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
from evonet.core.losses import mse_loss, ce_loss, softmax
from evonet.core.gpu_backend import get_device, is_gpu_available, GPUTensorWrapper
from evonet.core.layers import EvoAttentionLayer, ConfidenceHead

logger = logging.getLogger(__name__)


class MultiClassEvoNet:
    """
    Multi-Class Evolutionary Neural Network.

    Architecture:
        Layer 1: LEVEL1_NEURONS evolutionary neurons (each with input_dim weights)
        Layer 2: LEVEL2_NEURONS evolutionary neurons (each with LEVEL1_NEURONS weights)
        Layer 3: num_classes output neurons

    Training (Neuroevolution):
        For each sample, we evaluate ALL population members across all neurons by
        computing the full network output for each candidate index k, measuring
        cross-entropy loss, then running tournament selection + mutation on each
        neuron independently. This ensures neurons evolve toward configurations
        that help the NETWORK minimize the actual classification loss.

    Inference:
        Each neuron uses its stored elite (best-ever individual) weights.
        No label access during inference.
    """

    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.level1: List[EvoNeuron] = [
            EvoNeuron(input_dim) for _ in range(LEVEL1_NEURONS)
        ]
        self.level2: List[EvoNeuron] = [
            EvoNeuron(LEVEL1_NEURONS) for _ in range(LEVEL2_NEURONS)
        ]
        l3_input_dim = LEVEL2_NEURONS + (LEVEL1_NEURONS if USE_SKIP_CONNECTIONS else 0)
        self.level3: List[OutputNeuron] = [
            OutputNeuron(l3_input_dim) for _ in range(num_classes)
        ]

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
        """Adaptive mutation: higher error → stronger mutations."""
        return max(self.mut_strength_base * (self.global_error ** 2), MIN_MUT_STRENGTH)

    # ------------------------------------------------------------------
    # Core forward pass helpers
    # ------------------------------------------------------------------

    def _evaluate_all_candidates_batch(
        self,
        X_att_batch: np.ndarray,
        Y_true_oh_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized evaluation of all POP_SIZE candidate networks across a mini-batch.

        Returns:
            mean_losses: Mean CE loss for each of the POP_SIZE candidates (shape [POP_SIZE])
            l1_batch: L1 outputs for all candidates (shape [B, POP_SIZE, LEVEL1_NEURONS])
            l2_batch: L2 outputs for all candidates (shape [B, POP_SIZE, LEVEL2_NEURONS])
        """
        P = self.pop_size
        X64 = X_att_batch.astype(np.float64)  # [B, D]

        # --- Layer 1 ---
        W1 = np.array([[n.population[k]['weights'] for k in range(P)] for n in self.level1], dtype=np.float64)  # [L1, P, D]
        B1 = np.array([[n.population[k]['bias'] for k in range(P)] for n in self.level1], dtype=np.float64)    # [L1, P]
        
        # [L1, P, D] @ [B, D] -> [B, P, L1]
        l1_batch = np.einsum('lpd,bd->bpl', W1, X64) + B1.T
        l1_batch = np.maximum(0.0, l1_batch)

        # --- Layer 2 ---
        W2 = np.array([[n.population[k]['weights'] for k in range(P)] for n in self.level2], dtype=np.float64)  # [L2, P, L1]
        B2 = np.array([[n.population[k]['bias'] for k in range(P)] for n in self.level2], dtype=np.float64)    # [L2, P]
        
        # [L2, P, L1] @ [B, P, L1] -> [B, P, L2]
        l2_batch = np.einsum('jpl,bpl->bpj', W2, l1_batch) + B2.T
        l2_batch = np.maximum(0.0, l2_batch)

        # --- Layer 3 ---
        if USE_SKIP_CONNECTIONS:
            l3_in = np.concatenate([l2_batch, l1_batch], axis=2)  # [B, P, L2+L1]
        else:
            l3_in = l2_batch  # [B, P, L2]

        W3 = np.array([[n.population[k]['weights'] for k in range(P)] for n in self.level3], dtype=np.float64)  # [C, P, L3_dim]
        B3 = np.array([[n.population[k]['bias'] for k in range(P)] for n in self.level3], dtype=np.float64)    # [C, P]
        
        # [C, P, L] @ [B, P, L] -> [B, P, C]
        logits = np.einsum('cpl,bpl->bpc', W3, l3_in) + B3.T

        # Softmax per candidate, per sample
        logits = np.clip(logits, -1e4, 1e4)
        logits -= logits.max(axis=2, keepdims=True)
        exp_l = np.exp(logits)
        y_preds = exp_l / (exp_l.sum(axis=2, keepdims=True) + 1e-12)  # [B, P, C]

        # CE loss per candidate, per sample
        y_preds_clipped = np.clip(y_preds, 1e-8, 1 - 1e-8)
        Y_true_oh_exp = Y_true_oh_batch[:, np.newaxis, :]  # [B, 1, C]
        losses = -np.sum(Y_true_oh_exp * np.log(y_preds_clipped), axis=2)  # [B, P]
        
        mean_losses = losses.mean(axis=0) # [P]

        return mean_losses, l1_batch, l2_batch

    def _net_forward_with_pop_idx(
        self,
        x_att: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-candidate forward pass (kept for compatibility)."""
        x64 = x_att.astype(np.float64)
        l1_out = np.zeros(LEVEL1_NEURONS, dtype=np.float64)
        for i, neuron in enumerate(self.level1):
            ind = neuron.population[k]
            l1_out[i] = max(0.0, float(np.dot(ind['weights'].astype(np.float64), x64) + float(ind['bias'])))

        l2_out = np.zeros(LEVEL2_NEURONS, dtype=np.float64)
        for j, neuron in enumerate(self.level2):
            ind = neuron.population[k]
            l2_out[j] = max(0.0, float(np.dot(ind['weights'].astype(np.float64), l1_out) + float(ind['bias'])))

        l3_in = np.concatenate([l2_out, l1_out]) if USE_SKIP_CONNECTIONS else l2_out
        l3_logits = np.zeros(self.num_classes, dtype=np.float64)
        for c, neuron in enumerate(self.level3):
            ind = neuron.population[k]
            l3_logits[c] = float(np.dot(ind['weights'].astype(np.float64), l3_in) + float(ind['bias']))
        y_pred = softmax(l3_logits)
        return y_pred, l1_out, l2_out

    def _forward_elite(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Inference-only forward pass using elite weights.
        No label access. Returns (y_pred, l1_outputs, l2_outputs, confidence).
        """
        x_att = self.attention.forward(x, train=False)

        l1_out = np.zeros(LEVEL1_NEURONS, dtype=np.float64)
        for i, neuron in enumerate(self.level1):
            w = neuron.elite['weights'] if neuron.elite else neuron.population[0]['weights']
            b = neuron.elite['bias'] if neuron.elite else neuron.population[0]['bias']
            raw = float(np.dot(w.astype(np.float64), x_att.astype(np.float64)) + float(b))
            l1_out[i] = max(0.0, raw)

        l2_out = np.zeros(LEVEL2_NEURONS, dtype=np.float64)
        for j, neuron in enumerate(self.level2):
            w = neuron.elite['weights'] if neuron.elite else neuron.population[0]['weights']
            b = neuron.elite['bias'] if neuron.elite else neuron.population[0]['bias']
            raw = float(np.dot(w.astype(np.float64), l1_out) + float(b))
            l2_out[j] = max(0.0, raw)

        if USE_SKIP_CONNECTIONS:
            l3_in = np.concatenate([l2_out, l1_out])
        else:
            l3_in = l2_out

        l3_logits = np.zeros(self.num_classes, dtype=np.float64)
        for k, neuron in enumerate(self.level3):
            w = neuron.elite['weights'] if neuron.elite else neuron.population[0]['weights']
            b = neuron.elite['bias'] if neuron.elite else neuron.population[0]['bias']
            l3_logits[k] = float(np.dot(w.astype(np.float64), l3_in) + float(b))

        y_pred = softmax(l3_logits)
        confidence = self.confidence_head.get_confidence(l2_out)
        return y_pred, l1_out, l2_out, confidence

    def _train_one_batch(
        self,
        X_batch: np.ndarray,
        Y_true_oh_batch: np.ndarray,
        mut_strength: float
    ) -> float:
        """
        Core neuroevolution training step for a mini-batch of samples.

        Strategy:
        1. Evaluate the network for each candidate index k (0..POP_SIZE-1) across the ENTIRE batch.
        2. Compute average cross-entropy loss for each candidate across the batch.
        3. The best candidate index k* is the one with lowest average CE loss.
        4. Update each neuron's elite with population[k*] if it beats the neuron's current elite.
        5. Evolve each neuron's population using the fitness ranking from step 2.

        Returns: Mean CE loss of the best candidate (for global_error tracking)
        """
        X_att_batch = self.attention.forward(X_batch, train=True)
        vm_list = self.V_m.get()

        # --- Vectorized evaluation of all POP_SIZE candidates across the batch ---
        mean_losses, l1_batch, l2_batch = self._evaluate_all_candidates_batch(X_att_batch, Y_true_oh_batch)

        # Best candidate
        best_k = int(np.argmin(mean_losses))
        best_loss = mean_losses[best_k]

        # --- Update elites: each neuron's elite = population[best_k] if it's better ---
        for neuron in self.level1:
            err_k = mean_losses[best_k]
            if neuron.elite is None or err_k < neuron.elite_error:
                neuron.elite = {
                    'weights': neuron.population[best_k]['weights'].copy(),
                    'bias': neuron.population[best_k]['bias']
                }
                neuron.elite_error = float(err_k)
            neuron.last_error = float(err_k)
            neuron.last_weights = neuron.population[best_k]['weights'].copy()
            neuron.last_bias = float(neuron.population[best_k]['bias'])

        for neuron in self.level2:
            if neuron.elite is None or mean_losses[best_k] < neuron.elite_error:
                neuron.elite = {
                    'weights': neuron.population[best_k]['weights'].copy(),
                    'bias': neuron.population[best_k]['bias']
                }
                neuron.elite_error = float(mean_losses[best_k])
            neuron.last_error = float(mean_losses[best_k])
            neuron.last_weights = neuron.population[best_k]['weights'].copy()
            neuron.last_bias = float(neuron.population[best_k]['bias'])

        for neuron in self.level3:
            if neuron.elite is None or mean_losses[best_k] < neuron.elite_error:
                neuron.elite = {
                    'weights': neuron.population[best_k]['weights'].copy(),
                    'bias': neuron.population[best_k]['bias']
                }
                neuron.elite_error = float(mean_losses[best_k])

        # --- Evolve each neuron's population using network-level fitness ---
        # Convert loss to fitness (lower loss = higher fitness)
        sorted_idx = np.argsort(mean_losses)  # best (lowest loss) first
        fitness_ranks = np.zeros(self.pop_size)
        for rank, idx in enumerate(sorted_idx):
            fitness_ranks[idx] = self.pop_size - rank  # best gets highest rank

        # Evolve all neurons using their population indexed by the same k
        for neuron in self.level1:
            neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)

        for neuron in self.level2:
            neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)

        for neuron in self.level3:
            neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)

        # --- Update V_m with best L1/L2 directions ---
        l1_bench = self.level1[np.argmin([
            n.elite_error if n.elite_error != float('inf') else 999
            for n in self.level1
        ])]
        if l1_bench.last_error is not None and l1_bench.last_error < self.tau1:
            self.V_m.add(l1_bench.last_weights, l1_bench.last_bias)

        l2_bench = self.level2[np.argmin([
            n.elite_error if n.elite_error != float('inf') else 999
            for n in self.level2
        ])]
        if l2_bench.last_error is not None and l2_bench.last_error < self.tau2:
            self.V_m.add(l2_bench.last_weights, l2_bench.last_bias)

        return float(best_loss)

    def forward(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Forward pass — routes to training or inference.

        Args:
            x: Input features
            y_true: One-hot encoded target (only used during training)
            train: If True, runs evolutionary training step; if False, uses elite

        Returns:
            (y_pred, l1_placeholder, l2_placeholder, l3_logits_placeholder, confidence)
        """
        if train:
            mut_strength = self.get_mutation_strength()
            loss = self._train_one_sample(x, y_true, mut_strength)
            y_pred, _, _, confidence = self._forward_elite(x)
            return y_pred, np.zeros(LEVEL1_NEURONS), np.zeros(LEVEL2_NEURONS), np.zeros(self.num_classes), confidence
        else:
            y_pred, l1_out, l2_out, confidence = self._forward_elite(x)
            return y_pred, np.zeros(LEVEL1_NEURONS), np.zeros(LEVEL2_NEURONS), np.zeros(self.num_classes), confidence

    def predict(self, x: np.ndarray, pilot_index: int) -> Tuple[np.ndarray, float]:
        """Fast prediction for a specific population index (RL/trading use)."""
        x_att = self.attention.forward(x, train=False)
        k = pilot_index % self.pop_size

        l1_out = np.zeros(LEVEL1_NEURONS, dtype=np.float64)
        for i, neuron in enumerate(self.level1):
            ind = neuron.population[k]
            l1_out[i] = max(0.0, float(np.dot(x_att, ind['weights']) + ind['bias']))

        l2_out = np.zeros(LEVEL2_NEURONS, dtype=np.float64)
        for j, neuron in enumerate(self.level2):
            ind = neuron.population[k]
            l2_out[j] = max(0.0, float(np.dot(l1_out, ind['weights']) + ind['bias']))

        if USE_SKIP_CONNECTIONS:
            l3_in = np.concatenate([l2_out, l1_out])
        else:
            l3_in = l2_out

        l3_logits = np.zeros(self.num_classes, dtype=np.float64)
        for c, neuron in enumerate(self.level3):
            ind = neuron.population[k]
            l3_logits[c] = float(np.dot(l3_in, ind['weights']) + ind['bias'])

        y_pred = softmax(l3_logits)
        confidence = self.confidence_head.get_confidence(l2_out)
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
        Train using neuroevolution with mini-batches.

        Each epoch: shuffle data, slice into mini-batches, then run _train_one_batch()
        which evaluates all POP_SIZE candidates across the entire batch to find true fitness.
        Accuracy/loss reported using elite weights (honest, no label leakage).
        """
        batch_size = 32
        
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(X.shape[0])
            X_shuffled = X[perm]
            Y_shuffled = y_oh[perm]
            
            total_loss = 0.0
            num_batches = 0

            for i in range(0, X.shape[0], batch_size):
                end = min(i + batch_size, X.shape[0])
                X_batch = X_shuffled[i:end]
                Y_batch = Y_shuffled[i:end]
                
                loss = self._train_one_batch(X_batch, Y_batch, self.get_mutation_strength())
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches
            self.global_error = avg_loss

            if epoch % PRINT_INTERVAL == 0:
                train_acc, train_loss = self.evaluate(X, y, y_oh)
                print(f"Epoch {epoch}: Train Acc: {train_acc*100:.2f}%  Elite Loss: {train_loss:.4f}")

            if epoch % THOUGHT_INTERVAL == 0:
                print(f"[Thought] Epoch {epoch}: Avg CE={self.global_error:.4f}, "
                      f"mut_strength={self.get_mutation_strength():.4f}, "
                      f"V_m entries={len(self.V_m)}")

            if X_val is not None and epoch % PRINT_INTERVAL == 0:
                val_acc, val_loss = self.evaluate(X_val, y_val, y_val_oh)
                print(f"  [Val] Acc: {val_acc*100:.2f}%  Loss: {val_loss:.4f}")

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_oh: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate using elite weights only. No label access."""
        correct = 0
        total_loss = 0.0
        for i in range(X.shape[0]):
            y_pred, _, _, _ = self._forward_elite(X[i])
            if np.argmax(y_pred) == y[i]:
                correct += 1
            total_loss += ce_loss(y_pred, y_oh[i])
        return correct / X.shape[0], total_loss / X.shape[0]


class RegressionEvoNet:
    """
    Regression Evolutionary Neural Network.

    Same neuroevolution design as MultiClassEvoNet but for regression.
    Output is a single scalar, loss is MSE.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.level1: List[EvoNeuron] = [EvoNeuron(input_dim) for _ in range(LEVEL1_NEURONS)]
        self.level2: List[EvoNeuron] = [EvoNeuron(LEVEL1_NEURONS) for _ in range(LEVEL2_NEURONS)]
        l3_dim = LEVEL2_NEURONS + (LEVEL1_NEURONS if USE_SKIP_CONNECTIONS else 0)
        self.output_neuron = RegressionOutputNeuron(l3_dim)
        self.V_m = SignificantMutationVector()
        self.tau1 = TAU1
        self.tau2 = TAU2
        self.mut_strength_base = MUT_STRENGTH_BASE
        self.global_error: float = 1.0
        self.device = get_device(use_gpu=USE_GPU)
        self.tensor_wrapper = GPUTensorWrapper(self.device)

    def get_mutation_strength(self) -> float:
        return max(self.mut_strength_base * (self.global_error ** 2), MIN_MUT_STRENGTH)

    def _forward_elite(self, x: np.ndarray) -> float:
        """Inference using elite weights."""
        l1_out = np.zeros(LEVEL1_NEURONS, dtype=np.float64)
        for i, neuron in enumerate(self.level1):
            w = neuron.elite['weights'] if neuron.elite else neuron.population[0]['weights']
            b = neuron.elite['bias'] if neuron.elite else neuron.population[0]['bias']
            l1_out[i] = max(0.0, float(np.dot(w.astype(np.float64), x.astype(np.float64)) + float(b)))

        l2_out = np.zeros(LEVEL2_NEURONS, dtype=np.float64)
        for j, neuron in enumerate(self.level2):
            w = neuron.elite['weights'] if neuron.elite else neuron.population[0]['weights']
            b = neuron.elite['bias'] if neuron.elite else neuron.population[0]['bias']
            l2_out[j] = max(0.0, float(np.dot(w.astype(np.float64), l1_out) + float(b)))

        l3_in = np.concatenate([l2_out, l1_out]) if USE_SKIP_CONNECTIONS else l2_out
        w = self.output_neuron.elite['weights'] if self.output_neuron.elite else self.output_neuron.population[0]['weights']
        b = self.output_neuron.elite['bias'] if self.output_neuron.elite else self.output_neuron.population[0]['bias']
        return float(np.dot(w.astype(np.float64), l3_in) + float(b))

    def _net_forward_with_pop_idx(self, x: np.ndarray, k: int) -> float:
        """Full regression forward pass with population[k]."""
        l1_out = np.zeros(LEVEL1_NEURONS, dtype=np.float64)
        for i, neuron in enumerate(self.level1):
            ind = neuron.population[k]
            l1_out[i] = max(0.0, float(np.dot(ind['weights'].astype(np.float64), x.astype(np.float64)) + float(ind['bias'])))

        l2_out = np.zeros(LEVEL2_NEURONS, dtype=np.float64)
        for j, neuron in enumerate(self.level2):
            ind = neuron.population[k]
            l2_out[j] = max(0.0, float(np.dot(ind['weights'].astype(np.float64), l1_out) + float(ind['bias'])))

        l3_in = np.concatenate([l2_out, l1_out]) if USE_SKIP_CONNECTIONS else l2_out
        ind = self.output_neuron.population[k]
        return float(np.dot(ind['weights'].astype(np.float64), l3_in) + float(ind['bias']))

    def forward(
        self,
        x: np.ndarray,
        y_true: float,
        train: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Forward pass for regression."""
        mut_strength = self.get_mutation_strength()
        vm_list = self.V_m.get()

        if not train:
            return self._forward_elite(x), np.zeros(LEVEL1_NEURONS), np.zeros(LEVEL2_NEURONS)

        # Evaluate all candidates
        losses = np.array([mse_loss(self._net_forward_with_pop_idx(x, k), y_true) for k in range(self.output_neuron.pop_size)])
        best_k = int(np.argmin(losses))
        best_loss = losses[best_k]

        # Update elites
        for neuron in self.level1:
            if neuron.elite is None or best_loss < neuron.elite_error:
                neuron.elite = {'weights': neuron.population[best_k]['weights'].copy(), 'bias': neuron.population[best_k]['bias']}
                neuron.elite_error = float(best_loss)
            neuron.last_error = float(best_loss)
            neuron.last_weights = neuron.population[best_k]['weights'].copy()
            neuron.last_bias = float(neuron.population[best_k]['bias'])

        for neuron in self.level2:
            if neuron.elite is None or best_loss < neuron.elite_error:
                neuron.elite = {'weights': neuron.population[best_k]['weights'].copy(), 'bias': neuron.population[best_k]['bias']}
                neuron.elite_error = float(best_loss)
            neuron.last_error = float(best_loss)
            neuron.last_weights = neuron.population[best_k]['weights'].copy()
            neuron.last_bias = float(neuron.population[best_k]['bias'])

        if self.output_neuron.elite is None or best_loss < self.output_neuron.elite_error:
            self.output_neuron.elite = {'weights': self.output_neuron.population[best_k]['weights'].copy(), 'bias': self.output_neuron.population[best_k]['bias']}
            self.output_neuron.elite_error = float(best_loss)

        # Evolve
        sorted_idx = np.argsort(losses)
        fitness_ranks = np.zeros(len(losses))
        for rank, idx in enumerate(sorted_idx):
            fitness_ranks[idx] = len(losses) - rank

        for neuron in self.level1:
            neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)
        for neuron in self.level2:
            neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)
        self.output_neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)

        # V_m update
        l1_bench = self.level1[np.argmin([n.elite_error if n.elite_error != float('inf') else 999 for n in self.level1])]
        if l1_bench.last_error is not None and l1_bench.last_error < self.tau1:
            self.V_m.add(l1_bench.last_weights, l1_bench.last_bias)

        return self._forward_elite(x), np.zeros(LEVEL1_NEURONS), np.zeros(LEVEL2_NEURONS)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = EPOCHS,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Train the regression network."""
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(X.shape[0])
            total_loss = 0.0
            for i in perm:
                y_pred, _, _ = self.forward(X[i], y[i], train=True)
                total_loss += mse_loss(y_pred, y[i])
            avg_loss = total_loss / X.shape[0]
            self.global_error = avg_loss

            if epoch % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch}: MSE Loss: {avg_loss:.4f}")
            if epoch % THOUGHT_INTERVAL == 0:
                print(f"[Thought] Epoch {epoch}: error={self.global_error:.4f}, mut={self.get_mutation_strength():.4f}")
            if X_val is not None and epoch % PRINT_INTERVAL == 0:
                val_loss, _ = self.evaluate(X_val, y_val)
                print(f"  [Val] MSE: {val_loss:.4f}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate using elite weights."""
        preds = [self._forward_elite(X[i]) for i in range(X.shape[0])]
        total_loss = sum(mse_loss(preds[i], y[i]) for i in range(len(preds)))
        return total_loss / X.shape[0], np.array(preds)
