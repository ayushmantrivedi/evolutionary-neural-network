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
import warnings
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

        # Adaptive tau: scale thresholds by log(num_classes) so V_m works
        # for high-class problems (e.g. Digits: 10 classes → tau scales up)
        import math
        tau_scale = math.log(num_classes + 1) / math.log(3)  # normalized to 3-class baseline
        self.tau1 = TAU1 * tau_scale
        self.tau2 = TAU2 * tau_scale
        self.pop_size = POP_SIZE
        self.mut_strength_base = MUT_STRENGTH_BASE
        self.global_error: float = 1.0
        self.best_global_loss: float = float('inf')  # tracks true best-ever network loss
        self._best_state: Optional[Dict[str, Any]] = None
        self.device = get_device(use_gpu=USE_GPU)
        self.tensor_wrapper = GPUTensorWrapper(self.device)

        # Accessibility Anchor memory: stored penultimate features for prior tasks.
        self.anchor_inputs: Optional[np.ndarray] = None
        self.anchor_features: Optional[np.ndarray] = None
        self.anchor_weight: float = 0.15
        self.anchor_max_examples: int = 64
        self.route_count: int = 3
        self.route_best_idx: int = 0
        self.route_strength: float = 0.04
        self.diversity_archive: List[np.ndarray] = []
        self.diversity_archive_size: int = 3
        self.diversity_bonus: float = 0.02
        
        # Data robustness
        self._imputer_medians: Optional[np.ndarray] = None
        self._variance_guards: Optional[np.ndarray] = None

    def _apply_data_guards(self, X: np.ndarray) -> np.ndarray:
        """Applies imputation and variance guards to data (Test 2 & 9)."""
        X_clean = X.copy()
        # 1. Missing Values
        nan_mask = np.isnan(X_clean)
        if np.any(nan_mask) and self._imputer_medians is not None:
            # Broadcast medians to missing locations
            for j in range(X_clean.shape[1]):
                col_nans = nan_mask[:, j]
                if np.any(col_nans):
                    X_clean[col_nans, j] = self._imputer_medians[j]
                    
        # 2. Edge Cases (Zero Variance)
        if self._variance_guards is not None and np.any(self._variance_guards):
            # Add tiny noise to constant columns to prevent attention deadlocks
            noise = np.random.normal(0, 1e-6, size=(X_clean.shape[0], np.sum(self._variance_guards)))
            X_clean[:, self._variance_guards] += noise
            
        return X_clean

    def _route_inputs(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a lightweight route adapter based on sample volatility."""
        X_arr = np.asarray(X, dtype=np.float32)
        is_single = (X_arr.ndim == 1)
        if is_single:
            X_arr = X_arr.reshape(1, -1)

        volatility = np.std(X_arr, axis=1)
        if X_arr.shape[0] > 1:
            low, high = np.quantile(volatility, [0.33, 0.66])
        else:
            low = 0.5
            high = 1.0

        route_idx = np.zeros(X_arr.shape[0], dtype=int)
        route_idx[volatility >= low] = 1
        route_idx[volatility >= high] = 2

        routed = X_arr.copy()
        routed[route_idx == 0] = routed[route_idx == 0] * (1.0 - self.route_strength)
        routed[route_idx == 1] = routed[route_idx == 1] * (1.0 + self.route_strength) + np.tanh(routed[route_idx == 1]) * (self.route_strength * 0.5)
        routed[route_idx == 2] = routed[route_idx == 2] * (1.0 - (self.route_strength * 0.5)) - np.sign(routed[route_idx == 2]) * (self.route_strength * 0.25)

        if is_single:
            return routed[0], route_idx[0:1]
        return routed, route_idx

    def _flatten_candidate_state(self, k: int) -> np.ndarray:
        """Flatten the network parameters for candidate index k."""
        weights = []
        for layer in (self.level1, self.level2, self.level3):
            for neuron in layer:
                individual = neuron.population[k]
                weights.append(np.asarray(individual['weights'], dtype=np.float64).ravel())
                weights.append(np.asarray([individual['bias']], dtype=np.float64))
        return np.concatenate(weights)

    def _diversity_score(self, candidate_vector: np.ndarray) -> float:
        """Compute the minimum cosine distance from the elite archive."""
        if not self.diversity_archive:
            return 0.0

        candidate_norm = np.linalg.norm(candidate_vector) + 1e-12
        best_distance = 0.0
        for archived in self.diversity_archive:
            archived_norm = np.linalg.norm(archived) + 1e-12
            similarity = float(np.dot(candidate_vector, archived) / (candidate_norm * archived_norm))
            distance = 1.0 - np.clip(similarity, -1.0, 1.0)
            best_distance = max(best_distance, distance)
        return best_distance

    def _update_diversity_archive(self, candidate_vector: np.ndarray, candidate_loss: float) -> None:
        """Keep a small archive of distinct high-quality candidate states."""
        if not self.diversity_archive:
            self.diversity_archive.append(candidate_vector.copy())
            return

        candidate_norm = np.linalg.norm(candidate_vector) + 1e-12
        distinct = True
        for archived in self.diversity_archive:
            archived_norm = np.linalg.norm(archived) + 1e-12
            similarity = float(np.dot(candidate_vector, archived) / (candidate_norm * archived_norm))
            if similarity > 0.995:
                distinct = False
                break

        if distinct:
            self.diversity_archive.append(candidate_vector.copy())
            if len(self.diversity_archive) > self.diversity_archive_size:
                self.diversity_archive.pop(0)

    def get_mutation_strength(self) -> float:
        """Adaptive mutation: higher error → stronger mutations, but bounded."""
        return min(max(self.mut_strength_base * (self.global_error ** 2), MIN_MUT_STRENGTH), 1.0)

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

        W1 = np.array([[n.population[k]['weights'] for k in range(P)] for n in self.level1], dtype=np.float64)  # [L1, P, D]
        B1 = np.array([[n.population[k]['bias'] for k in range(P)] for n in self.level1], dtype=np.float64)    # [L1, P]
        
        W2 = np.array([[n.population[k]['weights'] for k in range(P)] for n in self.level2], dtype=np.float64)  # [L2, P, L1]
        B2 = np.array([[n.population[k]['bias'] for k in range(P)] for n in self.level2], dtype=np.float64)    # [L2, P]
        
        W3 = np.array([[n.population[k]['weights'] for k in range(P)] for n in self.level3], dtype=np.float64)  # [C, P, L3_dim]
        B3 = np.array([[n.population[k]['bias'] for k in range(P)] for n in self.level3], dtype=np.float64)    # [C, P]

        if self.device != 'cpu':
            import torch
            X_t = self.tensor_wrapper.to_tensor(X_att_batch)
            W1_t = self.tensor_wrapper.to_tensor(W1)
            B1_t = self.tensor_wrapper.to_tensor(B1)
            l1_batch_t = torch.einsum('lpd,bd->bpl', W1_t, X_t) + B1_t.T
            l1_batch_t = torch.relu(l1_batch_t)
            
            W2_t = self.tensor_wrapper.to_tensor(W2)
            B2_t = self.tensor_wrapper.to_tensor(B2)
            l2_batch_t = torch.einsum('jpl,bpl->bpj', W2_t, l1_batch_t) + B2_t.T
            l2_batch_t = torch.relu(l2_batch_t)
            
            if USE_SKIP_CONNECTIONS:
                l3_in_t = torch.cat([l2_batch_t, l1_batch_t], dim=2)
            else:
                l3_in_t = l2_batch_t
                
            W3_t = self.tensor_wrapper.to_tensor(W3)
            B3_t = self.tensor_wrapper.to_tensor(B3)
            logits_t = torch.einsum('cpl,bpl->bpc', W3_t, l3_in_t) + B3_t.T
            
            logits_t = torch.clamp(logits_t, min=-1e4, max=1e4)
            logits_t = logits_t - logits_t.max(dim=2, keepdim=True)[0]
            exp_l_t = torch.exp(logits_t)
            y_preds_t = exp_l_t / (exp_l_t.sum(dim=2, keepdim=True) + 1e-12)
            
            y_preds_clipped_t = torch.clamp(y_preds_t, min=1e-8, max=1-1e-8)
            Y_true_oh_t = self.tensor_wrapper.to_tensor(Y_true_oh_batch).unsqueeze(1)
            
            # Focal Loss: -(1-p_t)^gamma * log(p_t)
            p_t_torch = torch.sum(y_preds_clipped_t * Y_true_oh_t, dim=2)
            losses_t = -torch.pow(1.0 - p_t_torch, 2.0) * torch.log(p_t_torch)

            mean_losses_t = losses_t.mean(dim=0)
            
            return self.tensor_wrapper.to_numpy(mean_losses_t), self.tensor_wrapper.to_numpy(l1_batch_t), self.tensor_wrapper.to_numpy(l2_batch_t)

        # --- Numpy Fallback ---
        
        # [L1, P, D] @ [B, D] -> [B, P, L1]
        l1_batch = np.einsum('lpd,bd->bpl', W1, X64) + B1.T
        l1_batch = np.maximum(0.0, l1_batch)

        # [L2, P, L1] @ [B, P, L1] -> [B, P, L2]
        l2_batch = np.einsum('jpl,bpl->bpj', W2, l1_batch) + B2.T
        l2_batch = np.maximum(0.0, l2_batch)

        # --- Layer 3 ---
        if USE_SKIP_CONNECTIONS:
            l3_in = np.concatenate([l2_batch, l1_batch], axis=2)  # [B, P, L2+L1]
        else:
            l3_in = l2_batch  # [B, P, L2]

        # [C, P, L] @ [B, P, L] -> [B, P, C]
        logits = np.einsum('cpl,bpl->bpc', W3, l3_in) + B3.T

        # Softmax per candidate, per sample
        logits = np.clip(logits, -1e4, 1e4)
        logits -= logits.max(axis=2, keepdims=True)
        exp_l = np.exp(logits)
        y_preds = exp_l / (exp_l.sum(axis=2, keepdims=True) + 1e-12)  # [B, P, C]

        # Focal loss per candidate, per sample
        y_preds_clipped = np.clip(y_preds, 1e-8, 1 - 1e-8)
        Y_true_oh_exp = Y_true_oh_batch[:, np.newaxis, :]  # [B, 1, C]
        
        p_t = np.sum(y_preds_clipped * Y_true_oh_exp, axis=2) # [B, P]
        losses = -(np.power(1.0 - p_t, 2.0)) * np.log(p_t)    # [B, P]

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

    def _forward_elite(self, X_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Inference-only batched forward pass using elite weights.
        Returns (y_preds, l1_batch, l2_batch, confidences).
        Handles both single samples (1D) and batches (2D) natively by always expanding to 2D.
        """
        is_single = (X_batch.ndim == 1)
        if is_single:
            X_batch = X_batch.reshape(1, -1)
        
        # Always apply data guards so NaN handling is consistent at inference time
        if self._imputer_medians is not None:
            X_batch = self._apply_data_guards(X_batch)

        X_batch, _ = self._route_inputs(X_batch)
            
        x_att = self.attention.forward(X_batch, train=False) # [B, D]
        
        # Build stacked elite weights
        W1 = np.array([n.elite['weights'] if n.elite else n.population[0]['weights'] for n in self.level1], dtype=np.float64) # [L1, D]
        B1 = np.array([n.elite['bias'] if n.elite else n.population[0]['bias'] for n in self.level1], dtype=np.float64)       # [L1]
        W2 = np.array([n.elite['weights'] if n.elite else n.population[0]['weights'] for n in self.level2], dtype=np.float64) # [L2, L1]
        B2 = np.array([n.elite['bias'] if n.elite else n.population[0]['bias'] for n in self.level2], dtype=np.float64)       # [L2]
        W3 = np.array([n.elite['weights'] if n.elite else n.population[0]['weights'] for n in self.level3], dtype=np.float64) # [C, L]
        B3 = np.array([n.elite['bias'] if n.elite else n.population[0]['bias'] for n in self.level3], dtype=np.float64)       # [C]

        if self.device != 'cpu':
            import torch
            X_t = self.tensor_wrapper.to_tensor(x_att)
            W1_t = self.tensor_wrapper.to_tensor(W1)
            B1_t = self.tensor_wrapper.to_tensor(B1)
            
            l1_batch = torch.relu(torch.matmul(X_t, W1_t.T) + B1_t)
            
            W2_t = self.tensor_wrapper.to_tensor(W2)
            B2_t = self.tensor_wrapper.to_tensor(B2)
            l2_batch = torch.relu(torch.matmul(l1_batch, W2_t.T) + B2_t)
            
            if USE_SKIP_CONNECTIONS:
                l3_in = torch.cat([l2_batch, l1_batch], dim=1)
            else:
                l3_in = l2_batch
                
            W3_t = self.tensor_wrapper.to_tensor(W3)
            B3_t = self.tensor_wrapper.to_tensor(B3)
            logits = torch.matmul(l3_in, W3_t.T) + B3_t
            
            logits = torch.clamp(logits, min=-1e4, max=1e4)
            logits = logits - logits.max(dim=1, keepdim=True)[0]
            exp_l = torch.exp(logits)
            y_preds = exp_l / (exp_l.sum(dim=1, keepdim=True) + 1e-12)
            
            y_preds = self.tensor_wrapper.to_numpy(y_preds)
            l1_batch = self.tensor_wrapper.to_numpy(l1_batch)
            l2_batch = self.tensor_wrapper.to_numpy(l2_batch)
        else:
            X64 = x_att.astype(np.float64)
            l1_batch = np.maximum(0.0, np.dot(X64, W1.T) + B1)
            l2_batch = np.maximum(0.0, np.dot(l1_batch, W2.T) + B2)
            
            if USE_SKIP_CONNECTIONS:
                l3_in = np.concatenate([l2_batch, l1_batch], axis=1)
            else:
                l3_in = l2_batch
                
            logits = np.dot(l3_in, W3.T) + B3
            logits = np.clip(logits, -1e4, 1e4)
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            y_preds = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-12)

        confidences = self.confidence_head.get_confidence(l2_batch)
        
        if is_single:
            return y_preds[0], l1_batch[0], l2_batch[0], confidences[0]
        return y_preds, l1_batch, l2_batch, confidences

    def capture_anchor_memory(self, X_anchor: np.ndarray, max_examples: Optional[int] = None) -> None:
        """Store anchor inputs and their penultimate features for later accessibility regularization."""
        X_anchor = np.asarray(X_anchor, dtype=np.float32)
        if X_anchor.ndim == 1:
            X_anchor = X_anchor.reshape(1, -1)

        limit = self.anchor_max_examples if max_examples is None else int(max_examples)
        if X_anchor.shape[0] > limit:
            anchor_idx = np.linspace(0, X_anchor.shape[0] - 1, limit, dtype=int)
            X_anchor = X_anchor[anchor_idx]

        if self._imputer_medians is not None:
            X_anchor = self._apply_data_guards(X_anchor)

        _, _, l2_anchor, _ = self._forward_elite(X_anchor)
        self.anchor_inputs = X_anchor.copy()
        self.anchor_features = np.asarray(l2_anchor, dtype=np.float32).copy()

    def clear_anchor_memory(self) -> None:
        """Remove any stored accessibility anchors."""
        self.anchor_inputs = None
        self.anchor_features = None

    def _evaluate_anchor_memory_loss(self) -> np.ndarray:
        """Return anchor reconstruction loss for each candidate in the population."""
        if self.anchor_inputs is None or self.anchor_features is None:
            return np.zeros(self.pop_size, dtype=np.float64)

        X_anchor = self.anchor_inputs
        target_l2 = self.anchor_features
        X_att = self.attention.forward(X_anchor, train=False)

        losses = np.zeros(self.pop_size, dtype=np.float64)
        for k in range(self.pop_size):
            sample_losses = []
            for sample_idx in range(X_att.shape[0]):
                _, _, l2_out = self._net_forward_with_pop_idx(X_att[sample_idx], k)
                sample_losses.append(float(np.mean((l2_out - target_l2[sample_idx]) ** 2)))
            losses[k] = float(np.mean(sample_losses)) if sample_losses else 0.0

        return losses

    def _capture_best_state(self) -> Dict[str, Any]:
        """Store the current inference state so training can restore the best network seen."""
        return {
            'attention_best_idx': self.attention.best_idx,
            'attention_best_mask': self.attention.population[self.attention.best_idx]['weights'].copy(),
            'level1_elites': [
                None if neuron.elite is None else {
                    'weights': neuron.elite['weights'].copy(),
                    'bias': neuron.elite['bias'],
                    'elite_error': neuron.elite_error,
                }
                for neuron in self.level1
            ],
            'level2_elites': [
                None if neuron.elite is None else {
                    'weights': neuron.elite['weights'].copy(),
                    'bias': neuron.elite['bias'],
                    'elite_error': neuron.elite_error,
                }
                for neuron in self.level2
            ],
            'level3_elites': [
                None if neuron.elite is None else {
                    'weights': neuron.elite['weights'].copy(),
                    'bias': neuron.elite['bias'],
                    'elite_error': neuron.elite_error,
                }
                for neuron in self.level3
            ],
        }

    def _restore_best_state(self, state: Dict[str, Any]) -> None:
        """Restore a previously captured inference state."""
        best_idx = int(state['attention_best_idx'])
        self.attention.best_idx = best_idx
        self.attention.population[best_idx]['weights'] = state['attention_best_mask'].copy()

        for neuron, saved in zip(self.level1, state['level1_elites']):
            if saved is None:
                neuron.elite = None
                neuron.elite_error = float('inf')
            else:
                neuron.elite = {
                    'weights': saved['weights'].copy(),
                    'bias': saved['bias'],
                }
                neuron.elite_error = float(saved['elite_error'])

        for neuron, saved in zip(self.level2, state['level2_elites']):
            if saved is None:
                neuron.elite = None
                neuron.elite_error = float('inf')
            else:
                neuron.elite = {
                    'weights': saved['weights'].copy(),
                    'bias': saved['bias'],
                }
                neuron.elite_error = float(saved['elite_error'])

        for neuron, saved in zip(self.level3, state['level3_elites']):
            if saved is None:
                neuron.elite = None
                neuron.elite_error = float('inf')
            else:
                neuron.elite = {
                    'weights': saved['weights'].copy(),
                    'bias': saved['bias'],
                }
                neuron.elite_error = float(saved['elite_error'])

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
        vm_list = self.V_m.get()

        X_routed_batch, route_idx = self._route_inputs(X_batch)
        X_att_batch = self.attention.forward(X_routed_batch, train=True)

        # --- Vectorized evaluation of all POP_SIZE candidates across the batch ---
        mean_losses, l1_batch, l2_batch = self._evaluate_all_candidates_batch(X_att_batch, Y_true_oh_batch)

        if self.anchor_inputs is not None and self.anchor_features is not None:
            anchor_losses = self._evaluate_anchor_memory_loss()
            mean_losses = mean_losses + (self.anchor_weight * anchor_losses)

        if self.diversity_archive:
            diversity_scores = np.array([self._diversity_score(self._flatten_candidate_state(k)) for k in range(self.pop_size)])
            mean_losses = mean_losses - (self.diversity_bonus * diversity_scores)

        # Best candidate
        best_k = int(np.argmin(mean_losses))
        best_loss = mean_losses[best_k]
        if best_loss < self.best_global_loss:
            self.best_global_loss = float(best_loss)
            self._best_state = self._capture_best_state()

        self.route_best_idx = int(np.bincount(route_idx, minlength=self.route_count).argmax())
        self._update_diversity_archive(self._flatten_candidate_state(best_k), float(best_loss))

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

        # Keep the attention mask adaptive instead of freezing the initial mask.
        self.attention.evolve(fitness_ranks.tolist(), mut_strength)

        # Evolve all neurons using their population indexed by the same k
        for neuron in self.level1:
            neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)

        for neuron in self.level2:
            neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)

        for neuron in self.level3:
            neuron.evolve_rl(fitness_ranks.tolist(), mut_strength, vm_list)

        # --- Update V_m with best L1/L2 directions ---
        # Use RELATIVE trigger: update V_m when batch-best is below 60% of
        # running global_error. This works for any num_classes (no fixed tau needed).
        relative_threshold = self.global_error * 0.6
        
        l1_bench = self.level1[np.argmin([
            n.elite_error if n.elite_error != float('inf') else 999
            for n in self.level1
        ])]
        if l1_bench.last_error is not None and l1_bench.last_error < relative_threshold:
            self.V_m.add(l1_bench.last_weights, l1_bench.last_bias)

        l2_bench = self.level2[np.argmin([
            n.elite_error if n.elite_error != float('inf') else 999
            for n in self.level2
        ])]
        if l2_bench.last_error is not None and l2_bench.last_error < relative_threshold:
            self.V_m.add(l2_bench.last_weights, l2_bench.last_bias)

        return float(best_loss)

    def _train_one_sample(
        self,
        x: np.ndarray,
        y_true_oh: np.ndarray,
        mut_strength: float
    ) -> float:
        """Train the network on a single sample using the batch trainer."""
        x_batch = x.reshape(1, -1) if x.ndim == 1 else x
        y_batch = y_true_oh.reshape(1, -1) if y_true_oh.ndim == 1 else y_true_oh
        return self._train_one_batch(x_batch, y_batch, mut_strength)

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
        x_clean = self._apply_data_guards(x.reshape(1, -1))[0] if x.ndim == 1 else self._apply_data_guards(x)
        x_clean, _ = self._route_inputs(x_clean)
        x_att = self.attention.forward(x_clean, train=False)
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
        
        # Impute missing values and compute variance guards
        if self._imputer_medians is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self._imputer_medians = np.nanmedian(X, axis=0)
                # If a column is ALL NaNs, median is NaN, fallback to 0
                self._imputer_medians = np.nan_to_num(self._imputer_medians, nan=0.0)
                
                stds = np.nanstd(X, axis=0)
                self._variance_guards = (stds == 0)

        # Apply imputer and guards
        X_clean = self._apply_data_guards(X)
        if X_val is not None:
            X_val_clean = self._apply_data_guards(X_val)
        else:
            X_val_clean = None
        
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(X_clean.shape[0])
            X_shuffled = X_clean[perm]
            Y_shuffled = y_oh[perm]
            
            total_loss = 0.0
            num_batches = 0

            for i in range(0, X_clean.shape[0], batch_size):
                end = min(i + batch_size, X_clean.shape[0])
                X_batch = X_shuffled[i:end]
                Y_batch = Y_shuffled[i:end]
                
                loss = self._train_one_batch(X_batch, Y_batch, self.get_mutation_strength())
                total_loss += loss
                num_batches += 1
                self.global_error = total_loss / num_batches

            avg_loss = total_loss / num_batches
            self.global_error = avg_loss

            if epoch % PRINT_INTERVAL == 0:
                train_acc, train_loss = self.evaluate(X_clean, y, y_oh)
                print(f"Epoch {epoch}: Train Acc: {train_acc*100:.2f}%  Elite Loss: {train_loss:.4f}")

            if epoch % THOUGHT_INTERVAL == 0:
                print(f"[Thought] Epoch {epoch}: Avg CE={self.global_error:.4f}, "
                      f"mut_strength={self.get_mutation_strength():.4f}, "
                      f"V_m entries={len(self.V_m)}")

            if X_val_clean is not None and epoch % PRINT_INTERVAL == 0:
                val_acc, val_loss = self.evaluate(X_val_clean, y_val, y_val_oh)
                print(f"  [Val] Acc: {val_acc*100:.2f}%  Loss: {val_loss:.4f}")

        if self._best_state is not None:
            self._restore_best_state(self._best_state)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_oh: np.ndarray,
        batch_size: int = 512
    ) -> Tuple[float, float]:
        """Evaluate using elite weights in batches for massive speedup. No label access."""
        X_clean = self._apply_data_guards(X)
        correct = 0
        total_loss = 0.0
        
        for i in range(0, X_clean.shape[0], batch_size):
            end = min(i + batch_size, X_clean.shape[0])
            X_batch = X_clean[i:end]
            Y_oh_batch = y_oh[i:end]
            Y_batch = y[i:end]
            
            y_preds, _, _, _ = self._forward_elite(X_batch)
            preds_idx = np.argmax(y_preds, axis=1)
            correct += np.sum(preds_idx == Y_batch)
            
            # CE loss
            y_preds_clipped = np.clip(y_preds, 1e-8, 1 - 1e-8)
            batch_loss = -np.sum(Y_oh_batch * np.log(y_preds_clipped))
            total_loss += batch_loss
            
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
