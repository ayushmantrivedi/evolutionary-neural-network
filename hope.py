import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import multiprocessing as mp
from multiprocessing import Pool, Manager
import random
import sys
import time
import os
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_iris, load_wine, load_digits
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Set, Dict
import copy
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import json
from datetime import datetime
from collections import deque

# Import SMOTE first and handle it properly
SMOTE_AVAILABLE = False
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("✅ SMOTE successfully imported and available!")
except ImportError as e:
    print(f"⚠️  SMOTE not available: {e}")
    SMOTE_AVAILABLE = False
except Exception as e:
    print(f"⚠️  SMOTE import failed: {e}")
    SMOTE_AVAILABLE = False



try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: numba not available. Using regular Python functions.")
    NUMBA_AVAILABLE = False
    def njit(func):
        return func
import warnings
warnings.filterwarnings('ignore')

# Global variable to track target scaling
target_scaler_global = None

def preprocess_dataset(X, y, dataset_name="Dataset"):
    """
    Centralized preprocessing function for all datasets.
    Handles cleaning, scaling, and SMOTE balancing if needed.
    
    Args:
        X: Features
        y: Target variable
        dataset_name: Name of the dataset for logging
    
    Returns:
        tuple: (X_processed, y_processed, feature_names)
    """
    print(f"\n=== {dataset_name} Preprocessing ===")
    
    # Convert to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y).flatten()  # Ensure y is 1D
    
    # Detect problem type
    unique_values = len(np.unique(y))
    is_regression = unique_values > 10  # If more than 10 unique values, likely regression
    is_multi_class = unique_values > 2 and unique_values <= 10  # Multi-class classification
    
    if is_regression:
        print("Regression dataset detected. Skipping SMOTE (not applicable).")
        # Standardize features for regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        return X_scaled, y, [f"Feature_{i}" for i in range(X_scaled.shape[1])]
    
    elif is_multi_class:
        print("Multi-class classification detected. Skipping SMOTE to avoid confusion.")
        print("Using original data distribution for better model learning.")
        
        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Standardize features without SMOTE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        return X_scaled, y, [f"Feature_{i}" for i in range(X_scaled.shape[1])]
    
    else:
        # Binary classification - check for imbalance
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Calculate imbalance ratio
        min_count = np.min(counts)
        max_count = np.max(counts)
        imbalance_ratio = min_count / max_count
        print(f"Imbalance ratio: {imbalance_ratio:.3f}")
        
        # Apply SMOTE only for binary classification if imbalanced
        if imbalance_ratio < 0.3:
            print("⚠️  WARNING: Binary classification dataset is imbalanced!")
            if SMOTE_AVAILABLE:
                print("Applying SMOTE to balance the dataset...")
                try:
                    smote = SMOTE(random_state=42)
                    X_balanced, y_balanced = smote.fit_resample(X, y)
                    
                    # Show new distribution
                    unique_after, counts_after = np.unique(y_balanced, return_counts=True)
                    print(f"After SMOTE - Class distribution: {dict(zip(unique_after, counts_after))}")
                    print("✅ SMOTE applied successfully!")
                    
                    X, y = X_balanced, y_balanced
                except Exception as e:
                    print(f"❌ Error applying SMOTE: {e}")
                    print("⚠️  Continuing with original imbalanced dataset...")
            else:
                print("❌ SMOTE not available. Cannot balance imbalanced dataset.")
                print("⚠️  WARNING: Training on imbalanced dataset may lead to poor performance!")
                print("💡 Suggestion: Install imbalanced-learn package: pip install imbalanced-learn")
        else:
            print("✅ Binary classification dataset is balanced. No SMOTE needed.")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Preprocessing complete. Final shape: {X_scaled.shape}")
        return X_scaled, y, [f"Feature_{i}" for i in range(X_scaled.shape[1])]

# Multi-class Neural Network Hyperparameters
LEVEL1_NEURONS = 50
LEVEL2_NEURONS = 20
POP_SIZE = 20
VM_HISTORY = 20
VM_INFLUENCE_PROB = 0.2
VM_IMPROVEMENT_THRESH = 0.15
TAU1 = 0.15
TAU2 = 0.10
MUT_STRENGTH_BASE = 0.1
EPOCHS = 50
PRINT_INTERVAL = 5
THOUGHT_INTERVAL = 10
MIN_MUT_STRENGTH = 0.01

# Local refinement hyperparameters
LOCAL_GD_ENABLED = True
LOCAL_GD_LR = 0.05
LOCAL_LINESEARCH_ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
LOCAL_MIN_IMPROVEMENT = 1e-8

# Multi-class Neural Network Classes
@njit
def population_forward(x, weights, biases):
    return np.dot(weights.astype(np.float32), x.astype(np.float32)) + biases.astype(np.float32)

class EvoNeuron:
    def __init__(self, input_dim, pop_size=POP_SIZE):
        self.input_dim = input_dim
        self.pop_size = pop_size
        self.population = [self._random_individual() for _ in range(pop_size)]
        self.best_idx = 0
        self.last_error = None
        self.last_output = None
        self.last_weights = None
        self.last_bias = None
        self.elite = None
        self.elite_error = float('inf')

    def _random_individual(self):
        return {
            'weights': np.random.randn(self.input_dim).astype(np.float32),
            'bias': np.float32(np.random.randn()),
        }

    def _check_population_shapes(self):
        for ind in self.population:
            if ind['weights'].shape != (self.input_dim,):
                print(f"[DEBUG] Population shape mismatch detected. Reinitializing population for input_dim={self.input_dim}.")
                self.population = [self._random_individual() for _ in range(self.pop_size)]
                break

    def forward(self, x, y_true, error_fn, mutation_strength, V_m=None):
        self._check_population_shapes()
        weights = np.array([ind['weights'] for ind in self.population])
        biases = np.array([ind['bias'] for ind in self.population])
        outs = population_forward(x, weights, biases)
        errors = np.array([error_fn(out, y_true) for out in outs])
        best_idx = np.argmin(errors)
        self.best_idx = best_idx
        self.last_error = errors[best_idx]
        self.last_output = outs[best_idx]
        self.last_weights = self.population[best_idx]['weights'].copy()
        self.last_bias = self.population[best_idx]['bias']
        if self.last_error < self.elite_error:
            self.elite_error = self.last_error
            self.elite = {
                'weights': self.last_weights.copy(),
                'bias': self.last_bias
            }
        return self.last_output, self.last_error

    def _activate(self, x):
        return x

    def evolve(self, x, y_true, error_fn, mutation_strength, V_m=None, tau=None):
        self._check_population_shapes()
        errors = []
        for ind in self.population:
            out = self._activate(np.dot(x, ind['weights']) + ind['bias'])
            err = error_fn(out, y_true)
            errors.append(err)
        idx_sorted = np.argsort(errors)
        survivors = [self.population[i] for i in idx_sorted[:2]]
        if self.elite is not None:
            survivors.append({'weights': self.elite['weights'].copy(), 'bias': self.elite['bias']})
        # Local GD micro-step aligned with mean V_m direction on best survivor
        if LOCAL_GD_ENABLED and len(survivors) > 0:
            best = survivors[0]
            w = best['weights']
            b = best['bias']
            out = self._activate(np.dot(x, w) + b)
            err_current = error_fn(out, y_true)
            trigger = (tau is None) or (err_current >= tau)
            if trigger:
                d_hat_w = None
                d_hat_b = None
                if V_m and len(V_m) > 0:
                    vm_filtered = [v for v in V_m if isinstance(v.get('weights', None), np.ndarray) and v['weights'].shape == w.shape]
                    if len(vm_filtered) > 0:
                        w_bar = np.mean([v['weights'] for v in vm_filtered], axis=0)
                        b_bar = float(np.mean([v['bias'] for v in vm_filtered]))
                        d_w = (w_bar - w).astype(np.float32)
                        d_b = np.float32(b_bar - b)
                        denom = np.sqrt(np.dot(d_w, d_w) + float(d_b) * float(d_b)) + 1e-12
                        d_hat_w = d_w / denom
                        d_hat_b = d_b / denom
                if isinstance(y_true, np.ndarray):
                    y_bar = float(np.mean(y_true))
                else:
                    y_bar = float(y_true)
                grad_signal = 2.0 * (float(out) - y_bar)
                s_w = (-LOCAL_GD_LR * grad_signal) * x.astype(np.float32)
                s_b = np.float32(-LOCAL_GD_LR * grad_signal)
                if d_hat_w is not None:
                    dot_sb = float(np.dot(s_w, d_hat_w) + float(s_b) * float(d_hat_b))
                    if dot_sb < 0.0:
                        s_w = s_w - dot_sb * d_hat_w
                        s_b = np.float32(float(s_b) - dot_sb * float(d_hat_b))
                w_prop = (w + s_w).astype(np.float32)
                b_prop = np.float32(b + s_b)
                out_prop = self._activate(np.dot(x, w_prop) + b_prop)
                err_prop = error_fn(out_prop, y_true)
                improved = (err_current - err_prop) > LOCAL_MIN_IMPROVEMENT
                if (not improved) and (d_hat_w is not None):
                    best_alpha = 0.0
                    best_err = err_current
                    for alpha in LOCAL_LINESEARCH_ALPHAS:
                        if alpha <= 0.0:
                            continue
                        w_ls = (w + alpha * d_hat_w).astype(np.float32)
                        b_ls = np.float32(b + alpha * float(d_hat_b))
                        out_ls = self._activate(np.dot(x, w_ls) + b_ls)
                        err_ls = error_fn(out_ls, y_true)
                        if err_ls + 1e-12 < best_err:
                            best_err = err_ls
                            best_alpha = alpha
                    if best_alpha > 0.0:
                        w_prop = (w + best_alpha * d_hat_w).astype(np.float32)
                        b_prop = np.float32(b + best_alpha * float(d_hat_b))
                        err_prop = best_err
                        improved = True
                if improved:
                    best['weights'] = w_prop
                    best['bias'] = b_prop
        new_pop = survivors.copy()
        while len(new_pop) < self.pop_size:
            parent = random.choice(survivors)
            assert parent['weights'].shape == (self.input_dim,)
            child = {
                'weights': parent['weights'] + np.random.randn(self.input_dim) * mutation_strength,
                'bias': parent['bias'] + np.random.randn() * mutation_strength
            }
            if V_m and len(V_m) > 0 and random.random() < VM_INFLUENCE_PROB:
                v = random.choice(V_m)
                if v['weights'].shape == (self.input_dim,):
                    child['weights'] += v['weights'] * 0.5
                child['bias'] += v['bias'] * 0.5
            new_pop.append(child)
        self.population = new_pop

class OutputNeuron(EvoNeuron):
    def _activate(self, x):
        return x

def mse_loss(pred, true):
    return np.mean((pred - true) ** 2)

def ce_loss(pred, true):
    pred = np.clip(pred, 1e-8, 1-1e-8)
    return -np.sum(true * np.log(pred))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def ce_loss_with_confidence(pred, true, reward_weight=0.1):
    pred = np.clip(pred, 1e-8, 1-1e-8)
    ce = -np.sum(true * np.log(pred))
    correct_prob = np.sum(pred * true)
    max_other_prob = np.max(pred * (1 - true))
    confidence_margin = correct_prob - max_other_prob
    reward = reward_weight * confidence_margin
    return ce - reward

class SignificantMutationVector:
    def __init__(self, maxlen=VM_HISTORY):
        self.deque = deque(maxlen=maxlen)
    def add(self, weights, bias):
        self.deque.append({'weights': weights.copy(), 'bias': bias})
    def get(self):
        return list(self.deque)

class MultiClassEvoNet:
    def __init__(self, input_dim, num_classes):
        self.level1 = [EvoNeuron(input_dim) for _ in range(LEVEL1_NEURONS)]
        self.level2 = [EvoNeuron(LEVEL1_NEURONS) for _ in range(LEVEL2_NEURONS)]
        self.level3 = [OutputNeuron(LEVEL2_NEURONS) for _ in range(num_classes)]
        self.num_classes = num_classes
        self.V_m = SignificantMutationVector()
        self.tau1 = TAU1
        self.tau2 = TAU2
        self.mut_strength_base = MUT_STRENGTH_BASE
        self.global_error = 1.0

    def get_mutation_strength(self):
        mut_strength = self.mut_strength_base * (self.global_error ** 2)
        return max(mut_strength, MIN_MUT_STRENGTH)

    def forward(self, x, y_true, train=True):
        mut_strength = self.get_mutation_strength()
        l1_outputs = []
        l1_errors = []
        l1_marks = []
        for neuron in self.level1:
            out, err = neuron.forward(x, y_true, mse_loss, mut_strength, self.V_m.get())
            l1_outputs.append(out)
            l1_errors.append(err)
            if err < self.tau1:
                l1_marks.append(out)
            else:
                l1_marks.append(('*', out))
        l1_outputs = np.array(l1_outputs, dtype=object)
        l1_errors = np.array(l1_errors)
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
            out, err = neuron.forward(np.full(LEVEL1_NEURONS, inp_val), y_true, mse_loss, mut_strength, self.V_m.get())
            l2_outputs.append(out)
            l2_errors.append(err)
            if err < self.tau2:
                l2_marks.append(out)
            else:
                l2_marks.append(('*', out))
        l2_outputs = np.array(l2_outputs, dtype=object)
        l2_errors = np.array(l2_errors)
        l3_inputs = l2_marks
        assert len(l3_inputs) == LEVEL2_NEURONS
        l3_outputs = []
        for i, (neuron, inp) in enumerate(zip(self.level3, l3_inputs)):
            if isinstance(inp, tuple) and inp[0] == '*':
                inp_val = inp[1]
            else:
                inp_val = inp
            out, _ = neuron.forward(np.full(LEVEL2_NEURONS, inp_val), y_true[i], mse_loss, mut_strength, self.V_m.get())
            l3_outputs.append(out)
        l3_outputs = np.array(l3_outputs)
        y_pred = softmax(l3_outputs)
        if train:
            for i, neuron in enumerate(self.level3):
                inp = l3_inputs[i]
                inp_val = inp[1] if isinstance(inp, tuple) and inp[0] == '*' else inp
                neuron.evolve(np.full(LEVEL2_NEURONS, inp_val), y_true[i], mse_loss, mut_strength, self.V_m.get(), tau=self.tau2)
        return y_pred, l1_errors, l2_errors, l3_outputs

    def train(self, X, y, y_oh, epochs=EPOCHS, X_val=None, y_val=None, y_val_oh=None):
        for epoch in range(1, epochs+1):
            print(f"Starting epoch {epoch}")
            correct = 0
            total_loss = 0
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
                l1_bench_idx = np.argmin(l1_errs)
                l1_bench_err = l1_errs[l1_bench_idx]
                l1_bench = self.level1[l1_bench_idx]
                if l1_bench.last_error is not None and l1_bench.last_error < self.tau1:
                    self.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
                l2_bench_idx = np.argmin(l2_errs)
                l2_bench_err = l2_errs[l2_bench_idx]
                l2_bench = self.level2[l2_bench_idx]
                if l2_bench.last_error is not None and l2_bench.last_error < self.tau2:
                    self.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
            acc = correct / X.shape[0]
            avg_loss = total_loss / X.shape[0]
            self.global_error = avg_loss
            print(f"Finished epoch {epoch}")
            if epoch % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch}: Global Accuracy: {acc*100:.2f}%, Loss: {avg_loss:.4f}")
            if epoch % THOUGHT_INTERVAL == 0:
                print(f"[Thought] Epoch {epoch}: Global error is {self.global_error:.4f}, mutation strength now {self.get_mutation_strength():.4f}")
            if X_val is not None and epoch % PRINT_INTERVAL == 0:
                val_acc, val_loss = self.evaluate(X_val, y_val, y_val_oh)
                print(f"[Validation] Accuracy: {val_acc*100:.2f}%, Loss: {val_loss:.4f}")

    def evaluate(self, X, y, y_oh):
        correct = 0
        total_loss = 0
        for i in range(X.shape[0]):
            x = X[i]
            y_true = y_oh[i]
            y_label = y[i]
            y_pred, _, _, _ = self.forward(x, y_true, train=False)
            pred_label = np.argmax(y_pred)
            if pred_label == y_label:
                correct += 1
            loss = ce_loss(y_pred, y_true)
            total_loss += loss
        acc = correct / X.shape[0]
        avg_loss = total_loss / X.shape[0]
        return acc, avg_loss

# Regression Neural Network Classes (Integrated from ex1.py)
class RegressionOutputNeuron(EvoNeuron):
    def _activate(self, x):
        return x  # Linear activation for regression

class RegressionEvoNet:
    def __init__(self, input_dim):
        # KEPT THE SAME: Layer 1 and Layer 2
        self.level1 = [EvoNeuron(input_dim) for _ in range(LEVEL1_NEURONS)]
        self.level2 = [EvoNeuron(LEVEL1_NEURONS) for _ in range(LEVEL2_NEURONS)]
        
        # CHANGED: Single output neuron instead of multiple class neurons
        self.output_neuron = RegressionOutputNeuron(LEVEL2_NEURONS)
        
        # KEPT THE SAME: All evolutionary mechanisms
        self.V_m = SignificantMutationVector()
        self.tau1 = TAU1
        self.tau2 = TAU2
        self.mut_strength_base = MUT_STRENGTH_BASE
        self.global_error = 1.0

    def get_mutation_strength(self):
        # KEPT THE SAME: Adaptive mutation strength
        mut_strength = self.mut_strength_base * (self.global_error ** 2)
        return max(mut_strength, MIN_MUT_STRENGTH)

    def forward(self, x, y_true, train=True):
        mut_strength = self.get_mutation_strength()
        
        # KEPT THE SAME: Level 1 (sequential for speed)
        l1_outputs = []
        l1_errors = []
        l1_marks = []
        for neuron in self.level1:
            out, err = neuron.forward(x, y_true, mse_loss, mut_strength, self.V_m.get())
            l1_outputs.append(out)
            l1_errors.append(err)
            if err < self.tau1:
                l1_marks.append(out)
            else:
                l1_marks.append(('*', out))  # Mark failed neuron output as tuple
        l1_outputs = np.array(l1_outputs, dtype=object)
        l1_errors = np.array(l1_errors)
        
        # KEPT THE SAME: Level 1: pass all outputs, but mark failed ones
        l2_inputs = l1_marks
        assert len(l2_inputs) == LEVEL1_NEURONS
        
        # KEPT THE SAME: Level 2 (sequential for speed)
        l2_outputs = []
        l2_errors = []
        l2_marks = []
        for neuron, inp in zip(self.level2, l2_inputs):
            # If input is marked, extract value for computation
            if isinstance(inp, tuple) and inp[0] == '*':
                inp_val = inp[1]
            else:
                inp_val = inp
            out, err = neuron.forward(np.full(LEVEL1_NEURONS, inp_val), y_true, mse_loss, mut_strength, self.V_m.get())
            l2_outputs.append(out)
            l2_errors.append(err)
            if err < self.tau2:
                l2_marks.append(out)
            else:
                l2_marks.append(('*', out))  # Mark failed neuron output as tuple
        l2_outputs = np.array(l2_outputs, dtype=object)
        l2_errors = np.array(l2_errors)
        
        # KEPT THE SAME: Level 2: pass all outputs, but mark failed ones
        l3_inputs = l2_marks
        assert len(l3_inputs) == LEVEL2_NEURONS
        
        # CHANGED: Single output neuron instead of multiple class neurons
        l3_input = l3_inputs[0] if not isinstance(l3_inputs[0], tuple) else l3_inputs[0][1]
        out, _ = self.output_neuron.forward(np.full(LEVEL2_NEURONS, l3_input), y_true, mse_loss, mut_strength, self.V_m.get())
        
        # CHANGED: No softmax needed for regression
        y_pred = out  # Single continuous value
        
        # KEPT THE SAME: Evolution for output neuron
        if train:
            self.output_neuron.evolve(np.full(LEVEL2_NEURONS, l3_input), y_true, mse_loss, mut_strength, self.V_m.get(), tau=self.tau2)
        
        return y_pred, l1_errors, l2_errors

    def train(self, X, y, epochs=EPOCHS, X_val=None, y_val=None):
        for epoch in range(1, epochs+1):
            print(f"Starting epoch {epoch}")
            total_loss = 0
            for i in range(X.shape[0]):
                x = X[i]
                y_true = y[i]
                y_pred, l1_errs, l2_errs = self.forward(x, y_true, train=True)
                
                # CHANGED: MSE loss instead of cross-entropy
                loss = mse_loss(y_pred, y_true)
                total_loss += loss
                
                # KEPT THE SAME: Benchmark neuron and V_m update
                # Level 1
                l1_bench_idx = np.argmin(l1_errs)
                l1_bench_err = l1_errs[l1_bench_idx]
                l1_bench = self.level1[l1_bench_idx]
                if l1_bench.last_error is not None and l1_bench.last_error < self.tau1:
                    self.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
                
                # Level 2
                l2_bench_idx = np.argmin(l2_errs)
                l2_bench_err = l2_errs[l2_bench_idx]
                l2_bench = self.level2[l2_bench_idx]
                if l2_bench.last_error is not None and l2_bench.last_error < self.tau2:
                    self.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
            
            avg_loss = total_loss / X.shape[0]
            self.global_error = avg_loss  # Use for mutation scaling
            print(f"Finished epoch {epoch}")
            
            if epoch % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch}: MSE Loss: {avg_loss:.4f}")
            
            if epoch % THOUGHT_INTERVAL == 0:
                print(f"[Thought] Epoch {epoch}: Global error is {self.global_error:.4f}, mutation strength now {self.get_mutation_strength():.4f}")
            
            # Optionally, evaluate on validation set
            if X_val is not None and epoch % PRINT_INTERVAL == 0:
                val_loss, _ = self.evaluate(X_val, y_val)
                print(f"[Validation] MSE Loss: {val_loss:.4f}")

    def evaluate(self, X, y):
        total_loss = 0
        predictions = []
        for i in range(X.shape[0]):
            x = X[i]
            y_true = y[i]
            y_pred, _, _ = self.forward(x, y_true, train=False)
            
            # CHANGED: MSE loss instead of cross-entropy
            loss = mse_loss(y_pred, y_true)
            total_loss += loss
            predictions.append(y_pred)
        
        avg_loss = total_loss / X.shape[0]
        return avg_loss, np.array(predictions)

# Faster training methods for regression
def mini_batch_evolution_training(model, X, y, batch_size=32, iterations=1000, X_val=None, y_val=None):
    """
    Fast training using mini-batch evolution instead of epochs
    """
    print(f"Starting mini-batch evolution training...")
    print(f"Batch size: {batch_size}, Iterations: {iterations}")
    
    n_samples = X.shape[0]
    total_loss = 0
    
    for iteration in range(iterations):
        # Sample a mini-batch
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        batch_loss = 0
        for i in range(batch_size):
            x = X_batch[i]
            y_true = y_batch[i]
            y_pred, l1_errs, l2_errs = model.forward(x, y_true, train=True)
            loss = mse_loss(y_pred, y_true)
            batch_loss += loss
            
            # Update V_m for good neurons (same as before)
            l1_bench_idx = np.argmin(l1_errs)
            l1_bench = model.level1[l1_bench_idx]
            if l1_bench.last_error is not None and l1_bench.last_error < model.tau1:
                model.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
            
            l2_bench_idx = np.argmin(l2_errs)
            l2_bench = model.level2[l2_bench_idx]
            if l2_bench.last_error is not None and l2_bench.last_error < model.tau2:
                model.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
        
        avg_batch_loss = batch_loss / batch_size
        total_loss += avg_batch_loss
        model.global_error = avg_batch_loss
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Batch MSE Loss: {avg_batch_loss:.4f}")
            
        if iteration % 500 == 0:
            print(f"[Thought] Iteration {iteration}: Global error is {model.global_error:.4f}, mutation strength now {model.get_mutation_strength():.4f}")
            
        if X_val is not None and iteration % 200 == 0:
            val_loss, _ = model.evaluate(X_val, y_val)
            print(f"[Validation] MSE Loss: {val_loss:.4f}")
    
    return total_loss / iterations

def early_stopping_training(model, X, y, patience=20, min_delta=0.001, X_val=None, y_val=None):
    """
    Training with early stopping to avoid overfitting and reduce training time
    """
    print(f"Starting early stopping training...")
    print(f"Patience: {patience}, Min delta: {min_delta}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    iteration = 0
    
    while patience_counter < patience:
        # Sample a batch
        batch_indices = np.random.choice(len(X), min(64, len(X)), replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        batch_loss = 0
        for i in range(len(X_batch)):
            x = X_batch[i]
            y_true = y_batch[i]
            y_pred, l1_errs, l2_errs = model.forward(x, y_true, train=True)
            loss = mse_loss(y_pred, y_true)
            batch_loss += loss
            
            # Update V_m
            l1_bench_idx = np.argmin(l1_errs)
            l1_bench = model.level1[l1_bench_idx]
            if l1_bench.last_error is not None and l1_bench.last_error < model.tau1:
                model.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
            
            l2_bench_idx = np.argmin(l2_errs)
            l2_bench = model.level2[l2_bench_idx]
            if l2_bench.last_error is not None and l2_bench.last_error < model.tau2:
                model.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
        
        avg_loss = batch_loss / len(X_batch)
        model.global_error = avg_loss
        
        # Early stopping check
        if X_val is not None:
            val_loss, _ = model.evaluate(X_val, y_val)
            
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: Train MSE: {avg_loss:.4f}, Val MSE: {val_loss:.4f}, Patience: {patience_counter}")
        else:
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: MSE Loss: {avg_loss:.4f}")
        
        iteration += 1
        
        if iteration > 2000:  # Max iterations
            break
    
    print(f"Training stopped after {iteration} iterations")
    return avg_loss

def clean_dataset(df):
    print("\n--- Cleaning Dataset ---")
    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")
    # Drop columns with >50% missing values
    thresh = int(0.5 * len(df))
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=thresh)
    print(f"Dropped {before_cols - df.shape[1]} columns with >50% missing values.")
    # Remove constant columns
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=const_cols)
    if const_cols:
        print(f"Removed constant columns: {const_cols}")
    # Impute missing values
    for col in df.columns:
        if df[col].dtype == 'O':
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
        else:
            mean = df[col].mean()
            df[col] = df[col].fillna(mean)
    print("Imputed missing values (mean for numeric, mode for categorical).")
    return df

def load_housing_dataset(custom_path=None):
    """Load and preprocess the California Housing Dataset (sklearn or custom CSV)"""
    if custom_path is not None:
        print(f"\nLoading custom dataset from: {custom_path}")
        try:
            df = pd.read_csv(custom_path)
        except FileNotFoundError:
            print(f"Error: File not found at {custom_path}")
            print("Please check the file path and try again.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV file {custom_path}: {e}")
            sys.exit(1)
        df = clean_dataset(df)
        feature_df = df.iloc[:, :-1]
        feature_df = pd.get_dummies(feature_df)
        scaler = StandardScaler()
        X = scaler.fit_transform(feature_df.values.astype(np.float64))
        target_col = df.iloc[:, -1]
        y = target_col.values.reshape(-1, 1).astype(np.float64)
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        print("\nFeature names:")
        for i, name in enumerate(feature_df.columns):
            print(f"{i+1}. {name}")
        print("\nTarget: Housing Value")
        
        # Apply centralized preprocessing (no SMOTE for regression)
        X, y, feature_names = preprocess_dataset(X, y, "Custom Housing")
        
        return X, y
    else:
        housing = fetch_california_housing()
        X = housing.data
        y = housing.target.reshape(-1, 1)
        print("\nDataset Information:")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        print("\nFeature names:")
        for i, name in enumerate(housing.feature_names):
            print(f"{i+1}. {name}")
        print("\nTarget: Housing Value")
        
        # Apply centralized preprocessing (no SMOTE for regression)
        X, y, feature_names = preprocess_dataset(X, y, "California Housing")
        
        return X, y

def load_cancer_dataset():
    """Load and preprocess the Breast Cancer Dataset (sklearn)"""
    print("\nLoading breast cancer dataset (sklearn)...")
    data = load_breast_cancer()
    X = data.data
    y = data.target.reshape(-1, 1)
    feature_names = data.feature_names
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i+1}. {name}")
    print("\nTarget: Cancer (0=malignant, 1=benign)")
    
    # Apply centralized preprocessing
    X, y, feature_names = preprocess_dataset(X, y, "Breast Cancer")
    
    return X, y, feature_names

def load_iris_dataset():
    """Load and preprocess the Iris Dataset (sklearn) - Multi-class classification"""
    print("\nLoading iris dataset (sklearn)...")
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i+1}. {name}")
    print("\nTarget classes:")
    for i, name in enumerate(target_names):
        print(f"  Class {i}: {name}")
    
    # Create visualization of the dataset
    desktop_dir = r'C:/Users/ayush/Desktop'
    os.makedirs(desktop_dir, exist_ok=True)
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar([target_names[i] for i in unique], counts, color=['red', 'green', 'blue'])
    plt.xlabel('Iris Species')
    plt.ylabel('Count')
    plt.title('Iris Dataset Class Distribution')
    
    # Feature distributions by class
    plt.subplot(1, 2, 2)
    for i, target_name in enumerate(target_names):
        class_data = X[y.ravel() == i]
        plt.scatter(class_data[:, 0], class_data[:, 1], label=target_name, alpha=0.7)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Iris Dataset: Sepal Length vs Sepal Width')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_dir, 'iris_dataset_analysis.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Iris dataset analysis saved to: iris_dataset_analysis.png")
    
    # Apply centralized preprocessing
    X, y, feature_names = preprocess_dataset(X, y, "Iris")
    
    return X, y, feature_names

def load_wine_dataset():
    """Load and preprocess the Wine Dataset (sklearn) - Multi-class classification"""
    print("\nLoading wine dataset (sklearn)...")
    data = load_wine()
    X = data.data
    y = data.target.reshape(-1, 1)
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i+1}. {name}")
    print("\nTarget classes:")
    for i, name in enumerate(target_names):
        print(f"  Class {i}: {name}")
    
    # Create visualization of the dataset
    desktop_dir = r'C:/Users/ayush/Desktop'
    os.makedirs(desktop_dir, exist_ok=True)
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.bar([target_names[i] for i in unique], counts, color=['red', 'green', 'blue'])
    plt.xlabel('Wine Type')
    plt.ylabel('Count')
    plt.title('Wine Dataset Class Distribution')
    plt.xticks(rotation=45)
    
    # Feature distributions by class
    plt.subplot(1, 3, 2)
    for i, target_name in enumerate(target_names):
        class_data = X[y.ravel() == i]
        plt.scatter(class_data[:, 0], class_data[:, 1], label=target_name, alpha=0.7)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Wine Dataset: Alcohol vs Malic Acid')
    plt.legend()
    
    # Feature correlation heatmap
    plt.subplot(1, 3, 3)
    correlation_matrix = np.corrcoef(X.T)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_dir, 'wine_dataset_analysis.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Wine dataset analysis saved to: wine_dataset_analysis.png")
    
    # Apply centralized preprocessing
    X, y, feature_names = preprocess_dataset(X, y, "Wine")
    
    return X, y, feature_names

def load_digits_dataset():
    """Load and preprocess the Digits Dataset (sklearn) - Multi-class classification"""
    print("\nLoading digits dataset (sklearn)...")
    data = load_digits()
    X = data.data
    y = data.target.reshape(-1, 1)
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print("\nTarget: Digit recognition (0-9)")
    
    # Create visualization of the dataset
    desktop_dir = r'C:/Users/ayush/Desktop'
    os.makedirs(desktop_dir, exist_ok=True)
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 6))
    plt.bar(unique.ravel(), counts, color='skyblue')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.title('Digits Dataset Class Distribution')
    plt.xticks(unique.ravel())
    
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_dir, 'digits_dataset_analysis.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Digits dataset analysis saved to: digits_dataset_analysis.png")
    
    # Apply centralized preprocessing
    X, y, feature_names = preprocess_dataset(X, y, "Digits")
    
    return X, y, feature_names



def load_custom_csv_dataset(custom_path):
    print(f"\nLoading custom dataset from: {custom_path}")
    try:
        df = pd.read_csv(custom_path, encoding='latin1', nrows=2000)
    except FileNotFoundError:
        print(f"Error: File not found at {custom_path}")
        print("Please check the file path and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {custom_path}: {e}")
        sys.exit(1)
    
    # Check if DataFrame is empty
    if df.empty:
        print("Error: No data found in the CSV file.")
        sys.exit(1)
    
    df = clean_dataset(df)
    
    # Check if DataFrame is empty after cleaning
    if df.empty:
        print("Error: No data remaining after cleaning.")
        sys.exit(1)
    
    # Exclude likely ID columns
    cols_to_exclude = [col for col in df.columns if 'id' in col.lower() or 'ID' in col]
    print(f"Excluding columns from features (ID-like): {cols_to_exclude}")
    
    # Get feature columns (all except the last column which is the target)
    feature_df = df.iloc[:, :-1].copy()
    
    # Remove excluded columns
    feature_df = feature_df.drop(columns=cols_to_exclude, errors='ignore')
    
    # Check if feature_df is empty after removing ID columns
    if feature_df.empty:
        print("Error: No features remaining after removing ID columns.")
        sys.exit(1)
    
    # Exclude high-cardinality columns (>50 unique values), but keep stock-related columns
    protected_cols = ['booking_origin', 'purchase_lead', 'open', 'high', 'low', 'close', 'volume', 'price', 'amount']
    # Also protect columns that contain these terms (case-insensitive)
    stock_terms = ['open', 'high', 'low', 'close', 'volume', 'price', 'amount', 'value', 'stock', 'market']
    
    # Check if this looks like stock data
    is_stock_data = any(term in col.lower() for col in feature_df.columns for term in stock_terms)
    
    if is_stock_data:
        print("📈 Stock data detected! Protecting OHLCV and price-related columns from high-cardinality exclusion.")
        # For stock data, only exclude obvious non-feature columns
        exclude_cols = [col for col in feature_df.columns if any(term in col.lower() for term in ['id', 'index', 'row'])]
        if exclude_cols:
            print(f"Excluding non-feature columns: {exclude_cols}")
            feature_df = feature_df.drop(columns=exclude_cols, errors='ignore')
    else:
        # For non-stock data, use original logic
        high_card_cols = [col for col in feature_df.columns if feature_df[col].nunique() > 50 and col not in protected_cols]
        print(f"Excluding high-cardinality columns (>50 unique values, except protected): {high_card_cols}")
        feature_df = feature_df.drop(columns=high_card_cols, errors='ignore')
    
    # Check if feature_df is empty after removing high-cardinality columns
    if feature_df.empty:
        print("Error: No features remaining after removing high-cardinality columns.")
        sys.exit(1)
    
    # --- Visualize feature cardinality ---
    desktop_dir = r'C:/Users/ayush/Desktop'
    os.makedirs(desktop_dir, exist_ok=True)
    cardinalities = [feature_df[col].nunique() for col in feature_df.columns]
    plt.figure(figsize=(min(16, max(8, len(feature_df.columns)//2)), 4))
    plt.bar(feature_df.columns, cardinalities)
    plt.xticks(rotation=90)
    plt.ylabel('Unique Values')
    plt.title('Feature Cardinality (Unique Values per Feature)')
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_dir, 'feature_cardinality.png'), bbox_inches='tight')
    plt.close()
    
    # Convert categorical variables to dummy variables with error handling
    try:
        # Separate numeric and categorical columns
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
        
        # Process numeric columns
        numeric_df = feature_df[numeric_cols] if numeric_cols else pd.DataFrame()
        
        # Process categorical columns
        if categorical_cols:
            categorical_df = feature_df[categorical_cols]
            # Ensure categorical columns have values
            categorical_df = categorical_df.dropna()
            if not categorical_df.empty:
                dummies_df = pd.get_dummies(categorical_df, drop_first=False)
            else:
                print("Warning: No valid categorical data found.")
                dummies_df = pd.DataFrame()
        else:
            dummies_df = pd.DataFrame()
        
        # Combine numeric and dummy variables
        if not numeric_df.empty and not dummies_df.empty:
            feature_df = pd.concat([numeric_df, dummies_df], axis=1)
        elif not numeric_df.empty:
            feature_df = numeric_df
        elif not dummies_df.empty:
            feature_df = dummies_df
        else:
            print("Error: No valid features found after processing.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error processing features: {e}")
        print("Trying alternative approach...")
        # Fallback: use only numeric columns
        feature_df = feature_df.select_dtypes(include=[np.number])
        if feature_df.empty:
            print("Error: No numeric features found.")
            sys.exit(1)
    
    # Check if feature_df is empty
    if feature_df.empty:
        print("Error: No features available after processing.")
        sys.exit(1)
    
    # Handle large datasets by sampling if too large
    max_samples = 2000  # Match nrows=2000 for consistency
    if len(feature_df) > max_samples:
        print(f"⚠️  Large dataset detected ({len(feature_df)} samples). Sampling {max_samples} random samples for efficiency.")
        sampled = feature_df.sample(n=max_samples, random_state=42)
        feature_df = sampled
        # Also sample the target column using the same indices
        df = df.loc[sampled.index]
        print(f"✅ Sampled {len(feature_df)} samples for processing.")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values.astype(np.float64))
    # X = feature_df.values.astype(np.float64)
    
    target_col = df.iloc[:, -1]
    
    # Encode target column if it's categorical
    if target_col.dtype == 'object':
        print(f"Target column '{df.columns[-1]}' is categorical. Encoding...")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(target_col.values)
        y = y_encoded.reshape(-1, 1)
        print(f"Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    else:
        # For regression problems, also scale the target to prevent numerical issues
        y_raw = target_col.values.reshape(-1, 1)
        
        # Check if target values are very large (like volume in trillions)
        y_max = np.max(np.abs(y_raw))
        if y_max > 1e6:  # If max value > 1 million
            print(f"⚠️  Large target values detected (max: {y_max:.2e}). Scaling target for numerical stability.")
            target_scaler = StandardScaler()
            y = target_scaler.fit_transform(y_raw)
            print(f"✅ Target scaled to range [{np.min(y):.3f}, {np.max(y):.3f}]")
            # Store target scaler for later use
            global target_scaler_global
            target_scaler_global = target_scaler
        else:
            y = y_raw
            print(f"✅ Target values in reasonable range [{np.min(y):.3f}, {np.max(y):.3f}]")
            target_scaler_global = None
    
    feature_names = feature_df.columns.tolist()
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i+1}. {name}")
    print(f"\nTarget: {df.columns[-1]}")
    # --- Visualize class distribution ---
    y_flat = y.ravel()
    unique, counts = np.unique(y_flat, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.bar([str(u) for u in unique], counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Target')
    plt.savefig(os.path.join(desktop_dir, 'class_distribution.png'), bbox_inches='tight')
    plt.close()
    # print("Class distribution in target:")
    # for val, count in zip(unique, counts):
    #     print(f"Class {val}: {count} samples")
    
    print(f"🔄 Processing {len(X)} samples...")
    X, y, feature_names = preprocess_dataset(X, y, "Custom CSV")
    
    return X, y, feature_names

def load_telemetry_dataset(json_path):
    """Load and preprocess the telemetry JSON dataset"""
    print(f"\nLoading telemetry dataset from: {json_path}")
    
    try:
        # Load JSON data with proper encoding
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        print("Please check the file path and try again.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_path}")
        print(f"JSON Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {json_path}: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(data)} telemetry records")
    
    # Convert to DataFrame
    records = []
    for record in data:
        # Extract features
        record_dict = {
            'deviceID': record['deviceID'],
            'deviceType': record['deviceType'],
            'timestamp': record['timestamp'],
            'country': record['location']['country'],
            'city': record['location']['city'],
            'area': record['location']['area'],
            'factory': record['location']['factory'],
            'section': record['location']['section'],
            'temperature': record['data']['temperature'],
            'status': record['data']['status']
        }
        records.append(record_dict)
    
    df = pd.DataFrame(records)
    
    # Check if DataFrame is empty
    if df.empty:
        print("Error: No data found in the JSON file.")
        sys.exit(1)
    
    # Convert timestamp to datetime features
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Create target variable (1 for healthy, 0 for unhealthy)
    df['target'] = (df['status'] == 'healthy').astype(int)
    
    # Clean the dataset
    df = clean_dataset(df)
    
    # Check if DataFrame is empty after cleaning
    if df.empty:
        print("Error: No data remaining after cleaning.")
        sys.exit(1)
    
    # Select features for modeling (exclude deviceID, timestamp, datetime, status, target)
    feature_cols = ['deviceType', 'country', 'city', 'area', 'factory', 'section', 
                   'temperature', 'hour', 'day_of_week', 'month']
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        print("Error: No valid feature columns found.")
        sys.exit(1)
    
    feature_df = df[feature_cols].copy()
    
    # Convert categorical variables to dummy variables with error handling
    try:
        # Separate numeric and categorical columns
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
        
        # Process numeric columns
        numeric_df = feature_df[numeric_cols] if numeric_cols else pd.DataFrame()
        
        # Process categorical columns
        if categorical_cols:
            categorical_df = feature_df[categorical_cols]
            # Ensure categorical columns have values
            categorical_df = categorical_df.dropna()
            if not categorical_df.empty:
                dummies_df = pd.get_dummies(categorical_df, drop_first=False)
            else:
                print("Warning: No valid categorical data found.")
                dummies_df = pd.DataFrame()
        else:
            dummies_df = pd.DataFrame()
        
        # Combine numeric and dummy variables
        if not numeric_df.empty and not dummies_df.empty:
            feature_df = pd.concat([numeric_df, dummies_df], axis=1)
        elif not numeric_df.empty:
            feature_df = numeric_df
        elif not dummies_df.empty:
            feature_df = dummies_df
        else:
            print("Error: No valid features found after processing.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error processing features: {e}")
        print("Trying alternative approach...")
        # Fallback: use only numeric columns
        feature_df = feature_df.select_dtypes(include=[np.number])
        if feature_df.empty:
            print("Error: No numeric features found.")
            sys.exit(1)
    
    # Check if feature_df is empty
    if feature_df.empty:
        print("Error: No features available after processing.")
        sys.exit(1)
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values.astype(np.float64))
    y = df['target'].values.reshape(-1, 1)
    
    feature_names = feature_df.columns.tolist()
    
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i+1}. {name}")
    
    # Analyze the target distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nTarget distribution:")
    for val, count in zip(unique, counts):
        status = "healthy" if val == 1 else "unhealthy"
        print(f"{status}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Only create visualizations that help neural network learning
    desktop_dir = r'C:/Users/ayush/Desktop'
    os.makedirs(desktop_dir, exist_ok=True)
    
    # Class distribution - HELPS: Shows if we need SMOTE balancing
    plt.figure(figsize=(8, 6))
    status_labels = ['unhealthy', 'healthy']
    plt.pie(counts, labels=status_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Device Health Status Distribution (Helps decide SMOTE balancing)')
    plt.savefig(os.path.join(desktop_dir, 'telemetry_class_distribution.png'), bbox_inches='tight')
    plt.close()
    
    print(f"\nTarget: Device Health Status (0=unhealthy, 1=healthy)")
    print("Learning-focused visualization saved:")
    print("- telemetry_class_distribution.png (helps decide if SMOTE needed)")
    
    # Apply centralized preprocessing
    X, y, feature_names = preprocess_dataset(X, y, "Telemetry")
    
    return X, y, feature_names

# --- Feature Selection Function ---
def select_best_features(X, y, feature_names=None, min_accuracy=0.0):
    """
    For each feature, evaluate its individual accuracy using a threshold rule.
    Keep only features whose accuracy is above min_accuracy.
    Returns filtered X, selected feature indices, and a summary dict.
    """
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    selected_indices = []
    feature_scores = {}
    dropped_features = []
    for i in range(n_features):
        feature = X[:, i]
        best_acc = 0
        best_thresh = feature.mean()
        best_dir = 1
        # Try thresholds between min and max
        for thresh in np.linspace(feature.min(), feature.max(), 100):
            for direction in [1, -1]:
                if direction == 1:
                    pred = (feature > thresh).astype(int)
                else:
                    pred = (feature < thresh).astype(int)
                acc = np.mean(pred == y.ravel())
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
                    best_dir = direction
        feature_scores[feature_names[i]] = best_acc
        if best_acc >= min_accuracy:
            selected_indices.append(i)
        else:
            dropped_features.append((feature_names[i], best_acc))
    print("\n--- Feature Selection Report ---")
    print(f"Minimum accuracy threshold: {min_accuracy}")
    print(f"Selected features ({len(selected_indices)}/{n_features}):")
    for i in selected_indices:
        print(f"  {feature_names[i]}: accuracy={feature_scores[feature_names[i]]:.3f}")
    if dropped_features:
        print(f"Dropped features ({len(dropped_features)}):")
        for name, acc in dropped_features:
            print(f"  {name}: accuracy={acc:.3f}")
    print("-------------------------------\n")
    X_selected = X[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]
    return X_selected, selected_indices, selected_names, feature_scores

def detect_problem_type(y):
    """Detect if the problem is binary classification, multi-class classification, or regression."""
    unique_vals = np.unique(y)
    if len(unique_vals) == 2:
        return 'binary_classification'
    elif len(unique_vals) <= 10:  # Multi-class if 3-10 unique values
        return 'multi_class_classification'
    else:
        return 'regression'



# --- Layer One: Parallel neurons (no backprop) ---
def parallel_neuron_predict(X, weights, biases, n_classes=2):
    """
    X: (n_samples, n_features)
    weights: (n_neurons, n_features)
    biases: (n_neurons,)
    n_classes: number of classes for multi-class classification
    Returns: (n_samples,) final prediction by voting
    """
    if n_classes == 2:
        # Binary classification
        neuron_outputs = (np.dot(X, weights.T) + biases) > 0  # shape: (n_samples, n_neurons)
        votes = np.sum(neuron_outputs, axis=1)
        return (votes >= (weights.shape[0] / 2)).astype(int)
    else:
        # Multi-class classification
        neuron_outputs = np.dot(X, weights.T) + biases  # shape: (n_samples, n_neurons)
        # Use argmax for multi-class prediction
        predictions = np.argmax(neuron_outputs, axis=1)
        return predictions

# --- Layer Two: Evolutionary/random search for best weights/biases for all neurons ---
def evolutionary_search_parallel(X_train, y_train, X_val, y_val, n_neurons=200, n_iter=100, pop_size=30, early_stop_thresh=0.005, patience=10, n_classes=2):
    n_features = X_train.shape[1]
    best_acc = 0
    best_weights = None
    best_biases = None
    rng = np.random.default_rng(42)
    no_improve_count = 0
    last_best_acc = 0
    for iter_idx in range(n_iter):
        improved = False
        for _ in range(pop_size):
            weights = rng.normal(0, 1, (n_neurons, n_features))
            biases = rng.normal(0, 1, n_neurons)
            val_pred = parallel_neuron_predict(X_val, weights, biases, n_classes)
            val_acc = np.mean(val_pred.ravel() == y_val.ravel())
            if val_acc > best_acc + early_stop_thresh:
                best_acc = val_acc
                best_weights = weights.copy()
                best_biases = biases.copy()
                improved = True
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        if no_improve_count >= patience:
            print(f"Early stopping at iteration {iter_idx+1} due to no improvement > {early_stop_thresh} for {patience} iterations.")
            break
    return best_weights, best_biases, best_acc



# =========================
# Non-Backprop Ensemble Model
# =========================
if __name__ == "__main__":
    try:
        debug = True  # Set to False for full accuracy mode
        np.random.seed(42)
        print("="*50)
        print("NON-BACKPROP ENSEMBLE NEURAL NETWORK")
        print("="*50)
        
        # Choose dataset
        print("\nAvailable datasets:")
        print("1. California Housing (Regression)")
        print("2. Breast Cancer (Binary Classification)")
        print("3. Iris Dataset (Multi-Class Classification)")
        print("4. Wine Dataset (Multi-Class Classification)")
        print("5. Digits Dataset (Multi-Class Classification)")
        print("6. Custom CSV Dataset")
        print("7. Telemetry Dataset")
        
        print("\n💡 TIP: You can enter a file path directly instead of choosing a number!")
        print("Examples:")
        print("- Enter '7' then provide path, OR")
        print("- Enter path directly: C:/Users/ayush/Desktop/daikibo-telemetry-data.json")
        print("- Enter path directly: ./data/my_dataset.csv")
        print("\nSupported file types:")
        print("- .json files → Telemetry dataset")
        print("- .csv files → Custom dataset")
        
        choice = input("\nEnter your choice (1-7) or a file path: ").strip()
        
        # Check if input looks like a file path
        def is_file_path(input_str):
            # Remove quotes and check for file extensions or path indicators
            cleaned = input_str.strip('"').strip("'")
            return ('.json' in cleaned.lower() or 
                   '.csv' in cleaned.lower() or 
                   '\\' in cleaned or 
                   '/' in cleaned or
                   os.path.exists(cleaned))
        
        if is_file_path(choice):
            # User entered a file path directly
            file_path = choice.strip('"').strip("'")
            print(f"\nDetected file path: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"Error: File not found at {file_path}")
                print("Please check the path and try again.")
                sys.exit(1)
            
            # Determine file type and load accordingly
            if file_path.lower().endswith('.json'):
                print("Loading as telemetry JSON dataset...")
                X, y, feature_names = load_telemetry_dataset(file_path)
            elif file_path.lower().endswith('.csv'):
                print("Loading as custom CSV dataset...")
                X, y, feature_names = load_custom_csv_dataset(file_path)
            else:
                print("Unknown file type. Trying as CSV...")
                try:
                    X, y, feature_names = load_custom_csv_dataset(file_path)
                except:
                    print("Error: Could not load file. Please ensure it's a valid CSV or JSON file.")
                    sys.exit(1)
        
        elif choice == "1":
            # California Housing Dataset
            custom_path = input("Enter path to custom housing CSV (or press Enter for sklearn dataset): ").strip()
            if custom_path:
                # Clean and validate the path
                custom_path = custom_path.strip('"').strip("'")  # Remove quotes if present
                if not os.path.exists(custom_path):
                    print(f"Error: File not found at {custom_path}")
                    print("Please check the path and try again.")
                    sys.exit(1)
                X, y = load_housing_dataset(custom_path)
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            else:
                X, y = load_housing_dataset()
                feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        elif choice == "2":
            # Breast Cancer Dataset
            X, y, feature_names = load_cancer_dataset()
        
        elif choice == "3":
            # Iris Dataset (Multi-Class Classification)
            X, y, feature_names = load_iris_dataset()
        
        elif choice == "4":
            # Wine Dataset (Multi-Class Classification)
            X, y, feature_names = load_wine_dataset()
        
        elif choice == "5":
            # Digits Dataset (Multi-Class Classification)
            X, y, feature_names = load_digits_dataset()
        
        elif choice == "6":
            # Custom CSV Dataset
            custom_path = input("Enter path to your CSV file: ").strip()
            # Clean and validate the path
            custom_path = custom_path.strip('"').strip("'")  # Remove quotes if present
            if not os.path.exists(custom_path):
                print(f"Error: File not found at {custom_path}")
                print("Please check the path and try again.")
                sys.exit(1)
            X, y, feature_names = load_custom_csv_dataset(custom_path)
        
        elif choice == "7":
            # Telemetry Dataset
            json_path = input("Enter path to telemetry JSON file: ").strip()
            # Clean and validate the path
            json_path = json_path.strip('"').strip("'")  # Remove quotes if present
            if not os.path.exists(json_path):
                print(f"Error: File not found at {json_path}")
                print("Please check the path and try again.")
                sys.exit(1)
            X, y, feature_names = load_telemetry_dataset(json_path)
        

        
        else:
            print("Invalid choice. Using California Housing dataset.")
            X, y = load_housing_dataset()
            feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        # --- Detect problem type ---
        problem_type = detect_problem_type(y)
        print(f"\nDetected problem type: {problem_type}")
        
        if problem_type == 'binary_classification':
            print("\n[Layer One] Activating binary parallel neuron (no backprop) and evolutionary search...")
            # Feature selection: use top 10 features for neurons
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y.ravel())
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            X_top = X[:, top_indices]
            top_feature_names = [feature_names[i] for i in top_indices]
            print(f"Top 10 features for neurons: {top_feature_names}")
            # --- Visualize feature importances ---
            desktop_dir = r'C:/Users/ayush/Desktop'
            os.makedirs(desktop_dir, exist_ok=True)
            plt.figure(figsize=(8,6))
            plt.barh(np.array(top_feature_names)[::-1], importances[top_indices][::-1])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importances (Binary RandomForest)')
            plt.tight_layout()
            plt.savefig(os.path.join(desktop_dir, 'binary_feature_importances.png'), bbox_inches='tight')
            plt.close()
            # Split data
            from sklearn.model_selection import train_test_split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_top, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                print(f"Stratified split failed: {e}. Using regular split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_top, y, test_size=0.2, random_state=42
                )
            
            # Further split train into train/val for evolutionary search
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
            )
            n_neurons = 200
            # Layer Two: Evolutionary search for best weights/biases for all neurons
            best_weights, best_biases, best_val_acc = evolutionary_search_parallel(
                X_tr, y_tr, X_val, y_val, n_neurons=n_neurons, n_iter=300, pop_size=50, early_stop_thresh=0.005, patience=10, n_classes=2
            )
            print(f"Best validation accuracy from evolutionary search: {best_val_acc:.4f}")
            # Test prediction
            test_pred = parallel_neuron_predict(X_test, best_weights, best_biases, n_classes=2)
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            test_acc = accuracy_score(y_test, test_pred)
            print(f"Test set accuracy: {test_acc:.4f}")
            # Print detailed classification report
            print("\nClassification Report:")
            print(classification_report(y_test, test_pred))
            print("Confusion matrix:")
            print(confusion_matrix(y_test, test_pred))
            # Print summary of predictions
            print(f"\nPrediction Summary:")
            print(f"Total test samples: {len(y_test)}")
            for class_idx in range(2):
                print(f"Predicted class {class_idx}: {np.sum(test_pred == class_idx)}")
            # Create confusion matrix visualization
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, test_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Binary Classification Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(desktop_dir, 'binary_confusion_matrix.png'), bbox_inches='tight')
            plt.close()
            print(f"Binary confusion matrix saved to: binary_confusion_matrix.png")
            
        elif problem_type == 'multi_class_classification':
            print("\n[Multi-Class Layer] Activating evolutionary neural network for multi-class classification...")
            
            # Get number of classes
            n_classes = len(np.unique(y))
            print(f"Number of classes: {n_classes}")
            
            # Feature selection for multi-class
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import SelectKBest, f_classif
            
            # Use all features for small datasets, select top features for larger ones
            if X.shape[1] <= 20:
                X_top = X
                top_feature_names = feature_names
                print(f"Using all {X.shape[1]} features for multi-class classification")
            else:
                # Select top features for larger datasets
                k = min(20, X.shape[1])
                selector = SelectKBest(score_func=f_classif, k=k)
                X_top = selector.fit_transform(X, y.ravel())
                selected_indices = selector.get_support(indices=True)
                top_feature_names = [feature_names[i] for i in selected_indices]
                print(f"Selected top {k} features for multi-class classification: {top_feature_names}")
            
            # Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_top, y.ravel())
            importances = rf.feature_importances_
            
            # --- Enhanced visualizations for multi-class ---
            desktop_dir = r'C:/Users/ayush/Desktop'
            os.makedirs(desktop_dir, exist_ok=True)
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_feature_names)), importances)
            plt.yticks(range(len(top_feature_names)), top_feature_names)
            plt.xlabel('Feature Importance')
            plt.title('Multi-Class Feature Importances (RandomForest)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(desktop_dir, 'multiclass_feature_importances.png'), bbox_inches='tight')
            plt.close()
            
            # Class distribution visualization
            unique, counts = np.unique(y, return_counts=True)
            plt.figure(figsize=(8, 6))
            plt.pie(counts, labels=[f'Class {i}' for i in unique], autopct='%1.1f%%', startangle=90)
            plt.title('Multi-Class Dataset Distribution')
            plt.savefig(os.path.join(desktop_dir, 'multiclass_distribution.png'), bbox_inches='tight')
            plt.close()
            
            # Split data with enhanced error handling
            from sklearn.model_selection import train_test_split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_top, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                print(f"Stratified split failed: {e}. Using regular split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_top, y, test_size=0.2, random_state=42
                )
            
            # Further split for validation
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
            )
            
            # Prepare one-hot encoded targets for the neural network
            try:
                ohe = OneHotEncoder(sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(sparse=False)
            
            y_tr_oh = ohe.fit_transform(y_tr.reshape(-1, 1))
            y_val_oh = ohe.transform(y_val.reshape(-1, 1))
            y_test_oh = ohe.transform(y_test.reshape(-1, 1))
            
            # Enhanced neural network parameters for multi-class
            print(f"\n=== Multi-Class Neural Network Training ===")
            print(f"Number of classes: {n_classes}")
            print(f"Training samples: {X_tr.shape[0]}")
            print(f"Validation samples: {X_val.shape[0]}")
            print(f"Test samples: {X_test.shape[0]}")
            
            # Initialize and train the MultiClassEvoNet with fastest method
            model = MultiClassEvoNet(input_dim=X_tr.shape[1], num_classes=n_classes)
            
            print("\n=== Fast Multi-Class Training ===")
            print("Using mini-batch evolution training for speed...")
            
            # Fast training using mini-batch evolution
            n_samples = X_tr.shape[0]
            batch_size = min(64, n_samples)
            iterations = 1000
            
            print(f"Training with batch_size={batch_size}, iterations={iterations}")
            
            for iteration in range(iterations):
                # Sample a mini-batch
                batch_indices = np.random.choice(n_samples, batch_size, replace=False)
                X_batch = X_tr[batch_indices]
                y_batch = y_tr[batch_indices]
                y_batch_oh = y_tr_oh[batch_indices]
                
                batch_correct = 0
                batch_loss = 0
                
                for i in range(batch_size):
                    x = X_batch[i]
                    y_true = y_batch_oh[i]
                    y_label = y_batch[i]
                    y_pred, l1_errs, l2_errs, l3_outs = model.forward(x, y_true, train=True)
                    pred_label = np.argmax(y_pred)
                    if pred_label == y_label:
                        batch_correct += 1
                    loss = ce_loss_with_confidence(y_pred, y_true)
                    batch_loss += loss
                    
                    # Update V_m for good neurons
                    l1_bench_idx = np.argmin(l1_errs)
                    l1_bench = model.level1[l1_bench_idx]
                    if l1_bench.last_error is not None and l1_bench.last_error < model.tau1:
                        model.V_m.add(l1_bench.last_weights, l1_bench.last_bias)
                    
                    l2_bench_idx = np.argmin(l2_errs)
                    l2_bench = model.level2[l2_bench_idx]
                    if l2_bench.last_error is not None and l2_bench.last_error < model.tau2:
                        model.V_m.add(l2_bench.last_weights, l2_bench.last_bias)
                
                avg_batch_acc = batch_correct / batch_size
                avg_batch_loss = batch_loss / batch_size
                model.global_error = avg_batch_loss
                
                if iteration % 100 == 0:
                    print(f"Iteration {iteration}: Batch Accuracy: {avg_batch_acc*100:.2f}%, Loss: {avg_batch_loss:.4f}")
                    
                if iteration % 500 == 0:
                    print(f"[Thought] Iteration {iteration}: Global error is {model.global_error:.4f}, mutation strength now {model.get_mutation_strength():.4f}")
                    
                # Validation check
                if iteration % 200 == 0 and X_val is not None:
                    val_acc, val_loss = model.evaluate(X_val, y_val.ravel(), y_val_oh)
                    print(f"[Validation] Accuracy: {val_acc*100:.2f}%, Loss: {val_loss:.4f}")
            
            print("Fast training completed!")
            
            # Evaluate on test set
            test_acc, test_loss = model.evaluate(X_test, y_test.ravel(), y_test_oh)
            print(f"\n=== Multi-Class Classification Results ===")
            print(f"Test set accuracy: {test_acc:.4f}")
            print(f"Test set loss: {test_loss:.4f}")
            
            # Get predictions for detailed analysis
            predictions = []
            for i in range(X_test.shape[0]):
                x = X_test[i]
                y_true = y_test_oh[i]
                y_pred, _, _, _ = model.forward(x, y_true, train=False)
                predictions.append(np.argmax(y_pred))
            
            predictions = np.array(predictions)
            
            # Enhanced evaluation metrics
            from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
            
            precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average='weighted')
            
            print(f"Weighted precision: {precision:.4f}")
            print(f"Weighted recall: {recall:.4f}")
            print(f"Weighted F1-score: {f1:.4f}")
            
            # Detailed per-class metrics
            print(f"\nPer-class metrics:")
            for i in range(n_classes):
                class_precision = precision_recall_fscore_support(y_test, predictions, average=None)[0][i]
                class_recall = precision_recall_fscore_support(y_test, predictions, average=None)[1][i]
                class_f1 = precision_recall_fscore_support(y_test, predictions, average=None)[2][i]
                print(f"Class {i}: Precision={class_precision:.3f}, Recall={class_recall:.3f}, F1={class_f1:.3f}")
            
            # Enhanced classification report
            print(f"\nDetailed Classification Report:")
            print(classification_report(y_test, predictions, target_names=[f'Class_{i}' for i in range(n_classes)]))
            
            # Enhanced confusion matrix
            cm = confusion_matrix(y_test, predictions)
            print(f"\nConfusion Matrix:")
            print(cm)
            
            # Enhanced prediction summary
            print(f"\n=== Prediction Summary ===")
            print(f"Total test samples: {len(y_test)}")
            for class_idx in range(n_classes):
                predicted = np.sum(predictions == class_idx)
                actual = np.sum(y_test == class_idx)
                print(f"Class {class_idx}: Predicted={predicted}, Actual={actual}")
            
            # Enhanced visualizations
            # Confusion matrix heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=[f'Class_{i}' for i in range(n_classes)],
                       yticklabels=[f'Class_{i}' for i in range(n_classes)])
            plt.title(f'Multi-Class Confusion Matrix ({n_classes} classes)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(desktop_dir, 'multiclass_confusion_matrix.png'), bbox_inches='tight')
            plt.close()
            
            # Prediction vs Actual comparison
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.hist(y_test, bins=n_classes, alpha=0.7, label='Actual', color='blue')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Actual Class Distribution')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.hist(predictions, bins=n_classes, alpha=0.7, label='Predicted', color='red')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Predicted Class Distribution')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.bar(range(n_classes), [np.sum(y_test == i) for i in range(n_classes)], 
                   alpha=0.7, label='Actual', color='blue')
            plt.bar(range(n_classes), [np.sum(predictions == i) for i in range(n_classes)], 
                   alpha=0.7, label='Predicted', color='red')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Actual vs Predicted Comparison')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(desktop_dir, 'multiclass_prediction_analysis.png'), bbox_inches='tight')
            plt.close()
            
            print(f"\nMulti-class visualizations saved:")
            print(f"- multiclass_feature_importances.png")
            print(f"- multiclass_distribution.png")
            print(f"- multiclass_confusion_matrix.png")
            print(f"- multiclass_prediction_analysis.png")
            
        else:
            print("\n[Regression Layer] Activating evolutionary neural network for regression...")
            
            # Feature selection for regression
            from sklearn.feature_selection import SelectKBest, f_regression
            k = min(8, X.shape[1])
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y.ravel())
            selected_indices = selector.get_support(indices=True)
            selected_names = [feature_names[i] for i in selected_indices]
            print(f"Selected top {k} features for regression: {selected_names}")
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
            
            # Further split for validation
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.25, random_state=42
            )
            
            print(f"\n=== Regression Neural Network Training ===")
            print(f"Training samples: {X_tr.shape[0]}")
            print(f"Validation samples: {X_val.shape[0]}")
            print(f"Test samples: {X_test.shape[0]}")
            
            # Choose training method
            print("\nAvailable training methods:")
            print("1. Standard Epoch Training (Slow)")
            print("2. Mini-Batch Evolution Training (Fast)")
            print("3. Early Stopping Training (Efficient)")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            # Initialize regression neural network
            model = RegressionEvoNet(input_dim=X_tr.shape[1])
            
            if choice == "1":
                print("\n=== Standard Epoch Training ===")
                model.train(X_tr, y_tr.ravel(), epochs=EPOCHS, X_val=X_val, y_val=y_val.ravel())
            elif choice == "2":
                print("\n=== Mini-Batch Evolution Training ===")
                mini_batch_evolution_training(model, X_tr, y_tr.ravel(), batch_size=32, iterations=1000, X_val=X_val, y_val=y_val.ravel())
            elif choice == "3":
                print("\n=== Early Stopping Training ===")
                early_stopping_training(model, X_tr, y_tr.ravel(), patience=20, min_delta=0.001, X_val=X_val, y_val=y_val.ravel())
            else:
                print("Invalid choice. Using mini-batch training.")
                mini_batch_evolution_training(model, X_tr, y_tr.ravel(), batch_size=32, iterations=1000, X_val=X_val, y_val=y_val.ravel())
            
            # Evaluate on test set
            test_loss, test_predictions = model.evaluate(X_test, y_test.ravel())
            test_r2 = r2_score(y_test, test_predictions)
            
            print(f"\n=== Regression Neural Network Results ===")
            print(f"Test MSE Loss: {test_loss:.4f}")
            print(f"Test R² Score: {test_r2:.4f}")
            print(f"Test RMSE: {np.sqrt(test_loss):.4f}")
            
            # Check if target was scaled (for large values like volume)
            y_test_max = np.max(np.abs(y_test))
            if target_scaler_global is not None:
                print(f"📊 Note: Target values were scaled for numerical stability")
                print(f"   Original scale: mean={target_scaler_global.mean_[0]:.2e}, std={target_scaler_global.scale_[0]:.2e}")
                print(f"   Results shown are on scaled values. For original scale, multiply by {target_scaler_global.scale_[0]:.2e}")
            else:
                print(f"📊 Results shown on original scale (max value: {y_test_max:.2f})")
            
            # Compare with traditional models
            print(f"\n=== Comparison with Traditional Models ===")
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            
            # RandomForest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_tr, y_tr.ravel())
            rf_pred = rf.predict(X_test)
            rf_mse = mean_squared_error(y_test, rf_pred)
            rf_r2 = r2_score(y_test, rf_pred)
            print(f"RandomForest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")
            
            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_tr, y_tr.ravel())
            lr_pred = lr.predict(X_test)
            lr_mse = mean_squared_error(y_test, lr_pred)
            lr_r2 = r2_score(y_test, lr_pred)
            print(f"LinearRegression - MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}")
            
            # Create visualization
            desktop_dir = r'C:/Users/ayush/Desktop'
            os.makedirs(desktop_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 4))
            
            # Actual vs Predicted
            plt.subplot(1, 3, 1)
            plt.scatter(y_test, test_predictions, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Regression Neural Network\nActual vs Predicted')
            
            # Residuals
            plt.subplot(1, 3, 2)
            residuals = y_test.ravel() - test_predictions
            plt.scatter(test_predictions, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            
            # Model Comparison
            plt.subplot(1, 3, 3)
            models = ['EvoNet', 'RandomForest', 'LinearReg']
            r2_scores = [test_r2, rf_r2, lr_r2]
            plt.bar(models, r2_scores, color=['blue', 'green', 'orange'])
            plt.ylabel('R² Score')
            plt.title('Model Comparison (R² Scores)')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(desktop_dir, 'regression_neural_network_results.png'), bbox_inches='tight')
            plt.close()
            
            print(f"\nRegression neural network results saved to: regression_neural_network_results.png")
            print(f"\nFinal model output is from the evolutionary regression neural network.")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")




