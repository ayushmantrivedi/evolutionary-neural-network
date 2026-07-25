"""
EvoNet Research Benchmark Suite
================================
Compares EvoNet (full) vs EvoNet (no V_m) vs Random Forest vs PyTorch MLP
on standard ML datasets from scikit-learn.

Usage:
    python research_benchmark_suite.py          # Full run (30 epochs)
    python research_benchmark_suite.py --fast   # Fast run (10 epochs)
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Ensure evonet is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evonet.core.network import MultiClassEvoNet
from evonet.core.gpu_backend import is_gpu_available, get_device

# Optional PyTorch baseline
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. PyTorch baseline will be skipped.")


class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PyTorchMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def run_pytorch_baseline(X_train, y_train, X_test, y_test, num_classes, epochs, device_str):
    if not HAS_TORCH:
        return 0.0, 0.0

    device = torch.device('cpu')
    model = PyTorchMLP(X_train.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs.data, 1)
        correct = (predicted == y_test_t).sum().item()
        acc = correct / y_test_t.size(0)

    return acc, end_time - start_time


def run_rf_baseline(X_train, y_train, X_test, y_test):
    start_time = time.time()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    end_time = time.time()
    acc = clf.score(X_test, y_test)
    return acc, end_time - start_time


def run_evonet(X_train, y_train, y_oh_train, X_test, y_test, y_oh_test, num_classes, epochs, enable_vm=True, label="EvoNet"):
    start_time = time.time()
    net = MultiClassEvoNet(input_dim=X_train.shape[1], num_classes=num_classes)

    # Ablation: disable V_m influence by making its deque unable to store anything
    if not enable_vm:
        from collections import deque as _deque
        net.V_m._deque = _deque(maxlen=0)
        net.V_m.maxlen = 0

    net.train(X_train, y_train, y_oh_train, epochs=epochs)
    end_time = time.time()

    acc, loss = net.evaluate(X_test, y_test, y_oh_test)
    print(f"  [{label}] Test Accuracy: {acc*100:.2f}%, Time: {end_time - start_time:.1f}s")
    return acc, end_time - start_time


def benchmark_dataset(name, data_func, epochs=30, fast=False):
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name}")
    print(f"{'='*60}")

    data = data_func()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_oh = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test, y_oh_train, y_oh_test = train_test_split(
        X, y, y_oh, test_size=0.2, random_state=42
    )

    num_classes = y_oh.shape[1]
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]} | Classes: {num_classes}")
    print()

    results = {}

    # 1. EvoNet (Full — with V_m memory)
    print("Running EvoNet (Full, with V_m memory)...")
    acc, t = run_evonet(X_train, y_train, y_oh_train, X_test, y_test, y_oh_test,
                        num_classes, epochs, enable_vm=True, label="EvoNet Full")
    results['EvoNet (Full+V_m)'] = {'Accuracy': round(acc, 4), 'Time (s)': round(t, 2)}

    # 2. EvoNet (Ablation — no V_m memory)
    print("Running EvoNet (Ablation, no V_m memory)...")
    acc, t = run_evonet(X_train, y_train, y_oh_train, X_test, y_test, y_oh_test,
                        num_classes, epochs, enable_vm=False, label="EvoNet NoVm")
    results['EvoNet (No V_m)'] = {'Accuracy': round(acc, 4), 'Time (s)': round(t, 2)}

    # 3. Random Forest (strong gradient-free baseline)
    print("Running Random Forest baseline...")
    acc, t = run_rf_baseline(X_train, y_train, X_test, y_test)
    print(f"  [RF] Test Accuracy: {acc*100:.2f}%, Time: {t:.3f}s")
    results['Random Forest'] = {'Accuracy': round(acc, 4), 'Time (s)': round(t, 4)}

    # 4. PyTorch MLP (gradient-based baseline)
    if HAS_TORCH:
        pt_epochs = epochs * 10  # Give gradient methods many more steps to be fair
        print(f"Running PyTorch MLP ({pt_epochs} epochs)...")
        acc, t = run_pytorch_baseline(X_train, y_train, X_test, y_test, num_classes, pt_epochs, get_device())
        print(f"  [PyTorch] Test Accuracy: {acc*100:.2f}%, Time: {t:.3f}s")
        results[f'PyTorch MLP'] = {'Accuracy': round(acc, 4), 'Time (s)': round(t, 4)}

    df = pd.DataFrame(results).T
    print(f"\n  Results for {name}:")
    print(df.to_markdown())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoNet Research Benchmark Suite")
    parser.add_argument('--fast', action='store_true', help='Fast mode: fewer epochs')
    args = parser.parse_args()

    epochs = 10 if args.fast else 30

    print("EvoNet Research Benchmark Suite")
    print(f"Mode: {'FAST' if args.fast else 'FULL'} | Epochs: {epochs}")
    print(f"GPU Available: {is_gpu_available()} ({get_device()})")

    all_results = {}
    all_results['Iris'] = benchmark_dataset("Iris (150 samples, 4 feat, 3 cls)", load_iris, epochs=epochs, fast=args.fast)
    all_results['Breast Cancer'] = benchmark_dataset("Breast Cancer (569 samples, 30 feat, 2 cls)", load_breast_cancer, epochs=epochs, fast=args.fast)

    if not args.fast:
        all_results['Digits'] = benchmark_dataset("Digits (1797 samples, 64 feat, 10 cls)", load_digits, epochs=epochs, fast=args.fast)

    print("\n\nFINAL SUMMARY ACROSS ALL DATASETS")
    print("="*60)
    for ds_name, df in all_results.items():
        print(f"\n{ds_name}:")
        print(df[['Accuracy']].to_markdown())

    print("\nBenchmarks Complete.")
