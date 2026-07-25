"""
EvoNet Continual Learning & Catastrophic Forgetting Experiment
==============================================================
Demonstrates EvoNet's ability to resist catastrophic forgetting
through its V_m (Significant Mutation Vector) memory module.

Experiment:
    Phase 1: Train on Task A (digits 0-4)
    Phase 2: Train on Task B (digits 5-9) — this induces forgetting in standard nets
    Phase 3: Evaluate on Task A — EvoNet's elite + V_m should retain some Task A knowledge

Baseline comparison:
    - EvoNet (Full, with V_m): Should retain Task A via V_m memory
    - EvoNet (No V_m): Should forget Task A (catastrophic forgetting)

Usage:
    python research_continual_learning.py
    python research_continual_learning.py --fast
"""

import sys
import os
import copy
import argparse
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evonet.core.network import MultiClassEvoNet


def prepare_task(digits_data, class_filter, scaler=None, encoder=None):
    """
    Prepare a binary/multi-class task from a subset of digit classes.
    Labels are remapped to [0, N) for the subset.
    """
    X_all, y_all = digits_data.data, digits_data.target

    # Filter to the requested digit classes
    mask = np.isin(y_all, class_filter)
    X = X_all[mask]
    y = y_all[mask]

    # Remap labels to [0, len(class_filter))
    label_map = {cls: idx for idx, cls in enumerate(class_filter)}
    y_remapped = np.array([label_map[label] for label in y])

    # Fit or transform scaler
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # One-hot encode
    num_classes = len(class_filter)
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, categories=[range(num_classes)])
        y_oh = encoder.fit_transform(y_remapped.reshape(-1, 1))
    else:
        y_oh = encoder.transform(y_remapped.reshape(-1, 1))

    return X, y_remapped, y_oh, scaler, encoder, num_classes


def evaluate_on_task(net, X, y, y_oh, task_name):
    """Evaluate a network on a specific task using elite weights."""
    acc, loss = net.evaluate(X, y, y_oh)
    print(f"  [{task_name}] Accuracy: {acc*100:.2f}%  Loss: {loss:.4f}")
    return acc


def run_experiment(enable_vm, epochs_per_task, label):
    """
    Run the full continual learning experiment for one model configuration.
    """
    print(f"\n{'='*60}")
    print(f"  Model: {label}")
    print(f"{'='*60}")

    data = load_digits()

    # Task A: Digits 0-4 (5-class classification)
    task_a_classes = [0, 1, 2, 3, 4]
    # Task B: Digits 5-9 (5-class classification)
    task_b_classes = [5, 6, 7, 8, 9]

    # Prepare Task A
    Xa, ya, ya_oh, scaler_a, enc_a, nc_a = prepare_task(data, task_a_classes)
    Xa_train, Xa_test, ya_train, ya_test, ya_oh_train, ya_oh_test = train_test_split(
        Xa, ya, ya_oh, test_size=0.2, random_state=42
    )
    print(f"Task A (digits 0-4): {Xa_train.shape[0]} train, {Xa_test.shape[0]} test, {nc_a} classes")

    # Prepare Task B (same input_dim as Task A — same scaler)
    Xb, yb, yb_oh, scaler_b, enc_b, nc_b = prepare_task(data, task_b_classes, scaler=scaler_a)
    Xb_train, Xb_test, yb_train, yb_test, yb_oh_train, yb_oh_test = train_test_split(
        Xb, yb, yb_oh, test_size=0.2, random_state=42
    )
    print(f"Task B (digits 5-9): {Xb_train.shape[0]} train, {Xb_test.shape[0]} test, {nc_b} classes")

    # Create model — both tasks have same input_dim (64) and same number of classes (5)
    assert Xa.shape[1] == Xb.shape[1], "Input dims must match for continual learning"
    assert nc_a == nc_b, "Number of classes must match for this experiment"

    net = MultiClassEvoNet(input_dim=Xa.shape[1], num_classes=nc_a)
    if not enable_vm:
        # Disable V_m memory to simulate a baseline with no memory
        # Rebuild the deque with maxlen=0 (nothing can be stored)
        from collections import deque as _deque
        net.V_m._deque = _deque(maxlen=0)
        net.V_m.maxlen = 0

    # ------------------------------------------------------------------
    # Phase 1: Train on Task A
    # ------------------------------------------------------------------
    print(f"\n[Phase 1] Training on Task A ({epochs_per_task} epochs)...")
    net.train(Xa_train, ya_train, ya_oh_train, epochs=epochs_per_task)

    acc_a_after_a = evaluate_on_task(net, Xa_test, ya_test, ya_oh_test, "Task A after Phase 1")

    # Save a snapshot of V_m size to show memory accumulation
    vm_size_after_a = len(net.V_m.get())
    print(f"  V_m memory entries: {vm_size_after_a}")

    # ------------------------------------------------------------------
    # Phase 2: Train on Task B (catastrophic forgetting inducer)
    # ------------------------------------------------------------------
    print(f"\n[Phase 2] Training on Task B ({epochs_per_task} epochs — forgetting pressure)...")
    net.train(Xb_train, yb_train, yb_oh_train, epochs=epochs_per_task)

    # Evaluate on Task B (should be good — just trained)
    acc_b_after_b = evaluate_on_task(net, Xb_test, yb_test, yb_oh_test, "Task B after Phase 2")

    # Evaluate on Task A (this reveals catastrophic forgetting)
    acc_a_after_b = evaluate_on_task(net, Xa_test, ya_test, ya_oh_test, "Task A after Phase 2 (FORGETTING TEST)")

    vm_size_after_b = len(net.V_m.get())
    print(f"  V_m memory entries: {vm_size_after_b}")

    # Compute forgetting metric
    forgetting = acc_a_after_a - acc_a_after_b
    retention = acc_a_after_b / max(acc_a_after_a, 1e-9)

    print(f"\n  --- Results Summary for {label} ---")
    print(f"  Task A (after training A):       {acc_a_after_a*100:.2f}%")
    print(f"  Task B (after training B):       {acc_b_after_b*100:.2f}%")
    print(f"  Task A (after training B):       {acc_a_after_b*100:.2f}%  [Forgetting test]")
    print(f"  Forgetting:                      {forgetting*100:.2f}% points")
    print(f"  Retention:                       {retention*100:.2f}%")

    return {
        'task_a_after_a': acc_a_after_a,
        'task_b_after_b': acc_b_after_b,
        'task_a_after_b': acc_a_after_b,
        'forgetting': forgetting,
        'retention': retention,
        'vm_size_after_a': vm_size_after_a,
        'vm_size_after_b': vm_size_after_b,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoNet Continual Learning Experiment")
    parser.add_argument('--fast', action='store_true', help='Fast mode: fewer epochs')
    args = parser.parse_args()

    epochs_per_task = 5 if args.fast else 20

    print("EvoNet Continual Learning & Catastrophic Forgetting Experiment")
    print(f"Mode: {'FAST' if args.fast else 'FULL'} | Epochs per task: {epochs_per_task}")
    print()
    print("Hypothesis: EvoNet with V_m memory retains Task A knowledge after Task B training.")
    print("The V_m stores successful weight directions from Task A, guiding Task B evolution")
    print("away from parameter regions that destroy Task A representations.")

    # Run with V_m enabled
    results_vm = run_experiment(
        enable_vm=True,
        epochs_per_task=epochs_per_task,
        label="EvoNet (Full, with V_m memory)"
    )

    # Run without V_m (control / catastrophic forgetting baseline)
    results_no_vm = run_experiment(
        enable_vm=False,
        epochs_per_task=epochs_per_task,
        label="EvoNet (Ablation, no V_m memory)"
    )

    # Final comparison
    print("\n\n" + "="*60)
    print("FINAL COMPARISON: V_m vs No-V_m")
    print("="*60)
    print(f"\n{'Metric':<35} {'With V_m':>12} {'No V_m':>12} {'Delta':>12}")
    print("-"*70)

    metrics = [
        ("Task A accuracy (after train A)", 'task_a_after_a'),
        ("Task B accuracy (after train B)", 'task_b_after_b'),
        ("Task A accuracy (after train B)", 'task_a_after_b'),
        ("Forgetting (lower is better)", 'forgetting'),
        ("Retention % (higher is better)", 'retention'),
    ]

    for label, key in metrics:
        vm_val = results_vm[key]
        no_vm_val = results_no_vm[key]
        delta = vm_val - no_vm_val
        print(f"  {label:<33} {vm_val*100:>10.2f}% {no_vm_val*100:>10.2f}% {delta*100:>+11.2f}%")

    print()
    if results_vm['forgetting'] < results_no_vm['forgetting']:
        print("RESULT: V_m memory REDUCED catastrophic forgetting.")
        print(f"EvoNet (with V_m) retained {results_vm['retention']*100:.1f}% of Task A knowledge,")
        print(f"vs {results_no_vm['retention']*100:.1f}% without V_m memory.")
        print("This supports the hypothesis that V_m acts as a form of implicit task memory.")
    else:
        print("RESULT: V_m memory did NOT reduce forgetting in this run.")
        print("Consider increasing epochs_per_task or V_m history size.")

    print("\nExperiment Complete.")
