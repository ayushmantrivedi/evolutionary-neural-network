"""
EvoNet Imbalanced Data Benchmark
=================================
Proves EvoNet's behavior on a real imbalanced classification dataset.

Uses the real Bank Marketing dataset from OpenML and measures F1 on the
minority class.

Usage:
    python benchmark_imbalanced.py

Output:
    - Console: formatted comparison table
    - imbalanced_results.csv: machine-readable results
"""

import sys
import os
import time
import csv
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evonet.core.network import MultiClassEvoNet

# ------------------------------------------------------------------------------
TOTAL_SAMPLES = 5000
N_FEATURES    = 20
EVONET_EPOCHS = 20
RANDOM_SEED   = 42
RESULTS_FILE  = "imbalanced_results.csv"
# ------------------------------------------------------------------------------


def encode_categorical_columns(frame):
    """Encode object/category columns without pandas dtype warnings."""
    encoded = frame.copy()
    for col in encoded.columns:
        if encoded[col].dtype == object or str(encoded[col].dtype) == 'category':
            encoded[col] = encoded[col].astype(str).astype('category').cat.codes.astype(np.int32)
    return encoded


def load_real_imbalanced_dataset():
    """Load a real binary imbalanced dataset."""
    data = fetch_openml(data_id=1461, as_frame=True, parser='auto')
    X = data.data.copy()
    y = data.target.copy()
    raw_labels = list(pd.unique(y))
    X = encode_categorical_columns(X)
    X = X.fillna(X.median(numeric_only=True))

    if y.dtype == object or str(y.dtype) == 'category':
        y = y.astype(str).astype('category').cat.codes
    else:
        y = y.astype(int)
        y = y - y.min()

    y = np.array(y, dtype=int)
    class_counts = np.bincount(y)
    minority_label = int(np.argmin(class_counts))
    majority_label = int(np.argmax(class_counts))
    target_names = [str(label) for label in raw_labels]
    return X.values.astype(np.float32), y, minority_label, majority_label, target_names


def make_balanced_training_set(X_train, y_train, random_seed):
    """Oversample the minority class to match the majority class."""
    rng = np.random.default_rng(random_seed)
    class_counts = np.bincount(y_train)
    majority_label = int(np.argmax(class_counts))
    minority_label = int(np.argmin(class_counts))

    majority_idx = np.where(y_train == majority_label)[0]
    minority_idx = np.where(y_train == minority_label)[0]

    if len(minority_idx) == 0 or len(majority_idx) == 0:
        return X_train, y_train, np.eye(2)[y_train.astype(int)]

    extra_idx = rng.choice(
        minority_idx,
        size=max(len(majority_idx) - len(minority_idx), 0),
        replace=True,
    )
    balanced_idx = np.concatenate([majority_idx, minority_idx, extra_idx])
    rng.shuffle(balanced_idx)

    X_balanced = X_train[balanced_idx]
    y_balanced = y_train[balanced_idx]
    y_balanced_oh = np.eye(2)[y_balanced.astype(int)]
    return X_balanced, y_balanced, y_balanced_oh


def cap_training_slice(X_train, y_train, cap, random_seed):
    """Keep a smaller, class-aware training slice for faster benchmark runs."""
    if len(X_train) <= cap:
        return X_train, y_train

    rng = np.random.default_rng(random_seed)
    class_indices = {label: np.where(y_train == label)[0] for label in np.unique(y_train)}
    minority_label = min(class_indices, key=lambda label: len(class_indices[label]))
    majority_label = max(class_indices, key=lambda label: len(class_indices[label]))

    minority_idx = class_indices[minority_label]
    majority_idx = class_indices[majority_label]

    minority_keep = min(len(minority_idx), cap // 2)
    majority_keep = min(len(majority_idx), cap - minority_keep)

    chosen_minority = rng.choice(minority_idx, size=minority_keep, replace=False) if minority_keep else np.array([], dtype=int)
    chosen_majority = rng.choice(majority_idx, size=majority_keep, replace=False) if majority_keep else np.array([], dtype=int)

    chosen = np.concatenate([chosen_minority, chosen_majority])
    rng.shuffle(chosen)
    return X_train[chosen], y_train[chosen]


def run_competitors_imbalanced(X_train, X_test, y_train, y_test, minority_label):
    """Run XGBoost, RF, and MLP with imbalance-aware settings."""
    results = {}

    # Compute class weight for imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos = n_neg / max(n_pos, 1)

    # XGBoost (with scale_pos_weight)
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric='logloss',
        random_state=RANDOM_SEED, verbosity=0
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results['XGBoost'] = {
        'f1':  f1_score(y_test, preds, pos_label=minority_label, zero_division=0),
        'acc': accuracy_score(y_test, preds),
        'time': time.time() - t0
    }

    # Random Forest (with class_weight='balanced')
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced',
        random_state=RANDOM_SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results['RandomForest'] = {
        'f1':  f1_score(y_test, preds, pos_label=minority_label, zero_division=0),
        'acc': accuracy_score(y_test, preds),
        'time': time.time() - t0
    }

    # MLP (sklearn)
    t0 = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500,
        early_stopping=True, random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results['MLP'] = {
        'f1':  f1_score(y_test, preds, pos_label=minority_label, zero_division=0),
        'acc': accuracy_score(y_test, preds),
        'time': time.time() - t0
    }

    return results


def run_evonet_imbalanced(X_train, X_test, y_train, y_test, minority_label):
    """Train and evaluate EvoNet on the real imbalanced binary dataset."""
    X_fit, _, y_fit, _ = train_test_split(
        X_train, y_train, test_size=0.20, random_state=RANDOM_SEED, stratify=y_train
    )
    y_oh_fit = np.eye(2)[y_fit.astype(int)]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_fit, y_fit, test_size=0.20, random_state=RANDOM_SEED, stratify=y_fit
    )
    X_fit, y_fit = cap_training_slice(X_fit, y_fit, cap=6000, random_seed=RANDOM_SEED)
    X_fit, y_fit, y_oh_fit = make_balanced_training_set(X_fit, y_fit, RANDOM_SEED)

    t0 = time.time()
    net = MultiClassEvoNet(input_dim=X_train.shape[1], num_classes=2)

    net.train(X_fit, y_fit, y_oh_fit, epochs=EVONET_EPOCHS)
    elapsed = time.time() - t0

    val_probs = np.array([net._forward_elite(x)[0][minority_label] for x in X_val])
    best_threshold = 0.5
    best_val_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        val_preds = np.where(val_probs >= threshold, minority_label, 1 - minority_label)
        val_f1 = f1_score(y_val, val_preds, pos_label=minority_label, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = float(threshold)

    # Predict
    test_probs = np.array([net._forward_elite(x)[0][minority_label] for x in X_test])
    preds = np.where(test_probs >= best_threshold, minority_label, 1 - minority_label)

    f1  = f1_score(y_test, preds, pos_label=minority_label, zero_division=0)
    acc = accuracy_score(y_test, preds)
    print(f"  Calibrated EvoNet threshold: {best_threshold:.2f} | Val F1: {best_val_f1:.4f}")
    return f1, acc, elapsed


def run_real_imbalance_test():
    """Run the real-world imbalanced benchmark."""
    X, y, minority_label, majority_label, target_names = load_real_imbalanced_dataset()
    minority_name = target_names[minority_label]
    majority_name = target_names[majority_label]

    print(f"\n{'-'*60}")
    print("  Real Dataset: Bank Marketing (OpenML 1461)")
    print(f"  Minority class: {minority_name} ({minority_label})")
    print(f"  Majority class: {majority_name} ({majority_label})")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'-'*60}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    comp = run_competitors_imbalanced(X_train, X_test, y_train, y_test, minority_label)
    evo_f1, evo_acc, evo_time = run_evonet_imbalanced(X_train, X_test, y_train, y_test, minority_label)

    # Best competitor F1
    best_comp_f1   = max(r['f1'] for r in comp.values())
    best_comp_name = max(comp, key=lambda m: comp[m]['f1'])
    evonet_wins    = evo_f1 >= best_comp_f1
    delta          = (evo_f1 - best_comp_f1) * 100

    print(f"\n  {'Model':<15} {'F1 Score':>10}  {'Accuracy':>10}  {'Time':>8}")
    print(f"  {'-'*50}")
    marker = ' <- BEST' if evonet_wins else ''
    print(f"  {'EvoNet':<15} {evo_f1:>10.4f}  {evo_acc*100:>9.2f}%  {evo_time:>6.1f}s{marker}")
    for model, r in comp.items():
        marker = ' <- BEST' if model == best_comp_name and not evonet_wins else ''
        print(f"  {model:<15} {r['f1']:>10.4f}  {r['acc']*100:>9.2f}%  {r['time']:>6.2f}s{marker}")

    sign = '+' if delta >= 0 else ''
    print(f"\n  EvoNet vs best competitor: {sign}{delta:.1f}% F1")

    return {
        'dataset':            'Bank Marketing',
        'minority_class':     minority_name,
        'majority_class':     majority_name,
        'total_samples':      len(X),
        'EvoNet_F1':          round(evo_f1, 4),
        'EvoNet_Acc':         round(evo_acc, 4),
        'XGBoost_F1':         round(comp['XGBoost']['f1'], 4),
        'XGBoost_Acc':        round(comp['XGBoost']['acc'], 4),
        'RF_F1':              round(comp['RandomForest']['f1'], 4),
        'RF_Acc':             round(comp['RandomForest']['acc'], 4),
        'MLP_F1':             round(comp['MLP']['f1'], 4),
        'MLP_Acc':            round(comp['MLP']['acc'], 4),
        'EvoNet_vs_best_pct': round(delta, 2),
        'EvoNet_wins':        evonet_wins
    }


def main():
    print("\n" + "="*60)
    print("  EVONET IMBALANCED DATA BENCHMARK")
    print("  Real-world validation on Bank Marketing dataset")
    print("="*60)

    rows = [run_real_imbalance_test()]

    # Summary
    print(f"\n\n{'='*72}")
    print("  REAL IMBALANCED BENCHMARK SUMMARY — Minority Class F1-Score")
    print(f"{'='*72}")
    print(f"  {'Dataset':<16} {'EvoNet':>9} {'XGBoost':>9} {'RF':>9} {'MLP':>9}  {'EvoNet ?':>10}")
    print(f"  {'-'*60}")
    wins = 0
    for r in rows:
        win = '[OK] +' if r['EvoNet_wins'] else '[FAIL] '
        if r['EvoNet_wins']:
            wins += 1
        delta_str = f"{win}{abs(r['EvoNet_vs_best_pct']):.1f}%"
        print(f"  {r['dataset']:<16} {r['EvoNet_F1']:>9.4f} {r['XGBoost_F1']:>9.4f} "
              f"{r['RF_F1']:>9.4f} {r['MLP_F1']:>9.4f}  {delta_str:>10}")
    print(f"  {'-'*60}")
    print(f"  EvoNet won {wins}/{len(rows)} real imbalanced scenarios\n")

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
