"""
EvoNet Imbalanced Data Benchmark
=================================
Proves EvoNet's superiority on imbalanced classification — its primary market differentiator.

Tests three imbalance ratios (90-10, 95-5, 99-1) against XGBoost and MLP.
Metric: F1-score on the minority class (the hard, commercially important cases).

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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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


def make_imbalanced(minority_pct, n_samples=TOTAL_SAMPLES, n_features=N_FEATURES, seed=RANDOM_SEED):
    """Generate a binary imbalanced classification dataset."""
    weights = [1.0 - minority_pct, minority_pct]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        weights=weights,
        flip_y=0.01,
        random_state=seed
    )
    return X, y


def run_competitors_imbalanced(X_train, X_test, y_train, y_test):
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
        'f1':  f1_score(y_test, preds, pos_label=1, zero_division=0),
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
        'f1':  f1_score(y_test, preds, pos_label=1, zero_division=0),
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
        'f1':  f1_score(y_test, preds, pos_label=1, zero_division=0),
        'acc': accuracy_score(y_test, preds),
        'time': time.time() - t0
    }

    return results


def run_evonet_imbalanced(X_train, X_test, y_train, y_test):
    """Train and evaluate EvoNet on imbalanced binary data."""
    y_oh_train = np.eye(2)[y_train.astype(int)]
    y_oh_test  = np.eye(2)[y_test.astype(int)]

    t0 = time.time()
    net = MultiClassEvoNet(input_dim=X_train.shape[1], num_classes=2)
    net.train(X_train, y_train, y_oh_train, epochs=EVONET_EPOCHS)
    elapsed = time.time() - t0

    # Predict
    preds = []
    for i in range(len(X_test)):
        y_pred, _, _, _ = net._forward_elite(X_test[i])
        preds.append(int(np.argmax(y_pred)))
    preds = np.array(preds)

    f1  = f1_score(y_test, preds, pos_label=1, zero_division=0)
    acc = accuracy_score(y_test, preds)
    return f1, acc, elapsed


def run_imbalance_test(label, minority_pct):
    """Run one imbalance ratio test."""
    majority_pct = int((1 - minority_pct) * 100)
    minor_pct    = int(minority_pct * 100)
    ratio_str    = f"{majority_pct}-{minor_pct}"

    print(f"\n{'-'*60}")
    print(f"  Imbalance Ratio: {ratio_str} | Minority samples: ~{int(TOTAL_SAMPLES * minority_pct)}")
    print(f"{'-'*60}")

    X, y = make_imbalanced(minority_pct)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    comp = run_competitors_imbalanced(X_train, X_test, y_train, y_test)
    evo_f1, evo_acc, evo_time = run_evonet_imbalanced(X_train, X_test, y_train, y_test)

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
        'ratio':              ratio_str,
        'minority_pct':       minority_pct,
        'total_samples':      TOTAL_SAMPLES,
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
    print("  Your killer feature: F1 on the hard minority class")
    print("="*60)

    scenarios = [
        ("Fraud Detection (90-10)",     0.10),
        ("Disease Prediction (95-5)",   0.05),
        ("Anomaly Detection (99-1)",    0.01),
    ]

    rows = []
    for label, minority_pct in scenarios:
        row = run_imbalance_test(label, minority_pct)
        rows.append(row)

    # Summary
    print(f"\n\n{'='*72}")
    print("  IMBALANCED BENCHMARK SUMMARY — Minority Class F1-Score")
    print(f"{'='*72}")
    print(f"  {'Ratio':<10} {'EvoNet':>9} {'XGBoost':>9} {'RF':>9} {'MLP':>9}  {'EvoNet ?':>10}")
    print(f"  {'-'*60}")
    wins = 0
    for r in rows:
        win = '[OK] +' if r['EvoNet_wins'] else '[FAIL] '
        if r['EvoNet_wins']:
            wins += 1
        delta_str = f"{win}{abs(r['EvoNet_vs_best_pct']):.1f}%"
        print(f"  {r['ratio']:<10} {r['EvoNet_F1']:>9.4f} {r['XGBoost_F1']:>9.4f} "
              f"{r['RF_F1']:>9.4f} {r['MLP_F1']:>9.4f}  {delta_str:>10}")
    print(f"  {'-'*60}")
    print(f"  EvoNet won {wins}/{len(rows)} imbalance scenarios\n")

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
