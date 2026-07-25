"""
EvoNet Standard Classification Benchmarks
==========================================
Compares EvoNet against XGBoost, Random Forest, and MLP on 4 standard datasets.

Usage:
    python evaluate_benchmarks.py

Output:
    - Console: formatted comparison table
    - benchmark_results.csv: machine-readable results
"""

import sys
import os
import time
import csv
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evonet.core.network import MultiClassEvoNet
from evonet.core.gpu_backend import is_gpu_available, get_device

# ------------------------------------------------------------------------------
EVONET_EPOCHS = 30
RANDOM_SEED   = 42
TEST_SIZE     = 0.20
RESULTS_FILE  = "benchmark_results.csv"
# ------------------------------------------------------------------------------

def run_competitors(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost, Random Forest, and MLP."""
    results = {}

    # XGBoost
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_SEED, verbosity=0
    )
    clf.fit(X_train, y_train)
    results['XGBoost'] = {
        'acc': accuracy_score(y_test, clf.predict(X_test)),
        'time': time.time() - t0
    }

    # Random Forest
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    results['RandomForest'] = {
        'acc': accuracy_score(y_test, clf.predict(X_test)),
        'time': time.time() - t0
    }

    # MLP (sklearn)
    t0 = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500,
        early_stopping=True, random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)
    results['MLP'] = {
        'acc': accuracy_score(y_test, clf.predict(X_test)),
        'time': time.time() - t0
    }

    return results


def run_evonet(X_train, X_test, y_train, y_test, y_oh_train, y_oh_test, input_dim, num_classes):
    """Train and evaluate EvoNet."""
    t0 = time.time()
    net = MultiClassEvoNet(input_dim=input_dim, num_classes=num_classes)
    net.train(X_train, y_train, y_oh_train, epochs=EVONET_EPOCHS)
    acc, _ = net.evaluate(X_test, y_test, y_oh_test)
    elapsed = time.time() - t0
    return acc, elapsed


def run_benchmark(name, loader, epochs=EVONET_EPOCHS):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {name}")
    print(f"{'='*60}")

    data = loader()
    X, y = data.data, data.target
    num_classes = len(np.unique(y))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_oh = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test, y_oh_train, y_oh_test = train_test_split(
        X, y, y_oh, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Classes: {num_classes}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Competitors
    comp_results = run_competitors(X_train, X_test, y_train, y_test)

    # EvoNet
    evonet_acc, evonet_time = run_evonet(
        X_train, X_test, y_train, y_test, y_oh_train, y_oh_test,
        X.shape[1], num_classes
    )

    # Determine winner
    all_accs = {m: r['acc'] for m, r in comp_results.items()}
    all_accs['EvoNet'] = evonet_acc
    winner = max(all_accs, key=all_accs.get)

    row = {
        'dataset':        name,
        'samples':        len(X),
        'features':       X.shape[1],
        'classes':        num_classes,
        'EvoNet_acc':     round(evonet_acc * 100, 2),
        'EvoNet_time_s':  round(evonet_time, 1),
        'XGBoost_acc':    round(comp_results['XGBoost']['acc'] * 100, 2),
        'XGBoost_time_s': round(comp_results['XGBoost']['time'], 2),
        'RF_acc':         round(comp_results['RandomForest']['acc'] * 100, 2),
        'RF_time_s':      round(comp_results['RandomForest']['time'], 2),
        'MLP_acc':        round(comp_results['MLP']['acc'] * 100, 2),
        'MLP_time_s':     round(comp_results['MLP']['time'], 2),
        'winner':         winner
    }

    # Pretty print
    print(f"\n  {'Model':<15} {'Accuracy':>10}  {'Time':>8}")
    print(f"  {'-'*36}")
    marker = ' <- WINNER' if winner == 'EvoNet' else ''
    print(f"  {'EvoNet':<15} {evonet_acc*100:>9.2f}%  {evonet_time:>6.1f}s{marker}")
    for model, r in comp_results.items():
        marker = ' <- WINNER' if winner == model else ''
        print(f"  {model:<15} {r['acc']*100:>9.2f}%  {r['time']:>6.2f}s{marker}")

    return row


def main():
    print("\n" + "="*60)
    print("  EVONET STANDARD BENCHMARKS")
    print(f"  Device: {'GPU OK' if is_gpu_available() else 'CPU'}")
    print(f"  EvoNet epochs: {EVONET_EPOCHS} | Seed: {RANDOM_SEED}")
    print("="*60)

    benchmarks = [
        ("Iris",          load_iris),
        ("Wine",          load_wine),
        ("Breast Cancer", load_breast_cancer),
        ("Digits",        load_digits),
    ]

    rows = []
    for name, loader in benchmarks:
        row = run_benchmark(name, loader)
        rows.append(row)

    # Summary table
    print(f"\n\n{'='*72}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Dataset':<16} {'EvoNet':>8} {'XGBoost':>8} {'RF':>8} {'MLP':>8}  {'Winner':<12}")
    print(f"  {'-'*68}")
    evonet_wins = 0
    for r in rows:
        win_marker = '[WIN] EvoNet' if r['winner'] == 'EvoNet' else r['winner']
        if r['winner'] == 'EvoNet':
            evonet_wins += 1
        print(f"  {r['dataset']:<16} {r['EvoNet_acc']:>7.2f}% {r['XGBoost_acc']:>7.2f}% "
              f"{r['RF_acc']:>7.2f}% {r['MLP_acc']:>7.2f}%  {win_marker}")
    print(f"  {'-'*68}")
    print(f"  EvoNet won {evonet_wins}/{len(rows)} datasets\n")

    # Save CSV
    fieldnames = list(rows[0].keys())
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
