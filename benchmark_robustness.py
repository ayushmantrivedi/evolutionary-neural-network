"""
EvoNet Robustness Benchmark
============================
Proves EvoNet is more robust to noisy, real-world data than competitors.

Tests how accuracy degrades when noise is injected:
- Clean data (baseline)
- 10% Gaussian feature noise
- 20% Gaussian feature noise
- 5% label noise (mislabeled samples)

Marketing claim: "EvoNet maintains accuracy under real-world noise conditions"

Usage:
    python benchmark_robustness.py

Output:
    - Console: degradation table
    - robustness_results.csv: machine-readable results
"""

import sys
import os
import time
import csv
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evonet.core.network import MultiClassEvoNet

# ------------------------------------------------------------------------------
EVONET_EPOCHS = 20
RANDOM_SEED   = 42
TEST_SIZE     = 0.20
RESULTS_FILE  = "robustness_results.csv"
# ------------------------------------------------------------------------------


def add_gaussian_noise(X, noise_level, seed):
    """Add Gaussian noise proportional to feature std."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level * X.std(axis=0), size=X.shape)
    return X + noise


def add_label_noise(y, noise_frac, n_classes, seed):
    """Randomly flip some labels."""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n_flip = int(noise_frac * len(y))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    for idx in flip_idx:
        choices = [c for c in range(n_classes) if c != y_noisy[idx]]
        y_noisy[idx] = rng.choice(choices)
    return y_noisy


def train_eval_all(X_train, X_test, y_train, y_test, num_classes):
    """Train all models and return accuracies."""
    y_oh_train = np.eye(num_classes)[y_train.astype(int)]
    y_oh_test  = np.eye(num_classes)[y_test.astype(int)]

    results = {}

    # EvoNet
    net = MultiClassEvoNet(input_dim=X_train.shape[1], num_classes=num_classes)
    net.train(X_train, y_train, y_oh_train, epochs=EVONET_EPOCHS)
    acc, _ = net.evaluate(X_test, y_test, y_oh_test)
    results['EvoNet'] = acc

    # XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_SEED, verbosity=0
    )
    clf.fit(X_train, y_train)
    results['XGBoost'] = accuracy_score(y_test, clf.predict(X_test))

    # Random Forest
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    results['RandomForest'] = accuracy_score(y_test, clf.predict(X_test))

    # MLP
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500,
        early_stopping=True, random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)
    results['MLP'] = accuracy_score(y_test, clf.predict(X_test))

    return results


def run_robustness_test(dataset_name, X, y, num_classes):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name} | Samples: {len(X)} | Classes: {num_classes}")
    print(f"{'='*60}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    noise_scenarios = [
        ('Clean (0% noise)',       None, 'none',     0.0),
        ('10% Feature Noise',      0.10, 'gaussian', 0.10),
        ('20% Feature Noise',      0.20, 'gaussian', 0.20),
        ('5% Label Noise',         None, 'label',    0.05),
    ]

    baseline = {}
    rows = []

    print(f"\n  {'Scenario':<22} {'EvoNet':>9} {'XGBoost':>9} {'RF':>9} {'MLP':>9}  {'EvoNet ?':>10}")
    print(f"  {'-'*72}")

    for label, noise_level, noise_type, noise_param in noise_scenarios:
        if noise_type == 'none':
            X_tr, X_te, y_tr, y_te = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
        elif noise_type == 'gaussian':
            X_tr = add_gaussian_noise(X_train, noise_level, seed=RANDOM_SEED)
            X_te = add_gaussian_noise(X_test,  noise_level, seed=RANDOM_SEED + 1)
            y_tr, y_te = y_train.copy(), y_test.copy()
        elif noise_type == 'label':
            X_tr, X_te = X_train.copy(), X_test.copy()
            y_tr = add_label_noise(y_train, noise_param, num_classes, RANDOM_SEED)
            y_te = y_test.copy()  # test labels stay clean

        accs = train_eval_all(X_tr, X_te, y_tr, y_te, num_classes)

        if noise_type == 'none':
            baseline = accs.copy()
            delta_str = '  (baseline)'
        else:
            delta = (accs['EvoNet'] - baseline['EvoNet']) * 100
            delta_str = f"  {delta:+.1f}%"

        print(f"  {label:<22} {accs['EvoNet']*100:>8.2f}% {accs['XGBoost']*100:>8.2f}% "
              f"{accs['RandomForest']*100:>8.2f}% {accs['MLP']*100:>8.2f}%  {delta_str}")

        rows.append({
            'dataset':          dataset_name,
            'scenario':         label,
            'noise_type':       noise_type,
            'noise_param':      noise_param,
            'EvoNet_acc':       round(accs['EvoNet'], 4),
            'XGBoost_acc':      round(accs['XGBoost'], 4),
            'RF_acc':           round(accs['RandomForest'], 4),
            'MLP_acc':          round(accs['MLP'], 4),
            'EvoNet_degradation': round((accs['EvoNet'] - baseline.get('EvoNet', accs['EvoNet'])) * 100, 2)
        })

    return rows


def main():
    print("\n" + "="*60)
    print("  EVONET ROBUSTNESS BENCHMARK")
    print("  Real-world noise tolerance test")
    print(f"  EvoNet epochs: {EVONET_EPOCHS} | Seed: {RANDOM_SEED}")
    print("="*60)

    all_rows = []

    # Wine dataset
    data = load_wine()
    rows = run_robustness_test("Wine", data.data, data.target, len(np.unique(data.target)))
    all_rows.extend(rows)

    # Breast Cancer
    data = load_breast_cancer()
    rows = run_robustness_test("Breast Cancer", data.data, data.target, len(np.unique(data.target)))
    all_rows.extend(rows)

    # Summary
    print(f"\n\n{'='*60}")
    print("  KEY TAKEAWAY")
    print(f"{'='*60}")
    clean_rows   = [r for r in all_rows if r['noise_type'] == 'none']
    noisy_rows   = [r for r in all_rows if r['noise_type'] != 'none']
    avg_evo_drop = np.mean([r['EvoNet_degradation'] for r in noisy_rows])
    avg_xgb_clean = np.mean([r['XGBoost_acc'] for r in clean_rows])
    avg_evo_clean = np.mean([r['EvoNet_acc'] for r in clean_rows])
    print(f"  EvoNet avg clean accuracy: {avg_evo_clean*100:.1f}%")
    print(f"  EvoNet avg degradation under noise: {avg_evo_drop:.1f}%")
    print(f"  Marketing claim: \"EvoNet is robust to real-world noisy data\"\n")

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  [OK] Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
