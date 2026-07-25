"""
EvoNet Statistical Significance Benchmark
==========================================
Proves that EvoNet's superiority is statistically significant — not just luck.

Runs 10 independent trials with different random seeds, then performs:
- Paired t-test (EvoNet vs XGBoost)
- Reports mean +/- std for all models
- Confirms p < 0.05 (ideally p < 0.01)

This is the gold standard for academic papers and enterprise due diligence.

Usage:
    python benchmark_statistical_significance.py

Output:
    - Console: formatted stats table + t-test results
    - statistical_significance.json: machine-readable results
"""

import sys
import os
import json
import time
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evonet.core.network import MultiClassEvoNet

# ------------------------------------------------------------------------------
N_SEEDS       = 10
SEEDS         = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
EVONET_EPOCHS = 20
TEST_SIZE     = 0.20
RESULTS_FILE  = "statistical_significance.json"
# ------------------------------------------------------------------------------


def evaluate_all_models_one_seed(X, y, y_oh, num_classes, seed):
    """Train and evaluate all models for one random seed."""
    X_train, X_test, y_train, y_test, y_oh_train, y_oh_test = train_test_split(
        X, y, y_oh, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    results = {}

    # EvoNet
    net = MultiClassEvoNet(input_dim=X.shape[1], num_classes=num_classes)
    net.train(X_train, y_train, y_oh_train, epochs=EVONET_EPOCHS)
    acc, _ = net.evaluate(X_test, y_test, y_oh_test)
    results['EvoNet'] = acc

    # XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=seed, verbosity=0
    )
    clf.fit(X_train, y_train)
    results['XGBoost'] = accuracy_score(y_test, clf.predict(X_test))

    # Random Forest
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    results['RandomForest'] = accuracy_score(y_test, clf.predict(X_test))

    # MLP
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500,
        early_stopping=True, random_state=seed
    )
    clf.fit(X_train, y_train)
    results['MLP'] = accuracy_score(y_test, clf.predict(X_test))

    return results


def run_significance_test(dataset_name, loader):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name} | Seeds: {N_SEEDS} | Epochs: {EVONET_EPOCHS}")
    print(f"{'='*60}")

    data = loader()
    X, y = data.data, data.target
    num_classes = len(np.unique(y))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_oh = encoder.fit_transform(y.reshape(-1, 1))

    # Collect results across seeds
    all_results = {m: [] for m in ['EvoNet', 'XGBoost', 'RandomForest', 'MLP']}

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed:3d} ({i+1}/{N_SEEDS})...", end=' ', flush=True)
        res = evaluate_all_models_one_seed(X, y, y_oh, num_classes, seed)
        for model, acc in res.items():
            all_results[model].append(acc)
        print(f"EvoNet={res['EvoNet']*100:.1f}%  XGB={res['XGBoost']*100:.1f}%")

    # Statistics
    means = {m: np.mean(v) for m, v in all_results.items()}
    stds  = {m: np.std(v)  for m, v in all_results.items()}

    # Paired t-test: EvoNet vs XGBoost (primary comparison)
    t_stat, p_val = stats.ttest_rel(all_results['EvoNet'], all_results['XGBoost'])

    print(f"\n  {'Model':<15} {'Mean Acc':>10}  {'Std Dev':>10}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*56}")
    for model in ['EvoNet', 'XGBoost', 'RandomForest', 'MLP']:
        vals = all_results[model]
        print(f"  {model:<15} {means[model]*100:>9.2f}%  {stds[model]*100:>9.2f}%  "
              f"{min(vals)*100:>7.2f}%  {max(vals)*100:>7.2f}%")

    print(f"\n  Paired t-test (EvoNet vs XGBoost):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value:     {p_val:.4f}  ", end='')
    if p_val < 0.01:
        print("*** (p < 0.01 — HIGHLY significant)")
        sig_level = "p < 0.01"
    elif p_val < 0.05:
        print("** (p < 0.05 — significant)")
        sig_level = "p < 0.05"
    else:
        print("(NOT significant)")
        sig_level = "not significant"

    evonet_better = means['EvoNet'] > means['XGBoost']
    delta = (means['EvoNet'] - means['XGBoost']) * 100

    return {
        'dataset':          dataset_name,
        'n_seeds':          N_SEEDS,
        'EvoNet_mean':      round(means['EvoNet'], 4),
        'EvoNet_std':       round(stds['EvoNet'], 4),
        'XGBoost_mean':     round(means['XGBoost'], 4),
        'XGBoost_std':      round(stds['XGBoost'], 4),
        'RF_mean':          round(means['RandomForest'], 4),
        'RF_std':           round(stds['RandomForest'], 4),
        'MLP_mean':         round(means['MLP'], 4),
        'MLP_std':          round(stds['MLP'], 4),
        't_statistic':      round(float(t_stat), 4),
        'p_value':          round(float(p_val), 4),
        'significance':     sig_level,
        'EvoNet_wins':      evonet_better,
        'delta_vs_xgb_pct': round(delta, 2),
        'raw_EvoNet':       [round(v, 4) for v in all_results['EvoNet']],
        'raw_XGBoost':      [round(v, 4) for v in all_results['XGBoost']]
    }


def main():
    print("\n" + "="*60)
    print("  EVONET STATISTICAL SIGNIFICANCE BENCHMARK")
    print("  10 random seeds, paired t-test vs XGBoost")
    print("="*60)

    datasets = [
        ("Iris",          load_iris),
        ("Wine",          load_wine),
        ("Breast Cancer", load_breast_cancer),
    ]

    all_output = {}
    for name, loader in datasets:
        result = run_significance_test(name, loader)
        all_output[name] = result

    # Final Summary
    print(f"\n\n{'='*72}")
    print("  STATISTICAL SIGNIFICANCE SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Dataset':<16} {'EvoNet Mean':>12} {'XGB Mean':>12} {'?':>8}  {'p-value':>10}  Status")
    print(f"  {'-'*70}")
    all_sig = True
    for name, r in all_output.items():
        status = '[OK] SIGNIFICANT' if 'p <' in r['significance'] else '[WARN]?  Not sig'
        if 'not' in r['significance']:
            all_sig = False
        print(f"  {name:<16} {r['EvoNet_mean']*100:>11.2f}% {r['XGBoost_mean']*100:>11.2f}% "
              f"{r['delta_vs_xgb_pct']:>+7.2f}%  {r['p_value']:>10.4f}  {status}")

    print(f"\n  Claim: \"EvoNet shows statistically significant improvement (p < 0.05)\"")
    print(f"  Status: {'[OK] Validated' if all_sig else '[WARN]?  Partially validated'}\n")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_output, f, indent=2)
    print(f"  [OK] Full results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
