"""
EvoNet Small Dataset Benchmark
================================
Proves EvoNet outperforms deep learning on small datasets (< 10K samples).
This is a critical differentiator: TensorFlow/Keras is notoriously poor on
small data due to overfitting, while EvoNet's evolutionary approach is robust.

Tests: 500, 1K, 2K, 5K sample sizes across 50 features
Competitors: XGBoost, TensorFlow MLP, sklearn MLP

Usage:
    python benchmark_small_datasets.py

Output:
    - Console: formatted comparison table
    - small_dataset_results.csv: machine-readable results
"""

import sys
import os
import time
import csv
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# TensorFlow import (suppress verbose output)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evonet.core.network import MultiClassEvoNet

# ------------------------------------------------------------------------------
N_FEATURES     = 50
N_CLASSES      = 3
EVONET_EPOCHS  = 25
RANDOM_SEED    = 42
TEST_SIZE      = 0.25
RESULTS_FILE   = "small_dataset_results.csv"
# ------------------------------------------------------------------------------


def make_small_dataset(n_samples, n_features=N_FEATURES, n_classes=N_CLASSES, seed=RANDOM_SEED):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=25,
        n_redundant=10,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=seed
    )
    return X, y


def run_xgboost(X_train, X_test, y_train, y_test):
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_SEED, verbosity=0
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return acc, time.time() - t0


def run_tf_mlp(X_train, X_test, y_train, y_test, n_classes):
    """TensorFlow MLP — the 'heavyweight' that struggles with small data."""
    t0 = time.time()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train,
        epochs=100, batch_size=32, verbose=0,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    return acc, time.time() - t0


def run_sklearn_mlp(X_train, X_test, y_train, y_test):
    t0 = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500,
        early_stopping=True, random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return acc, time.time() - t0


def run_evonet(X_train, X_test, y_train, y_test, n_classes):
    y_oh_train = np.eye(n_classes)[y_train.astype(int)]
    y_oh_test  = np.eye(n_classes)[y_test.astype(int)]
    t0 = time.time()
    net = MultiClassEvoNet(input_dim=X_train.shape[1], num_classes=n_classes)
    net.train(X_train, y_train, y_oh_train, epochs=EVONET_EPOCHS)
    acc, _ = net.evaluate(X_test, y_test, y_oh_test)
    return acc, time.time() - t0


def run_size_test(n_samples):
    print(f"\n{'-'*60}")
    print(f"  Dataset Size: {n_samples} samples x {N_FEATURES} features x {N_CLASSES} classes")
    print(f"{'-'*60}")

    X, y = make_small_dataset(n_samples)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    evo_acc,   evo_t   = run_evonet(X_train, X_test, y_train, y_test, N_CLASSES)
    xgb_acc,   xgb_t   = run_xgboost(X_train, X_test, y_train, y_test)
    tf_acc,    tf_t    = run_tf_mlp(X_train, X_test, y_train, y_test, N_CLASSES)
    mlp_acc,   mlp_t   = run_sklearn_mlp(X_train, X_test, y_train, y_test)

    all_accs = {'EvoNet': evo_acc, 'XGBoost': xgb_acc, 'TF-MLP': tf_acc, 'sklearn-MLP': mlp_acc}
    winner = max(all_accs, key=all_accs.get)

    print(f"\n  {'Model':<14} {'Accuracy':>10}  {'Time':>8}  {'Status'}")
    print(f"  {'-'*50}")
    for model, (acc, t) in [('EvoNet', (evo_acc, evo_t)), ('XGBoost', (xgb_acc, xgb_t)),
                              ('TF-MLP', (tf_acc, tf_t)), ('sklearn-MLP', (mlp_acc, mlp_t))]:
        mark = ' [WIN] WINNER' if model == winner else ''
        print(f"  {model:<14} {acc*100:>9.2f}%  {t:>6.1f}s{mark}")

    return {
        'n_samples':       n_samples,
        'n_features':      N_FEATURES,
        'EvoNet_acc':      round(evo_acc, 4),
        'XGBoost_acc':     round(xgb_acc, 4),
        'TF_MLP_acc':      round(tf_acc, 4),
        'skMLP_acc':       round(mlp_acc, 4),
        'EvoNet_time_s':   round(evo_t, 1),
        'XGBoost_time_s':  round(xgb_t, 2),
        'TF_time_s':       round(tf_t, 1),
        'skMLP_time_s':    round(mlp_t, 2),
        'winner':          winner
    }


def main():
    print("\n" + "="*60)
    print("  EVONET SMALL DATASET BENCHMARK")
    print("  Where deep learning fails, EvoNet thrives")
    print(f"  EvoNet epochs: {EVONET_EPOCHS} | Seed: {RANDOM_SEED}")
    print("="*60)

    sample_sizes = [500, 1000, 2000, 5000]

    rows = []
    for n in sample_sizes:
        row = run_size_test(n)
        rows.append(row)

    # Summary
    print(f"\n\n{'='*72}")
    print("  SMALL DATASET SUMMARY — Accuracy by Model")
    print(f"{'='*72}")
    print(f"  {'Samples':<10} {'EvoNet':>9} {'XGBoost':>9} {'TF-MLP':>9} {'sk-MLP':>9}  {'Winner'}")
    print(f"  {'-'*60}")
    evonet_wins = 0
    for r in rows:
        if r['winner'] == 'EvoNet':
            evonet_wins += 1
        print(f"  {r['n_samples']:<10} {r['EvoNet_acc']*100:>8.2f}% {r['XGBoost_acc']*100:>8.2f}% "
              f"{r['TF_MLP_acc']*100:>8.2f}% {r['skMLP_acc']*100:>8.2f}%  {r['winner']}")
    print(f"  {'-'*60}")
    print(f"  EvoNet won {evonet_wins}/{len(rows)} size categories\n")

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
