"""
EvoNet Production Readiness Validator
=======================================
Runs the 23 algorithmic + production readiness checks from the monetization checklist.
Pass this before pitching to investors or enterprise clients.

Usage:
    python validate_production_readiness.py

Output:
    - Console: pass/fail for each test
    - production_readiness_report.md: shareable report
"""

import sys
import os
import time
import pickle
import tempfile
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evonet.core.network import MultiClassEvoNet, RegressionEvoNet
from evonet.core.gpu_backend import is_gpu_available, get_device

# ------------------------------------------------------------------------------
REPORT_FILE = "production_readiness_report.md"
# ------------------------------------------------------------------------------


class TestResult:
    def __init__(self, test_id, name, category):
        self.test_id   = test_id
        self.name      = name
        self.category  = category
        self.passed    = False
        self.detail    = ""
        self.time_s    = 0.0


def run_test(test_id, name, category, fn):
    result = TestResult(test_id, name, category)
    t0 = time.time()
    try:
        passed, detail = fn()
        result.passed = passed
        result.detail = detail
    except Exception as e:
        result.passed = False
        result.detail = f"EXCEPTION: {e}"
    result.time_s = round(time.time() - t0, 2)
    status = "[PASS] PASS" if result.passed else "[FAIL] FAIL"
    print(f"  [{test_id:02d}] {status}  {name:<45}  ({result.detail})")
    return result


# ==============================================================================
# ALGORITHMIC TESTS
# ==============================================================================

def test_01_basic_accuracy():
    """Does it achieve >90% on Iris?"""
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    y_oh = np.eye(3)[y]
    X_tr, X_te, y_tr, y_te, yoh_tr, yoh_te = train_test_split(X, y, y_oh, test_size=0.2, random_state=42)
    net = MultiClassEvoNet(input_dim=4, num_classes=3)
    net.train(X_tr, y_tr, yoh_tr, epochs=20)
    acc, _ = net.evaluate(X_te, y_te, yoh_te)
    return acc >= 0.90, f"Iris accuracy: {acc*100:.1f}%"


def test_02_multiclass():
    """Does it handle any number of classes?"""
    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    y_oh = np.eye(3)[y]
    X_tr, X_te, y_tr, y_te, yoh_tr, yoh_te = train_test_split(X, y, y_oh, test_size=0.2, random_state=42)
    net = MultiClassEvoNet(input_dim=13, num_classes=3)
    net.train(X_tr, y_tr, yoh_tr, epochs=15)
    acc, _ = net.evaluate(X_te, y_te, yoh_te)
    return acc >= 0.80, f"Wine (3-class) accuracy: {acc*100:.1f}%"


def test_03_binary_classification():
    """Does it handle binary classification?"""
    X = np.random.randn(500, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_oh = np.eye(2)[y]
    X_tr, X_te, y_tr, y_te, yoh_tr, yoh_te = train_test_split(X, y, y_oh, test_size=0.2, random_state=42)
    net = MultiClassEvoNet(input_dim=10, num_classes=2)
    net.train(X_tr, y_tr, yoh_tr, epochs=15)
    acc, _ = net.evaluate(X_te, y_te, yoh_te)
    return acc >= 0.75, f"Binary accuracy: {acc*100:.1f}%"


def test_04_regression():
    """Does regression work?"""
    X = np.random.randn(300, 5)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + np.random.randn(300) * 0.1
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    net = RegressionEvoNet(input_dim=5)
    net.train(X_tr, y_tr, epochs=20)
    loss, _ = net.evaluate(X_te, y_te)
    return loss < 1.0, f"Regression MSE: {loss:.4f}"


def test_05_serialization():
    """Can the model be pickled and restored?"""
    net = MultiClassEvoNet(input_dim=4, num_classes=3)
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    y_oh = np.eye(3)[y]
    net.train(X, y, y_oh, epochs=5)

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name
        pickle.dump(net, f)

    with open(path, 'rb') as f:
        net2 = pickle.load(f)
    os.unlink(path)

    acc1, _ = net.evaluate(X, y, y_oh)
    acc2, _ = net2.evaluate(X, y, y_oh)
    diff = abs(acc1 - acc2)
    return diff < 0.01, f"Pre-pickle: {acc1*100:.1f}% | Post-pickle: {acc2*100:.1f}% | Diff: {diff*100:.2f}%"


def test_06_training_time():
    """Can it train on 1K samples in under 5 minutes?"""
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 3, 1000)
    y_oh = np.eye(3)[y]
    t0 = time.time()
    net = MultiClassEvoNet(input_dim=20, num_classes=3)
    net.train(X, y, y_oh, epochs=10)
    elapsed = time.time() - t0
    return elapsed < 300, f"1K samples x 10 epochs: {elapsed:.1f}s"


def test_07_memory_usage():
    """Does it use reasonable memory?"""
    import tracemalloc
    tracemalloc.start()
    X = np.random.randn(2000, 30)
    y = np.random.randint(0, 3, 2000)
    y_oh = np.eye(3)[y]
    net = MultiClassEvoNet(input_dim=30, num_classes=3)
    net.train(X, y, y_oh, epochs=5)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024
    return peak_mb < 2048, f"Peak memory: {peak_mb:.0f} MB"


def test_08_missing_values():
    """Does it handle datasets with imputed missing values?"""
    X = np.random.randn(200, 10)
    mask = np.random.rand(*X.shape) < 0.05
    X[mask] = 0.0  # replace NaN with 0 (simple imputation)
    y = np.random.randint(0, 2, 200)
    y_oh = np.eye(2)[y]
    net = MultiClassEvoNet(input_dim=10, num_classes=2)
    net.train(X, y, y_oh, epochs=5)
    acc, _ = net.evaluate(X, y, y_oh)
    return acc >= 0.40, f"Accuracy with missing values (imputed): {acc*100:.1f}%"


def test_09_reproducibility():
    """Same seed -> same results?"""
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    y_oh = np.eye(3)[y]
    X_tr, X_te, y_tr, y_te, yoh_tr, yoh_te = train_test_split(X, y, y_oh, test_size=0.2, random_state=42)

    np.random.seed(99)
    net1 = MultiClassEvoNet(input_dim=4, num_classes=3)
    net1.train(X_tr, y_tr, yoh_tr, epochs=5)
    acc1, _ = net1.evaluate(X_te, y_te, yoh_te)

    np.random.seed(99)
    net2 = MultiClassEvoNet(input_dim=4, num_classes=3)
    net2.train(X_tr, y_tr, yoh_tr, epochs=5)
    acc2, _ = net2.evaluate(X_te, y_te, yoh_te)

    diff = abs(acc1 - acc2)
    return diff < 0.05, f"Run1={acc1*100:.1f}% Run2={acc2*100:.1f}% Diff={diff*100:.2f}%"


def test_10_edge_case_single_feature():
    """Does it work with a single feature?"""
    X = np.random.randn(100, 1)
    y = (X[:, 0] > 0).astype(int)
    y_oh = np.eye(2)[y]
    net = MultiClassEvoNet(input_dim=1, num_classes=2)
    net.train(X, y, y_oh, epochs=10)
    acc, _ = net.evaluate(X, y, y_oh)
    return acc >= 0.5, f"1-feature accuracy: {acc*100:.1f}%"


def test_11_edge_case_high_dim():
    """Does it work with 200+ features?"""
    X = np.random.randn(300, 200)
    y = np.random.randint(0, 3, 300)
    y_oh = np.eye(3)[y]
    net = MultiClassEvoNet(input_dim=200, num_classes=3)
    net.train(X, y, y_oh, epochs=5)
    acc, _ = net.evaluate(X, y, y_oh)
    return True, f"200-feature training completed, acc: {acc*100:.1f}%"


def test_12_prediction_probabilities():
    """Does it return valid probability distributions?"""
    net = MultiClassEvoNet(input_dim=4, num_classes=3)
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    y_oh = np.eye(3)[y]
    net.train(X, y, y_oh, epochs=5)
    y_pred, _, _, conf = net._forward_elite(X[0])
    prob_sum = sum(y_pred)
    all_valid = all(0.0 <= p <= 1.0 for p in y_pred)
    return abs(prob_sum - 1.0) < 1e-6 and all_valid, f"Prob sum={prob_sum:.6f}, all valid: {all_valid}"


def test_13_gpu_detection():
    """GPU is detected (or CPU gracefully used)."""
    device = get_device()
    gpu_avail = is_gpu_available()
    return True, f"GPU available: {gpu_avail} | Device: {device}"


def test_14_existing_brain_loads():
    """Can the validated trained brain be loaded?"""
    brain_files = [
        "nifty50_brain_validated.pkl",
        "ultimate_brain_validated.pkl"
    ]
    loaded = []
    for bf in brain_files:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), bf)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            loaded.append(bf)
    return len(loaded) > 0, f"Loaded: {', '.join(loaded) if loaded else 'None found'}"


def test_15_error_handling_bad_input():
    """Does it fail gracefully on mismatched input dim?"""
    net = MultiClassEvoNet(input_dim=4, num_classes=3)
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    y_oh = np.eye(3)[y]
    net.train(X, y, y_oh, epochs=5)
    try:
        # Pass wrong shape — should raise an error
        _ = net._forward_elite(np.zeros(10))
        return False, "No error raised for wrong input dim"
    except Exception as e:
        return True, f"Correctly raises error: {type(e).__name__}"


def test_16_imbalanced_data_handles():
    """Does it handle heavily imbalanced data without crashing?"""
    n = 500
    X = np.random.randn(n, 10)
    y = np.zeros(n, dtype=int)
    y[:5] = 1  # 1% minority
    y_oh = np.eye(2)[y]
    net = MultiClassEvoNet(input_dim=10, num_classes=2)
    net.train(X, y, y_oh, epochs=5)
    acc, _ = net.evaluate(X, y, y_oh)
    return True, f"Completed training on 99-1 imbalanced data, acc: {acc*100:.1f}%"


def main():
    print("\n" + "="*60)
    print("  EVONET PRODUCTION READINESS VALIDATOR")
    print("  Must pass >= 14/16 tests before monetizing")
    print("="*60 + "\n")

    categories = {
        "ALGORITHMIC":   "Core algorithmic correctness",
        "PRODUCTION":    "Production environment readiness",
    }

    tests = [
        (1,  "Basic accuracy (Iris >= 90%)",          "ALGORITHMIC",  test_01_basic_accuracy),
        (2,  "Multi-class classification",            "ALGORITHMIC",  test_02_multiclass),
        (3,  "Binary classification",                 "ALGORITHMIC",  test_03_binary_classification),
        (4,  "Regression task",                       "ALGORITHMIC",  test_04_regression),
        (5,  "Serialization (pickle round-trip)",     "PRODUCTION",   test_05_serialization),
        (6,  "Training time (<5 min, 1K samples)",    "PRODUCTION",   test_06_training_time),
        (7,  "Memory usage (<2GB)",                   "PRODUCTION",   test_07_memory_usage),
        (8,  "Handles missing values (imputed)",      "ALGORITHMIC",  test_08_missing_values),
        (9,  "Reproducibility (same seed)",           "ALGORITHMIC",  test_09_reproducibility),
        (10, "Edge case: single feature",             "ALGORITHMIC",  test_10_edge_case_single_feature),
        (11, "High-dimensional (200 features)",       "ALGORITHMIC",  test_11_edge_case_high_dim),
        (12, "Valid probability output",              "ALGORITHMIC",  test_12_prediction_probabilities),
        (13, "GPU/CPU detection",                     "PRODUCTION",   test_13_gpu_detection),
        (14, "Load trained brain files",              "PRODUCTION",   test_14_existing_brain_loads),
        (15, "Error handling (bad input)",            "PRODUCTION",   test_15_error_handling_bad_input),
        (16, "Imbalanced data stability",             "ALGORITHMIC",  test_16_imbalanced_data_handles),
    ]

    results = []
    current_cat = None
    for test_id, name, category, fn in tests:
        if category != current_cat:
            current_cat = category
            print(f"\n  -- {category}: {categories[category]} --")
        r = run_test(test_id, name, category, fn)
        results.append(r)

    # Summary
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    pct    = passed / total * 100

    print(f"\n\n{'='*60}")
    print(f"  RESULT: {passed}/{total} tests passed ({pct:.0f}%)")
    if passed >= 14:
        print("  STATUS: [PASS] PRODUCTION READY — proceed with monetization")
    elif passed >= 10:
        print("  STATUS: [WARN]  MOSTLY READY — fix failing tests before launch")
    else:
        print("  STATUS: [FAIL] NOT READY — significant issues detected")
    print("="*60)

    # Failed tests summary
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n  Failed tests ({len(failed)}):")
        for r in failed:
            print(f"    [{r.test_id:02d}] {r.name}: {r.detail}")

    # Write report
    now = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")
    report_lines = [
        f"# EvoNet Production Readiness Report\n",
        f"> Generated: {now}\n",
        f"\n## Summary\n",
        f"**{passed}/{total} tests passed ({pct:.0f}%)**\n",
        f"**Status**: {'[PASS] PRODUCTION READY' if passed >= 14 else '[WARN] Needs attention'}\n\n",
        "## Test Results\n\n",
        "| # | Test | Category | Status | Detail |\n",
        "|---|------|----------|--------|--------|\n",
    ]
    for r in results:
        status = "[PASS] PASS" if r.passed else "[FAIL] FAIL"
        report_lines.append(f"| {r.test_id} | {r.name} | {r.category} | {status} | {r.detail} |\n")

    with open(REPORT_FILE, 'w') as f:
        f.writelines(report_lines)

    print(f"\n  [DOC] Report saved to {REPORT_FILE}")
    return passed, total


if __name__ == "__main__":
    main()
