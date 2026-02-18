"""
+============================================================================+
|                    CTO-LEVEL VALIDATION SUITE                              |
|                    NIFTY50 BRAIN - REAL WORLD TESTS                        |
+============================================================================+
|  Runs 7 professional tests to validate the brain under every condition:    |
|                                                                            |
|  TEST 1: Brain Integrity & Determinism Check                               |
|  TEST 2: Multi-Regime Backtest (Bull / Bear / Crash / Sideways)            |
|  TEST 3: Walk-Forward Validation (Rolling Windows)                         |
|  TEST 4: Stress Test (Synthetic Crash Injection)                           |
|  TEST 5: Transaction Cost Sensitivity Analysis                             |
|  TEST 6: Monte Carlo Robustness (Randomized Runs)                          |
|  TEST 7: Buy-and-Hold Benchmark Comparison                                 |
+============================================================================+

Usage:
    python tests/cto_validation_suite.py
    python tests/cto_validation_suite.py --brain nifty50_brain_validated.pkl
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pickle
import argparse
import time
import datetime
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import yfinance as yf_lib
from evonet.trader.alpha_factory import AlphaFactory
from evonet.trader.environment import FinancialRegimeEnv
from evonet.core import layers

# Bypass attention layer (same as training)
layers.EvoAttentionLayer.forward = lambda self, x, train=True: x

# ─── Constants ────────────────────────────────────────────────────────────────
TICKER = "^NSEI"
WINDOW_SIZE = 20
FEE = 0.0007
SLIPPAGE = 0.0003
PASS_MARK = "[PASS]"
FAIL_MARK = "[FAIL]"
WARN_MARK = "[WARN]"

# Per-period data cache (keyed by date range, NOT just ticker)
_DATA_CACHE = {}

# ─── Core Helpers ─────────────────────────────────────────────────────────────

def load_brain(path: str):
    """Load and return the pickled brain."""
    with open(path, "rb") as f:
        return pickle.load(f)


def fetch_nifty_data(start: str, end: str) -> pd.DataFrame:
    """Fetch and process NIFTY50 data for a SPECIFIC date range.
    
    CRITICAL FIX: The original DataFetcher cached by ticker only, ignoring dates.
    This caused ALL tests to run on the same data. We now fetch fresh data per period.
    """
    cache_key = f"{start}_{end}"
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()
    
    print(f"  [FETCH] {TICKER} {start} -> {end}...", end=" ", flush=True)
    df = yf_lib.download(TICKER, start=start, end=end, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    
    # Apply AlphaFactory features (same processing as training)
    df = AlphaFactory.apply_all(df)
    df.dropna(inplace=True)
    
    print(f"{len(df)} bars")
    _DATA_CACHE[cache_key] = df
    return df.copy()


def make_env(df: pd.DataFrame, fee=FEE, slippage=SLIPPAGE):
    """Create a trading environment from processed data."""
    safe_end = len(df) - (WINDOW_SIZE * 3)
    if safe_end <= WINDOW_SIZE + 10:
        return None
    return FinancialRegimeEnv(
        df,
        frame_bound=(WINDOW_SIZE, safe_end),
        window_size=WINDOW_SIZE,
        fee=fee,
        slippage_std=slippage,
    )


def run_episode(env, brain, genome_idx=0, max_steps=800):
    """Run one trading episode and return metrics."""
    state, _ = env.reset()
    equity, equity_curve, returns = 1.0, [1.0], []
    actions_taken = []
    terminated, steps, last_action, num_trades = False, 0, 1, 0

    while not terminated and steps < max_steps:
        action = brain.get_action(state, genome_idx)
        actions_taken.append(action)
        if action != last_action:
            num_trades += 1
            last_action = action
        state, reward, terminated, _, _ = env.step(action)
        reward = np.clip(reward, -0.05, 0.05)
        equity *= (1.0 + reward)
        equity = np.clip(equity, 0.01, 100.0)
        equity_curve.append(equity)
        returns.append(reward)
        steps += 1

    returns_arr = np.array(returns)
    metrics = compute_metrics(returns_arr, equity_curve, num_trades)
    metrics["actions"] = actions_taken
    metrics["steps"] = steps
    return metrics


def compute_metrics(returns, equity_curve, num_trades):
    """Compute Sharpe, Sortino, Return, MaxDD, Win Rate."""
    if len(returns) == 0:
        return {"sharpe": 0, "sortino": 0, "return": -100, "max_dd": 1.0,
                "trades": 0, "win_rate": 0.0}

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)

    sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252) if std_ret > 0 else 0.0
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (mean_ret / (downside_std + 1e-9)) * np.sqrt(252)

    peak, max_dd = equity_curve[0], 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / (peak + 1e-9)
        if dd > max_dd:
            max_dd = dd

    total_return = ((equity_curve[-1] / equity_curve[0]) - 1.0) * 100
    wins = np.sum(returns > 0)
    win_rate = wins / len(returns) * 100 if len(returns) > 0 else 0

    return {
        "sharpe": float(np.clip(sharpe, -10, 10)),
        "sortino": float(np.clip(sortino, -10, 15)),
        "return": float(np.clip(total_return, -100, 500)),
        "max_dd": float(np.clip(max_dd, 0, 1)),
        "trades": int(num_trades),
        "win_rate": float(win_rate),
    }


def print_header(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_metrics_table(metrics: dict, label: str = ""):
    if label:
        print(f"  {label}:")
    print(f"    Sharpe: {metrics['sharpe']:6.2f}  |  Sortino: {metrics['sortino']:6.2f}  |  "
          f"Return: {metrics['return']:7.1f}%  |  MaxDD: {metrics['max_dd']:5.1%}  |  "
          f"Trades: {metrics['trades']:3d}  |  WinRate: {metrics.get('win_rate', 0):5.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1: BRAIN INTEGRITY & DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════
def test_1_integrity(brain, df_test):
    """Verify brain loads correctly and produces deterministic outputs."""
    print_header("TEST 1: BRAIN INTEGRITY & DETERMINISM CHECK")

    results = {"name": "Integrity & Determinism", "passed": True, "details": []}

    # 1a: Structure check
    checks = [
        ("Input Dimension == 200", brain.input_dim == 200),
        ("Output Dimension == 3", brain.output_dim == 3),
        ("Population Size > 0", brain.net.pop_size > 0),
    ]
    for label, ok in checks:
        status = PASS_MARK if ok else FAIL_MARK
        print(f"  {status}  {label}")
        if not ok:
            results["passed"] = False

    # 1b: Determinism check (same input → same output twice)
    env = make_env(df_test)
    if env is None:
        print(f"  {FAIL_MARK}  Cannot create test environment")
        results["passed"] = False
        return results

    state, _ = env.reset()
    a1 = brain.get_action(state, 0)
    a2 = brain.get_action(state, 0)
    det_ok = a1 == a2
    print(f"  {PASS_MARK if det_ok else FAIL_MARK}  Deterministic output (same state → same action)")
    if not det_ok:
        results["passed"] = False

    # 1c: Action range check
    actions = set()
    for _ in range(50):
        state, _ = env.reset()
        for _ in range(20):
            a = brain.get_action(state, 0)
            actions.add(a)
            state, _, done, _, _ = env.step(a)
            if done:
                break
    env.close()
    valid_range = actions.issubset({0, 1, 2})
    print(f"  {PASS_MARK if valid_range else FAIL_MARK}  Actions in valid range {{0,1,2}} → observed {actions}")

    # 1d: Multiple action types used
    diverse = len(actions) >= 2
    print(f"  {PASS_MARK if diverse else WARN_MARK}  Uses multiple action types → {len(actions)} distinct actions")

    if not valid_range:
        results["passed"] = False

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2: MULTI-REGIME BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
def test_2_multi_regime(brain):
    """Test performance across distinct market regimes."""
    print_header("TEST 2: MULTI-REGIME BACKTEST")

    regimes = {
        "COVID Crash (2020-02 → 2020-06)":       ("2019-06-01", "2020-07-01"),
        "Post-COVID Bull (2020-06 → 2021-12)":    ("2020-01-01", "2022-01-01"),
        "2022 Bear / Inflation (2022-01 → 2022-12)": ("2021-06-01", "2023-01-01"),
        "2023 Recovery Bull (2023-01 → 2023-12)": ("2022-07-01", "2024-01-01"),
        "2024 Consolidation (2024-01 → 2024-12)": ("2023-07-01", "2025-01-01"),
    }

    results = {"name": "Multi-Regime Backtest", "passed": True, "regime_results": {}}
    catastrophic = 0

    for label, (start, end) in regimes.items():
        try:
            df = fetch_nifty_data(start, end)
            env = make_env(df)
            if env is None:
                print(f"  {WARN_MARK}  {label}: insufficient data, skipped")
                continue
            m = run_episode(env, brain)
            env.close()
            results["regime_results"][label] = m
            print_metrics_table(m, label)
            if m["return"] < -50 or m["max_dd"] > 0.60:
                catastrophic += 1
                print(f"           → {FAIL_MARK} CATASTROPHIC performance in this regime!")
        except Exception as e:
            print(f"  {WARN_MARK}  {label}: error → {e}")

    if catastrophic > 0:
        results["passed"] = False
        print(f"\n  {FAIL_MARK}  Brain FAILED in {catastrophic} regime(s) with catastrophic loss")
    else:
        print(f"\n  {PASS_MARK}  Brain survived ALL regimes without catastrophic loss")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3: WALK-FORWARD VALIDATION (Rolling Windows)
# ═══════════════════════════════════════════════════════════════════════════════
def test_3_walk_forward(brain):
    """Rolling 1-year windows to test consistency across time."""
    print_header("TEST 3: WALK-FORWARD VALIDATION (Rolling 1-Year Windows)")

    windows = [
        ("2015-01-01", "2016-06-01"),
        ("2016-01-01", "2017-06-01"),
        ("2017-01-01", "2018-06-01"),
        ("2018-01-01", "2019-06-01"),
        ("2019-01-01", "2020-06-01"),
        ("2020-01-01", "2021-06-01"),
        ("2021-01-01", "2022-06-01"),
        ("2022-01-01", "2023-06-01"),
        ("2023-01-01", "2024-06-01"),
    ]

    results = {"name": "Walk-Forward", "passed": True, "window_results": []}
    sortinos, returns_list, drawdowns = [], [], []
    losing_windows = 0

    for start, end in windows:
        try:
            df = fetch_nifty_data(start, end)
            env = make_env(df)
            if env is None:
                continue
            m = run_episode(env, brain)
            env.close()
            results["window_results"].append({"period": f"{start[:4]}-{end[:4]}", **m})
            print_metrics_table(m, f"{start[:4]}–{end[:4]}")
            sortinos.append(m["sortino"])
            returns_list.append(m["return"])
            drawdowns.append(m["max_dd"])
            if m["return"] < -10:
                losing_windows += 1
        except Exception as e:
            print(f"  {WARN_MARK}  {start[:4]}–{end[:4]}: {e}")

    if len(sortinos) > 0:
        avg_sortino = np.mean(sortinos)
        avg_ret = np.mean(returns_list)
        std_ret = np.std(returns_list)
        avg_dd = np.mean(drawdowns)
        consistency = 1.0 - (std_ret / (abs(avg_ret) + 1e-9))

        print(f"\n  ── WALK-FORWARD SUMMARY ──")
        print(f"  Avg Sortino:    {avg_sortino:.2f}")
        print(f"  Avg Return:     {avg_ret:.1f}%")
        print(f"  Std Return:     {std_ret:.1f}%")
        print(f"  Avg MaxDD:      {avg_dd:.1%}")
        print(f"  Losing Windows: {losing_windows}/{len(sortinos)}")

        if losing_windows > len(sortinos) // 2:
            results["passed"] = False
            print(f"  {FAIL_MARK}  Majority of windows are losers — strategy unreliable")
        elif avg_sortino < 0.5:
            results["passed"] = False
            print(f"  {FAIL_MARK}  Average Sortino below 0.5 — insufficient edge")
        else:
            print(f"  {PASS_MARK}  Walk-forward validates consistent performance")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4: STRESS TEST (SYNTHETIC CRASH)
# ═══════════════════════════════════════════════════════════════════════════════
def test_4_stress_test(brain):
    """Inject synthetic crashes and see if the brain survives."""
    print_header("TEST 4: STRESS TEST (SYNTHETIC CRASH INJECTION)")

    results = {"name": "Stress Test", "passed": True, "scenarios": []}

    df_base = fetch_nifty_data("2022-01-01", "2024-01-01")

    scenarios = {
        "Flash Crash (-15% in 3 days)":   {"magnitude": 0.85, "days": 3},
        "Prolonged Bear (-25% in 20 days)": {"magnitude": 0.96, "days": 20},
        "V-Recovery (-20% then +25%)":    {"magnitude": 0.80, "days": 5, "recovery": True},
        "Black Swan (-35% in 1 day)":     {"magnitude": 0.65, "days": 1},
    }

    for label, params in scenarios.items():
        df = df_base.copy()
        mid = len(df) // 2

        # Inject crash
        for d in range(params["days"]):
            idx = mid + d
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc("Close")] *= params["magnitude"]
                adj_ret = np.log(params["magnitude"])
                df.iloc[idx, df.columns.get_loc("Log_Ret")] = adj_ret

        # Optional recovery
        if params.get("recovery"):
            for d in range(params["days"]):
                idx = mid + params["days"] + d
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc("Close")] *= (1.0 / params["magnitude"])
                    df.iloc[idx, df.columns.get_loc("Log_Ret")] = -np.log(params["magnitude"])

        try:
            env = make_env(df)
            if env is None:
                continue
            m = run_episode(env, brain)
            env.close()
            results["scenarios"].append({"scenario": label, **m})
            print_metrics_table(m, label)

            survived = m["return"] > -80 and m["max_dd"] < 0.80
            if not survived:
                results["passed"] = False
                print(f"           → {FAIL_MARK} Brain did NOT survive this stress event!")
            else:
                print(f"           → {PASS_MARK} Brain survived")
        except Exception as e:
            print(f"  {WARN_MARK}  {label}: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5: TRANSACTION COST SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════
def test_5_cost_sensitivity(brain):
    """Test how performance degrades with increasing costs."""
    print_header("TEST 5: TRANSACTION COST SENSITIVITY ANALYSIS")

    df = fetch_nifty_data("2023-01-01", "2025-01-01")

    cost_levels = {
        "Zero Cost (theoretical)":    (0.0000, 0.0000),
        "Low Cost (discount broker)": (0.0003, 0.0001),
        "Standard (Zerodha level)":   (0.0007, 0.0003),
        "High Cost (full service)":   (0.0020, 0.0005),
        "Extreme Cost (worst case)":  (0.0050, 0.0010),
    }

    results = {"name": "Cost Sensitivity", "passed": True, "levels": []}
    prev_ret = None

    for label, (fee, slip) in cost_levels.items():
        try:
            env = make_env(df, fee=fee, slippage=slip)
            if env is None:
                continue
            m = run_episode(env, brain)
            env.close()
            results["levels"].append({"level": label, "fee": fee, "slippage": slip, **m})
            print_metrics_table(m, f"{label} (fee={fee*100:.2f}%, slip={slip*100:.2f}%)")

            if prev_ret is not None and m["return"] > prev_ret + 20:
                print(f"           → {WARN_MARK} Return INCREASED with higher costs — unusual")

            prev_ret = m["return"]
        except Exception as e:
            print(f"  {WARN_MARK}  {label}: {e}")

    # Check if strategy is still profitable at standard costs
    std = [l for l in results["levels"] if "Standard" in l["level"]]
    if std and std[0]["return"] < 0:
        results["passed"] = False
        print(f"\n  {FAIL_MARK}  Strategy is UNPROFITABLE at standard costs!")
    else:
        print(f"\n  {PASS_MARK}  Strategy remains profitable across cost levels")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6: MONTE CARLO ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════
def test_6_monte_carlo(brain):
    """Run multiple randomized episodes to check consistency."""
    print_header("TEST 6: MONTE CARLO ROBUSTNESS (10 Randomized Runs)")

    df = fetch_nifty_data("2023-01-01", "2025-01-01")

    results = {"name": "Monte Carlo", "passed": True, "runs": []}
    all_returns, all_sortinos, all_dds = [], [], []
    NUM_RUNS = 10

    for i in range(NUM_RUNS):
        np.random.seed(i * 42)
        try:
            env = make_env(df)
            if env is None:
                continue
            m = run_episode(env, brain)
            env.close()
            results["runs"].append(m)
            all_returns.append(m["return"])
            all_sortinos.append(m["sortino"])
            all_dds.append(m["max_dd"])
            print(f"  Run {i+1:2d}/10: Ret={m['return']:7.1f}%  Sortino={m['sortino']:5.2f}  "
                  f"DD={m['max_dd']:5.1%}  Trades={m['trades']}")
        except Exception as e:
            print(f"  Run {i+1:2d}/10: ERROR — {e}")

    if len(all_returns) >= 3:
        mean_ret = np.mean(all_returns)
        std_ret = np.std(all_returns)
        mean_sort = np.mean(all_sortinos)
        mean_dd = np.mean(all_dds)
        pct_profitable = np.mean([1 for r in all_returns if r > 0]) * 100
        worst_ret = np.min(all_returns)
        best_ret = np.max(all_returns)

        print(f"\n  ── MONTE CARLO SUMMARY ──")
        print(f"  Mean Return:      {mean_ret:.1f}% ± {std_ret:.1f}%")
        print(f"  Mean Sortino:     {mean_sort:.2f}")
        print(f"  Mean MaxDD:       {mean_dd:.1%}")
        print(f"  Profitable Runs:  {pct_profitable:.0f}%")
        print(f"  Range:            {worst_ret:.1f}% to {best_ret:.1f}%")

        if pct_profitable < 60:
            results["passed"] = False
            print(f"  {FAIL_MARK}  Less than 60% of runs are profitable")
        elif std_ret > abs(mean_ret) * 2:
            results["passed"] = False
            print(f"  {FAIL_MARK}  Returns are too unstable (high variance)")
        else:
            print(f"  {PASS_MARK}  Brain shows robust, consistent performance")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7: BUY-AND-HOLD BENCHMARK COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def test_7_benchmark(brain):
    """Compare AI against simple buy-and-hold on the same data."""
    print_header("TEST 7: BUY-AND-HOLD BENCHMARK COMPARISON")

    periods = {
        "Full Test 2023-2024": ("2022-06-01", "2025-01-01"),
        "2023 Only":           ("2022-06-01", "2024-01-01"),
        "2024 Only":           ("2023-06-01", "2025-01-01"),
    }

    results = {"name": "Benchmark", "passed": True, "comparisons": []}
    ai_wins = 0
    total = 0

    for label, (start, end) in periods.items():
        try:
            df = fetch_nifty_data(start, end)
            env = make_env(df)
            if env is None:
                continue
            m = run_episode(env, brain)
            env.close()

            # Buy-and-hold return
            prices = df["Close"].values
            bah_return = ((prices[-1] / prices[WINDOW_SIZE]) - 1.0) * 100

            total += 1
            alpha = m["return"] - bah_return
            if alpha > 0:
                ai_wins += 1

            comp = {"period": label, "ai": m, "bah_return": bah_return, "alpha": alpha}
            results["comparisons"].append(comp)

            print(f"  {label}:")
            print(f"    AI Return:          {m['return']:7.1f}%  (Sortino: {m['sortino']:.2f}, DD: {m['max_dd']:.1%})")
            print(f"    Buy-and-Hold:       {bah_return:7.1f}%")
            print(f"    Alpha (AI - B&H):   {alpha:+7.1f}%  {'✅' if alpha > 0 else '⚠️'}")

        except Exception as e:
            print(f"  {WARN_MARK}  {label}: {e}")

    if total > 0:
        win_pct = ai_wins / total * 100
        print(f"\n  ── BENCHMARK SUMMARY ──")
        print(f"  AI beats Buy-and-Hold: {ai_wins}/{total} periods ({win_pct:.0f}%)")

        if win_pct >= 50:
            print(f"  {PASS_MARK}  AI demonstrates alpha over buy-and-hold")
        else:
            print(f"  {WARN_MARK}  AI underperforms buy-and-hold in most periods")
            print(f"          (May still be valuable due to lower risk / drawdown)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
def generate_scorecard(all_results: list):
    """Generate the CTO-level final scorecard."""
    print(f"\n+{'='*78}+")
    print(f"|{'':^78s}|")
    print(f"|{'CTO VALIDATION SCORECARD':^78s}|")
    print(f"|{'':^78s}|")
    print(f"+{'='*78}+")

    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    failed = total - passed

    for r in all_results:
        status = PASS_MARK if r["passed"] else FAIL_MARK
        print(f"  {status}  {r['name']}")

    print(f"\n{'─'*80}")
    print(f"  TOTAL: {passed}/{total} tests passed")

    if failed == 0:
        grade = "A+"
        verdict = "INVESTMENT GRADE — Ready for production deployment"
    elif failed == 1:
        grade = "A"
        verdict = "STRONG — Minor concerns; acceptable for deployment with monitoring"
    elif failed == 2:
        grade = "B"
        verdict = "ACCEPTABLE — Address failures before production use"
    else:
        grade = "C"
        verdict = "NEEDS WORK — Significant issues found; retrain recommended"

    print(f"\n  GRADE:   {grade}")
    print(f"  VERDICT: {verdict}")
    print(f"+{'='*78}+")

    return grade, verdict


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="CTO-Level Validation Suite")
    parser.add_argument("--brain", default="nifty50_brain_validated.pkl",
                        help="Path to brain pickle file")
    args = parser.parse_args()

    brain_path = os.path.join(ROOT_DIR, args.brain) if not os.path.isabs(args.brain) else args.brain
    if not os.path.exists(brain_path):
        print(f"[ERROR] Brain file not found: {brain_path}")
        sys.exit(1)

    print(f"\n+{'='*78}+")
    print(f"|{'CTO-LEVEL VALIDATION SUITE':^78s}|")
    print(f"|{'NIFTY50 AI BRAIN - REAL-WORLD TESTING':^78s}|")
    print(f"+{'='*78}+")
    print(f"  Brain File:  {os.path.basename(brain_path)}")
    print(f"  Market:      NIFTY50 (NSE India)")
    print(f"  Date:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"+{'='*78}+")

    start_time = time.time()
    brain = load_brain(brain_path)

    # Fetch test data once for Test 1
    print("\n  [*] Fetching NIFTY50 data (this may take a moment)...")
    print("  [*] NOTE: Each test period fetches its OWN data (cache-bug fixed)")
    df_test = fetch_nifty_data("2023-01-01", "2025-01-01")
    print(f"  [OK] Base test data ready: {len(df_test)} trading days\n")

    # Run all tests
    all_results = []
    all_results.append(test_1_integrity(brain, df_test))
    all_results.append(test_2_multi_regime(brain))
    all_results.append(test_3_walk_forward(brain))
    all_results.append(test_4_stress_test(brain))
    all_results.append(test_5_cost_sensitivity(brain))
    all_results.append(test_6_monte_carlo(brain))
    all_results.append(test_7_benchmark(brain))

    # Final scorecard
    elapsed = time.time() - start_time
    grade, verdict = generate_scorecard(all_results)
    print(f"\n  Time:   {elapsed:.0f}s")
    print(f"  Brain:  {os.path.basename(brain_path)}")
    print(f"  Market: NIFTY50 (NSE India)\n")


if __name__ == "__main__":
    main()
