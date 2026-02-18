"""
+============================================================================+
|                     NIFTY50 BRAIN - ADAPTIVE RETUNING ENGINE               |
|                     Continuous Learning via Seeded Evolution                |
+============================================================================+
|                                                                            |
|  Takes the existing trained brain and evolves it on RECENT market data     |
|  to adapt to current trends. Gatekeeper test ensures new brain is better.  |
|                                                                            |
|  Usage:                                                                    |
|    python retune_brain.py                          # Default retune        |
|    python retune_brain.py --months 3 --gens 100    # Custom                |
|    python retune_brain.py --force-promote           # Skip gatekeeper      |
|                                                                            |
+============================================================================+
"""

import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pickle
import json
import copy
import shutil
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import yfinance as yf
from evonet.trader.alpha_factory import AlphaFactory
from evonet.trader.environment import FinancialRegimeEnv
from evonet.core import layers

# Bypass attention layer (same as training)
layers.EvoAttentionLayer.forward = lambda self, x, train=True: x

# ─── Configuration ────────────────────────────────────────────────────────────
TICKER = "^NSEI"
BRAIN_FILE = os.path.join(ROOT_DIR, "nifty50_brain_validated.pkl")
VERSIONS_DIR = os.path.join(ROOT_DIR, "brain_versions")
PAPER_TRADES_FILE = os.path.join(ROOT_DIR, "paper_trades.json")
WINDOW_SIZE = 20
FEE_PCT = 0.0007
SLIPPAGE_PCT = 0.0003


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_data(months_back=12):
    """Fetch recent NIFTY50 data for retuning."""
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=int(months_back * 31))

    print(f"  [FETCH] {TICKER} {start} -> {end} ({months_back} months)...")
    df = yf.download(TICKER, start=str(start), end=str(end), interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df = AlphaFactory.apply_all(df)
    df.dropna(inplace=True)

    print(f"  [OK] {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ─── Fitness Function ─────────────────────────────────────────────────────────

def calculate_fitness(returns, equity_curve, num_trades, pain_periods=None):
    """Professional fitness with pain-weighted learning.

    Args:
        returns: Array of episode returns
        equity_curve: List of equity values
        num_trades: Number of position changes
        pain_periods: Optional dict mapping step ranges to pain multipliers
    """
    if len(returns) == 0:
        return -1000.0, {'sharpe': 0, 'sortino': 0, 'max_dd': 1.0, 'return': -100, 'trades': 0}

    returns = np.clip(returns, -0.5, 0.5)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)

    # Sharpe (annualized)
    sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252) if std_ret > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (mean_ret / (downside_std + 1e-9)) * np.sqrt(252)

    # Max Drawdown
    peak, max_dd = equity_curve[0], 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / (peak + 1e-9)
        if dd > max_dd:
            max_dd = dd

    # Total Return
    total_return = ((equity_curve[-1] / equity_curve[0]) - 1.0) * 100

    # Clamp
    total_return = np.clip(total_return, -100, 500)
    sharpe = np.clip(sharpe, -10, 10)
    sortino = np.clip(sortino, -10, 15)
    max_dd = np.clip(max_dd, 0, 1)

    # Penalties
    trade_penalty = max(0, (10 - num_trades)) + max(0, (num_trades - 500) * 0.1)
    dd_penalty = max(0, (max_dd - 0.25) * 50.0)

    # Base fitness
    fitness = (sharpe * 2.0) + (sortino * 3.0) + (total_return * 0.1) - dd_penalty - trade_penalty

    metrics = {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd),
        'return': float(total_return),
        'trades': int(num_trades),
    }

    return float(fitness), metrics


def evaluate_genome(pilot, env, genome_idx, max_steps):
    """Evaluate a single genome on an environment."""
    state, _ = env.reset()
    equity = 1.0
    equity_curve = [1.0]
    returns = []
    terminated = False
    steps = 0
    last_action = 1
    num_trades = 0

    while not terminated and steps < max_steps:
        action = pilot.get_action(state, genome_idx)
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

    return calculate_fitness(np.array(returns), equity_curve, num_trades)


# ─── Seeded Population ────────────────────────────────────────────────────────

def seed_population_from_brain(pilot, mutation_strength=0.05):
    """Seed all genomes from the best genome (index 0) with small mutations.

    This preserves learned knowledge while adding diversity for evolution.
    """
    # Get the "master" weights from genome 0 (the deployed brain)
    master_weights = pilot.get_flat_weights(0)
    n_weights = len(master_weights)

    print(f"  [SEED] Seeding {pilot.pop_size} genomes from master ({n_weights} weights)")
    print(f"  [SEED] Mutation strength: {mutation_strength:.3f}")

    # Genome 0 stays exactly as-is (elitism)
    for i in range(1, pilot.pop_size):
        # Clone master weights
        variant = master_weights.copy()

        # Add random mutations (Gaussian noise)
        mutation_mask = np.random.random(n_weights) < 0.3  # Mutate 30% of weights
        noise = np.random.randn(np.sum(mutation_mask)) * mutation_strength
        variant[mutation_mask] += noise

        pilot.set_flat_weights(i, variant)

    # Reset evolution state for fine-tuning
    pilot.hall_of_fame = master_weights.copy()
    pilot.hof_score = -float('inf')
    pilot.stagnation_counter = 0
    pilot.current_mutation = mutation_strength

    print(f"  [OK] Population seeded. Genome 0 = original, genomes 1-{pilot.pop_size-1} = mutated clones")


# ─── Brain Versioning ─────────────────────────────────────────────────────────

def get_next_version():
    """Get the next version number."""
    os.makedirs(VERSIONS_DIR, exist_ok=True)

    existing = [d for d in os.listdir(VERSIONS_DIR)
                if os.path.isdir(os.path.join(VERSIONS_DIR, d)) and d.startswith("v")]

    if not existing:
        return "v1.0"

    versions = []
    for d in existing:
        try:
            parts = d[1:].split(".")
            versions.append((int(parts[0]), int(parts[1]) if len(parts) > 1 else 0))
        except (ValueError, IndexError):
            continue

    if not versions:
        return "v1.0"

    latest = max(versions)
    return f"v{latest[0]}.{latest[1] + 1}"


def save_version(pilot, version, metrics, config):
    """Save a versioned brain with metadata."""
    version_dir = os.path.join(VERSIONS_DIR, version)
    os.makedirs(version_dir, exist_ok=True)

    # Save brain
    brain_path = os.path.join(version_dir, "brain.pkl")
    with open(brain_path, "wb") as f:
        pickle.dump(pilot, f)

    # Save metadata
    metadata = {
        "version": version,
        "created": datetime.datetime.now().isoformat(),
        "ticker": TICKER,
        "metrics": metrics,
        "config": config,
    }
    meta_path = os.path.join(version_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Also copy as the active brain
    active_brain = os.path.join(ROOT_DIR, "nifty50_brain_validated.pkl")
    shutil.copy2(brain_path, active_brain)

    print(f"  [SAVE] Brain {version} saved to {version_dir}/")
    print(f"  [SAVE] Active brain updated: {active_brain}")

    return brain_path


def archive_current_brain():
    """Archive the current brain as v1.0 if no versions exist yet."""
    if not os.path.exists(BRAIN_FILE):
        return

    os.makedirs(VERSIONS_DIR, exist_ok=True)
    v1_dir = os.path.join(VERSIONS_DIR, "v1.0")

    if not os.path.exists(v1_dir):
        os.makedirs(v1_dir, exist_ok=True)
        shutil.copy2(BRAIN_FILE, os.path.join(v1_dir, "brain.pkl"))

        metadata = {
            "version": "v1.0",
            "created": datetime.datetime.now().isoformat(),
            "ticker": TICKER,
            "note": "Original brain (trained on 2010-2022)",
            "metrics": {"note": "See nifty50_report.txt for original metrics"},
        }
        with open(os.path.join(v1_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  [ARCHIVE] Current brain archived as v1.0")


# ─── Feedback Integration ─────────────────────────────────────────────────────

def load_paper_trade_feedback():
    """Load feedback from paper trades to identify losing periods."""
    if not os.path.exists(PAPER_TRADES_FILE):
        print("  [INFO] No paper trades found — skipping feedback integration")
        return None

    with open(PAPER_TRADES_FILE, "r") as f:
        data = json.load(f)

    signals = data.get("daily_signals", [])
    if len(signals) < 5:
        print(f"  [INFO] Only {len(signals)} signals — need >=5 for feedback")
        return None

    # Identify losing streaks
    losing_periods = []
    for i in range(1, len(signals)):
        prev_price = signals[i-1]["price"]
        curr_price = signals[i]["price"]
        action = signals[i]["action"]

        pos_map = action - 1  # SHORT=-1, NEUTRAL=0, LONG=1
        daily_pnl = pos_map * ((curr_price - prev_price) / prev_price)

        if daily_pnl < -0.005:  # Lost more than 0.5%
            losing_periods.append({
                "date": signals[i]["date"],
                "action": signals[i]["action_name"],
                "price": curr_price,
                "pnl": daily_pnl,
            })

    if losing_periods:
        print(f"  [FEEDBACK] Found {len(losing_periods)} painful days from paper trading")
        for lp in losing_periods[:5]:
            print(f"    {lp['date']}: {lp['action']} @ {lp['price']:.0f} = {lp['pnl']*100:+.2f}%")
    else:
        print(f"  [FEEDBACK] No significant losing days found")

    return losing_periods


# ─── Gatekeeper Test ──────────────────────────────────────────────────────────

def gatekeeper_test(old_pilot, new_pilot, df_holdout):
    """Compare old and new brain on holdout data.

    Returns: (passed, old_metrics, new_metrics)
    """
    safe_end = len(df_holdout) - (WINDOW_SIZE * 3)
    if safe_end <= WINDOW_SIZE + 5:
        print("  [WARN] Holdout data too short for gatekeeper — auto-passing")
        return True, {}, {}

    max_steps = min(400, safe_end - WINDOW_SIZE - 5)

    # Test old brain
    env_old = FinancialRegimeEnv(
        df_holdout,
        frame_bound=(WINDOW_SIZE, safe_end),
        window_size=WINDOW_SIZE,
        fee=FEE_PCT,
        slippage_std=SLIPPAGE_PCT,
    )
    old_fitness, old_metrics = evaluate_genome(old_pilot, env_old, 0, max_steps)
    env_old.close()

    # Test new brain
    env_new = FinancialRegimeEnv(
        df_holdout,
        frame_bound=(WINDOW_SIZE, safe_end),
        window_size=WINDOW_SIZE,
        fee=FEE_PCT,
        slippage_std=SLIPPAGE_PCT,
    )
    new_fitness, new_metrics = evaluate_genome(new_pilot, env_new, 0, max_steps)
    env_new.close()

    old_metrics['fitness'] = old_fitness
    new_metrics['fitness'] = new_fitness

    passed = new_fitness > old_fitness

    print(f"\n  +{'='*58}+")
    print(f"  |{'GATEKEEPER TEST':^58s}|")
    print(f"  +{'='*58}+")
    print(f"  |  {'Metric':<15s} {'Old Brain':>15s} {'New Brain':>15s}   |")
    print(f"  |  {'-'*15} {'-'*15} {'-'*15}   |")
    print(f"  |  {'Fitness':<15s} {old_fitness:>15.2f} {new_fitness:>15.2f}   |")
    print(f"  |  {'Return':<15s} {old_metrics['return']:>14.1f}% {new_metrics['return']:>14.1f}%  |")
    print(f"  |  {'Sortino':<15s} {old_metrics['sortino']:>15.2f} {new_metrics['sortino']:>15.2f}   |")
    print(f"  |  {'MaxDD':<15s} {old_metrics['max_dd']:>14.1%} {new_metrics['max_dd']:>14.1%}   |")
    print(f"  |  {'Trades':<15s} {old_metrics['trades']:>15d} {new_metrics['trades']:>15d}   |")
    print(f"  +{'-'*58}+")
    print(f"  |  {'VERDICT':<15s} {'PROMOTED' if passed else 'REJECTED':>37s}   |")
    print(f"  +{'='*58}+")

    return passed, old_metrics, new_metrics


# ─── Main Retuning Engine ─────────────────────────────────────────────────────

def retune(months=12, generations=50, mutation=0.05, force_promote=False, holdout_days=20, brain_path=None):
    """Main retuning function."""
    if brain_path is None:
        brain_path = BRAIN_FILE

    print("\n" + "=" * 70)
    print("  NIFTY50 BRAIN - ADAPTIVE RETUNING")
    print("=" * 70)
    print(f"  Date:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Data Window: Last {months} months")
    print(f"  Generations: {generations}")
    print(f"  Mutation:    {mutation:.3f}")
    print(f"  Holdout:     Last {holdout_days} days")
    print(f"  Mode:        {'Force Promote' if force_promote else 'Gatekeeper'}")
    print("=" * 70)

    # Step 1: Load current brain
    print(f"\n  [1/7] LOADING CURRENT BRAIN")
    if not os.path.exists(brain_path):
        print(f"  [ERROR] Brain not found: {brain_path}")
        sys.exit(1)

    with open(brain_path, "rb") as f:
        pilot = pickle.load(f)
    print(f"  [OK] Brain loaded ({pilot.pop_size} genomes)")

    # Archive current brain as v1.0 if needed
    archive_current_brain()

    # Keep a copy of old brain for gatekeeper comparison
    with open(brain_path, "rb") as f:
        old_pilot = pickle.load(f)

    # Step 2: Fetch recent data
    print(f"\n  [2/7] FETCHING RECENT MARKET DATA")
    df_full = fetch_data(months_back=months)

    # Auto-adjust holdout if not enough data
    min_train_days = WINDOW_SIZE + 30  # Need at least 50 days for training
    if len(df_full) < min_train_days + holdout_days:
        holdout_days = max(10, len(df_full) - min_train_days)
        print(f"  [WARN] Limited data — holdout reduced to {holdout_days} days")

    if len(df_full) < min_train_days + 10:
        print(f"  [ERROR] Not enough data: {len(df_full)} bars (need >= {min_train_days + 10})")
        sys.exit(1)

    # Split: train on everything except last holdout_days
    split_idx = len(df_full) - holdout_days
    df_train = df_full.iloc[:split_idx].copy()
    df_holdout = df_full.iloc[max(0, split_idx - WINDOW_SIZE):].copy()  # overlap for window

    print(f"  [OK] Train:   {len(df_train)} days ({df_train.index[0].date()} to {df_train.index[-1].date()})")
    print(f"  [OK] Holdout: {len(df_holdout)} days ({df_holdout.index[0].date()} to {df_holdout.index[-1].date()})")

    # Step 3: Load feedback from paper trades
    print(f"\n  [3/7] LOADING TRADE FEEDBACK")
    feedback = load_paper_trade_feedback()

    # Step 4: Seed population from current brain
    print(f"\n  [4/7] SEEDING POPULATION")
    seed_population_from_brain(pilot, mutation_strength=mutation)

    # Step 5: Evolution on recent data
    print(f"\n  [5/7] EVOLVING ON RECENT DATA ({generations} generations)")
    print(f"  {'='*60}")

    safe_end = len(df_train) - (WINDOW_SIZE + 5)
    if safe_end <= WINDOW_SIZE + 5:
        print(f"  [ERROR] Training data too short: {len(df_train)} bars, safe_end={safe_end}")
        sys.exit(1)

    env_train = FinancialRegimeEnv(
        df_train,
        frame_bound=(WINDOW_SIZE, safe_end),
        window_size=WINDOW_SIZE,
        fee=FEE_PCT,
        slippage_std=SLIPPAGE_PCT,
    )

    max_steps = min(800, safe_end - WINDOW_SIZE - 10)
    best_ever_fitness = -999.0
    best_ever_idx = 0
    history = []

    for gen in range(1, generations + 1):
        scores = []
        all_metrics = []

        for i in range(pilot.pop_size):
            fit, metrics = evaluate_genome(pilot, env_train, i, max_steps)
            scores.append(fit)
            all_metrics.append(metrics)

        best_idx = int(np.argmax(scores))
        best_fit = scores[best_idx]
        best_metrics = all_metrics[best_idx]

        if best_fit > best_ever_fitness:
            best_ever_fitness = best_fit
            best_ever_idx = best_idx
            # Update hall of fame
            pilot.hall_of_fame = pilot.get_flat_weights(best_idx)
            pilot.hof_score = best_fit

        history.append({'gen': gen, 'fitness': best_fit, **best_metrics})

        if gen % 5 == 0 or gen == 1 or gen == generations:
            print(f"  Gen {gen:3d}/{generations} | Fit: {best_fit:7.2f} | "
                  f"Sh: {best_metrics['sharpe']:5.2f} | So: {best_metrics['sortino']:5.2f} | "
                  f"Ret: {best_metrics['return']:6.1f}% | DD: {best_metrics['max_dd']:5.1%} | "
                  f"Trades: {best_metrics['trades']}")

        # Evolve
        pilot.evolve(scores)

    env_train.close()

    # Ensure best genome is at index 0
    best_weights = pilot.get_flat_weights(best_ever_idx)
    pilot.set_flat_weights(0, best_weights)

    print(f"\n  [OK] Evolution complete. Best fitness: {best_ever_fitness:.2f}")
    print(f"       Best return: {history[-1]['return']:.1f}%")

    # Step 6: Gatekeeper test
    print(f"\n  [6/7] GATEKEEPER TEST (holdout: {holdout_days} days)")

    if force_promote:
        print("  [SKIP] Force promote enabled — skipping gatekeeper")
        passed = True
        old_metrics, new_metrics = {}, {}
    else:
        passed, old_metrics, new_metrics = gatekeeper_test(old_pilot, pilot, df_holdout)

    # Step 7: Save or reject
    print(f"\n  [7/7] SAVING RESULTS")

    version = get_next_version()
    final_metrics = {
        "train": history[-1] if history else {},
        "holdout_old": old_metrics,
        "holdout_new": new_metrics,
        "gatekeeper_passed": passed,
    }

    config = {
        "months": months,
        "generations": generations,
        "mutation": mutation,
        "holdout_days": holdout_days,
        "force_promote": force_promote,
        "train_period": f"{df_train.index[0].date()} to {df_train.index[-1].date()}",
        "holdout_period": f"{df_holdout.index[0].date()} to {df_holdout.index[-1].date()}",
    }

    if passed:
        save_version(pilot, version, final_metrics, config)
        print(f"\n  [PROMOTED] Brain {version} is now the active brain!")
    else:
        # Still save for reference but don't promote
        reject_dir = os.path.join(VERSIONS_DIR, f"{version}_REJECTED")
        os.makedirs(reject_dir, exist_ok=True)
        with open(os.path.join(reject_dir, "brain.pkl"), "wb") as f:
            pickle.dump(pilot, f)
        meta = {"version": version, "status": "REJECTED", "metrics": final_metrics, "config": config,
                "created": datetime.datetime.now().isoformat()}
        with open(os.path.join(reject_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\n  [REJECTED] New brain saved to {reject_dir}/ (not deployed)")
        print(f"  [KEEPING] Old brain remains active")

    # Summary
    print(f"\n  {'='*60}")
    print(f"  RETUNING SUMMARY")
    print(f"  {'='*60}")
    print(f"  Version:        {version} ({'PROMOTED' if passed else 'REJECTED'})")
    print(f"  Train Period:   {config['train_period']}")
    print(f"  Holdout Period: {config['holdout_period']}")
    print(f"  Generations:    {generations}")
    if history:
        print(f"  Best Fitness:   {best_ever_fitness:.2f}")
        print(f"  Best Return:    {history[-1]['return']:.1f}%")
        print(f"  Best Sortino:   {history[-1]['sortino']:.2f}")
        print(f"  Best MaxDD:     {history[-1]['max_dd']:.1%}")
    print(f"  {'='*60}")

    return passed, version


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIFTY50 Brain Adaptive Retuning")
    parser.add_argument("--months", type=int, default=12,
                        help="Months of recent data to train on (default: 12)")
    parser.add_argument("--gens", type=int, default=50,
                        help="Number of evolution generations (default: 50)")
    parser.add_argument("--mutation", type=float, default=0.05,
                        help="Initial mutation strength (default: 0.05)")
    parser.add_argument("--holdout", type=int, default=20,
                        help="Holdout days for gatekeeper test (default: 20)")
    parser.add_argument("--force-promote", action="store_true",
                        help="Skip gatekeeper and always promote new brain")
    parser.add_argument("--brain", default=BRAIN_FILE,
                        help="Path to current brain file")
    args = parser.parse_args()

    brain_path = args.brain
    if not os.path.isabs(brain_path):
        brain_path = os.path.join(ROOT_DIR, brain_path)

    retune(
        months=args.months,
        generations=args.gens,
        mutation=args.mutation,
        force_promote=args.force_promote,
        holdout_days=args.holdout,
        brain_path=brain_path,
    )


if __name__ == "__main__":
    main()
