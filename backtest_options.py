"""
backtest_options.py — Deep Options Backtest & Analytics
=======================================================
Runs a full options paper-trade simulation and outputs a comprehensive
performance report: returns, Greeks tracking, Sharpe, drawdown,
win rate, regime breakdown, strategy breakdown, and more.

Usage: python backtest_options.py [--days N]   (default 252 = ~1 trading year)
"""

import os, sys, io, json, pickle, datetime, warnings, math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import yfinance as yf
from options_cloud_signal import (
    add_indicators, get_directional_signal, compute_ivr, classify_regime,
    select_strategy, compute_rentech_score, get_nearest_expiry_T,
    ACTION_NAMES, FEATURE_COLS,
    TICKER, VIX_TICKER, BRAIN_FILE,
    WINDOW_SIZE, LOOKBACK_DAYS, INITIAL_CAPITAL, RISK_FREE_RATE, LOT_SIZE, FEE_PCT,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=252)
args = parser.parse_args()
N_DAYS = args.days

print(f"\n{'='*70}")
print(f"  EVOTRADER OPTIONS AI — DEEP BACKTEST  ({N_DAYS} trading days)")
print(f"  Running till yesterday: {(datetime.date.today()-datetime.timedelta(days=1)).isoformat()}")
print(f"{'='*70}\n")

# ── 1. Load brain ─────────────────────────────────────────────────────────────
with open(BRAIN_FILE, "rb") as f:
    brain = pickle.load(f)
print(f"[1/4] Brain loaded: {brain['input_dim']}->{brain['layer_sizes']}->{brain['output_dim']}")

# ── 2. Fetch data ─────────────────────────────────────────────────────────────
end   = datetime.date.today()
start = end - datetime.timedelta(days=int((N_DAYS + LOOKBACK_DAYS + 50) * 1.6))

print(f"[2/4] Fetching NIFTY50 + India VIX ({start} → {end})...")
df_nifty = yf.download(TICKER,     start=str(start), end=str(end), interval="1d", progress=False)
df_vix   = yf.download(VIX_TICKER, start=str(start), end=str(end), interval="1d", progress=False)

for df in [df_nifty, df_vix]:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:               df.index = df.index.tz_convert(None)

df_nifty = add_indicators(df_nifty)
df_nifty.dropna(inplace=True)
print(f"       {len(df_nifty)} clean bars  |  VIX bars: {len(df_vix)}")

# ── 3. Backtest loop ──────────────────────────────────────────────────────────
print(f"[3/4] Simulating {N_DAYS} trading days...\n")

start_idx   = max(WINDOW_SIZE + 5, len(df_nifty) - N_DAYS)
trade_dates = df_nifty.index[start_idx:]

capital    = float(INITIAL_CAPITAL)
position   = 1          # current AI position
records    = []         # one row per day
equity_curve = [capital]

# Track open options leg for P&L
open_leg = None          # {entry_price, delta, theta, regime, strategy, T_at_entry}

for i, date in enumerate(trade_dates):
    iloc_idx = df_nifty.index.get_loc(date)
    df_slice = df_nifty.iloc[:iloc_idx + 1]
    if len(df_slice) < WINDOW_SIZE + 5:
        continue

    vix_slice = df_vix[df_vix.index <= date]
    vix       = float(vix_slice["Close"].iloc[-1]) if len(vix_slice) > 0 else 15.0
    spot      = float(df_slice["Close"].iloc[-1])
    ivr       = compute_ivr(vix_slice["Close"])
    regime    = classify_regime(ivr, vix)
    sigma     = vix / 100.0
    T         = get_nearest_expiry_T(1)

    action, probs = get_directional_signal(brain, df_slice, position)
    strategy      = select_strategy(action, regime, spot, sigma, T, RISK_FREE_RATE)
    score         = compute_rentech_score(probs, df_slice, ivr, regime)

    # ── Options P&L simulation ─────────────────────────────────────────────
    day_pnl = 0.0

    if open_leg is not None:
        # Time decay earned/paid (Theta * 1 day)
        theta_pnl = open_leg["net_theta"]   # already per day, per lot
        # Delta P&L from price move
        prev_spot  = open_leg["spot"]
        delta_pnl  = open_leg["net_delta"] * (spot - prev_spot)
        day_pnl    = theta_pnl + delta_pnl
        # Cap loss at 2% of capital (hard stop)
        max_loss   = -capital * 0.02
        day_pnl    = max(day_pnl, max_loss)
        capital   += day_pnl

    # Close old position and open new one every week (weekly options)
    week_num = date.isocalendar()[1]
    if open_leg is None or week_num != open_leg.get("week_num", -1):
        if open_leg is not None and strategy["legs"]:
            # Transaction cost
            capital -= capital * FEE_PCT
        if strategy["legs"]:
            open_leg = {
                "net_delta": strategy["net_delta"],
                "net_theta": strategy["net_theta"],
                "spot":      spot,
                "week_num":  week_num,
                "strategy":  strategy["name"],
                "regime":    regime,
            }
        else:
            open_leg = None   # CASH — no trade

    capital = max(capital, 1.0)
    equity_curve.append(capital)
    position = action

    records.append({
        "date":          date.date().isoformat(),
        "spot":          round(spot, 2),
        "vix":           round(vix, 2),
        "ivr":           round(ivr, 3),
        "regime":        regime,
        "action":        ACTION_NAMES[action],
        "strategy":      strategy["name"],
        "net_delta":     strategy["net_delta"],
        "net_theta":     strategy["net_theta"],
        "net_vega":      strategy["net_vega"],
        "net_premium":   strategy["net_premium"],
        "day_pnl":       round(day_pnl, 2),
        "capital":       round(capital, 2),
        "score":         score["composite"],
        "conviction":    score["conviction"],
        "probs_long":    round(float(probs[2]), 3),
        "probs_short":   round(float(probs[0]), 3),
    })

df_bt = pd.DataFrame(records)

# ── 4. Analytics ─────────────────────────────────────────────────────────────
print(f"[4/4] Computing analytics...\n")

eq = np.array(equity_curve, dtype=float)
daily_rets = np.diff(eq) / eq[:-1]

# Core metrics
total_return   = (capital / INITIAL_CAPITAL - 1) * 100
ann_return     = ((capital / INITIAL_CAPITAL) ** (252 / len(records)) - 1) * 100 if len(records) > 0 else 0
sharpe         = (daily_rets.mean() / daily_rets.std() * math.sqrt(252)) if daily_rets.std() > 0 else 0
sortino_down   = daily_rets[daily_rets < 0].std()
sortino        = (daily_rets.mean() / sortino_down * math.sqrt(252)) if sortino_down > 0 else 0

# Drawdown
peak     = np.maximum.accumulate(eq)
dd_arr   = (eq - peak) / peak * 100
max_dd   = float(dd_arr.min())
avg_dd   = float(dd_arr[dd_arr < 0].mean()) if (dd_arr < 0).any() else 0.0

# Drawdown duration
in_dd = False
dd_start = 0
dd_durations = []
for i, d in enumerate(dd_arr):
    if d < 0 and not in_dd:
        in_dd = True
        dd_start = i
    elif d >= 0 and in_dd:
        dd_durations.append(i - dd_start)
        in_dd = False
max_dd_dur = max(dd_durations) if dd_durations else 0

# Win / Loss
prof_days  = (df_bt["day_pnl"] > 0).sum()
loss_days  = (df_bt["day_pnl"] < 0).sum()
flat_days  = (df_bt["day_pnl"] == 0).sum()
win_rate   = prof_days / max(prof_days + loss_days, 1) * 100

avg_win    = df_bt[df_bt["day_pnl"] > 0]["day_pnl"].mean() if prof_days > 0 else 0
avg_loss   = df_bt[df_bt["day_pnl"] < 0]["day_pnl"].mean() if loss_days > 0 else 0
profit_factor = (avg_win * prof_days) / max(abs(avg_loss * loss_days), 1)

# Calmar ratio
calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

# Buy & Hold benchmark
first_price = float(df_nifty["Close"].iloc[start_idx])
last_price  = float(df_nifty["Close"].iloc[-1])
bah_return  = (last_price / first_price - 1) * 100
bah_ann     = ((last_price / first_price) ** (252 / len(records)) - 1) * 100

# NIFTY B&H daily returns for Sharpe comparison
nifty_prices = df_nifty["Close"].iloc[start_idx:].values
nifty_rets   = np.diff(nifty_prices) / nifty_prices[:-1]
bah_sharpe   = (nifty_rets.mean() / nifty_rets.std() * math.sqrt(252)) if nifty_rets.std() > 0 else 0

# Regime breakdown
regime_stats = df_bt.groupby("regime").agg(
    days=("date", "count"),
    avg_score=("score", "mean"),
    total_pnl=("day_pnl", "sum"),
    win_days=("day_pnl", lambda x: (x > 0).sum()),
).reset_index()

# Strategy breakdown
strat_stats = df_bt.groupby("strategy").agg(
    days=("date", "count"),
    total_pnl=("day_pnl", "sum"),
    avg_theta=("net_theta", "mean"),
    avg_delta=("net_delta", "mean"),
    win_days=("day_pnl", lambda x: (x > 0).sum()),
).reset_index()
strat_stats["win_rate"] = strat_stats["win_days"] / strat_stats["days"] * 100

# AI direction accuracy (did AI direction align with actual move?)
df_bt["next_move"] = df_nifty["Close"].iloc[start_idx:].diff().shift(-1).values[:len(df_bt)]
df_bt["direction_correct"] = (
    ((df_bt["action"] == "LONG")    & (df_bt["next_move"] > 0)) |
    ((df_bt["action"] == "SHORT")   & (df_bt["next_move"] < 0)) |
    ((df_bt["action"] == "NEUTRAL") & (df_bt["next_move"].abs() < df_bt["next_move"].abs().median()))
)
direction_acc = df_bt["direction_correct"].mean() * 100

# Score → outcome correlation
score_outcome_corr = df_bt[["score", "day_pnl"]].corr().iloc[0, 1]

# VIX / IV correlation
vix_pnl_corr = df_bt[["vix", "day_pnl"]].corr().iloc[0, 1]

# ── Print Report ──────────────────────────────────────────────────────────────
sep  = "─" * 70
sep2 = "═" * 70

print(sep2)
print("  EVOTRADER OPTIONS AI — FULL BACKTEST REPORT")
print(f"  Period: {df_bt['date'].iloc[0]}  →  {df_bt['date'].iloc[-1]}  ({len(df_bt)} trading days)")
print(sep2)

print("\n  ┌─── RETURNS ────────────────────────────────────────────────────┐")
print(f"  │  Total Return:        {total_return:>+8.2f}%  (AI Options)          │")
print(f"  │  Annualised Return:   {ann_return:>+8.2f}%                          │")
print(f"  │  NIFTY B&H Return:   {bah_return:>+8.2f}%  (benchmark)            │")
print(f"  │  NIFTY B&H Ann.:     {bah_ann:>+8.2f}%                          │")
print(f"  │  Alpha vs B&H:       {total_return - bah_return:>+8.2f}%                          │")
print(f"  └────────────────────────────────────────────────────────────────┘")

print("\n  ┌─── RISK-ADJUSTED METRICS ────────────────────────────────────────┐")
print(f"  │  Sharpe Ratio:       {sharpe:>8.3f}   (B&H Sharpe: {bah_sharpe:.3f})         │")
print(f"  │  Sortino Ratio:      {sortino:>8.3f}                              │")
print(f"  │  Calmar Ratio:       {calmar:>8.3f}   (Ann.Ret / Max DD)          │")
print(f"  │  Max Drawdown:       {max_dd:>8.2f}%                             │")
print(f"  │  Avg Drawdown:       {avg_dd:>8.2f}%                             │")
print(f"  │  Max DD Duration:    {max_dd_dur:>8d}  days                        │")
print(f"  └────────────────────────────────────────────────────────────────┘")

print("\n  ┌─── WIN / LOSS STATISTICS ────────────────────────────────────────┐")
print(f"  │  Win Rate:           {win_rate:>8.2f}%  ({prof_days}W / {loss_days}L / {flat_days}flat)        │")
print(f"  │  Avg Win Day:        Rs {avg_win:>+9,.0f}                         │")
print(f"  │  Avg Loss Day:       Rs {avg_loss:>+9,.0f}                         │")
print(f"  │  Profit Factor:      {profit_factor:>8.3f}   (>1.5 is institutional)       │")
print(f"  └────────────────────────────────────────────────────────────────┘")

print("\n  ┌─── AI SIGNAL QUALITY ────────────────────────────────────────────┐")
print(f"  │  Direction Accuracy: {direction_acc:>8.2f}%  (vs 50% random)          │")
print(f"  │  Score→PnL Corr.:    {score_outcome_corr:>8.4f}   (>0 = score predicts pnl)    │")
print(f"  │  VIX→PnL Corr.:      {vix_pnl_corr:>8.4f}   (<0 = high-IV helps us)     │")
print(f"  └────────────────────────────────────────────────────────────────┘")

print("\n  ┌─── CAPITAL SUMMARY ──────────────────────────────────────────────┐")
print(f"  │  Initial Capital:   Rs {INITIAL_CAPITAL:>12,.0f}                    │")
print(f"  │  Final Capital:     Rs {capital:>12,.2f}                    │")
print(f"  │  Peak Capital:      Rs {eq.max():>12,.2f}                    │")
print(f"  │  Trough Capital:    Rs {eq.min():>12,.2f}                    │")
print(f"  └────────────────────────────────────────────────────────────────┘")

print(f"\n  ─── REGIME BREAKDOWN ────────────────────────────────────────────")
print(f"  {'Regime':<12}  {'Days':>6}  {'Win Days':>9}  {'Win%':>7}  {'Total PnL':>14}  {'Avg Score':>10}")
print(f"  {'-'*12}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*14}  {'-'*10}")
for _, r in regime_stats.iterrows():
    wr = r["win_days"] / r["days"] * 100 if r["days"] > 0 else 0
    print(f"  {r['regime']:<12}  {r['days']:>6}  {r['win_days']:>9}  "
          f"{wr:>7.1f}%  Rs {r['total_pnl']:>11,.0f}  {r['avg_score']:>+10.3f}")

print(f"\n  ─── STRATEGY BREAKDOWN ──────────────────────────────────────────")
print(f"  {'Strategy':<30}  {'Days':>5}  {'WinRate':>8}  {'Total PnL':>14}  {'Avg Theta':>10}")
print(f"  {'-'*30}  {'-'*5}  {'-'*8}  {'-'*14}  {'-'*10}")
for _, s in strat_stats.sort_values("total_pnl", ascending=False).iterrows():
    print(f"  {s['strategy'][:30]:<30}  {s['days']:>5}  {s['win_rate']:>8.1f}%  "
          f"Rs {s['total_pnl']:>11,.0f}  {s['avg_theta']:>+10.2f}")

print(f"\n  ─── ACTION DISTRIBUTION ─────────────────────────────────────────")
action_counts = df_bt["action"].value_counts()
for act, cnt in action_counts.items():
    pct = cnt / len(df_bt) * 100
    bar = "█" * int(pct / 2)
    print(f"  {act:>8}:  {cnt:>4} days ({pct:5.1f}%)  {bar}")

print(f"\n  ─── SCORE DISTRIBUTION ──────────────────────────────────────────")
conv_counts = df_bt["conviction"].value_counts()
for conv, cnt in conv_counts.items():
    pct = cnt / len(df_bt) * 100
    print(f"  {str(conv).replace('🔥','').replace('✅','').replace('🔸','').replace('⚪','').strip():>12}:  "
          f"{cnt:>4} days ({pct:5.1f}%)")

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  OVERALL VERDICT")
print(f"{'='*70}")

grades = []
if ann_return > bah_ann:        grades.append(("Return vs B&H",     "PASS ✅", f"{total_return-bah_return:+.1f}% alpha"))
else:                           grades.append(("Return vs B&H",     "FAIL ❌", f"{total_return-bah_return:+.1f}% alpha"))
if sharpe > 1.0:                grades.append(("Sharpe > 1.0",      "PASS ✅", f"{sharpe:.3f}"))
elif sharpe > 0.5:              grades.append(("Sharpe > 1.0",      "WEAK ⚠️",  f"{sharpe:.3f}"))
else:                           grades.append(("Sharpe > 1.0",      "FAIL ❌", f"{sharpe:.3f}"))
if profit_factor > 1.5:        grades.append(("Profit Factor>1.5", "PASS ✅", f"{profit_factor:.3f}"))
elif profit_factor > 1.0:      grades.append(("Profit Factor>1.5", "WEAK ⚠️",  f"{profit_factor:.3f}"))
else:                           grades.append(("Profit Factor>1.5", "FAIL ❌", f"{profit_factor:.3f}"))
if direction_acc > 55:         grades.append(("Direction Acc>55%", "PASS ✅", f"{direction_acc:.1f}%"))
elif direction_acc > 50:       grades.append(("Direction Acc>55%", "WEAK ⚠️",  f"{direction_acc:.1f}%"))
else:                           grades.append(("Direction Acc>55%", "FAIL ❌", f"{direction_acc:.1f}%"))
if max_dd > -15:               grades.append(("Max DD < 15%",      "PASS ✅", f"{max_dd:.2f}%"))
elif max_dd > -25:             grades.append(("Max DD < 15%",      "WEAK ⚠️",  f"{max_dd:.2f}%"))
else:                           grades.append(("Max DD < 15%",      "FAIL ❌", f"{max_dd:.2f}%"))
if calmar > 1.5:               grades.append(("Calmar > 1.5",      "PASS ✅", f"{calmar:.3f}"))
elif calmar > 0.5:             grades.append(("Calmar > 1.5",      "WEAK ⚠️",  f"{calmar:.3f}"))
else:                           grades.append(("Calmar > 1.5",      "FAIL ❌", f"{calmar:.3f}"))

passes = sum(1 for g in grades if "PASS" in g[1])
weaks  = sum(1 for g in grades if "WEAK" in g[1])
fails  = sum(1 for g in grades if "FAIL" in g[1])

for test, result, detail in grades:
    print(f"  {test:<22}   {result:<10}   {detail}")

overall = "🏆 EXCELLENT" if passes >= 5 else "✅ GOOD" if passes >= 4 else "⚠️ MODERATE" if passes >= 3 else "❌ UNDERPERFORMING"
print(f"\n  Overall Grade:  {overall}  ({passes} PASS / {weaks} WEAK / {fails} FAIL)")
print(f"\n{'='*70}")

# Save results to JSON
results = {
    "period":            f"{df_bt['date'].iloc[0]} to {df_bt['date'].iloc[-1]}",
    "trading_days":      len(df_bt),
    "total_return_pct":  round(total_return, 3),
    "ann_return_pct":    round(ann_return, 3),
    "bah_return_pct":    round(bah_return, 3),
    "alpha_pct":         round(total_return - bah_return, 3),
    "sharpe":            round(sharpe, 4),
    "sortino":           round(sortino, 4),
    "calmar":            round(calmar, 4),
    "max_drawdown_pct":  round(max_dd, 3),
    "win_rate_pct":      round(win_rate, 2),
    "profit_factor":     round(profit_factor, 4),
    "direction_acc_pct": round(direction_acc, 2),
    "final_capital":     round(capital, 2),
    "passes":            passes,
    "weaks":             weaks,
    "fails":             fails,
    "overall_grade":     overall,
    "daily_records":     records,
}
with open(os.path.join(ROOT_DIR, "options_backtest_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Full data saved: options_backtest_results.json")
print("  Done.\n")
