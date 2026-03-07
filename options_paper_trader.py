"""
+============================================================================+
|                  NIFTY50 OPTIONS AI - LIVE PAPER TRADER                    |
|                  Greeks-Aware · Multi-Leg · RenTech-Scored                 |
+============================================================================+

Usage:
    python options_paper_trader.py                     # Today's entry signal
    python options_paper_trader.py --exit              # Exit/adjustment check
    python options_paper_trader.py --dashboard         # Full portfolio view
    python options_paper_trader.py --backfill 30       # Simulate last 30 days
    python options_paper_trader.py --reset             # Reset trade log
"""

import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, pickle, json, argparse, datetime, warnings, math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import yfinance as yf

# ── Import options engine (reuse all logic from cloud signal) ─────────────────
from options_cloud_signal import (
    add_indicators, get_directional_signal, compute_ivr, classify_regime,
    select_strategy, compute_rentech_score, get_nearest_expiry_T,
    load_log, save_log, update_log, build_entry_message, build_exit_message,
    FEATURE_COLS, ACTION_NAMES, ACTION_EMOJI, REGIME_EMOJI,
    TICKER, VIX_TICKER, BRAIN_FILE, TRADE_LOG_FILE,
    WINDOW_SIZE, LOOKBACK_DAYS, INITIAL_CAPITAL, RISK_FREE_RATE,
)

# ── Core fetch + signal pipeline ──────────────────────────────────────────────
def fetch_data(days_back: int = LOOKBACK_DAYS + 50):
    end   = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=int(days_back * 1.6))
    print(f"  [FETCH] {TICKER} + {VIX_TICKER}  {start} → {end}...")

    df_nifty = yf.download(TICKER, start=str(start), end=str(end),
                            interval="1d", progress=False)
    df_vix   = yf.download(VIX_TICKER, start=str(start), end=str(end),
                            interval="1d", progress=False)

    for df in [df_nifty, df_vix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

    df_nifty = add_indicators(df_nifty)
    df_nifty.dropna(inplace=True)

    print(f"  [OK] {len(df_nifty)} bars  |  "
          f"NIFTY ₹{df_nifty['Close'].iloc[-1]:,.2f}  "
          f"VIX {df_vix['Close'].iloc[-1]:.2f}")
    return df_nifty, df_vix

def run_signal_pipeline(brain, df_nifty, df_vix, log):
    """Full pipeline: direction → regime → strategy → score."""
    prev_action = {v: k for k, v in ACTION_NAMES.items()}.get(
        log["daily_signals"][-1]["action"] if log["daily_signals"] else "NEUTRAL", 1)

    action, probs = get_directional_signal(brain, df_nifty, prev_action)
    spot          = float(df_nifty["Close"].iloc[-1])
    vix           = float(df_vix["Close"].iloc[-1]) if len(df_vix) > 0 else 15.0
    ivr           = compute_ivr(df_vix["Close"])
    regime        = classify_regime(ivr, vix)
    sigma         = vix / 100.0
    T             = get_nearest_expiry_T(1)

    strategy = select_strategy(action, regime, spot, sigma, T, RISK_FREE_RATE)
    score    = compute_rentech_score(probs, df_nifty, ivr, regime)

    return action, probs, spot, vix, ivr, regime, strategy, score


# ── Commands ─────────────────────────────────────────────────────────────────
def cmd_signal(brain, log, mode="entry"):
    print("\n" + "="*70)
    print("  NIFTY50 OPTIONS AI — TODAY'S SIGNAL")
    print("="*70)

    df_nifty, df_vix = fetch_data()
    today = df_nifty.index[-1].date().isoformat()

    action, probs, spot, vix, ivr, regime, strategy, score = \
        run_signal_pipeline(brain, df_nifty, df_vix, log)

    log = update_log(log, today, action, spot, strategy, score, regime, vix, ivr)
    save_log(log)

    # ── Display ──────────────────────────────────────────────────────────────
    print(f"\n  Date:        {today}")
    print(f"  NIFTY50:     ₹{spot:,.2f}")
    print(f"  India VIX:   {vix:.2f}  ({REGIME_EMOJI[regime]})")
    print(f"  IV Rank:     {ivr*100:.0f}th percentile")
    print(f"\n  AI Signal:   {ACTION_EMOJI[action]}")
    print(f"  Probs:       SHORT={probs[0]:.2%}  NEUTRAL={probs[1]:.2%}  "
          f"LONG={probs[2]:.2%}")
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Strategy:  {strategy['name']:<42}│")
    print(f"  │  {strategy['description']:<52}│")
    print(f"  └─────────────────────────────────────────────────────┘")

    if strategy["legs"]:
        print(f"\n  Legs:")
        for leg in strategy["legs"]:
            g = leg.get("greeks", {})
            arrow = "  ▲ LONG " if leg["position"] == "LONG" else "  ▼ SHORT"
            print(f"  {arrow} {leg['type'].upper():4s} @{int(leg['strike'])}  "
                  f"δ={g.get('delta',0):+.4f}  θ={g.get('theta',0):+.2f}/d  "
                  f"ν={g.get('vega',0):+.2f}/%  price≈₹{g.get('price',0):.2f}")
    else:
        print(f"\n  ⛔ {strategy.get('caution', 'No trade — skip this session')}")

    print(f"\n  Portfolio Greeks:")
    print(f"    Δ Net Delta:  {strategy['net_delta']:+.2f}")
    print(f"    Θ Net Theta:  {strategy['net_theta']:+.2f} / day  "
          f"({'earning' if strategy['net_theta'] > 0 else 'paying'} decay)")
    print(f"    ν Net Vega:   {strategy['net_vega']:+.2f} / 1% IV")
    print(f"    Net Premium:  ₹{strategy['net_premium']:+,.2f}  "
          f"({'credit' if strategy['net_premium'] < 0 else 'debit'})")

    print(f"\n  RenTech Score:  {score['composite']:+.3f}  {score['conviction']}")
    print(f"  {score['score_bar']}")

    pnl = ((log["current_capital"] / log["initial_capital"]) - 1) * 100
    print(f"\n  Portfolio:  ₹{log['current_capital']:,.2f}  ({pnl:+.2f}%)")
    if log.get("open_position"):
        op = log["open_position"]
        print(f"  Open Trade: {op['strategy']}  (entered {op['date_entered']})")

    print(f"\n  Trade log: {TRADE_LOG_FILE}")
    print("="*70)


def cmd_exit_check(brain, log):
    print("\n" + "="*70)
    print("  NIFTY50 OPTIONS AI — EXIT CHECK")
    print("="*70)

    df_nifty, df_vix = fetch_data()
    today = df_nifty.index[-1].date().isoformat()
    action, probs, spot, vix, ivr, regime, strategy, score = \
        run_signal_pipeline(brain, df_nifty, df_vix, log)

    op = log.get("open_position")
    if not op:
        print(f"\n  ℹ️  No open position to evaluate.")
        print(f"  NIFTY: ₹{spot:,.2f}  |  Regime: {REGIME_EMOJI[regime]}")
        print("="*70)
        return

    spot_entry = op["spot_at_entry"]
    move_pct   = (spot - spot_entry) / spot_entry * 100
    theta_gain = abs(op.get("net_theta", 0))
    pnl_est    = op.get("net_delta", 0) * (spot - spot_entry) + theta_gain

    exit_now = abs(score["composite"]) < 0.1 or abs(move_pct) > 3.0
    reason = ("Score reversed to neutral" if abs(score["composite"]) < 0.1 else
              "3% stop-loss triggered" if abs(move_pct) > 3.0 else
              "Trend intact — hold")

    print(f"\n  Date:        {today}")
    print(f"  NIFTY50:     ₹{spot:,.2f}  ({move_pct:+.1f}% from entry)")
    print(f"  Regime:      {REGIME_EMOJI[regime]}")
    print(f"\n  Open Trade:  {op['strategy']}")
    print(f"  Entered:     {op['date_entered']}  @  ₹{spot_entry:,.2f}")
    print(f"  Est. P&L:    ₹{pnl_est:+,.0f}")
    print(f"\n  Score:       {score['composite']:+.3f}  {score['conviction']}")
    print(f"\n  ► {'⛔ CLOSE POSITION' if exit_now else '✅ HOLD POSITION'}")
    print(f"    Reason: {reason}")
    print("="*70)


def cmd_dashboard(log):
    print("\n" + "="*70)
    print("  NIFTY50 OPTIONS AI — PORTFOLIO DASHBOARD")
    print("="*70)

    cap   = log["current_capital"]
    init  = log["initial_capital"]
    pnl   = ((cap / init) - 1) * 100
    stats = log.get("stats", {})

    print(f"\n  Initial Capital:  ₹{init:>12,.2f}")
    print(f"  Current Capital:  ₹{cap:>12,.2f}")
    print(f"  Total P&L:        {pnl:>+8.2f}%")
    print(f"  Total Trades:     {stats.get('total_trades', 0)}")
    print(f"  Win Rate:         "
          f"{stats.get('winning_trades',0)}/{max(stats.get('total_trades',1),1)*100:.0f}%")

    # Open position
    op = log.get("open_position")
    if op:
        print(f"\n  ┌── OPEN POSITION ──────────────────────────────────────┐")
        print(f"  │  {op['strategy']:<53}│")
        print(f"  │  Entered: {op['date_entered']}  Regime: {op['regime']:<23}│")
        print(f"  │  Spot@entry: ₹{op['spot_at_entry']:,.2f}  "
              f"ATM: {int(op['atm_at_entry'])}  {'':20}│")
        print(f"  │  Δ={op['net_delta']:+.2f}  Θ={op['net_theta']:+.2f}/d  "
              f"ν={op['net_vega']:+.2f}/%IV  {'':19}│")
        print(f"  └──────────────────────────────────────────────────────┘")
    else:
        print(f"\n  Open Position:    None (cash)")

    # Recent signals
    signals = log["daily_signals"]
    if signals:
        print(f"\n  {'Date':>12}  {'NIFTY':>10}  {'Signal':>8}  "
              f"{'Regime':>10}  {'Strategy':>25}  {'Score':>7}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*25}  {'-'*7}")
        for s in signals[-12:]:
            print(f"  {s['date']:>12}  {s['price']:>10,.2f}  {s['action']:>8}  "
                  f"{s['regime']:>10}  {s.get('strategy','—')[:25]:>25}  "
                  f"{s.get('composite_score', 0):>+7.3f}")

    # Equity curve (last 5)
    ec = log.get("equity_curve", [])
    if ec:
        print(f"\n  Recent Equity Curve:")
        print(f"  {'Date':>12}  {'Equity':>14}  {'Regime':>10}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*10}")
        for e in ec[-5:]:
            print(f"  {e['date']:>12}  ₹{e['equity']:>12,.2f}  "
                  f"{e.get('regime','—'):>10}")

    print(f"\n  Log: {TRADE_LOG_FILE}")
    print("="*70)


def cmd_backfill(brain, days: int):
    """Simulate options signals over last N market days."""
    print("\n" + "="*70)
    print(f"  NIFTY50 OPTIONS AI — BACKFILL ({days} DAYS)")
    print("="*70)

    # Fresh log
    log = {
        "initial_capital": INITIAL_CAPITAL,
        "current_capital": INITIAL_CAPITAL,
        "open_position":   None,
        "trades":          [],
        "daily_signals":   [],
        "equity_curve":    [],
        "stats": {"total_trades": 0, "winning_trades": 0, "total_pnl": 0.0},
    }

    df_nifty, df_vix = fetch_data(days_back=days + LOOKBACK_DAYS + 30)
    if len(df_nifty) < days + WINDOW_SIZE + 10:
        days = len(df_nifty) - WINDOW_SIZE - 10
        print(f"  [WARN] Adjusted to {days} available days")

    start_idx = max(WINDOW_SIZE + 5, len(df_nifty) - days)
    trade_dates = df_nifty.index[start_idx:]

    print(f"  Period: {trade_dates[0].date()} → {trade_dates[-1].date()}")
    print(f"  Days:   {len(trade_dates)}\n")
    print(f"  {'Date':>12}  {'NIFTY':>10}  {'VIX':>6}  {'Regime':>8}  "
          f"{'Signal':>8}  {'Strategy':>25}  {'Score':>7}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*25}  {'-'*7}")

    for date in trade_dates:
        date_str  = date.date().isoformat()
        iloc_idx  = df_nifty.index.get_loc(date)

        # Slice up to this day's data
        df_slice  = df_nifty.iloc[:iloc_idx + 1]
        if len(df_slice) < WINDOW_SIZE + 5:
            continue

        # VIX slice
        vix_slice = df_vix[df_vix.index <= date]
        vix       = float(vix_slice["Close"].iloc[-1]) if len(vix_slice) > 0 else 15.0

        prev_action = {v: k for k, v in ACTION_NAMES.items()}.get(
            log["daily_signals"][-1]["action"] if log["daily_signals"]
            else "NEUTRAL", 1)

        action, probs = get_directional_signal(brain, df_slice, prev_action)
        spot          = float(df_slice["Close"].iloc[-1])
        ivr           = compute_ivr(vix_slice["Close"])
        regime        = classify_regime(ivr, vix)
        sigma         = vix / 100.0
        T             = get_nearest_expiry_T(1)

        strategy = select_strategy(action, regime, spot, sigma, T, RISK_FREE_RATE)
        score    = compute_rentech_score(probs, df_slice, ivr, regime)

        log = update_log(log, date_str, action, spot, strategy, score,
                         regime, vix, ivr)

        print(f"  {date_str:>12}  {spot:>10,.2f}  {vix:>6.2f}  "
              f"{regime:>8}  {ACTION_NAMES[action]:>8}  "
              f"{strategy['name'][:25]:>25}  {score['composite']:>+7.3f}")

    save_log(log)

    # Summary
    pnl = ((log["current_capital"] / log["initial_capital"]) - 1) * 100
    first_price = float(df_nifty.loc[trade_dates[0], "Close"])
    last_price  = float(df_nifty.loc[trade_dates[-1], "Close"])
    bah         = ((last_price / first_price) - 1) * 100

    print(f"\n  {'='*60}")
    print(f"  BACKFILL SUMMARY")
    print(f"  {'='*60}")
    print(f"  Period:           {trade_dates[0].date()} → {trade_dates[-1].date()}")
    print(f"  Trading Days:     {len(trade_dates)}")
    print(f"  Initial Capital:  ₹{INITIAL_CAPITAL:>12,.2f}")
    print(f"  Final Capital:    ₹{log['current_capital']:>12,.2f}")
    print(f"  Options P&L:      {pnl:>+8.2f}%")
    print(f"  NIFTY B&H:        {bah:>+8.2f}%")
    print(f"  Alpha vs B&H:     {pnl - bah:>+8.2f}%")
    print(f"  Total Trades:     {log['stats']['total_trades']}")
    print(f"\n  Trade log saved: {TRADE_LOG_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NIFTY50 Options AI · Local Paper Trader")
    parser.add_argument("--exit",      action="store_true",
                        help="Run exit/adjustment check (near-market-close)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Show full portfolio dashboard")
    parser.add_argument("--backfill",  type=int, default=0,
                        help="Backfill last N trading days")
    parser.add_argument("--reset",     action="store_true",
                        help="Reset options trade log")
    parser.add_argument("--brain",     default=BRAIN_FILE,
                        help="Path to brain pickle file")
    args = parser.parse_args()

    brain_path = args.brain if os.path.isabs(args.brain) \
        else os.path.join(ROOT_DIR, args.brain)

    if not os.path.exists(brain_path):
        print(f"[ERROR] Brain not found: {brain_path}")
        sys.exit(1)

    with open(brain_path, "rb") as f:
        brain = pickle.load(f)

    if args.reset:
        if os.path.exists(TRADE_LOG_FILE):
            os.remove(TRADE_LOG_FILE)
            print("[OK] Options trade log reset.")

    log = load_log()

    if args.dashboard:
        cmd_dashboard(log)
    elif args.backfill > 0:
        cmd_backfill(brain, args.backfill)
    elif args.exit:
        cmd_exit_check(brain, log)
    else:
        cmd_signal(brain, log)


if __name__ == "__main__":
    main()
