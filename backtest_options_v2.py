"""
backtest_options_v2.py — EvoTrader Options AI v2.0 Deep Backtester
==================================================================
Full regime-by-regime backtest with:
  • Proper mark-to-market P&L:  Δ×move + Θ×days + ½Γ×move²
  • Smart position holding (2–4 weeks, smart exit rules)
  • Per-regime breakdown (LOW / NORMAL / HIGH / EXTREME)
  • Comparison: v2 vs NIFTY B&H
  • All institutional metrics: Sharpe, Sortino, Calmar, Profit Factor

Usage: python backtest_options_v2.py [--days 252]
"""

import os, sys, io, json, pickle, datetime, warnings, math, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import yfinance as yf
from options_cloud_signal import (
    add_indicators, get_directional_signal,
    compute_ivr, compute_vix_momentum, compute_vol_term_structure, classify_regime,
    build_strategy, compute_evo_score, should_exit, evo_position_size,
    get_expiry_T, bs_greeks,
    ACTION_NAMES, FEATURE_COLS,
    TICKER, VIX_TICKER, BRAIN_FILE,
    WINDOW_SIZE, LOOKBACK_DAYS, INITIAL_CAPITAL, RISK_FREE_RATE, LOT_SIZE, FEE_PCT,
    STOP_LOSS_PCT, TARGET_PROFIT_PCT, MIN_HOLD_DAYS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=252)
args   = parser.parse_args()
N      = args.days

print(f"\n{'='*70}")
print(f"  EVOTRADER OPTIONS AI v2.0 — DEEP BACKTEST  ({N} trading days)")
print(f"  Period ends: {(datetime.date.today()-datetime.timedelta(days=1)).isoformat()}")
print(f"  Fixes: Credit-priority | MTM P&L | Smart exit | VIX term-structure")
print(f"{'='*70}\n")

# ── 1. Load brain ─────────────────────────────────────────────────────────────
with open(BRAIN_FILE,"rb") as f: brain=pickle.load(f)
print(f"[1/4] Brain: {brain['input_dim']}→{brain['layer_sizes']}→{brain['output_dim']}")

# ── 2. Fetch data ─────────────────────────────────────────────────────────────
end  = datetime.date.today()
strt = end - datetime.timedelta(days=int((N+LOOKBACK_DAYS+60)*1.7))
print(f"[2/4] Fetching NIFTY50 + VIX  ({strt} → {end})...")
dfn  = yf.download(TICKER,    start=str(strt),end=str(end),interval="1d",progress=False)
dfv  = yf.download(VIX_TICKER,start=str(strt),end=str(end),interval="1d",progress=False)
for d in [dfn,dfv]:
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    if d.index.tz is not None: d.index=d.index.tz_convert(None)
dfn=add_indicators(dfn); dfn.dropna(inplace=True)
print(f"      {len(dfn)} NIFTY bars  |  {len(dfv)} VIX bars")

# ── 3. Backtest loop ──────────────────────────────────────────────────────────
print(f"[3/4] Simulating {N} days with smart position holding...\n")

start_idx   = max(WINDOW_SIZE+5, len(dfn)-N)
trade_dates = dfn.index[start_idx:]
capital     = float(INITIAL_CAPITAL)
prev_action = 1
equity_curve = [capital]
records     = []
closed_trades = []

# Active position tracking
open_pos    = None   # None = in cash
pos_pnl     = 0.0   # cumulative P&L for current open position
pos_entry_cap= capital

print(f"  {'Date':>12}  {'NIFTY':>8}  {'VIX':>5}  {'Regime':>8}  {'Signal':>7}  {'Strategy':>28}  {'DayPnL':>9}  {'Capital':>12}")
print(f"  {'-'*12}  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*28}  {'-'*9}  {'-'*12}")

for date in trade_dates:
    iloc_idx  = dfn.index.get_loc(date)
    df_slice  = dfn.iloc[:iloc_idx+1]
    if len(df_slice) < WINDOW_SIZE+5: continue

    vix_slice = dfv[dfv.index <= date]
    vix       = float(vix_slice["Close"].iloc[-1]) if len(vix_slice)>0 else 15.
    spot      = float(df_slice["Close"].iloc[-1])

    ivr         = compute_ivr(vix_slice["Close"])
    vix_mom     = compute_vix_momentum(vix_slice["Close"])
    term_struct = compute_vol_term_structure(vix_slice["Close"])
    regime, regime_conf = classify_regime(ivr, vix, vix_mom, term_struct)
    sigma       = vix/100.

    action, probs = get_directional_signal(brain, df_slice, prev_action)
    brain_conf    = float(max(probs))
    lots = evo_position_size(capital, brain_conf, regime_conf, regime)

    strategy = build_strategy(action, regime, spot, sigma, RISK_FREE_RATE, lots=lots)
    score    = compute_evo_score(probs, df_slice, ivr, regime, vix_mom)

    # ── MTM P&L for open position: Taylor-series approximation ──────────────
    day_pnl = 0.0
    action_taken = "HOLD"

    if open_pos is not None:
        prev_spot = open_pos.get("last_spot", open_pos["entry_spot"])
        move      = spot - prev_spot

        # Second-order Taylor: ΔP ≈ δ×ΔS + θ×Δt + ½γ×ΔS²
        delta_pnl = open_pos["net_delta"] * move
        theta_pnl = open_pos["net_theta"]           # per calendar day
        gamma_pnl = 0.5 * open_pos["net_gamma"] * (move**2)

        day_pnl   = delta_pnl + theta_pnl + gamma_pnl
        pos_pnl  += day_pnl
        open_pos["days_held"] += 1
        open_pos["last_spot"]  = spot

        # Hard stop-loss: cap single-day loss at 2% of capital
        max_single_day = -capital * 0.02
        day_pnl = max(day_pnl, max_single_day)
        capital = max(capital + day_pnl, 1.)

        # Check exit conditions
        max_risk = abs(open_pos.get("net_premium",1.))
        pnl_pct  = pos_pnl / max(max_risk, 1.)
        ex, reason = should_exit(open_pos, score["composite"], regime,
                                  open_pos["days_held"], pnl_pct)
        if ex:
            closed_trades.append({
                "date_closed":  date.date().isoformat(),
                "date_entered": open_pos["date_entered"],
                "strategy":     open_pos["strategy"],
                "regime":       open_pos["entry_regime"],
                "days_held":    open_pos["days_held"],
                "lots":         open_pos.get("lots",1),
                "pnl":          round(pos_pnl, 2),
                "pnl_pct":      round(pnl_pct, 3),
                "exit_reason":  reason,
                "result":       "WIN" if pos_pnl > 0 else "LOSS",
                "theta_sign":   open_pos.get("theta_sign",0),
            })
            # Pay close transaction cost
            capital -= capital * FEE_PCT
            open_pos    = None
            pos_pnl     = 0.
            action_taken = "CLOSE"

    # Open new position if none active
    if open_pos is None and strategy["legs"]:
        capital -= capital * FEE_PCT   # pay open cost
        pos_pnl  = 0.
        open_pos = {
            "date_entered":     date.date().isoformat(),
            "strategy":         strategy["name"],
            "entry_spot":       spot,
            "last_spot":        spot,
            "net_delta":        strategy["net_delta"],
            "net_gamma":        strategy["net_gamma"],
            "net_theta":        strategy["net_theta"],
            "net_premium":      strategy["net_premium"],
            "theta_sign":       strategy["theta_sign"],
            "target_hold_days": strategy["target_hold_weeks"]*5,
            "entry_regime":     regime,
            "entry_score":      score["composite"],
            "lots":             lots,
            "days_held":        0,
        }
        action_taken = "OPEN"

    capital = max(capital, 1.)
    equity_curve.append(capital)
    prev_action = action

    strat_short = strategy["name"][:28]
    print(f"  {str(date.date()):>12}  {spot:>8,.0f}  {vix:>5.1f}  {regime:>8}  "
          f"{ACTION_NAMES[action]:>7}  {strat_short:>28}  "
          f"Rs {day_pnl:>+7,.0f}  Rs {capital:>10,.0f}")

    records.append({
        "date":        date.date().isoformat(),
        "spot":        round(spot,2),
        "vix":         round(vix,2),
        "ivr":         round(ivr,3),
        "regime":      regime,
        "regime_conf": round(regime_conf,3),
        "vix_mom":     round(vix_mom,3),
        "term_struct": term_struct,
        "action":      ACTION_NAMES[action],
        "strategy":    strategy["name"],
        "theta_sign":  strategy["theta_sign"],
        "net_theta":   strategy["net_theta"],
        "net_delta":   strategy["net_delta"],
        "net_vega":    strategy["net_vega"],
        "net_premium": strategy["net_premium"],
        "day_pnl":     round(day_pnl,2),
        "pos_pnl":     round(pos_pnl,2),
        "capital":     round(capital,2),
        "score":       score["composite"],
        "brain_conf":  score["brain_conf"],
        "conviction":  score["conviction"],
        "action_taken":action_taken,
        "lots":        lots,
    })


df_bt = pd.DataFrame(records)

# ── 4. Analytics ─────────────────────────────────────────────────────────────
print(f"\n[4/4] Computing analytics...")

eq  = np.array(equity_curve, dtype=float)
ret = np.diff(eq)/eq[:-1]

n   = len(records)
total_return  = (capital/INITIAL_CAPITAL-1)*100
ann_return    = ((capital/INITIAL_CAPITAL)**(252/max(n,1))-1)*100
sharpe        = (ret.mean()/ret.std()*math.sqrt(252)) if ret.std()>0 else 0.
down_std      = ret[ret<0].std() if (ret<0).any() else 1e-9
sortino       = ret.mean()/down_std*math.sqrt(252)
peak          = np.maximum.accumulate(eq)
dd_arr        = (eq-peak)/peak*100
max_dd        = float(dd_arr.min())
calmar        = ann_return/abs(max_dd) if max_dd!=0 else 0

# Trade-level stats
ct = pd.DataFrame(closed_trades)
n_trades  = len(ct)
wins      = int((ct["pnl"]>0).sum()) if n_trades>0 else 0
losses    = n_trades - wins
win_rate  = wins/max(n_trades,1)*100

avg_win   = float(ct[ct["pnl"]>0]["pnl"].mean()) if wins>0 else 0.
avg_loss  = float(ct[ct["pnl"]<0]["pnl"].mean()) if losses>0 else 0.
pf        = (avg_win*wins)/max(abs(avg_loss*losses),1)

# Theta strategies vs debit strategies
credit_trades = ct[ct["theta_sign"]==1] if n_trades>0 else pd.DataFrame()
debit_trades  = ct[ct["theta_sign"]==-1] if n_trades>0 else pd.DataFrame()
credit_win_r  = (credit_trades["pnl"]>0).mean()*100 if len(credit_trades)>0 else 0
debit_win_r   = (debit_trades["pnl"]>0).mean()*100  if len(debit_trades)>0  else 0

# B&H
bah_first = float(dfn["Close"].iloc[start_idx])
bah_last  = float(dfn["Close"].iloc[-1])
bah_ret   = (bah_last/bah_first-1)*100
bah_ann   = ((bah_last/bah_first)**(252/max(n,1))-1)*100
bah_rets  = np.diff(dfn["Close"].iloc[start_idx:].values)/dfn["Close"].iloc[start_idx:-1].values
bah_sharpe= (bah_rets.mean()/bah_rets.std()*math.sqrt(252)) if bah_rets.std()>0 else 0.

# Direction accuracy
df_bt["next_move"] = dfn["Close"].iloc[start_idx:].diff().shift(-1).values[:n]
dir_correct = (
    ((df_bt["action"]=="LONG")  & (df_bt["next_move"]>0)) |
    ((df_bt["action"]=="SHORT") & (df_bt["next_move"]<0))
)
dir_acc = dir_correct.mean()*100

# ── Print Report ──────────────────────────────────────────────────────────────
S70 = "─"*70; S70e = "═"*70
print(f"\n{S70e}")
print(f"  EVOTRADER OPTIONS AI v2.0 — BACKTEST REPORT")
print(f"  Period: {df_bt['date'].iloc[0]}  →  {df_bt['date'].iloc[-1]}  ({n} days)")
print(f"{S70e}")

print(f"\n  ┌─── RETURNS ──────────────────────────────────────────────────────┐")
print(f"  │  v2 Options AI Total Return:    {total_return:>+8.2f}%                   │")
print(f"  │  v2 Annualised Return:          {ann_return:>+8.2f}%                   │")
print(f"  │  NIFTY B&H Return:             {bah_ret:>+8.2f}%  (benchmark)      │")
print(f"  │  NIFTY B&H Annualised:         {bah_ann:>+8.2f}%                   │")
print(f"  │  Alpha vs B&H:                 {total_return-bah_ret:>+8.2f}%                   │")
print(f"  └────────────────────────────────────────────────────────────────────┘")

print(f"\n  ┌─── RISK-ADJUSTED ────────────────────────────────────────────────┐")
print(f"  │  Sharpe Ratio:         {sharpe:>+8.3f}    (B&H: {bah_sharpe:+.3f})             │")
print(f"  │  Sortino Ratio:        {sortino:>+8.3f}                               │")
print(f"  │  Calmar Ratio:         {calmar:>+8.3f}    (AnnRet / MaxDD)            │")
print(f"  │  Max Drawdown:          {max_dd:>8.2f}%                             │")
print(f"  └────────────────────────────────────────────────────────────────────┘")

print(f"\n  ┌─── TRADE-LEVEL STATS ────────────────────────────────────────────┐")
print(f"  │  Closed Trades:        {n_trades:>9d}                               │")
print(f"  │  Win Rate:              {win_rate:>8.1f}%  ({wins}W/{losses}L)                  │")
print(f"  │  Avg Win Trade:        Rs {avg_win:>+9,.0f}                         │")
print(f"  │  Avg Loss Trade:       Rs {avg_loss:>+9,.0f}                         │")
print(f"  │  Profit Factor:         {pf:>8.3f}    (>1.5 = institutional)      │")
print(f"  │  Credit Strat Win%:     {credit_win_r:>8.1f}%  ({len(credit_trades)} credit trades)     │")
print(f"  │  Debit Strat Win%:      {debit_win_r:>8.1f}%  ({len(debit_trades)} debit trades)       │")
print(f"  └────────────────────────────────────────────────────────────────────┘")

print(f"\n  ┌─── AI SIGNAL QUALITY ───────────────────────────────────────────┐")
print(f"  │  Direction Accuracy:    {dir_acc:>8.1f}%  (>55% = alpha)             │")
print(f"  │  Final Capital:        Rs {capital:>11,.0f}                      │")
print(f"  │  Peak Capital:         Rs {eq.max():>11,.0f}                      │")
print(f"  └────────────────────────────────────────────────────────────────────┘")

# ── Regime Breakdown ──────────────────────────────────────────────────────────
print(f"\n  ─── PERFORMANCE BY REGIME ──────────────────────────────────────────")
print(f"  Strategy theta types: (+θ)=credit(earn decay)  (-θ)=debit(pay decay)")
print(f"\n  {'Regime':>8}  {'Days':>5}  {'%Days':>6}  {'Avg VIX':>7}  {'Theta+':>7}  {'Theta-':>7}  {'DayPnL':>10}")
print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}")
for rg in ["LOW","NORMAL","HIGH","EXTREME"]:
    sub = df_bt[df_bt["regime"]==rg]
    if len(sub)==0: continue
    pct  = len(sub)/n*100
    avix = sub["vix"].mean()
    th_plus  = (sub["theta_sign"]==1).sum()
    th_minus = (sub["theta_sign"]==-1).sum()
    tot_pnl  = sub["day_pnl"].sum()
    print(f"  {rg:>8}  {len(sub):>5}  {pct:>5.1f}%  {avix:>7.1f}  {th_plus:>7}  {th_minus:>7}  Rs {tot_pnl:>+7,.0f}")

# ── Closed Trade Regime Breakdown ────────────────────────────────────────────
if n_trades > 0:
    print(f"\n  ─── CLOSED TRADES BY REGIME ────────────────────────────────────────")
    print(f"  {'Regime':>8}  {'Trades':>7}  {'Win%':>7}  {'TotalPnL':>12}  {'AvgDays':>8}  {'PF':>7}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*12}  {'-'*8}  {'-'*7}")
    for rg in ["LOW","NORMAL","HIGH","EXTREME"]:
        sub = ct[ct["regime"]==rg]
        if len(sub)==0: continue
        wr  = (sub["pnl"]>0).mean()*100
        tp  = sub["pnl"].sum()
        ad  = sub["days_held"].mean()
        w2  = (sub["pnl"]>0).sum(); l2=(sub["pnl"]<0).sum()
        aw  = sub[sub["pnl"]>0]["pnl"].mean() if w2>0 else 0
        al  = sub[sub["pnl"]<0]["pnl"].mean() if l2>0 else -1
        pf2 = (aw*w2)/max(abs(al*l2),1)
        print(f"  {rg:>8}  {len(sub):>7}  {wr:>6.1f}%  Rs {tp:>+9,.0f}  {ad:>8.1f}  {pf2:>7.3f}")

# ── Exit Reason Breakdown ─────────────────────────────────────────────────────
if n_trades > 0:
    print(f"\n  ─── EXIT REASON ANALYSIS ───────────────────────────────────────────")
    reason_groups = ct.groupby("exit_reason").agg(
        count=("pnl","count"),
        wins=("pnl",lambda x:(x>0).sum()),
        total_pnl=("pnl","sum"),
    ).reset_index()
    for _,r in reason_groups.iterrows():
        wr2=r["wins"]/r["count"]*100
        print(f"  {str(r['exit_reason'])[:42]:>42}:  {r['count']:>3} trades  "
              f"WR={wr2:.0f}%  PnL=Rs {r['total_pnl']:>+9,.0f}")

# ── Strategy Mix Breakdown ───────────────────────────────────────────────────
print(f"\n  ─── STRATEGY USAGE & PERFORMANCE ──────────────────────────────────")
print(f"  {'Strategy':>32}  {'Days':>5}  {'% credit':>9}  {'AvgTheta':>9}  {'DayPnL':>10}")
print(f"  {'-'*32}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*10}")
strat_g = df_bt.groupby("strategy").agg(
    days=("date","count"),
    theta_plus_pct=("theta_sign",lambda x:(x==1).mean()*100),
    avg_theta=("net_theta","mean"),
    total_pnl=("day_pnl","sum"),
).reset_index()
for _,s in strat_g.sort_values("total_pnl",ascending=False).iterrows():
    print(f"  {s['strategy'][:32]:>32}  {s['days']:>5}  {s['theta_plus_pct']:>8.0f}%  "
          f"{s['avg_theta']:>+9.1f}  Rs {s['total_pnl']:>+7,.0f}")

# ── Action Distribution ───────────────────────────────────────────────────────
print(f"\n  ─── AI SIGNAL DISTRIBUTION ─────────────────────────────────────────")
for act,cnt in df_bt["action"].value_counts().items():
    bar="█"*min(int(cnt/n*40),40)
    print(f"  {act:>8}  {cnt:>4}d ({cnt/n*100:5.1f}%)  {bar}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  OVERALL VERDICT — EvoTrader Options AI v2.0")
print(f"{'='*70}")
tests = [
    ("Return > B&H",      total_return>bah_ret,   abs(total_return-bah_ret)>2,  f"v2:{total_return:+.1f}%  B&H:{bah_ret:+.1f}%"),
    ("Sharpe > 0.5",      sharpe>0.5,             sharpe>0.3,                   f"{sharpe:+.3f}"),
    ("Profit Factor>1.2", pf>1.2,                 pf>1.0,                       f"{pf:.3f}"),
    ("Win Rate > 50%",    win_rate>50,            win_rate>45,                  f"{win_rate:.1f}%"),
    ("Max DD < 20%",      max_dd>-20,             max_dd>-30,                   f"{max_dd:.2f}%"),
    ("Dir. Acc > 50%",    dir_acc>50,             dir_acc>48,                   f"{dir_acc:.1f}%"),
]
passes=0; weaks=0; fails=0
for test,p,w,detail in tests:
    if p:   tag="PASS ✅"; passes+=1
    elif w: tag="WEAK ⚠️ "; weaks+=1
    else:   tag="FAIL ❌"; fails+=1
    print(f"  {test:<24}  {tag:<10}  {detail}")

ov=("🏆 EXCELLENT" if passes>=5 else "✅ GOOD" if passes>=4
    else "⚠️ MODERATE" if passes>=3 else "❌ UNDERPERFORMING")
print(f"\n  Grade:  {ov}  ({passes}P/{weaks}W/{fails}F)")
print(f"\n  CREDIT STRATEGY INSIGHT:")
print(f"  Theta-earning (credit) days: {(df_bt['theta_sign']==1).sum()}")
print(f"  Theta-paying  (debit)  days: {(df_bt['theta_sign']==-1).sum()}")
print(f"  Avg Theta/day this period:   Rs {df_bt['net_theta'].mean():+.2f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    "version":"2.0","period":f"{df_bt['date'].iloc[0]} to {df_bt['date'].iloc[-1]}",
    "trading_days":n,
    "total_return_pct":round(total_return,3),"ann_return_pct":round(ann_return,3),
    "bah_return_pct":round(bah_ret,3),"alpha_pct":round(total_return-bah_ret,3),
    "sharpe":round(sharpe,4),"sortino":round(sortino,4),"calmar":round(calmar,4),
    "max_drawdown_pct":round(max_dd,3),"win_rate_pct":round(win_rate,2),
    "profit_factor":round(pf,4),"direction_acc_pct":round(dir_acc,2),
    "n_trades":n_trades,"credit_win_rate":round(credit_win_r,1),"debit_win_rate":round(debit_win_r,1),
    "final_capital":round(capital,2),"grade":ov,
    "passes":passes,"weaks":weaks,"fails":fails,
    "closed_trades":closed_trades,"daily_records":records,
}
out_file = os.path.join(ROOT_DIR,"options_backtest_v2_results.json")
with open(out_file,"w") as f: json.dump(results,f,indent=2,default=str)
print(f"\n  Full results saved: options_backtest_v2_results.json")
print(f"{'='*70}\n")
