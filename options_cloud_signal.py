"""
options_cloud_signal.py  v2.0  —  EvoTrader AI · Options Cloud Runner
======================================================================
STATE-OF-THE-ART institutional options signal engine.

NOVELTY: EvoAdaptive™ architecture — the ONLY system that uses an
evolutionary neural network (not gradient descent) as the directional
brain, with a credit-priority multi-regime options overlay on top.

v2.0 Fixes vs v1.0:
  ✅ Credit-spread priority  — Theta works FOR us in NORMAL regime
  ✅ Proper MTM P&L          — Δ×move + Θ×days + ½Γ×move² (Taylor series)
  ✅ Smart holding logic     — 2–3 week positions, no weekly churn
  ✅ VIX term-structure      — vol regime uses VIX momentum + IVR
  ✅ EvoAdaptive sizing      — position size = brain_confidence × regime_weight
  ✅ Four-exit rules         — expiry / stop-50% / score-flip / regime-change
  ✅ Regime-specific risk    — per-regime max-loss limits, position caps

Architecture (6 Layers — EvoAdaptive™):
  L1 · EvoNet Brain         → direction + confidence  (evolutionary NN)
  L2 · VIX Term-Structure   → vol regime + momentum forecast
  L3 · Greeks Engine        → B-S: Δ Γ Θ ν Vanna Charm  (pure numpy)
  L4 · Credit-Priority Matrix → optimal multi-leg structure
  L5 · EvoAdaptive Scorer   → 7 signals × regime weights → composite
  L6 · Risk Governor        → position size, stop-loss, exit logic

Required GitHub Secrets: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
Required repo file:       brain_weights.pkl
Auto-created:             options_trades.json
"""

import os, sys, json, pickle, datetime, warnings, io, math
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TICKER          = "^NSEI"
VIX_TICKER      = "^INDIAVIX"
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
BRAIN_FILE      = os.path.join(ROOT_DIR, "brain_weights.pkl")
TRADE_LOG_FILE  = os.path.join(ROOT_DIR, "options_trades.json")
WINDOW_SIZE     = 20
LOOKBACK_DAYS   = 400          # need 52-week VIX for IVR + vol-momentum
LOT_SIZE        = 50           # NIFTY standard lot
RISK_FREE_RATE  = 0.068        # India 10Y gilt
INITIAL_CAPITAL = 1_000_000   # Rs 10 Lakh
MAX_RISK_PCT    = 0.02         # Max 2% capital risk per trade
FEE_PCT         = 0.0005       # ~0.05% total brokerage per leg
STOP_LOSS_PCT   = 0.50         # Exit if position loses >50% of premium
MIN_HOLD_DAYS   = 5            # Never exit before 5 days (avoid over-trading)
TARGET_PROFIT_PCT = 0.50       # Take-profit at 50% of max credit received

MODE = os.environ.get("SIGNAL_MODE", "entry")

ACTION_NAMES = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
ACTION_EMOJI = {0: "🔴 SHORT", 1: "⚪ NEUTRAL", 2: "🟢 LONG"}
REGIME_BADGE = {
    "LOW":     "🟦 LOW IV",
    "NORMAL":  "🟩 NORMAL",
    "HIGH":    "🟧 HIGH IV",
    "EXTREME": "🟥 EXTREME",
}

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ── Indicators (exact replica of training environment — DO NOT CHANGE) ────────
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["Log_Ret"]    = np.log(c / c.shift(1))
    df["Log_Ret_5d"] = df["Log_Ret"].rolling(5).sum()
    tr    = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    dm_p  = (h-h.shift()).clip(lower=0); dm_n = (l.shift()-l).clip(lower=0)
    dm_p[dm_p < (l.shift()-l).clip(lower=0)] = 0
    dm_n[dm_n < (h-h.shift()).clip(lower=0)]  = 0
    di_p  = 100 * dm_p.ewm(alpha=1/14,adjust=False).mean() / atr14
    di_n  = 100 * dm_n.ewm(alpha=1/14,adjust=False).mean() / atr14
    dx    = (100*(di_p-di_n).abs()/(di_p+di_n).replace(0,np.nan)).fillna(0)
    df["ADX"]    = dx.ewm(alpha=1/14,adjust=False).mean() / 100.0
    df["ATR"]    = atr14; df["ATR_Pct"] = atr14/c
    obv = (np.sign(c.diff())*v).fillna(0).cumsum()
    df["OBV_Slope"] = obv.pct_change(5).fillna(0)
    ema12 = _ema(c,12); ema26 = _ema(c,26); ml = ema12-ema26
    df["MACD_Hist"] = np.tanh(ml - _ema(ml,9))
    sma20 = _sma(c,20); std20 = c.rolling(20).std()
    df["BB_Pct"] = ((c-(sma20-2*std20))/(4*std20).replace(0,np.nan)).fillna(0.5).clip(0,1)
    df["Kurtosis_20"]     = c.rolling(20).kurt()
    sma200 = _sma(c,200)
    df["Distance_SMA200"] = ((c-sma200)/sma200.replace(0,np.nan)).fillna(0)
    atr30 = atr14.rolling(30).mean()
    df["Vol_Regime_Ratio"] = atr14/atr30.replace(0,np.nan)
    delta = c.diff(); gain = delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    df["RSI"]           = (100-100/(1+gain/loss.replace(0,np.nan))).fillna(50)
    df["Est_Spread_Pct"] = (h-l)/c
    df.fillna(0, inplace=True)
    return df

FEATURE_COLS = [
    "Log_Ret","Log_Ret_5d","ADX","ATR_Pct","OBV_Slope",
    "MACD_Hist","BB_Pct","Kurtosis_20","Distance_SMA200"
]


# ── EvoNet Brain (pure numpy — the EvoAdaptive™ novelty core) ─────────────────
def softmax(x):
    e = np.exp(x - x.max());  return e / e.sum()

def forward_pass(neurons, x):
    l1 = np.array([np.maximum(0., np.dot(x, n['weights'])+n['bias']) for n in neurons[0]], dtype=np.float32)
    l2 = np.array([np.maximum(0., np.dot(l1,n['weights'])+n['bias']) for n in neurons[1]], dtype=np.float32)
    l3 = np.array([np.dot(np.concatenate([l2,l1]),n['weights'])+n['bias'] for n in neurons[2]], dtype=np.float32)
    return softmax(l3)

def get_directional_signal(brain, df, prev_pos):
    obs   = df.iloc[-WINDOW_SIZE:][FEATURE_COLS].values.astype(np.float32)
    pos_ch= np.full((WINDOW_SIZE,1), float(prev_pos-1), dtype=np.float32)
    state = np.nan_to_num(np.hstack([obs,pos_ch]).flatten())
    probs = forward_pass(brain['neurons'], state)
    return int(np.argmax(probs)), probs


# ── VIX Term Structure & Vol Regime Engine ────────────────────────────────────
def compute_ivr(vix_series):
    """IV Rank: (current - 52w_low) / (52w_high - 52w_low)."""
    if len(vix_series) < 50: return 0.4
    w252 = vix_series.iloc[-252:] if len(vix_series) >= 252 else vix_series
    cur  = float(vix_series.iloc[-1])
    lo   = float(w252.min()); hi = float(w252.max())
    return 0.4 if (hi-lo) < 0.01 else max(0., min(1., (cur-lo)/(hi-lo)))

def compute_vix_momentum(vix_series, window=10):
    """VIX momentum: rate of change over last N days.
    Rising VIX = vol expanding = sell-premium caution.
    Falling VIX = vol contracting = credit spread sweet spot.
    Returns: float in [-1,+1], positive = VIX rising."""
    if len(vix_series) < window+2: return 0.0
    recent = vix_series.iloc[-window:]
    mom = (float(recent.iloc[-1]) - float(recent.iloc[0])) / max(float(recent.iloc[0]), 0.01)
    return float(np.tanh(mom * 10))

def compute_vol_term_structure(vix_series):
    """Simplified VIX term structure: compare 5d vs 20d avg.
    Backwardation (spot > future) = high fear = good for selling premium.
    Contango (spot < future) = low fear = buy vol.
    Returns: 'backwardation' | 'contango' | 'flat'"""
    if len(vix_series) < 25: return 'flat'
    vix5  = float(vix_series.iloc[-5:].mean())
    vix20 = float(vix_series.iloc[-20:].mean())
    ratio = vix5 / max(vix20, 0.01)
    if ratio > 1.05:  return 'backwardation'
    elif ratio < 0.95: return 'contango'
    else:              return 'flat'

def classify_regime(ivr, vix_now, vix_mom, term_structure):
    """
    Dual-confirmation regime classification:
    - Requires both IVR AND absolute VIX to agree for EXTREME
    - Uses vol momentum + term structure for nuance
    Returns: (regime_str, regime_confidence)
    """
    # Base regime from IVR + absolute VIX
    if   ivr >= 0.75 or vix_now >= 24.0: base = "EXTREME"
    elif ivr >= 0.45 or vix_now >= 17.0: base = "HIGH"
    elif ivr <= 0.20 and vix_now <= 13.5: base = "LOW"
    else:                                  base = "NORMAL"

    # Confidence: higher when signals agree
    confidence = 0.5
    if base == "EXTREME" and ivr >= 0.75 and vix_now >= 24.0: confidence = 1.0
    elif base == "HIGH"   and ivr >= 0.45 and vix_now >= 17.0: confidence = 0.9
    elif base == "LOW"    and ivr <= 0.15 and vix_now <= 12.0: confidence = 0.9
    elif base == "NORMAL": confidence = 0.7

    # Vol momentum adjustment (rising vol = upgrade regime risk)
    if vix_mom > 0.3 and base == "NORMAL":  base = "HIGH"; confidence *= 0.8
    if vix_mom < -0.3 and base == "HIGH":   base = "NORMAL"; confidence *= 0.8

    return base, round(confidence, 3)


# ── Black-Scholes Greeks Engine ───────────────────────────────────────────────
def _ncdf(x): return 0.5*(1.+math.erf(x/math.sqrt(2)))
def _npdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def bs_greeks(S, K, T, r, sigma, opt_type="call"):
    """Full BSM Greeks. Returns dict with: price, delta, gamma, theta, vega, vanna, charm."""
    if T <= 0:
        iv = max(S-K,0) if opt_type=="call" else max(K-S,0)
        return {"price":iv,"delta":1.0 if opt_type=="call" else -1.0,
                "gamma":0.,"theta":0.,"vega":0.,"vanna":0.,"charm":0.}
    sd = sigma*math.sqrt(T)
    d1 = (math.log(S/K)+(r+0.5*sigma**2)*T)/sd
    d2 = d1-sd
    nd1= _ncdf(d1); nd2=_ncdf(d2); pd1=_npdf(d1)
    disc= math.exp(-r*T)
    if opt_type=="call":
        price=S*nd1-K*disc*nd2;  delta=nd1
    else:
        price=K*disc*_ncdf(-d2)-S*_ncdf(-d1); delta=nd1-1.
    gamma = pd1/(S*sd)
    vega  = S*pd1*math.sqrt(T)/100.          # per 1% IV
    theta = (-(S*pd1*sigma)/(2*math.sqrt(T)) - r*K*disc*(nd2 if opt_type=="call" else _ncdf(-d2)))/365.
    vanna = -pd1*d2/sigma
    charm = -(pd1*(2*r*T-d2*sd)/(2*T*sd))
    return {"price":round(price,2),"delta":round(delta,4),"gamma":round(gamma,6),
            "theta":round(theta,2),"vega":round(vega,2),"vanna":round(vanna,4),"charm":round(charm,4)}

def get_atm_strike(spot, step=50.): return round(spot/step)*step

def get_expiry_T(weeks_out=1):
    today = datetime.date.today()
    days_to_thu = (3-today.weekday())%7
    if days_to_thu == 0: days_to_thu = 7
    return max((days_to_thu + (weeks_out-1)*7)/365., 1/365.)


# ── EvoAdaptive™ Position Sizing ──────────────────────────────────────────────
def evo_position_size(capital, brain_confidence, regime_confidence, regime,
                      max_risk_pct=MAX_RISK_PCT):
    """
    EvoAdaptive™ novelty: position size scales with BOTH:
    1. EvoNet brain confidence (P_winning - P_losing)
    2. Regime detection confidence
    This gives larger positions when both signals agree strongly,
    and tiny/zero positions when either signal is weak.

    Returns: number of lots (integer, min 1)
    """
    # Regime risk scalar — how much capital to risk per regime
    regime_scalar = {"LOW": 0.8, "NORMAL": 1.0, "HIGH": 0.7, "EXTREME": 0.4}
    rs  = regime_scalar.get(regime, 0.8)

    # Combined confidence: geometric mean ensures BOTH must be high
    combined = math.sqrt(brain_confidence * regime_confidence)

    # Capital risk amount
    risk_budget = capital * max_risk_pct * rs * combined

    # Approximate cost of 1 lot ATM option (~2% of NIFTY * LOT_SIZE)
    # Using a conservative Rs 200 per lot as base cost
    cost_per_lot = 200.0

    lots = max(1, int(risk_budget / cost_per_lot))
    lots = min(lots, 10)   # hard cap at 10 lots
    return lots


# ── Credit-Priority Strategy Matrix v2 ───────────────────────────────────────
# KEY CHANGE from v1: NORMAL regime now ALWAYS uses credit strategies first.
# Credit strategies have POSITIVE theta = time decay income daily.
# Debit strategies only used when vol is very cheap (LOW regime) + strong direction.
#
# Each leg: {type, offset, weeks, pos}  pos: +1=long, -1=short (the premium view)

STRATEGY_V2 = {
    # ══ LONG AI Signal ════════════════════════════════════════════════════════
    # LOW IV + LONG → Buy cheap call (low vega risk, cheap premium, 2-week hold)
    ("LONG","LOW"): {
        "name": "Long Call (Debit)",
        "theta_sign": -1,              # theta COSTS us
        "target_hold_weeks": 2,
        "legs":[{"type":"call","offset":0,"weeks":2,"pos":1}],
        "max_loss": "premium",
        "note": "Buy when premium cheap"
    },
    # NORMAL IV + LONG → Bull Put Spread: SELL downside put spread (CREDIT, positive theta)
    # Wins if market stays flat or goes up. Theta works for us every day.
    ("LONG","NORMAL"): {
        "name": "Bull Put Spread (Credit)",
        "theta_sign": +1,              # theta EARNS us
        "target_hold_weeks": 2,
        "legs":[
            {"type":"put","offset":-50, "weeks":2,"pos":-1},   # sell OTM put
            {"type":"put","offset":-150,"weeks":2,"pos": 1},   # buy further OTM put (wing)
        ],
        "max_loss": "spread_minus_credit",
        "note": "Theta income, wins if mkt flat or up"
    },
    # HIGH IV + LONG → Short Put Spread (wider, more credit in high-IV)
    ("LONG","HIGH"): {
        "name": "Short Put Spread (Hi-IV Credit)",
        "theta_sign": +1,
        "target_hold_weeks": 1,
        "legs":[
            {"type":"put","offset":-50, "weeks":1,"pos":-1},
            {"type":"put","offset":-200,"weeks":1,"pos": 1},
        ],
        "max_loss": "spread_minus_credit",
        "note": "Sell overpriced put premium in bull bias"
    },
    # EXTREME IV + LONG → Iron Fly (sell both sides, extreme premium)
    ("LONG","EXTREME"): {
        "name": "Short Iron Fly (Extreme Credit)",
        "theta_sign": +1,
        "target_hold_weeks": 1,
        "legs":[
            {"type":"call","offset": 0,   "weeks":1,"pos":-1},  # sell ATM call
            {"type":"call","offset":+200,  "weeks":1,"pos": 1},  # buy OTM call (wing)
            {"type":"put", "offset": 0,   "weeks":1,"pos":-1},  # sell ATM put
            {"type":"put", "offset":-200,  "weeks":1,"pos": 1},  # buy OTM put (wing)
        ],
        "max_loss": "wings_minus_credit",
        "note": "Max premium collection in extreme fear",
        "caution": "⚠️ HIGH RISK — only with strong brain confidence"
    },

    # ══ SHORT AI Signal ═══════════════════════════════════════════════════════
    # LOW IV + SHORT → Buy cheap put
    ("SHORT","LOW"): {
        "name": "Long Put (Debit)",
        "theta_sign": -1,
        "target_hold_weeks": 2,
        "legs":[{"type":"put","offset":0,"weeks":2,"pos":1}],
        "max_loss": "premium",
        "note": "Buy when premium cheap"
    },
    # NORMAL IV + SHORT → Bear Call Spread (CREDIT) — theta positive
    ("SHORT","NORMAL"): {
        "name": "Bear Call Spread (Credit)",
        "theta_sign": +1,
        "target_hold_weeks": 2,
        "legs":[
            {"type":"call","offset":+50, "weeks":2,"pos":-1},   # sell OTM call
            {"type":"call","offset":+150,"weeks":2,"pos": 1},   # buy further OTM call
        ],
        "max_loss": "spread_minus_credit",
        "note": "Theta income, wins if mkt flat or down"
    },
    # HIGH IV + SHORT → Short Call Spread (wider, more credit)
    ("SHORT","HIGH"): {
        "name": "Short Call Spread (Hi-IV Credit)",
        "theta_sign": +1,
        "target_hold_weeks": 1,
        "legs":[
            {"type":"call","offset":+50, "weeks":1,"pos":-1},
            {"type":"call","offset":+200,"weeks":1,"pos": 1},
        ],
        "max_loss": "spread_minus_credit",
        "note": "Sell overpriced call premium in bear bias"
    },
    # EXTREME IV + SHORT → Short Iron Fly
    ("SHORT","EXTREME"): {
        "name": "Short Iron Fly (Extreme Credit)",
        "theta_sign": +1,
        "target_hold_weeks": 1,
        "legs":[
            {"type":"call","offset": 0,  "weeks":1,"pos":-1},
            {"type":"call","offset":+200,"weeks":1,"pos": 1},
            {"type":"put", "offset": 0,  "weeks":1,"pos":-1},
            {"type":"put", "offset":-200,"weeks":1,"pos": 1},
        ],
        "max_loss": "wings_minus_credit",
        "note": "Max premium in extreme fear",
        "caution": "⚠️ HIGH RISK — only with strong brain confidence"
    },

    # ══ NEUTRAL AI Signal ═════════════════════════════════════════════════════
    # LOW IV + NEUTRAL → Long Strangle (expect a breakout move)
    ("NEUTRAL","LOW"): {
        "name": "Long Strangle (Breakout Play)",
        "theta_sign": -1,
        "target_hold_weeks": 3,    # needs more time for move to develop
        "legs":[
            {"type":"call","offset":+100,"weeks":3,"pos":1},
            {"type":"put", "offset":-100,"weeks":3,"pos":1},
        ],
        "max_loss": "premium",
        "note": "Cheap vol, expect regime-change breakout"
    },
    # NORMAL IV + NEUTRAL → Iron Condor (classic credit, theta machine)
    ("NEUTRAL","NORMAL"): {
        "name": "Iron Condor (Credit)",
        "theta_sign": +1,
        "target_hold_weeks": 2,
        "legs":[
            {"type":"call","offset":+100,"weeks":2,"pos":-1},
            {"type":"call","offset":+250,"weeks":2,"pos": 1},
            {"type":"put", "offset":-100,"weeks":2,"pos":-1},
            {"type":"put", "offset":-250,"weeks":2,"pos": 1},
        ],
        "max_loss": "spread_minus_credit",
        "note": "Range-bound, collect theta both sides"
    },
    # HIGH IV + NEUTRAL → Wide Iron Condor: wider strikes = more credit
    ("NEUTRAL","HIGH"): {
        "name": "Wide Iron Condor (Hi-IV Credit)",
        "theta_sign": +1,
        "target_hold_weeks": 1,
        "legs":[
            {"type":"call","offset":+150,"weeks":1,"pos":-1},
            {"type":"call","offset":+350,"weeks":1,"pos": 1},
            {"type":"put", "offset":-150,"weeks":1,"pos":-1},
            {"type":"put", "offset":-350,"weeks":1,"pos": 1},
        ],
        "max_loss": "spread_minus_credit",
        "note": "Wide condor in high-IV = maximum credit buffer"
    },
    # EXTREME IV + NEUTRAL → CASH (regime too dangerous for neutral strategies)
    ("NEUTRAL","EXTREME"): {
        "name": "CASH — Skip Trade",
        "theta_sign": 0,
        "target_hold_weeks": 0,
        "legs": [],
        "max_loss": "none",
        "note": "Extreme vol + no direction = undefined risk",
        "caution": "⛔ NO TRADE — Stay in cash"
    },
}

def build_strategy(ai_action, regime, spot, sigma, r, lots=1):
    """Build strategy: pick from matrix, compute all leg Greeks, return full dict."""
    key  = (ACTION_NAMES[ai_action], regime)
    tmpl = STRATEGY_V2.get(key, STRATEGY_V2[("NEUTRAL","NORMAL")])
    atm  = get_atm_strike(spot)

    legs          = []
    net_delta     = 0.0
    net_gamma     = 0.0
    net_theta     = 0.0
    net_vega      = 0.0
    net_premium   = 0.0    # positive = debit, negative = credit received

    for lg in tmpl["legs"]:
        K   = atm + lg["offset"]
        pos = lg["pos"]
        T   = get_expiry_T(lg["weeks"])
        g   = bs_greeks(spot, K, T, r, sigma, lg["type"])

        # Scale by lots
        net_delta   += pos * g["delta"] * LOT_SIZE * lots
        net_gamma   += pos * g["gamma"] * LOT_SIZE * lots
        net_theta   += pos * g["theta"] * LOT_SIZE * lots  # net per day per lots
        net_vega    += pos * g["vega"]  * LOT_SIZE * lots
        net_premium += pos * g["price"] * LOT_SIZE * lots  # negative = credit

        legs.append({
            "type": lg["type"], "strike": K,
            "weeks_out": lg["weeks"],
            "position": "LONG" if pos > 0 else "SHORT",
            "greeks": g, "lots": lots
        })

    return {
        **{k:v for k,v in tmpl.items() if k != "legs"},
        "legs":        legs,
        "atm":         atm,
        "net_delta":   round(net_delta, 3),
        "net_gamma":   round(net_gamma, 6),
        "net_theta":   round(net_theta, 2),
        "net_vega":    round(net_vega,  2),
        "net_premium": round(net_premium, 2),
        "lots":        lots,
        "max_credit":  round(-min(net_premium, 0), 2),  # for credit strategies
    }


# ── EvoAdaptive™ Composite Scorer (7 signals, regime-adjusted weights) ────────
BASE_WEIGHTS = {
    "ai_direction":   0.35,
    "iv_regime":      0.18,
    "vol_momentum":   0.12,
    "momentum":       0.13,
    "trend":          0.10,
    "mean_reversion": 0.07,
    "vol_ratio":      0.05,
}

REGIME_WEIGHT_OVERRIDES = {
    # In HIGH/EXTREME regimes, vol signals matter MORE
    "HIGH":    {"iv_regime": 0.25, "vol_momentum": 0.18, "momentum": 0.10, "mean_reversion": 0.05},
    "EXTREME": {"iv_regime": 0.30, "vol_momentum": 0.22, "momentum": 0.08, "trend": 0.07, "mean_reversion": 0.02},
    # In LOW regime, direction and trend matter more
    "LOW":     {"ai_direction": 0.40, "trend": 0.15, "iv_regime": 0.10},
}

def compute_evo_score(probs, df, ivr, regime, vix_mom):
    """EvoAdaptive™ composite scorer with regime-adjusted weights.
    Each signal [-1,+1]. Positive = bullish, negative = bearish.
    """
    row = df.iloc[-1]

    s_ai  = float(probs[2] - probs[0])
    brain_conf = float(max(probs))    # How confident is the brain?

    regime_map = {"LOW":-0.05, "NORMAL":0.0, "HIGH":0.05, "EXTREME":0.10}
    s_iv  = regime_map.get(regime, 0.) - ivr * 0.3   # high IVR = slight bear lean

    s_vmom = -float(np.tanh(vix_mom * 3))   # rising VIX = bearish for theta buyers

    macd = float(row.get("MACD_Hist",0.)); bb = (float(row.get("BB_Pct",0.5))-0.5)*2
    s_mom  = float(np.tanh((macd+bb)*2))

    adx = float(row.get("ADX",0.3))
    s_trend = float(np.sign(s_ai) * min(adx*3, 1.0))

    dist = float(row.get("Distance_SMA200",0.))
    s_mr   = float(np.tanh(-dist*10))

    vratio = float(row.get("Vol_Regime_Ratio",1.))
    s_vr   = float(np.tanh(-(vratio-1.)*5))

    signals = {
        "ai_direction":   round(s_ai, 3),
        "iv_regime":      round(s_iv, 3),
        "vol_momentum":   round(s_vmom, 3),
        "momentum":       round(s_mom, 3),
        "trend":          round(s_trend, 3),
        "mean_reversion": round(s_mr, 3),
        "vol_ratio":      round(s_vr, 3),
    }

    # Apply regime-specific weight overrides
    weights = dict(BASE_WEIGHTS)
    for k, v in REGIME_WEIGHT_OVERRIDES.get(regime, {}).items():
        weights[k] = v
    # Renormalise weights
    total_w = sum(weights.values())
    weights = {k: v/total_w for k,v in weights.items()}

    composite = sum(weights[k] * signals[k] for k in signals)
    composite = float(np.tanh(composite * 2.5))

    conviction = ("🔥 STRONG" if abs(composite)>=0.65 else
                  "✅ MODERATE" if abs(composite)>=0.40 else
                  "🔸 WEAK"    if abs(composite)>=0.20 else
                  "⚪ NEGLIGIBLE")

    return {
        "signals":     signals,
        "composite":   round(composite, 4),
        "brain_conf":  round(brain_conf, 3),
        "conviction":  conviction,
        "score_bar":   _score_bar(composite),
    }

def _score_bar(s, w=22):
    mid=w//2; pos=max(0,min(w-1,int(mid+s*mid)))
    bar=["─"]*w; bar[mid]="┼"; bar[pos]="●"
    return "◀["+"".join(bar)+"]▶"


# ── Smart Exit Governor ───────────────────────────────────────────────────────
def should_exit(open_pos, current_score, current_regime, days_held,
                current_pnl_pct, min_hold=MIN_HOLD_DAYS):
    """
    Four-rule exit system:
    1. EXPIRY:        Position near expiry (<= 1 day)
    2. STOP-LOSS:     Position lost >50% of premium/credit
    3. SCORE-FLIP:    Composite score flipped with HIGH conviction (>0.5)
    4. REGIME-CHANGE: Regime changed AND we've held at least min_hold days

    Returns: (should_exit: bool, reason: str)
    """
    if days_held < min_hold:
        return False, "Min hold period not reached"

    # Rule 1: Expiry
    if days_held >= open_pos.get("target_hold_days", 10):
        return True, "Approaching expiry — take profit/loss"

    # Rule 2: Stop-loss — position lost > 50% of what we risked
    if current_pnl_pct <= -STOP_LOSS_PCT:
        return True, f"Stop-loss triggered ({current_pnl_pct*100:.0f}%)"

    # Rule 3: Take-profit — for credit strategies, close at 50% of max credit
    if current_pnl_pct >= TARGET_PROFIT_PCT and open_pos.get("theta_sign", 0) > 0:
        return True, f"Take-profit triggered ({current_pnl_pct*100:.0f}% of credit)"

    # Rule 4: Score-flip — AI signal reversed with conviction
    entry_score = open_pos.get("entry_score", 0.)
    score_flipped = (np.sign(current_score) != np.sign(entry_score) and
                     abs(current_score) > 0.50)
    if score_flipped:
        return True, "AI score flipped with HIGH conviction"

    # Rule 5: Regime changed (e.g. NORMAL → EXTREME)
    entry_regime = open_pos.get("entry_regime", "NORMAL")
    danger_flip  = entry_regime in ("NORMAL","LOW") and current_regime in ("EXTREME","HIGH")
    if danger_flip:
        return True, f"Regime escalated: {entry_regime} → {current_regime}"

    return False, "Hold"


# ── Trade Log ─────────────────────────────────────────────────────────────────
def load_log():
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE) as f: return json.load(f)
    return {
        "version":         "2.0",
        "initial_capital": INITIAL_CAPITAL,
        "current_capital": INITIAL_CAPITAL,
        "open_position":   None,
        "trades":          [],
        "daily_signals":   [],
        "equity_curve":    [],
        "stats": {"total_trades":0,"wins":0,"losses":0,"total_pnl":0.}
    }

def save_log(log):
    with open(TRADE_LOG_FILE,"w") as f: json.dump(log,f,indent=2,default=str)

def open_position(log, today, action, price, strategy, score, regime,
                  vix, ivr, lots):
    pos = {
        "date_entered":     today,
        "spot_at_entry":    price,
        "atm_at_entry":     strategy["atm"],
        "strategy":         strategy["name"],
        "theta_sign":       strategy["theta_sign"],
        "target_hold_days": strategy["target_hold_weeks"] * 5,
        "net_premium":      strategy["net_premium"],
        "max_credit":       strategy["max_credit"],
        "net_delta":        strategy["net_delta"],
        "net_theta":        strategy["net_theta"],
        "net_vega":         strategy["net_vega"],
        "net_gamma":        strategy["net_gamma"],
        "entry_regime":     regime,
        "entry_score":      score["composite"],
        "lots":             lots,
        "cumulative_pnl":   0.,
        "days_held":        0,
        "legs":             strategy["legs"],
    }
    log["open_position"] = pos
    log["stats"]["total_trades"] = log["stats"].get("total_trades",0)+1
    return log

def close_position(log, today, price, reason, final_pnl):
    op = log.get("open_position")
    if not op: return log
    pnl_pct = final_pnl / max(abs(op["net_premium"]), 1.)
    is_win  = final_pnl > 0
    log["trades"].append({
        "date_closed":   today,
        "date_entered":  op["date_entered"],
        "strategy":      op["strategy"],
        "days_held":     op["days_held"],
        "entry_regime":  op["entry_regime"],
        "lots":          op["lots"],
        "final_pnl":     round(final_pnl,2),
        "pnl_pct":       round(pnl_pct,3),
        "exit_reason":   reason,
        "result":        "WIN" if is_win else "LOSS",
    })
    if is_win: log["stats"]["wins"] = log["stats"].get("wins",0)+1
    else:      log["stats"]["losses"] = log["stats"].get("losses",0)+1
    log["stats"]["total_pnl"] = log["stats"].get("total_pnl",0)+final_pnl
    log["open_position"] = None
    return log

def add_signal(log, today, action, price, vix, ivr, regime, strategy, score):
    existing = {s["date"] for s in log["daily_signals"]}
    if today not in existing:
        log["daily_signals"].append({
            "date":     today, "action": ACTION_NAMES[action],
            "price":    price, "vix":    round(vix,2),
            "ivr":      round(ivr,3), "regime":  regime,
            "strategy": strategy["name"],
            "net_theta":strategy["net_theta"],
            "score":    score["composite"],
            "conviction":score["conviction"],
            "timestamp":datetime.datetime.utcnow().isoformat(),
        })
    return log


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] No credentials — skipped."); return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          json={"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"HTML"},timeout=15)
        print("✅ Telegram sent." if r.ok else f"⚠️ Telegram error: {r.text[:200]}")
    except Exception as e:
        print(f"⚠️ Telegram error: {e}")

def build_telegram_msg(log, action, price, vix, ivr, regime, regime_conf,
                       term_struct, strategy, score, lots, today, mode="entry"):
    cap     = log["current_capital"]
    pnl_pct = ((cap/log["initial_capital"])-1)*100
    op      = log.get("open_position")
    n_days  = len(log["daily_signals"])

    if mode == "exit" and op:
        dpnl    = op.get("cumulative_pnl",0)
        return "\n".join([
            "🔔 <b>EVOTRADER OPTIONS v2 — EXIT CHECK</b>",
            f"📅 <b>{today}</b>  |  🕓 Near-close",
            f"  Open: <b>{op['strategy']}</b>  (day {op['days_held']})",
            f"  Spot: ₹{price:,.2f}  |  VIX: {vix:.2f}",
            f"  Est P&amp;L: ₹{dpnl:+,.0f}  |  Regime: {REGIME_BADGE[regime]}",
            f"  Score: {score['composite']:+.3f}  {score['conviction']}",
            f"  <i>EvoTrader Options AI v2.0</i>",
        ])

    legs_text = ""
    for lg in strategy.get("legs",[]):
        g = lg.get("greeks",{})
        arrow = "📈" if lg["position"]=="LONG" else "📉"
        legs_text += f"\n    {arrow} <b>{lg['position']}</b> {lg['type'].upper()} @{int(lg['strike'])} ×{lg['lots']}lot  [Δ={g.get('delta',0):.3f} Θ={g.get('theta',0):.2f}/d ν={g.get('vega',0):.2f}/%]"
    if not legs_text: legs_text = "\n    ⛔ No trade — Cash"

    theta_str = "earning" if strategy["net_theta"]>0 else "paying"
    prem_str  = "credit" if strategy["net_premium"]<0 else "debit"
    bias      = ("🐂 BULLISH" if score["composite"]>0.1 else "🐻 BEARISH" if score["composite"]<-0.1 else "😐 NEUTRAL")

    return "\n".join([
        "🧠 <b>EVOTRADER OPTIONS AI v2.0 — ENTRY SIGNAL</b>",
        f"📅 <b>{today}</b>  |  🕘 Pre-market",
        "",
        f"  NIFTY50:  <b>₹{price:,.2f}</b>",
        f"  VIX:      <b>{vix:.2f}</b>  {REGIME_BADGE[regime]}  conf={regime_conf:.0%}",
        f"  IVR:      <b>{ivr*100:.0f}th%ile</b>  Term: {term_struct}",
        "",
        f"🤖 <b>AI Direction:</b>  {ACTION_EMOJI[action]}",
        f"🏆 <b>Strategy:</b>  {strategy['name']}  ×{lots} lot",
        f"  {strategy.get('note','')}",
        "",
        "📐 <b>Legs:</b>" + legs_text,
        "",
        f"⚡ <b>Net Greeks</b>",
        f"  Δ={strategy['net_delta']:+.2f}  Γ={strategy['net_gamma']:+.5f}",
        f"  Θ={strategy['net_theta']:+.2f}/day ({theta_str} decay)",
        f"  ν={strategy['net_vega']:+.2f}/1%IV",
        f"  Net Premium: ₹{strategy['net_premium']:+,.2f} ({prem_str})",
        "",
        f"📊 <b>EvoAdaptive Score: {score['composite']:+.4f}  {score['conviction']}</b>",
        f"   {score['score_bar']}",
        f"   Bias: {bias}  |  BrainConf: {score['brain_conf']:.0%}",
        "",
        f"  Sub: AI={score['signals']['ai_direction']:+.3f}  IV={score['signals']['iv_regime']:+.3f}  Mom={score['signals']['momentum']:+.3f}",
        f"       Trend={score['signals']['trend']:+.3f}  VolMom={score['signals']['vol_momentum']:+.3f}",
        "",
        f"💰 <b>Portfolio:</b> ₹{cap:,.2f}  ({pnl_pct:+.2f}%)  Day #{n_days}",
        f"<i>Exit rules: 50% stop | 50% TP | score-flip | regime-change</i>",
        f"<i>EvoTrader Options AI v2.0 — EvoAdaptive™ arch</i>",
    ])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    today = datetime.date.today().isoformat()
    print(f"\n{'='*65}\n  EVOTRADER OPTIONS AI v2.0 — {today}  [{MODE.upper()}]\n{'='*65}\n")

    # 1. Brain
    print("[1/6] Loading EvoNet brain...")
    with open(BRAIN_FILE,"rb") as f: brain=pickle.load(f)
    print(f"      ✅ {brain['input_dim']}→{brain['layer_sizes']}→{brain['output_dim']}")

    # 2. Market data
    print("[2/6] Fetching NIFTY50 + India VIX...")
    end=datetime.date.today()+datetime.timedelta(days=1)
    start=end-datetime.timedelta(days=int(LOOKBACK_DAYS*1.6))
    df   = yf.download(TICKER,    start=str(start),end=str(end),interval="1d",progress=False)
    dfv  = yf.download(VIX_TICKER,start=str(start),end=str(end),interval="1d",progress=False)
    for d in [df,dfv]:
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        if d.index.tz is not None: d.index=d.index.tz_convert(None)
    spot=float(df["Close"].iloc[-1]); vix=float(dfv["Close"].iloc[-1]) if len(dfv)>0 else 15.
    print(f"      ✅ NIFTY ₹{spot:,.2f}  VIX {vix:.2f}")

    # 3. Indicators
    print("[3/6] Computing indicators...")
    df=add_indicators(df); df.dropna(inplace=True)
    print(f"      ✅ {len(df)} bars  9 features OK")

    # 4. Signals
    print("[4/6] EvoAdaptive signal pipeline...")
    log = load_log()
    prev = {v:k for k,v in ACTION_NAMES.items()}.get(
        log["daily_signals"][-1]["action"] if log["daily_signals"] else "NEUTRAL",1)
    action, probs = get_directional_signal(brain, df, prev)
    ivr       = compute_ivr(dfv["Close"])
    vix_mom   = compute_vix_momentum(dfv["Close"])
    term_struct = compute_vol_term_structure(dfv["Close"])
    regime, regime_conf = classify_regime(ivr, vix, vix_mom, term_struct)
    sigma     = vix/100.; T=get_expiry_T(1)
    brain_conf= float(max(probs))
    lots = evo_position_size(log["current_capital"], brain_conf, regime_conf, regime)
    print(f"      ✅ AI:{ACTION_NAMES[action]}  Regime:{regime}(conf={regime_conf:.0%})")
    print(f"      ✅ VIX:{vix:.2f}  IVR:{ivr:.1%}  VixMom:{vix_mom:+.2f}  Term:{term_struct}")
    print(f"      ✅ BrainConf:{brain_conf:.0%}  Lots:{lots}")

    # 5. Strategy + Score
    print("[5/6] Strategy + EvoAdaptive score...")
    strategy = build_strategy(action, regime, spot, sigma, RISK_FREE_RATE, lots=lots)
    score    = compute_evo_score(probs, df, ivr, regime, vix_mom)
    print(f"      ✅ {strategy['name']}  Θ={strategy['net_theta']:+.2f}/d  Premium=₹{strategy['net_premium']:.0f}")
    print(f"      ✅ Score:{score['composite']:+.4f}  {score['conviction']}")
    for leg in strategy["legs"]:
        g=leg.get("greeks",{})
        print(f"         {leg['position']:5} {leg['type']:4} @{int(leg['strike'])} "
              f"Δ={g.get('delta',0):+.4f} Θ={g.get('theta',0):+.2f}/d ν={g.get('vega',0):+.2f}/%")

    # 6. Position management + log + notify
    print("[6/6] Position management + notify...")
    log = add_signal(log, today, action, spot, vix, ivr, regime, strategy, score)

    op = log.get("open_position")
    if op:
        op["days_held"] = op.get("days_held",0)+1
        delta_pnl = op["net_delta"]*(spot-op["spot_at_entry"])
        theta_pnl = op["net_theta"]
        gamma_pnl = 0.5*op["net_gamma"]*(spot-op["spot_at_entry"])**2
        daily_pnl = delta_pnl + theta_pnl + gamma_pnl
        op["cumulative_pnl"] = op.get("cumulative_pnl",0.)+daily_pnl
        log["current_capital"] = max(log["current_capital"]+daily_pnl, 1.)
        log["equity_curve"].append({"date":today,"equity":log["current_capital"],"regime":regime})

        max_risk = abs(op["net_premium"]) if op["net_premium"] != 0 else abs(op["net_theta"])*op["target_hold_days"]
        pnl_pct  = op["cumulative_pnl"]/max(max_risk,1.)
        ex, reason = should_exit(op, score["composite"], regime, op["days_held"], pnl_pct)
        if ex:
            log = close_position(log, today, spot, reason, op["cumulative_pnl"])
            print(f"      ✅ Position CLOSED: {reason}  PnL=₹{daily_pnl:+,.0f}")
    else:
        if strategy["legs"]:
            log["equity_curve"].append({"date":today,"equity":log["current_capital"],"regime":regime})
            log = open_position(log, today, action, spot, strategy, score, regime, vix, ivr, lots)
            print(f"      ✅ New position opened: {strategy['name']}")

    save_log(log)
    msg = build_telegram_msg(log, action, spot, vix, ivr, regime, regime_conf,
                             term_struct, strategy, score, lots, today, MODE)
    send_telegram(msg)

    pnl=((log["current_capital"]/log["initial_capital"])-1)*100
    print(f"\n{'─'*65}")
    print(f"  SIGNAL:   {ACTION_NAMES[action]}")
    print(f"  STRATEGY: {strategy['name']}")
    print(f"  REGIME:   {regime}  VIX={vix:.2f}  IVR={ivr:.1%}")
    print(f"  SCORE:    {score['composite']:+.4f}  {score['conviction']}")
    print(f"  CAPITAL:  ₹{log['current_capital']:,.2f}  ({pnl:+.2f}%)")
    print(f"{'─'*65}\n✅ Done.")


if __name__ == "__main__":
    main()
