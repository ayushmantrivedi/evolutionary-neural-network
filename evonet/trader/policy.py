import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class TradingPolicy:
    """
    EvoTrader Professional Policy Layer.
    Implements the pipeline: Edge -> Regime -> Payoff -> Size -> Execution -> Exit
    """
    
    def __init__(self, cost_per_trade: float = 0.005, dd_limit: float = 0.10):
        self.cost_per_trade = cost_per_trade
        self.dd_limit = dd_limit
        
    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))
        
    def evaluate(self, 
                 df: pd.DataFrame, 
                 current_capital: float, 
                 peak_capital: float, 
                 raw_action: int, 
                 recent_trades: list = None) -> Tuple[int, Dict[str, Any]]:
        """
        Evaluate market conditions and return the final action and metadata.
        raw_action: 0=SHORT, 1=NEUTRAL, 2=LONG (from the neural network)
        """
        if df.empty or len(df) < 5:
            return 1, self._neutral_meta("NO_TRADE", "Insufficient data")
            
        latest = df.iloc[-1]
        
        # 1) Extract Inputs
        try:
            P = latest['Close']
            SMA5 = df['Close'].rolling(5).mean().iloc[-1]
            SMA20 = latest.get('SMA20', df['Close'].rolling(20).mean().iloc[-1])
            
            # Volatility
            VIX = latest.get('VIX_Level', 0.15) * 100
            VIX_prev = df['VIX_Level'].iloc[-2] * 100 if len(df) > 1 else VIX
            delta_VIX = (VIX - VIX_prev) / VIX_prev if VIX_prev > 0 else 0.0
            
            # Structure
            HH_HL = (latest['High'] > df['High'].iloc[-2]) and (latest['Low'] > df['Low'].iloc[-2])
            LL_LH = (latest['Low'] < df['Low'].iloc[-2]) and (latest['High'] < df['High'].iloc[-2])
            
            # Breakouts
            break_5D_low = P < df['Low'].iloc[-6:-1].min()
            break_5D_high = P > df['High'].iloc[-6:-1].max()
            
            # Edge
            VRP = latest.get('VRP', 0)
            VRP_series = df['VRP'] if 'VRP' in df.columns else pd.Series([0]*len(df))
            VRP_z = (VRP - VRP_series.rolling(60).mean().iloc[-1]) / (VRP_series.rolling(60).std().iloc[-1] + 1e-9)
            if np.isnan(VRP_z): VRP_z = 0.0
            
            # Meta
            DD = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0.0
            RV_5 = df['Log_Ret'].rolling(5).std().iloc[-1] * np.sqrt(252) if 'Log_Ret' in df.columns else 0.15
            ATR_5_slope = latest.get('ATR_Slope', 0) # proxy for ATR expanding
            
        except Exception as e:
            logger.warning(f"Feature extraction failed in policy: {e}")
            return 1, self._neutral_meta("NO_TRADE", "Data Error")

        # 2) Hard Guards (always evaluated first)
        if DD >= self.dd_limit:
            return 1, self._neutral_meta("NO_TRADE", "Drawdown limit reached")
            
        if delta_VIX >= 0.08 and P < SMA5:
            # Risk Off
            if raw_action == 0:
                raw_action = 1 # Block new shorts
                
        # 3) Regime Classifier
        UP_TREND = (P > SMA20) and (SMA5 > SMA20) and HH_HL and not break_5D_low
        DOWN_TREND = (P < SMA20) and (SMA5 < SMA20) and LL_LH and not break_5D_high
        RANGE = not UP_TREND and not DOWN_TREND
        
        VOL_EXPANSION = (delta_VIX > 0.04) or (ATR_5_slope > 0.05)
        VOL_CONTRACTION = (delta_VIX < -0.04)
        
        # Map to states
        if VOL_EXPANSION and (not UP_TREND and not DOWN_TREND):
            regime = "S4: Risk-Off"
        elif UP_TREND:
            regime = "S1: Long Trend"
        elif DOWN_TREND:
            regime = "S2: Short Trend"
        else:
            regime = "S3: Neutral/Theta"
            
        # 4) Edge Scoring (0-1)
        trend_strength = (P - SMA20) / SMA20
        edge_dir = self._sigmoid(abs(trend_strength) * 100) # structure
        edge_vrp = self._sigmoid(VRP_z) # mispricing
        edge_vol = 1.0 - self._sigmoid(delta_VIX * 100) # penalize rising vol
        
        w1, w2, w3 = 0.4, 0.4, 0.2
        edge_total = w1 * edge_dir + w2 * edge_vrp + w3 * edge_vol
        
        # Gate
        expected_edge = edge_total * 0.02 # Proxy for expected return
        if expected_edge < self.cost_per_trade * 1.5:
            edge_gate_passed = False
        else:
            edge_gate_passed = True

        if edge_total < 0.55 or not edge_gate_passed:
            # Not enough edge to trade
            if raw_action != 1:
                pass # Can log this

        # For the sake of the model's signal, if it says LONG/SHORT but edge is very low, we might override
        # However, we'll let the model decide direction, policy decides sizing and strategy.
        action = raw_action
        
        if regime == "S4: Risk-Off" and action != 1:
            action = 1 # Force neutral in risk-off
            
        # 5) Strategy Router
        strategy = "CASH"
        if action == 2: # Long
            if VOL_CONTRACTION and UP_TREND:
                strategy = "LONG CALL / CALL SPREAD"
            elif VRP_z > 0.5 and not VOL_EXPANSION:
                strategy = "BULL PUT SPREAD"
            else:
                strategy = "SMALL CALL SPREAD"
                
        elif action == 0: # Short
            if VOL_EXPANSION and DOWN_TREND:
                strategy = "LONG PUT / PUT SPREAD"
            elif VRP_z > 0.5 and not VOL_EXPANSION:
                strategy = "BEAR CALL SPREAD"
            else:
                strategy = "SMALL PUT SPREAD"
                
        elif action == 1: # Neutral
            if regime == "S3: Neutral/Theta" and VRP_z > 0.8 and not VOL_EXPANSION:
                strategy = "IRON CONDOR (SMALL)"
            else:
                strategy = "CASH"

        # Anti-repetition
        same_strategy_count = 0
        if recent_trades:
            for t in reversed(recent_trades):
                if t.get('strategy') == strategy:
                    same_strategy_count += 1
                else:
                    break
                    
        # 6) Position Sizing
        base = edge_total
        vol_adj = 1.0 / (1.0 + RV_5)
        dd_adj = 1.0 - np.clip(abs(DD) / self.dd_limit, 0.0, 0.5)
        
        size_raw = base * vol_adj * dd_adj
        
        # Strategy Caps
        if strategy in ["BULL PUT SPREAD", "BEAR CALL SPREAD", "IRON CONDOR (SMALL)"]:
            size_cap = 0.25
        elif "SPREAD" in strategy:
            size_cap = 0.35
        elif "LONG CALL" in strategy or "LONG PUT" in strategy:
            size_cap = 0.40
        else:
            size_cap = 0.20
            
        if strategy == "CASH":
            size_cap = 0.0
            
        size = min(size_raw, size_cap)
        
        # Brakes
        if delta_VIX > 0:
            size *= 0.7
        if DD > 0.01:
            size *= 0.7
        if same_strategy_count >= 2:
            size *= 0.8 # Penalty for repetition
            
        if size < 0.05 and action != 1:
            # If size is too small, just don't trade
            action = 1
            strategy = "CASH"
            size = 0.0
            
        # Diagnostics
        diagnostics = {
            "EdgeScore": round(edge_total, 2),
            "VRP_z": round(VRP_z, 1),
            "delta_VIX": round(delta_VIX * 100, 1),
            "TrendStrength": round(trend_strength * 100, 2),
            "Reason": f"Regime={regime}, Edge={edge_total:.2f}",
            "Strategy": strategy,
            "Regime": regime,
            "SizeDrivers": f"Cap={size_cap}, VolAdj={vol_adj:.2f}, DD={DD:.2%}",
            "Size": size
        }

        return action, diagnostics

    def _neutral_meta(self, strategy: str, reason: str) -> Dict[str, Any]:
        return {
            "EdgeScore": 0.0,
            "VRP_z": 0.0,
            "delta_VIX": 0.0,
            "TrendStrength": 0.0,
            "Reason": reason,
            "Strategy": strategy,
            "Regime": "Neutral / Guarded",
            "SizeDrivers": "N/A",
            "Size": 0.0
        }
