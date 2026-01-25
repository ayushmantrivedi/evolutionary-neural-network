
import numpy as np
import pandas as pd

class RiskGuard:
    """
    The Safety Layer of the EvoTrader.
    Responsible for Position Sizing and Trade Approval.
    
    Principles:
    1. Never Bet the Farm (Kelly Criterion).
    2. Don't Catch Falling Knives (Volatility Guard).
    3. Live to Fight Another Day (Drawdown Hard Stop).
    """
    def __init__(self, max_risk_per_trade=0.05, max_drawdown_limit=0.15):
        self.max_risk_per_trade = max_risk_per_trade # Max % of equity to risk
        self.max_drawdown_limit = max_drawdown_limit # 15% Hard Stop
        self.win_rate_history = [] # Rolling win rate
        self.avg_win_loss_ratio = 1.2 # Conservative initial guess
        
    def check_market_conditions(self, atr, price):
        """
        Returns True if market is safe to trade, False if too volatile.
        """
        # Heuristic: If ATR is > 5% of Price, it's a violently crashing market.
        # Wait for meaningful consolidation.
        volatility_pct = atr / price
        if volatility_pct > 0.05:
            return False, "Extreme Volatility (>5%)"
        return True, "Safe"
        
    def calculate_position_size(self, equity, confidence, volatility):
        """
        Calculates optimal position size using Half-Kelly Criterion.
        
        Kelly % = W - (1-W)/R
        W = Win Probability
        R = Win/Loss Ratio
        """
        # 1. Update Win Rate (Rolling 50 trade average or default 55%)
        # For now, we assume a slight edge for a trained agent
        W = 0.55 
        if len(self.win_rate_history) > 10:
             W = np.mean(self.win_rate_history[-50:])
             
        # 2. Ratio
        R = self.avg_win_loss_ratio
        
        # 3. Kelly Formula
        kelly_pct = W - (1-W)/R
        
        # 4. Safety: Use Half-Kelly (Standard Industry Practice) to reduce variance
        safe_size_pct = kelly_pct * 0.5
        
        # 5. Cap limit
        safe_size_pct = min(safe_size_pct, self.max_risk_per_trade)
        safe_size_pct = max(safe_size_pct, 0.0) # No negative size
        
        # 6. Volatility Adjustment (Inverse Volatility Sizing)
        # If High Volatility, reduce size further
        # Base vol = 2%
        current_vol = volatility
        if current_vol > 0.02:
            scalar = 0.02 / current_vol
            safe_size_pct *= scalar
            
        return safe_size_pct * equity
        
    def update_stats(self, win):
        self.win_rate_history.append(1.0 if win else 0.0)
