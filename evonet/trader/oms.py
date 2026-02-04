
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OrderManagementSystem:
    """
    Professional Trade Execution & Risk Management Layer.
    Handles 'Market Details' like slippage, fees, and ruins-prevention.
    """
    def __init__(self, initial_capital=10000, fee_pct=0.001, slippage_std=0.0001):
        self.capital = initial_capital
        self.equity = initial_capital
        self.fee_pct = fee_pct
        self.slippage_std = slippage_std
        
        # Risk Metrics
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital
        self.trades = []
        
        # Stats for Kelly Criterion
        self.win_count = 0
        self.loss_count = 0
        self.avg_win = 0.0
        self.avg_loss = 0.0

    def calculate_kelly_size(self) -> float:
        """
        Calculates optimal position size using Kelly Criterion.
        f* = p/a - q/b 
        where p = prob of win, q = prob of loss, a = loss pct, b = win pct
        Returns fraction of capital to risk.
        """
        total_trades = self.win_count + self.loss_count
        if total_trades < 5: 
            return 0.1 # Conservative starting size (10%)
            
        p = self.win_count / total_trades
        q = 1 - p
        
        # Risk-Reward Ratio
        if self.avg_loss == 0: return 0.2
        b = self.avg_win / abs(self.avg_loss)
        
        kelly_f = (p * b - q) / b
        
        # Fractional Kelly (Conservative: 0.5 of Kelly) capped at 50%
        return max(0.02, min(0.5 * kelly_f, 0.5))

    def calculate_slippage(self, price: float, atr: float) -> float:
        """
        Simulates 'Market Impact' and 'Liquidity Drying'.
        In volatile markets (high ATR), slippage is higher.
        """
        # Base slippage + volatility-adjusted penalty
        slippage_pct = np.abs(np.random.normal(0, self.slippage_std))
        volatility_penalty = (atr / price) * 0.1 # 10% of current volatility
        
        return price * (1.0 + slippage_pct + volatility_penalty)

    def execute_trade(self, action: int, current_price: float, atr: float, confidence: float):
        """
        Executes a trade with realistic friction.
        Action: 0=Short, 1=Neutral, 2=Long
        Confidence: From Model's ConfidenceHead (0 to 1)
        """
        if action == 1: # Neutral / Cash
            return 0.0
            
        # 1. Determine Position Size (Kelly * Confidence)
        # If the model is not confident, we scale down the risk significantly.
        base_size = self.calculate_kelly_size()
        real_size = base_size * (confidence ** 2)
        
        # 2. Apply Slippage & Fees
        entry_price = self.calculate_slippage(current_price, atr)
        fee_cost = self.capital * real_size * self.fee_pct
        
        # Log trade intent
        direction = "LONG" if action == 2 else "SHORT"
        logger.info(f"OMS: Executing {direction} | Price: {entry_price:.2f} | Size: {real_size:.2%}")
        
        # Update capital for fees
        self.capital -= fee_cost
        
        return real_size

    def update_performance(self, profit_loss_pct: float):
        """Updates internal stats for future sizing."""
        if profit_loss_pct > 0:
            self.win_count += 1
            self.avg_win = (self.avg_win * (self.win_count - 1) + profit_loss_pct) / self.win_count
        else:
            self.loss_count += 1
            self.avg_loss = (self.avg_loss * (self.loss_count - 1) + abs(profit_loss_pct)) / self.loss_count

        # Update Equity curve
        self.equity *= (1.0 + profit_loss_pct)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        dd = (self.peak_equity - self.equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, dd)
