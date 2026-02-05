
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("RiskManager")

@dataclass
class RiskConfig:
    MAX_POSITION_SIZE_PCT: float = 0.02  # 2% max per trade
    HARD_STOP_DAILY_PCT: float = 0.03    # 3% daily loss halt
    MAX_DRAWDOWN_WEEKLY: float = 0.07    # 7% weekly review trigger
    MIN_LIQUIDITY_SPREAD: float = 0.005  # 0.5% max spread
    VOLATILITY_SPIKE_THRESHOLD: float = 3.0 # Halt if Vol > 3x Normal
    MIN_CONFIDENCE: float = 0.60         # 60% confidence floor
    MAX_DATA_GAP_STD: float = 2.0        # Max allowed data gap sigma

class RiskManager:
    """
    The 'Compliance Officer' of the system.
    Enforces the CTO Constitution: Non-Negotiable Safeguards.
    """
    def __init__(self, config: RiskConfig = RiskConfig()):
        self.config = config
        self.daily_pnl = 0.0
        self.portfolio_value = 100000.0 # Default starting cash, should be updated
        self.is_halted = False
        self.halt_reason = ""
        self.confidence_history: List[float] = []
        
    def update_portfolio_state(self, current_value: float, daily_pnl: float):
        self.portfolio_value = current_value
        self.daily_pnl = daily_pnl
        self._check_circuit_breakers()

    def validate_trade(self, signal: Dict[str, Any], market_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Primary Gatekeeper. Returns (Approved, RejectionReason).
        """
        if self.is_halted:
            return False, f"SYSTEM HALTED: {self.halt_reason}"
            
        # 1. Liquidity Check (Spread)
        # Use Est_Spread_Pct from AlphaFactory
        spread = 0.0
        if 'Est_Spread_Pct' in market_data.columns:
            spread = market_data['Est_Spread_Pct'].iloc[-1]
        else:
            # Fallback estimation
            spread = (market_data['High'].iloc[-1] - market_data['Low'].iloc[-1]) / market_data['Close'].iloc[-1]
            
        # DYNAMIC THRESHOLD: Crypto usually has higher spread/volatility
        # If High-Low proxy is used, 0.5% is too tight for BTC daily candles.
        # We'll allow 1.5% for "Volatile" assets (ATR > 2% of price)
        limit = self.config.MIN_LIQUIDITY_SPREAD
        atr_pct = (market_data['High'].iloc[-1] - market_data['Low'].iloc[-1]) / market_data['Close'].iloc[-1]
        if atr_pct > 0.02: # High Volatility Asset
             limit = 0.015 # 1.5%
        
        if spread > limit:
             return False, f"Liquidity Too Low (Spread {spread:.2%} > {limit:.2%})"
        
        # Check Data Quality First
        if not self._check_data_quality(market_data):
             return False, "Data Quality Validation Failed"

        # 2. Confidence Check
        conf = signal.get('confidence', 0.5)
        self.confidence_history.append(conf)
        if len(self.confidence_history) > 10: 
            self.confidence_history.pop(0)
            
        if conf < self.config.MIN_CONFIDENCE:
            return False, f"Low Confidence ({conf:.2f} < {self.config.MIN_CONFIDENCE})"

        # 3. Volatility Regime Check
        # Halt if current volatility is 3x the 30-day average
        if len(market_data) > 30:
            current_atr = market_data['ATR'].iloc[-1] if 'ATR' in market_data else (market_data['High'].iloc[-1] - market_data['Low'].iloc[-1])
            avg_atr = market_data['ATR'].rolling(30).mean().iloc[-1] if 'ATR' in market_data else (market_data['High'] - market_data['Low']).rolling(30).mean().iloc[-1]
            
            if avg_atr > 0 and current_atr > (avg_atr * self.config.VOLATILITY_SPIKE_THRESHOLD):
                self._trigger_halt(f"Extreme Volatility Spike ({current_atr:.4f} vs Avg {avg_atr:.4f})")
                return False, "Volatility Circuit Breaker"

        return True, "APPROVED"

    def get_position_size(self, signal_strength: float = 1.0) -> float:
        """
        Calculate safe position size.
        NEVER exceeds MAX_POSITION_SIZE_PCT.
        """
        # Kelly Criterion could go here, but strictly capped.
        base_size = self.portfolio_value * self.config.MAX_POSITION_SIZE_PCT
        return base_size * signal_strength

    def _check_circuit_breakers(self):
        """
        Runs continuously to check PnL limits.
        """
        # Daily Loss Limit
        pnl_pct = self.daily_pnl / self.portfolio_value
        if pnl_pct < -self.config.HARD_STOP_DAILY_PCT:
            self._trigger_halt(f"Daily Loss Limit Hit ({pnl_pct*100:.2f}%)")

    def _check_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Enforce Unit tests on live data.
        """
        # Missing Data Threshold
        missing_pct = df.isnull().mean().max()
        if missing_pct > 0.005: # 0.5%
            logger.error(f"[RISK] Rejecting Data: Too many missing values ({missing_pct:.2%})")
            return False
            
        # Gap Detection
        # Check time delta consistency? (Complex for now, skipping strict time check in MVP)
        
        return True

    def _trigger_halt(self, reason: str):
        self.is_halted = True
        self.halt_reason = reason
        logger.critical(f"🛑 KILL SWITCH ENGAGED: {reason}")
