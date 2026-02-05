
import logging
import sys
import os
import pandas as pd
import numpy as np
import time

# Ensure evonet is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.trader.risk_manager import RiskManager, RiskConfig
from evonet.trader.alpha_factory import AlphaFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | [COMPLIANCE] %(message)s')
logger = logging.getLogger("ComplianceAudit")

class ComplianceTest:
    def __init__(self):
        self.rm = RiskManager()
        self.passed = 0
        self.total = 0
        
    def run_tests(self):
        self.total = 0
        self.passed = 0
        print("\n" + "="*80)
        print("COMPLIANCE & RISK CONSTITUTION AUDIT")
        print("="*80)
        
        self.test_daily_loss_limit()
        self.test_volatility_spike()
        self.test_low_confidence()
        self.test_data_quality_gap()
        self.test_liquidity_spread()
        
        print("\n" + "-"*80)
        print(f"AUDIT COMPLETE: {self.passed}/{self.total} Tests Passed.")
        
    def _assert(self, condition, name):
        self.total += 1
        if condition:
            print(f"✅ PASS: {name}")
            self.passed += 1
        else:
            print(f"❌ FAIL: {name}")
            
    def test_daily_loss_limit(self):
        print("\n[TEST] Daily Loss Limit (Hard Stop 3%)")
        self.rm.update_portfolio_state(100000, -2900) # -2.9% -> Safe
        self._assert(self.rm.is_halted == False, "System Active at -2.9% Loss")
        
        self.rm.update_portfolio_state(100000, -3100) # -3.1% -> HALT
        self._assert(self.rm.is_halted == True, "System HALTED at -3.1% Loss")
        
        # Reset for next test
        self.rm.is_halted = False 
        
    def test_volatility_spike(self):
        print("\n[TEST] Volatility Spike Protection (>3x Avg)")
        # Create mock DF with High Volatility
        df = pd.DataFrame({'Close': np.random.randn(50) + 100})
        df['High'] = df['Close'] + 5 # High Vol
        df['Low'] = df['Close'] - 5
        df['ATR'] = 10.0 # Huge ATR
        # Avg ATR simulated as 2.0 (so 10.0 is 5x spike)
        # Note: RiskManager calculates rolling avg internally if length > 30.
        # Let's mock a sequence where ATR jumps.
        
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(index=dates)
        df['Close'] = 100.0
        df['High'] = 101.0
        df['Low'] = 99.0
        df['ATR'] = 1.0 # Stable baseline
        
        # Inject Spike at end
        df.iloc[-1, df.columns.get_loc('ATR')] = 5.0 # 5x Spike
        
        signal = {'action': 'BULLISH', 'confidence': 0.8}
        approved, reason = self.rm.validate_trade(signal, df)
        
        self._assert(not approved, "Trade Rejected on Volatility Spike")
        self._assert("Volatility" in reason, f"Reason Correct: {reason}")
        
    def test_low_confidence(self):
        print("\n[TEST] Low Confidence Gate (<60%)")
        df = pd.DataFrame({'Close': [100]}) # Dummy
        
        # Reset Halt
        self.rm.is_halted = False
        
        signal = {'action': 'BULLISH', 'confidence': 0.55}
        approved, reason = self.rm.validate_trade(signal, df)
        self._assert(not approved, "Trade Rejected on 55% Confidence")
        
        signal = {'action': 'BULLISH', 'confidence': 0.70}
        approved, reason = self.rm.validate_trade(signal, df)
        self._assert(approved, "Trade Approved on 70% Confidence")
        
    def test_data_quality_gap(self):
        print("\n[TEST] Data Quality (Missing Data)")
        df = pd.DataFrame(np.random.randn(100, 4), columns=['Open','High','Low','Close'])
        # Inject NaNs > 0.5% (approx 1% here)
        df.iloc[0:2] = np.nan 
        
        self.rm.is_halted = False
        approved, reason = self.rm.validate_trade({}, df)
        self._assert(not approved, "Trade Rejected on Missing Data")

    def test_liquidity_spread(self):
        print("\n[TEST] Liquidity Gate (Spread > 0.5%)")
        
        # Test 1: Low Spread (Should Approve)
        df_good = pd.DataFrame({'Close': [100], 'High': [100.2], 'Low': [99.8], 'Est_Spread_Pct': [0.004]})
        self.rm.is_halted = False
        approved, _ = self.rm.validate_trade({'confidence': 0.8}, df_good)
        self._assert(approved, "Trade Approved on 0.4% Spread")
        
        # Test 2: High Spread (Should Reject)
        df_bad = pd.DataFrame({'Close': [100], 'High': [101.0], 'Low': [99.0], 'Est_Spread_Pct': [0.01]}) 
        approved, reason = self.rm.validate_trade({'confidence': 0.8}, df_bad)
        self._assert(not approved, "Trade Rejected on 1.0% Spread")
        self._assert("Liquidity" in reason, f"Reason Correct: {reason}")

if __name__ == "__main__":
    audit = ComplianceTest()
    audit.run_tests()
