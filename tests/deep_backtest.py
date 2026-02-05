
import logging
import sys
import os
import pandas as pd
import numpy as np
import time
from typing import Dict, List

# Ensure evonet is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.trader.risk_manager import RiskManager, RiskConfig
from evonet.trader.data_loader import DataFetcher
from evonet.trader.alpha_factory import AlphaFactory
from train_memory_autopilot import MemoryEvoPilot # [NEW] Import Brain Class
import pickle
import pandas_ta as ta # Ensure installed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | [BACKTEST] %(message)s')
logger = logging.getLogger("DeepBacktest")

class DeepBacktester:
    def __init__(self, ticker="BTC-USD", start_date="2018-01-01", end_date="2023-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.risk_manager = RiskManager()
        self.portfolio_value = 100000.0 # Start with $100k
        self.cash = 100000.0
        self.position = 0.0 # Units held
        self.trades = []
        self.history = []
        
        # Load Ultimate Brain if available
        self.brain = None
        if os.path.exists("ultimate_brain.pkl"):
            print("[BACKTEST] Loading Ultimate Brain...")
            with open("ultimate_brain.pkl", "rb") as f:
                self.brain = pickle.load(f)
        else:
            print("[WARN] No Brain found. Using Mock Strategy.")

    def _get_state_vector(self, slice_df):
        # Must match FinancialRegimeEnv exactly!
        # Features: [Log_Ret, ADX, MACD, BB_Pct, OBV_Slope, ATR_Pct, Kurtosis, Dist_SMA, Log_Ret_5d]
        columns = ['Log_Ret', 'ADX', 'MACD_Hist', 'BB_Pct', 'OBV_Slope', 'ATR_Pct', 'Kurtosis_20', 'Distance_SMA200', 'Log_Ret_5d']
        
        # We need the last WINDOW_SIZE rows
        window = slice_df.iloc[-20:] # Hardcoded WINDOW_SIZE=20
        
        # If window is small (<20), pad? or skip. 
        # DeepBacktester loop starts at 50, so we are safe.
        
        base_features = window[columns].values # Shape (20, 9)
        
        # Position Channel
        # Map: Short (-1), Neutral (0), Long (1)
        # We need to track 'Trade Mode'. 
        # DeepBacktester only has simple 'position > 0' long logic.
        # Let's assume Long=1, Neutral=0. No Shorting logic in Simple Backtester yet?
        # FinancialEnv maps 0->-1, 1->0, 2->1.
        # Let's use 1.0 if Long, 0.0 if Neutral.
        current_pos_val = 1.0 if self.position > 0 else 0.0
        
        pos_channel = np.full((20, 1), current_pos_val)
        
        final_obs = np.hstack([base_features, pos_channel]) # Shape (20, 10)
        return final_obs.flatten()
        
    def run_simulation(self):
        print(f"\n[BACKTEST] Starting Deep History Validation for {self.ticker}...")
        print(f"Period: {self.start_date} -> {self.end_date}")
        
        # 1. Fetch Data (Full History)
        fetcher = DataFetcher(self.ticker, self.start_date, self.end_date, interval="1d", provider="yf")
        df = fetcher.fetch_data(use_cache=False)
        df = fetcher.process() # Apply Alpha
        
        logger.info(f"Loaded {len(df)} candles. Beginning Event-Driven Loop...")
        
        # 2. Event-Driven Loop
        # We start at index 30 to allow for rolling windows (ATR, etc.)
        for i in range(50, len(df)):
            current_slice = df.iloc[:i+1] # Valid "Past" data only
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            date = current_candle.name
            price = current_candle['Close']
            
            # Update Portfolio Value (Mark-to-Market)
            current_pv = self.cash + (self.position * price)
            pnl_today = (price - prev_candle['Close']) * self.position
            self.risk_manager.update_portfolio_state(current_pv, pnl_today)
            
            if self.risk_manager.is_halted:
                 # Check if we can un-halt? For strict backtest, maybe require manual intervention (skip day)
                 # Simulating "cooling off" period - unhalt next day?
                 # No, stricter: Halt remains for the day. Simulation just holds.
                 logger.warning(f"[{date}] Market Halted. Holding positions.")
                 self.history.append({'date': date, 'pv': current_pv, 'action': 'HALT', 'drawdown': 0})
                 continue

            # Generate Mock Signal (Replacement for Slow Neural Net)
            if self.brain:
                state = self._get_state_vector(current_slice)
                # Prediction (Best Genome = Index 0)
                y_pred, conf = self.brain.net.predict(state, 0)
                action_idx = np.argmax(y_pred)
                
                # output_dim=3 -> [Short, Neutral, Long]? 
                # FinancialEnv: 0=Short, 1=Neutral, 2=Long
                actions_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
                raw_action = actions_map[action_idx]
                confidence = float(conf)
            else:
                # Mock Strategy
                sma50 = current_slice['Close'].rolling(50).mean().iloc[-1]
                raw_action = "BULLISH" if price > sma50 else "BEARISH"
                confidence = 0.85 # High confidence for stress testing
            
            # VALIDATE WITH RISK MANAGER
            signal_payload = {"action": raw_action, "confidence": confidence}
            is_approved, reason = self.risk_manager.validate_trade(signal_payload, current_slice)
            
            executed_action = "HOLD"
            if is_approved:
                # Execution Logic
                target_size_usd = self.risk_manager.get_position_size(1.0) # 2% max
                target_units = target_size_usd / price
                
                if raw_action == "BULLISH" and self.position == 0:
                    # Buy
                    cost = target_units * price * 1.001 # 0.1% Fee
                    if self.cash >= cost:
                        self.cash -= cost
                        self.position = target_units
                        executed_action = "BUY"
                        self.trades.append({'date': date, 'type': 'BUY', 'price': price, 'size': target_units})
                        
                elif raw_action == "BEARISH" and self.position > 0:
                    # Sell
                    revenue = self.position * price * 0.999 # 0.1% Fee
                    self.cash += revenue
                    self.position = 0
                    executed_action = "SELL"
                    self.trades.append({'date': date, 'type': 'SELL', 'price': price, 'size': target_units})
            else:
                 executed_action = f"REJECTED ({reason})"
            
            # Record History
            # Calc Drawdown
            peak = max([h['pv'] for h in self.history]) if self.history else current_pv
            dd = (peak - current_pv) / peak if peak > 0 else 0
            
            self.history.append({'date': date, 'pv': current_pv, 'action': executed_action, 'drawdown': dd})

        self._generate_report()

    def _generate_report(self):
        print("\n" + "="*80)
        print("DEEP HISTORY VALIDATION REPORT")
        print("="*80)
        
        hist_df = pd.DataFrame(self.history).set_index('date')
        
        start_pv = self.history[0]['pv']
        end_pv = self.history[-1]['pv']
        total_return = (end_pv - start_pv) / start_pv
        
        # Max Drawdown
        max_dd = hist_df['drawdown'].max()
        
        # Sharpe Ratio (Daily)
        hist_df['returns'] = hist_df['pv'].pct_change()
        sharpe = hist_df['returns'].mean() / hist_df['returns'].std() * np.sqrt(252)
        
        print(f"Total Period: {len(hist_df)} Days")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Max Drawdown: {max_dd*100:.2f}% (Limit: 15% Stocks / 25% Crypto)")
        print(f"Sharpe Ratio: {sharpe:.2f} (Target > 1.5)")
        print(f"Total Trades: {len(self.trades)}")
        
        print("-" * 40)
        print("RISK MANAGER INTERVENTIONS (Top 5):")
        rejections = hist_df[hist_df['action'].str.contains("REJECTED")]
        print(rejections['action'].value_counts().head(5))
        
        # Verdict
        if max_dd < 0.25 and sharpe > 1.0: # Relaxed initial target
             print("\nVERDICT: [PASS] System Survived.")
        else:
             print("\nVERDICT: [FAIL] Risk Parameters need tuning.")

if __name__ == "__main__":
    # Test on BTC (High Volatility)
    bt = DeepBacktester("BTC-USD", start_date="2020-01-01", end_date="2023-12-31")
    bt.run_simulation()
