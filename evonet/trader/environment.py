
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_anytrading.envs import StocksEnv

class FinancialRegimeEnv(StocksEnv):
    """
    Extended StocksEnv that supports specific market regimes (Bull/Bear).
    Wraps 'gym-anytrading' logic but adds specialized observation space for EvoNet.
    """
    def __init__(self, df, frame_bound, window_size, fee=0.001, slippage_std=0.0001):
        # Initialize parent StocksEnv
        super().__init__(df, window_size, frame_bound)
        
        # --- REAL WORLD FRICTION ---
        self.trade_fee_bid_percent = fee  # 0.1% Default
        self.trade_fee_ask_percent = fee  # 0.1% Default
        self.slippage_std = slippage_std  # Standard deviation of slippage (as % of price)
        
        # --- ACTION SPACE (Professional) ---
        # 0 = Short, 1 = Neutral (Cash), 2 = Long
        self.action_space = spaces.Discrete(3)
        
        # --- OBSERVATION SPACE ---
        # Features: [Log_Ret, ADX, MACD, BB_Pct, OBV_Slope, ATR_Pct, Position_Encoded]
        # Dimension = 6 + 1 = 7
        self.shape = (window_size, 7) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        
        # Alpha Signals from DataLoader
        log_ret = self.df.loc[:, 'Log_Ret'].to_numpy()
        adx = self.df.loc[:, 'ADX'].to_numpy()
        macd = self.df.loc[:, 'MACD_Hist'].to_numpy()
        bb_pct = self.df.loc[:, 'BB_Pct'].to_numpy()
        obv_slope = self.df.loc[:, 'OBV_Slope'].to_numpy()
        atr_pct = self.df.loc[:, 'ATR_Pct'].to_numpy()
        
        # Keep raw ATR for slippage calculation
        self.atr_raw = self.df.loc[:, 'ATR'].to_numpy()
        
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        
        if start < 0: start = 0
        
        valid_end = len(prices) - self.window_size + 1
        loop_end = min(end, valid_end)
        
        signal_features = []
        for i in range(start, loop_end):
            ws = i
            we = i + self.window_size
            
            # Stack features: (Window, Features)
            features = np.column_stack([
                log_ret[ws:we],
                adx[ws:we],
                macd[ws:we],
                bb_pct[ws:we],
                obv_slope[ws:we],
                atr_pct[ws:we]
            ])
            signal_features.append(features)
            
        self.prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        self.signal_features = np.array(signal_features)
        
        # Align ATR for slippage (same length as prices)
        self.atr_slice = self.atr_raw[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        
        return self.prices, self.signal_features

    def _get_observation(self):
        idx = self._current_tick - self.frame_bound[0]
        base_obs = self.signal_features[idx] # Shape (window_size, 6)
        
        # Position Encoding: Short=-1.0, Neutral=0.0, Long=1.0
        # Map Discrete(3) 0,1,2 -> -1, 0, 1
        current_pos_idx = self._position if hasattr(self, '_position') else 1 # Default neutral if missing
        
        # gym-anytrading uses value attribute for enum, but we use int 0,1,2
        # Let's standardize: self._position is the INT index of the action that caused this state
        # But wait, gym-anytrading stores position as specific Logic.
        # Let's override logic: We track self._current_position_int (0,1,2)
        
        pos_val = float(self._current_position_int - 1) # 0->-1, 1->0, 2->1
        
        position_channel = np.full((self.window_size, 1), pos_val, dtype=np.float32)
        final_obs = np.hstack([base_obs, position_channel]) # Shape (7)
        
        return final_obs.flatten()

    def reset(self, seed=None, options=None):
        # Override reset to init our custom position tracker
        # Start Neutral (1)
        self._current_position_int = 1 
        return super().reset(seed=seed, options=options)

    def _calculate_reward(self, action):
        """
        Calculates Pro Reward: PnL - Fees - Slippage
        Action: 0=Short, 1=Neutral, 2=Long
        """
        step_reward = 0.0
        
        current_price = self.prices[self._current_tick]
        last_price = self.prices[self._current_tick - 1]
        
        # Calculate Slippage (Random penalty based on Volatility/ATR)
        # Real slippage is usually against you.
        current_atr = self.atr_slice[self._current_tick]
        slippage = np.abs(np.random.normal(0, self.slippage_std * current_price))
        
        # 1. Calculate PnL from PREVIOUS position held to NOW
        # Previous position determines profit/loss on this price move
        prev_pos_map = self._current_position_int - 1 # -1, 0, 1
        
        # Log Return approximation for stability: ln(P_t/P_{t-1}) * Direction
        log_ret = np.log(current_price / last_price)
        step_reward = prev_pos_map * log_ret
        
        # 2. Handle State Transitions (Trading Costs)
        if action != self._current_position_int:
            # We traded!
            
            # Fee Cost (0.1% per trade value approx = 0.001 * 1.0)
            # Since we work in returns, we subtract fixed fee pct
            fee_cost = self.trade_fee_bid_percent
            
            # Slippage Cost (as percentage of return)
            possible_slippage_pct = (slippage / current_price)
            
            step_reward -= (fee_cost + possible_slippage_pct)
        
        # Update internal state for next step
        self._current_position_int = action
        
        return step_reward
