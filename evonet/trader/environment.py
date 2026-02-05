
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
        # CRITICAL FIX: Validate frame_bound to prevent IndexError
        # signal_features will have length = len(df) - window_size + 1
        # So frame_bound[1] must be <= len(df) - window_size
        max_valid_end = len(df) - window_size
        if frame_bound[1] > max_valid_end:
            print(f"   [WARNING] frame_bound[1]={frame_bound[1]} exceeds max_valid={max_valid_end}. Clamping.")
            frame_bound = (frame_bound[0], max_valid_end)
            
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
        # Features: [Log_Ret, ADX, MACD, BB_Pct, OBV_Slope, ATR_Pct, Kurtosis, Dist_SMA, Log_Ret_5d, Position]
        # Dimension = 9 + 1 = 10
        self.shape = (window_size, 10) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        
        # Alpha Signals from AlphaFactory
        log_ret = self.df.loc[:, 'Log_Ret'].to_numpy()
        adx = self.df.loc[:, 'ADX'].to_numpy()
        macd = self.df.loc[:, 'MACD_Hist'].to_numpy()
        bb_pct = self.df.loc[:, 'BB_Pct'].to_numpy()
        obv_slope = self.df.loc[:, 'OBV_Slope'].to_numpy()
        atr_pct = self.df.loc[:, 'ATR_Pct'].to_numpy()
        kurtosis = self.df.loc[:, 'Kurtosis_20'].to_numpy()
        dist_sma = self.df.loc[:, 'Distance_SMA200'].to_numpy()
        log_ret_5d = self.df['Log_Ret'].rolling(5).sum().fillna(0).to_numpy()
        
        # Keep raw ATR for slippage calculation
        self.atr_raw = self.df.loc[:, 'ATR'].to_numpy()
        
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        
        if start < 0: start = 0
        
        valid_end = len(prices) - self.window_size + 1
        loop_end = min(end, valid_end)
        
        
        # FIX: Generate signal_features for ALL valid windows in the data
        # This ensures signal_features aligns with the prices array
        signal_features = []
        max_index = len(prices) - self.window_size + 1
        for i in range(max_index):
            ws = i
            we = i + self.window_size
            
            # Stack features: (Window, Features)
            features = np.column_stack([
                log_ret[ws:we],
                adx[ws:we],
                macd[ws:we],
                bb_pct[ws:we],
                obv_slope[ws:we],
                atr_pct[ws:we],
                kurtosis[ws:we],
                dist_sma[ws:we],
                log_ret_5d[ws:we]
            ])
            signal_features.append(features)
            
        # Store FULL prices array to prevent IndexError on boundaries
        self.prices = prices 
        self.signal_features = np.array(signal_features)
        
        # Align ATR for slippage (Full Array)
        self.atr_slice = self.atr_raw
        
        return self.prices, self.signal_features

    def _get_observation(self):
        # FIX: Since signal_features now covers the ENTIRE DataFrame,
        # use _current_tick directly as the index (no offset subtraction)
        idx = self._current_tick
        
        # CRITICAL: Clamp idx to valid signal_features range
        # signal_features has length = len(prices) - window_size + 1
        max_valid_idx = len(self.signal_features) - 1
        if idx > max_valid_idx:
            idx = max_valid_idx
            
        base_obs = self.signal_features[idx] # Shape (window_size, 9)
        
        # Position Encoding: Short=-1.0, Neutral=0.0, Long=1.0
        # Map Discrete(3) 0,1,2 -> -1, 0, 1
        current_pos_idx = self._position if hasattr(self, '_position') else 1 # Default neutral if missing
        
        # gym-anytrading uses value attribute for enum, but we use int 0,1,2
        # Let's standardize: self._position is the INT index of the action that caused this state
        # But wait, gym-anytrading stores position as specific Logic.
        # Let's override logic: We track self._current_position_int (0,1,2)
        
        pos_val = float(self._current_position_int - 1) # 0->-1, 1->0, 2->1
        
        position_channel = np.full((self.window_size, 1), pos_val, dtype=np.float32)
        final_obs = np.hstack([base_obs, position_channel]) # Shape (10)
        
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
        
        # CRITICAL: Clamp tick indices to valid array ranges
        max_price_idx = len(self.prices) - 1
        current_tick_clamped = min(self._current_tick, max_price_idx)
        prev_tick_clamped = min(self._current_tick - 1, max_price_idx)
        
        current_price = self.prices[current_tick_clamped]
        last_price = self.prices[prev_tick_clamped]
        
        # Calculate Slippage (Random penalty based on Volatility/ATR)
        # Real slippage is usually against you.
        current_atr = self.atr_slice[current_tick_clamped]
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
