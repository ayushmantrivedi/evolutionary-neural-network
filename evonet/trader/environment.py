
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_anytrading.envs import StocksEnv

class FinancialRegimeEnv(StocksEnv):
    """
    Extended StocksEnv that supports specific market regimes (Bull/Bear).
    Wraps 'gym-anytrading' logic but adds specialized observation space for EvoNet.
    """
    def __init__(self, df, frame_bound, window_size):
        # Initialize parent StocksEnv
        super().__init__(df, window_size, frame_bound)
        
        # Extended Observation Space (Price + Indicators)
        # We assume df has SMA, RSI, ATR columns from DataLoader
        self.trade_fee_bid_percent = 0.0  # 0.0% fee (Training Wheels)
        self.trade_fee_ask_percent = 0.0  # 0.0% fee
        
        # State: [Price_Diff, SMA20_Diff, SMA50_Diff, RSI, ATR, Position_Encoded]
        # dimension = 5 + 1 = 6
        self.shape = (window_size, 6) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        
        # New Advanced Features
        log_ret = self.df.loc[:, 'Log_Ret'].to_numpy()
        volatility = self.df.loc[:, 'Volatility'].to_numpy() # Already normalizedish (0.01-0.10)
        vol_osc = self.df.loc[:, 'Vol_Osc'].to_numpy()       # -1 to 1 mostly
        rsi = self.df.loc[:, 'RSI'].to_numpy()               # 0 to 1
        trend = self.df.loc[:, 'Trend_50'].to_numpy()        # -0.2 to 0.2
        
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        
        if start < 0: start = 0 # Safety
        
        # Fix window bounds
        valid_end = len(prices) - self.window_size + 1
        loop_end = min(end, valid_end)
        
        # Create sliding windows
        signal_features = []
        for i in range(start, loop_end):
            window_start = i
            window_end = i + self.window_size
            
            # Slice window
            w_log_ret = log_ret[window_start:window_end]
            w_vol = volatility[window_start:window_end]
            w_vol_osc = vol_osc[window_start:window_end]
            w_rsi = rsi[window_start:window_end]
            w_trend = trend[window_start:window_end]
            
            # Stack features: (Window, Features)
            # Columns: [Log_Ret, Volatility, Vol_Osc, RSI, Trend]
            features = np.column_stack([
                w_log_ret, 
                w_vol, 
                w_vol_osc, 
                w_rsi, 
                w_trend
            ])
            
            signal_features.append(features)
            
        self.prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        self.signal_features = np.array(signal_features)
        
        # --- DEBUG ---
        print(f"   🔎 Env DEBUG: Prices Range [{np.min(self.prices):.2f}, {np.max(self.prices):.2f}]")
        print(f"   🔎 Env DEBUG: Signal Shape: {self.signal_features.shape}")
        # -------------
        
        return self.prices, self.signal_features

    def _get_observation(self):
        # Get base features from signal_features
        # Index is relative to frame_bound[0]
        idx = self._current_tick - self.frame_bound[0]
        base_obs = self.signal_features[idx] # Shape (window_size, 5)
        
        # Add Position Info (Long=1, Short=0) as a feature channel
        # We expand it to match window size
        position_val = 1.0 if (self._position == 1 or (hasattr(self._position, 'value') and self._position.value == 1)) else 0.0
        position_channel = np.full((self.window_size, 1), position_val, dtype=np.float32)
        
        final_obs = np.hstack([base_obs, position_channel]) # Shape (window_size, 6)
        
        # Flatten for the MLP pilot (which expects 1D input)
        # Pilot Input Dim = window_size * 6
        return final_obs.flatten()

    def _calculate_reward(self, action):
        # Standard Profit reward
        step_reward = 0
        
        current_price = self.prices[self._current_tick]
        last_price = self.prices[self._current_tick - 1]
        price_diff = current_price - last_price
        
        if self._position == 1 or (hasattr(self._position, 'value') and self._position.value == 1): # Long
            step_reward = price_diff
        
        # Simple Transaction Cost
        if action != self._position:
            # We switched
            cost = current_price * self.trade_fee_bid_percent
            step_reward -= cost
            
        return step_reward
