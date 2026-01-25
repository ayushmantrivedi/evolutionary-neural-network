
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import logging

class DataFetcher:
    """
    Fetches and processes financial data for EvoTrader.
    Splits data into regimes (Bull/Bear) for specialist training.
    """
    def __init__(self, ticker="BTC-USD", start_date="2020-01-01", end_date="2023-12-31", interval="1d"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.df = None
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def fetch_data(self):
        # Cache Filename: ticker_interval_start_end.csv
        safe_start = self.start_date.replace(":", "").replace("-", "")
        safe_end = self.end_date.replace(":", "").replace("-", "")
        cache_file = os.path.join(self.cache_dir, f"{self.ticker}_{self.interval}_{safe_start}_{safe_end}.csv")
        
        if os.path.exists(cache_file):
            print(f"⚡ Loading cached data from {cache_file}...")
            self.df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"   -> Loaded {len(self.df)} rows.")
            return self.df
            
        print(f"📉 Fetching {self.ticker} [{self.interval}] from {self.start_date} to {self.end_date}...")
        
        # YFinance Limitation: 1h data only available for last 730 days.
        # If interval is 1h and date range > 730 days, warn user.
        if self.interval == "1h":
             print("⚠️ WARNING: Public API (yfinance) limits 1h data to last 730 days.")
             
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.interval)
        
        if self.df.empty:
            raise ValueError("No data fetched! Check internet/ticker or yfinance 730-day limit for 1h data.")
            
        # Clean multi-index if present (yfinance update)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
            
        print(f"   -> Fetched {len(self.df)} rows. Caching...")
        self.df.to_csv(cache_file)
        return self.df
        
    def add_advanced_features(self):
        """Adds professional financial features (Alpha Signals)."""
        if self.df is None:
            self.fetch_data()
            
        # Ensure flat columns
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
            
        # --- 1. BASE LOG RETURNS ---
        # ln(P_t / P_{t-1})
        self.df['Log_Ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # --- 1.5 REGIME FILTERS (SMAs) ---
        self.df['SMA_50'] = ta.sma(self.df['Close'], length=50)
        self.df['SMA_200'] = ta.sma(self.df['Close'], length=200)
        
        # --- 2. TREND FILTER (ADX) ---
        # ADX > 25 = Trending, ADX < 20 = Chopping
        adx_df = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        self.df['ADX'] = adx_df['ADX_14'] / 100.0  # Normalize 0-1
        
        # --- 3. MOMENTUM (MACD) ---
        # MACD Histogram captures acceleration
        macd_df = ta.macd(self.df['Close'])
        # Normalize MACD Histogram roughly (-1 to 1 for crypto) - using hyperbolic tangent to bound
        self.df['MACD_Hist'] = np.tanh(macd_df['MACDh_12_26_9']) 
        
        # --- 4. MEAN REVERSION (Bollinger Bands %B) ---
        # %B > 1 (Overbought), %B < 0 (Oversold)
        try:
            bb_df = ta.bbands(self.df['Close'], length=20, std=2)
            # Auto-detect the %B column (it usually ends with P or BBP)
            bb_cols = bb_df.columns.tolist()
            bbp_col = next((c for c in bb_cols if c.startswith('BBP')), None)
            if bbp_col:
                 self.df['BB_Pct'] = bb_df[bbp_col]
            else:
                 print(f"Warning: Could not find BBP column in {bb_cols}. Using fallback.")
                 self.df['BB_Pct'] = 0.5 
        except Exception as e:
            print(f"Error computing BB: {e}")
            self.df['BB_Pct'] = 0.5
        
        # --- 5. SMART MONEY FLOW (On-Balance Volume Slope) ---
        # Is volume supporting the price?
        obv = ta.obv(self.df['Close'], self.df['Volume'])
        self.df['OBV_Slope'] = obv.pct_change(5).fillna(0) * 100 # 5-day slope scaled
        
        # --- 6. VOLATILITY NORMALIZER (ATR %) ---
        # Critical for sizing positions across different years
        atr = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        self.df['ATR_Pct'] = atr / self.df['Close']
        self.df['ATR'] = atr # Keep raw ATR for slippage model
        
        # Clean NaNs (Rolling windows create NaNs at start)
        self.df.dropna(inplace=True)
        
        # Fill strict infinite values if any
        self.df.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        print(f"   -> Advanced Alpha Features Added: {len(self.df)} rows ready.")
        print("      [ADX, MACD, BB%, OBV, ATR]")
        return self.df
        
    def add_indicators(self):
        """Legacy wrapper for backward compatibility."""
        return self.add_advanced_features()
        
    def get_regime_dates(self):
        """Returns hardcoded date ranges for known regimes (for training)."""
        regimes = {
            "bull_2020": ("2020-04-01", "2021-04-01"), # Post-Covid Pump
            "bear_2022": ("2021-11-10", "2022-11-10"), # The Great Crash
            "chop_2023": ("2023-01-01", "2023-10-01")  # Sideways / Accumulation
        }
        return regimes
        
    def split_by_regime(self, regime_name):
        """Returns a subset of data corresponding to a specific regime."""
        if self.df is None:
            self.add_indicators()
            
        regimes = self.get_regime_dates()
        if regime_name not in regimes:
            raise ValueError(f"Unknown regime: {regime_name}. Available: {list(regimes.keys())}")
            
        start, end = regimes[regime_name]
        mask = (self.df.index >= start) & (self.df.index <= end)
        subset = self.df.loc[mask]
        print(f"   -> Extracted Regime '{regime_name}': {len(subset)} rows ({start} to {end})")
        return subset

if __name__ == "__main__":
    # Test
    fetcher = DataFetcher()
    fetcher.fetch_data()
    fetcher.add_indicators()
    bull = fetcher.split_by_regime("bull_2020")
    bear = fetcher.split_by_regime("bear_2022")
    print("✅ Data Loader Verified.")
