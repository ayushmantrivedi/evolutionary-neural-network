
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
    def __init__(self, ticker="BTC-USD", start_date="2020-01-01", end_date="2023-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        
    def fetch_data(self):
        print(f"📉 Fetching {self.ticker} data from {self.start_date} to {self.end_date}...")
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if self.df.empty:
            raise ValueError("No data fetched! Check internet connection or ticker symbol.")
            
        print(f"   -> Fetched {len(self.df)} rows.")
        return self.df
        
    def add_advanced_features(self):
        """Adds professional financial features (Log-Returns, Volatility, Volume)."""
        if self.df is None:
            self.fetch_data()
            
        # Ensure flat columns
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
            
        # 1. Log Returns (Stationary Price Movement)
        # ln(P_t / P_{t-1})
        self.df['Log_Ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # 2. Realized Volatility (Risk Signal)
        # Rolling Std Dev of Log Returns (20 days)
        self.df['Volatility'] = self.df['Log_Ret'].rolling(window=20).std()
        
        # 3. Volume Oscillator (Activity)
        # (Vol - AvgVol) / AvgVol
        avg_vol = self.df['Volume'].rolling(window=20).mean()
        self.df['Vol_Osc'] = (self.df['Volume'] - avg_vol) / (avg_vol + 1e-8)
        
        # 4. RSI (Momentum) - Keep this, it's good
        self.df['RSI'] = ta.rsi(self.df['Close'], length=14) / 100.0 # Normalize 0-1
        
        # 5. Price Trend (Distance from SMA50)
        # (Price - SMA50) / SMA50 -> Normalized trend strength
        sma50 = ta.sma(self.df['Close'], length=50)
        self.df['Trend_50'] = (self.df['Close'] - sma50) / (sma50 + 1e-8)
        
        # Clean NaNs (Rolling windows create NaNs at start)
        self.df.dropna(inplace=True)
        
        # Fill strict infinite values if any
        self.df.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        print(f"   -> Advanced Features Added: {len(self.df)} rows ready.")
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
