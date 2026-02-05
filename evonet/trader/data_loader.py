
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import logging


from abc import ABC, abstractmethod
from evonet.trader.alpha_factory import AlphaFactory

class BaseFetcher(ABC):
    """Abstract base for all data providers."""
    def __init__(self, ticker: str, start_date: str, end_date: str, interval: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.df = None

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        pass

class YFinanceFetcher(BaseFetcher):
    """Fetches data from Yahoo Finance."""
    def fetch(self) -> pd.DataFrame:
        print(f"[FETCH] Fetching {self.ticker} [{self.interval}] via YFinance...")
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.interval)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
        # Ensure UTC and Timezone-Naive for consistent comparisons
        if self.df.index.tz is not None:
            self.df.index = self.df.index.tz_convert(None)
        return self.df

class CCXTFetcher(BaseFetcher):
    """Generic Crypto Fetcher via CCXT (FinRL-Meta approach)."""
    def fetch(self) -> pd.DataFrame:
        try:
            import ccxt
            print(f"[FETCH] Fetching {self.ticker} via CCXT Generic...")
            # Placeholder for actual exchange selection
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(self.ticker, self.interval)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.df = df
            return self.df
        except ImportError:
            print("[WARN] CCXT not installed. Falling back to YFinance mock data.")
            return YFinanceFetcher(self.ticker, self.start_date, self.end_date, self.interval).fetch()

class BinanceFetcher(BaseFetcher):
    """Specialized Binance Fetcher for high-frequency data."""
    def fetch(self) -> pd.DataFrame:
        print(f"[FETCH] Fetching {self.ticker} via specialized Binance API...")
        # In professional production, use Binance Python SDK or REST directly
        return CCXTFetcher(self.ticker, self.start_date, self.end_date, self.interval).fetch()

class DataFetcher:
    """
    Main Orchestrator for Data. 
    Can switch providers and applies AlphaFactory insights.
    """
    def __init__(self, ticker="BTC-USD", start_date="2018-01-01", end_date=None, interval="1d", provider="yf"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date # None = Now
        self.interval = interval
        
        if provider == "yf":
            self.fetcher = YFinanceFetcher(ticker, start_date, end_date, interval)
        elif provider == "binance":
            self.fetcher = BinanceFetcher(ticker, start_date, end_date, interval)
        elif provider == "ccxt":
            self.fetcher = CCXTFetcher(ticker, start_date, end_date, interval)
        else:
            raise ValueError(f"Provider {provider} not supported yet. Suggested: [yf, binance, ccxt]")
        
        self.df = None
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(self, use_cache=True):
        cache_file = os.path.join(self.cache_dir, f"{self.ticker}_{self.interval}.csv")
        
        if use_cache and os.path.exists(cache_file):
            print(f"[CACHE] Loading cached data...")
            self.df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self.df
            
        self.df = self.fetcher.fetch()
        self.df.to_csv(cache_file)
        return self.df

    def process(self):
        """Applies AlphaFactory features & Market Dynamics."""
        if self.df is None:
            self.fetch_data()
            
        print("[ALPHA] Applying Modular Market Intelligence via AlphaFactory...")
        self.df = AlphaFactory.apply_all(self.df)
        
        # Clean & Prep
        self.df.dropna(inplace=True)
        return self.df

    def inject_stress_scenario(self, scenario_type="crash"):
        """
        [STRESS TESTER]
        Manually modifies data to simulate extreme market events.
        Used to ensure the 'Growth' of the model into robust territory.
        """
        if self.df is None: self.process()
        
        if scenario_type == "crash":
            print("[STRESS] INJECTING SYNTHETIC CRASH SCENARIO (-30% in 5 steps)")
            # Take a slice of current data and drop it significantly
            last_idx = len(self.df) // 2
            for i in range(5):
                self.df.iloc[last_idx + i, self.df.columns.get_loc('Close')] *= 0.85
                self.df.iloc[last_idx + i, self.df.columns.get_loc('Log_Ret')] = -0.15
        
        return self.df

    def split_by_regime(self, regime_name):
        """Legacy helper for specialist training."""
        regimes = {
            "bull_2020": ("2020-04-01", "2021-04-01"),
            "bear_2022": ("2021-11-10", "2022-11-10"),
            "chop_2023": ("2023-01-01", "2023-10-01")
        }
        start, end = regimes[regime_name]
        mask = (self.df.index >= start) & (self.df.index <= end)
        return self.df.loc[mask]

if __name__ == "__main__":
    df_loader = DataFetcher()
    df_loader.fetch_data()
    processed = df_loader.process()
    print(f"[OK] Data processed. Features: {processed.columns.tolist()}")
    
    # Showcase 'Growth' potential with Stress Injection
    crashed = df_loader.inject_stress_scenario("crash")
    print(f"[STRESS] Stress Test Injected. Min Log_Ret: {crashed['Log_Ret'].min():.4f}")

if __name__ == "__main__":
    # Test
    fetcher = DataFetcher()
    fetcher.fetch_data()
    fetcher.add_indicators()
    bull = fetcher.split_by_regime("bull_2020")
    bear = fetcher.split_by_regime("bear_2022")
    print("[OK] Data Loader Verified.")
