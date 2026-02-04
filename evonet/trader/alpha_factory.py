
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Type, Protocol
import logging

logger = logging.getLogger(__name__)

class AlphaFeature(Protocol):
    """Protocol for all market insight features."""
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

class AlphaFactory:
    """
    A registry for market features. 
    Allows the system to 'grow' by simply adding new feature classes.
    """
    _registry: Dict[str, AlphaFeature] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(feature_cls: Type[AlphaFeature]):
            cls._registry[name] = feature_cls()
            return feature_cls
        return wrapper

    @classmethod
    def apply_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all registered features to the dataframe."""
        for name, feature in cls._registry.items():
            try:
                df = feature.calculate(df)
                logger.info(f"Successfully applied feature: {name}")
            except Exception as e:
                logger.error(f"Failed to apply feature {name}: {e}")
        return df

# --- CORE FEATURES (PRE-REGISTERED) ---

@AlphaFactory.register("stationarity")
class StationarityFeature:
    """Converts price to log-returns (Stationary patterns)."""
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        return df

@AlphaFactory.register("trend_strength")
class TrendStrengthFeature:
    """ADX helps detect if the market is trending (>25) or chopping (<20)."""
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx['ADX_14'] / 100.0 if adx is not None else 0.5
        return df

@AlphaFactory.register("volatility_clustering")
class VolatilityClusteringFeature:
    """ATR measures market fear/uncertainty."""
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Pct'] = df['ATR'] / df['Close']
        return df

@AlphaFactory.register("smart_money_flow")
class SmartMoneyFlowFeature:
    """OBV Slope shows if volume supports the price move."""
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        obv = ta.obv(df['Close'], df['Volume'])
        df['OBV_Slope'] = obv.pct_change(5).fillna(0)
        return df

@AlphaFactory.register("momentum_macd")
class MACDFeature:
    """MACD Histogram captures acceleration/deceleration of the trend."""
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            macd = ta.macd(df['Close'])
            if macd is not None and not macd.empty:
                col = [c for c in macd.columns if 'h' in c.lower()]
                if col:
                    df['MACD_Hist'] = np.tanh(macd[col[0]])
                else:
                    df['MACD_Hist'] = 0
            else:
                df['MACD_Hist'] = 0
        except Exception:
            df['MACD_Hist'] = 0
        return df

@AlphaFactory.register("mean_reversion_bb")
class MeanReversionFeature:
    """Bollinger Bands %B shows if the price is overextended."""
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            bb = ta.bbands(df['Close'], length=20, std=2)
            if bb is not None and not bb.empty:
                 col = [c for c in bb.columns if 'BBP' in c or 'pct' in c.lower()]
                 if col:
                     df['BB_Pct'] = bb[col[0]]
                 else:
                     df['BB_Pct'] = 0.5
            else:
                df['BB_Pct'] = 0.5
        except Exception:
            df['BB_Pct'] = 0.5
        return df

@AlphaFactory.register("tail_risk_vix")
class TailRiskFeature:
    """
    Approximates tail risk using Rolling Kurtosis and Volatility Spikes.
    Markets with high Kurtosis have 'Fat Tails' (more likely to crash).
    """
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # High kurtosis = high chance of outlier moves (Crashes)
        df['Kurtosis_20'] = df['Close'].rolling(20).kurt()
        # Distance from SMA200 as a 'Mean Reversion' or 'Bubble' metric
        sma200 = ta.sma(df['Close'], length=200)
        df['Distance_SMA200'] = (df['Close'] - sma200) / sma200 if sma200 is not None else 0
        return df.fillna(0)

if __name__ == "__main__":
    # Test stub
    data = pd.DataFrame({
        'Open': np.random.randn(300) + 100,
        'High': np.random.randn(300) + 102,
        'Low': np.random.randn(300) + 98,
        'Close': np.random.randn(300) + 100,
        'Volume': np.random.randint(1000, 5000, 300)
    })
    processed = AlphaFactory.apply_all(data)
    print(processed.columns)
