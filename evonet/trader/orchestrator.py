import pandas as pd
import numpy as np
import logging
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from evonet.trader.data_loader import DataFetcher
from evonet.trader.alpha_factory import AlphaFactory
from evonet.trader.reports import ReportGenerator, ScenarioAnalyzer
from evonet.trader.risk_manager import RiskManager, RiskConfig
from evonet.core.network import MultiClassEvoNet
from evonet.config import WINDOW_SIZE
from train_memory_autopilot import MemoryEvoPilot
import pickle

logger = logging.getLogger(__name__)

class OmniOrchestrator:
    """
    The master 'Financial Mind' that coordinates multi-source data feeds,
    cross-field alpha generation, and unified analyst reporting.
    """
    def __init__(self, brain_path: Optional[str] = "evotrader_brain.pkl"):
        self.fetchers: Dict[str, DataFetcher] = {}
        self.report_gen = ReportGenerator()
        self.scenario_analyzer = ScenarioAnalyzer()
        
        # [NEW] Risk Manager Integration
        self.risk_manager = RiskManager()
        
        # Load the Master Brain (The Pilot)
        self.pilot = None
        if os.path.exists(brain_path):
            try:
                with open(brain_path, 'rb') as f:
                    self.pilot = pickle.load(f)
                logger.info(f"Master Brain loaded from {brain_path}")
            except Exception as e:
                logger.error(f"Failed to load Master Brain: {e}")

    def analyze_market_unified(self, tickers: List[str], providers: List[str]) -> Dict[str, Any]:
        """
        Runs a parallelized full-pipeline analysis across multiple market fields.
        
        Args:
            tickers: List of assets (e.g. ['BTC-USD', 'AAPL'])
            providers: List of data sources (e.g. ['binance', 'yf'])
        """
        if len(tickers) != len(providers):
            raise ValueError("Ticker and Provider lists must be of the same length.")

        logger.info(f"Omni-Orchestration started for: {tickers}")
        
        results = {}
        
        def process_asset(ticker, provider):
            try:
                fetcher = DataFetcher(ticker, provider=provider)
                df = fetcher.fetch_data()
                df = fetcher.process() # Modular AlphaFactory
                
                if df is None or df.empty:
                    logger.warning(f"Orchestrator: No data found for {ticker}")
                    return None
                
                # --- AI Outlook ---
                ai_action = "NEUTRAL"
                confidence = 0.5
                if self.pilot:
                    # Prepare state from last window
                    from evonet.config import WINDOW_SIZE
                    if len(df) >= WINDOW_SIZE:
                        # Extract state (simplified for proof, real uses env.observation_space)
                        state_slice = df.tail(WINDOW_SIZE)
                        # State should be flattened 10-feature vector
                        features = ['Log_Ret', 'ADX', 'MACD_Hist', 'BB_Pct', 'OBV_Slope', 'ATR_Pct', 'Kurtosis_20', 'Distance_SMA200', 'Log_Ret_5d']
                        
                        # Correct State Construction (must matches FinancialRegimeEnv)
                        # 1. Base Features (20, 9)
                        base_obs = state_slice[features].values
                        
                        # 2. Position Embedding (Neutral = 0.0) -> (20, 1)
                        position_col = np.zeros((len(base_obs), 1))
                        
                        # 3. Stack & Flatten -> (20, 10) -> (200,)
                        full_state = np.hstack([base_obs, position_col]).flatten()
                        
                        y_pred, conf = self.pilot.net.predict(full_state, 0)
                        action_idx = np.argmax(y_pred)
                        actions = ["BEARISH (Short)", "NEUTRAL", "BULLISH (Long)"]
                        ai_action = actions[action_idx]
                        confidence = float(conf)
                        
                        # [NEW] COMPLIANCE GATE
                        # Validate trade with Risk Manager
                        signal_payload = {"action": ai_action, "confidence": confidence}
                        is_approved, reason = self.risk_manager.validate_trade(signal_payload, df)
                        
                        if not is_approved:
                            logger.warning(f"[RISK MANAGER]: Trade Halted for {ticker}. Reason: {reason}")
                            ai_action = f"NEUTRAL ({reason})"
                            # Override action to Neutral

                last_price = df['Close'].iloc[-1]
                
                return {
                    "ticker": ticker,
                    "price": last_price,
                    "outlook": ai_action,
                    "confidence": confidence,
                    "df": df
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to orchestrate {ticker}: {e}")
                return None

        # Execute in parallel to save time (Real-World requirement for speed)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_asset, t, p) for t, p in zip(tickers, providers)]
            for future in futures:
                res = future.result()
                if res:
                    results[res['ticker']] = res

        # Generate Unified Intelligence Summary
        summary = self._generate_meta_analysis(results)
        
        return {
            "assets": results,
            "meta_analysis": summary
        }

    def _generate_meta_analysis(self, results: Dict[str, Any]) -> str:
        """
        Synthesizes cross-field intelligence.
        """
        if not results: return "No data available."
        
        analysis = "Omi-Market Intelligence Summary:\n"
        for ticker, data in results.items():
            analysis += f"- {ticker}: Price {data['price']:.2f} | AI Outlook: {data['outlook']} (Conf: {data['confidence']:.2%})\n"
        
        # Cross-Field Logic (e.g. BTC vs Equities)
        if "BTC-USD" in results and "AAPL" in results:
            btc_ret = results["BTC-USD"]["df"]['Log_Ret'].iloc[-5:].sum()
            aapl_ret = results["AAPL"]["df"]['Log_Ret'].iloc[-5:].sum()
            
            if btc_ret > 0 and aapl_ret > 0:
                analysis += "\n[Meta Insight] Risk-ON Sentiment detected across both Equities and Crypto."
            elif btc_ret < 0 and aapl_ret < 0:
                analysis += "\n[Meta Insight] Risk-OFF Liquidity drain detected globally."
            else:
                analysis += "\n[Meta Insight] Market divergence: Crypto and Stocks are decoupled."
                
        return analysis

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    orch = OmniOrchestrator()
    summary = orch.analyze_market_unified(["BTC-USD", "AAPL"], ["binance", "yf"])
    print(summary["meta_analysis"])
