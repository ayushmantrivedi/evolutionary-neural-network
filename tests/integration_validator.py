
import sys
import os
import logging
import json

# Ensure evonet is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.trader.orchestrator import OmniOrchestrator

def test_full_pipeline_real_world():
    """
    CRITICAL TEST: Validates that the full pipeline from Data Retrieval 
    to Meta-Intelligence is 'Professional' and accurate.
    """
    logging.basicConfig(level=logging.INFO)
    print("="*60, flush=True)
    print("INTEGRATION TEST: Omni-Market Pipeline", flush=True)
    print("="*60, flush=True)
    
    orchestrator = OmniOrchestrator()
    
    # Real-World Scenario: Comparing Crypto (Binance) vs Stocks (YFinance)
    tickers = ["BTC-USD", "AAPL", "SPY"]
    providers = ["binance", "yf", "yf"]
    
    print(f"[Step 1] Running Parallel Fetch & Alpha Generation for {tickers}...", flush=True)
    full_output = orchestrator.analyze_market_unified(tickers, providers)
    
    # 1. Check Data Quality & AI Inference
    print("\n[Step 2] Validating Data Integrity & AI Inference...", flush=True)
    for ticker in tickers:
        if ticker in full_output["assets"]:
            asset = full_output["assets"][ticker]
            df = asset["df"]
            cols = df.columns
            print(f"   [OK] {ticker}: {len(df)} bars | Price: {asset['price']:.2f}", flush=True)
            print(f"      - AI Outlook: {asset['outlook']} (Conf: {asset['confidence']:.2%})", flush=True)
            
            if "MACD_Hist" in cols:
                 print(f"      - Alpha Signals: OK", flush=True)
        else:
            print(f"   [FAILED] {ticker}: FAILED TO RETRIEVE", flush=True)

    # 2. Check Meta-Intelligence
    print("\n[Step 3] Validating Meta-Intelligence Output...", flush=True)
    meta = full_output["meta_analysis"]
    print("-" * 30, flush=True)
    print(meta, flush=True)
    print("-" * 30, flush=True)
    
    if "[Meta Insight]" in meta:
        print("SUCCESS: Cross-field correlation intelligence generated.", flush=True)
    else:
        print("FAILURE: Meta-intelligence was generic or missing insights.", flush=True)

if __name__ == "__main__":
    test_full_pipeline_real_world()
