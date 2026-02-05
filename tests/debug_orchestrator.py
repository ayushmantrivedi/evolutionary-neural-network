
import logging
import sys
import os
import traceback

# Ensure evonet is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.trader.orchestrator import OmniOrchestrator

# Configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugOrchestrator")

if __name__ == "__main__":
    print("[DEBUG] Starting Orchestrator Check...")
    try:
        # TEST 1: Direct DataFetcher Check
        from evonet.trader.data_loader import DataFetcher
        print("   - [STEP 1] Testing DataFetcher for BTC-USD...")
        f = DataFetcher("BTC-USD", provider="binance") # maps to YF in code? No, binance uses CCXT
        # Wait, if binance uses CCXT and CCXT is missing, it mocks YF.
        # Let's see what happens.
        df = f.fetch_data(use_cache=False)
        print(f"   - Fetch complete. Rows: {len(df) if df is not None else 'None'}")
        
        if df is None or len(df) < 200:
             print("   [CRITICAL] Data Fetch invalid length!")
        
        proc_df = f.process()
        print(f"   - Process complete. Rows: {len(proc_df) if proc_df is not None else 'None'}")
        
        # TEST 2: Orchestrator
        print("   - [STEP 2] Running full Orchestrator...")
        orch = OmniOrchestrator()
        results = orch.analyze_market_unified(["BTC-USD"], ["binance"])
        
        if "BTC-USD" in results["assets"]:
            print("   [SUCCESS] Asset processed.")
        else:
            print("   [FAILURE] Asset missing from results.")
            
    except Exception:
        print("   [CRITICAL EXCEPTION Traceback]:")
        traceback.print_exc()
