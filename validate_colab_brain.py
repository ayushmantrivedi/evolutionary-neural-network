"""
BRAIN VALIDATOR - Test your Colab-trained brain locally
=========================================================

Run this after downloading ultimate_brain_colab.pkl from Google Colab
to validate the model before pitching to investors.

Usage:
    python validate_colab_brain.py ultimate_brain_colab.pkl
"""

import pickle
import sys
import numpy as np
from evonet.trader.data_loader import DataFetcher
from evonet.trader.environment import FinancialRegimeEnv

def validate_brain(brain_path: str):
    """Validates a Colab-trained brain on fresh data"""
    
    print("\n" + "="*70)
    print("🔍 BRAIN VALIDATION REPORT")
    print("="*70 + "\n")
    
    # Load brain
    print(f"📥 Loading brain: {brain_path}")
    try:
        with open(brain_path, 'rb') as f:
            pilot = pickle.load(f)
        print(f"✅ Brain loaded successfully")
        print(f"   - Input Dimension: {pilot.input_dim}")
        print(f"   - Output Dimension: {pilot.output_dim}")
        print(f"   - Population Size: {pilot.net.pop_size}\n")
    except Exception as e:
        print(f"❌ Error loading brain: {e}")
        return
        
    # Fetch validation data (out-of-sample: 2024 only)
    print("📊 Fetching Validation Data (2024)...")
    fetcher = DataFetcher("BTC-USD", start_date="2024-01-01", end_date="2024-12-31", provider="yf")
    df = fetcher.fetch_data()
    df = fetcher.process()
    print(f"✅ Loaded {len(df)} days of fresh data\n")
    
    # Create environment
    print("🌍 Setting up validation environment...")
    window_size = 20
    safe_end = len(df) - (window_size * 3)
    env = FinancialRegimeEnv(df, frame_bound=(window_size, safe_end), 
                            window_size=window_size, fee=0.001)
    print("✅ Environment ready\n")
    
    # Run validation episode
    print("🧪 Running validation episode...")
    state, _ = env.reset()
    equity = 1.0
    equity_curve = [1.0]
    returns = []
    actions_taken = {'Short': 0, 'Neutral': 0, 'Long': 0}
    terminated = False
    steps = 0
    max_steps = 1000
    
    while not terminated and steps < max_steps:
        # Use genome 0 (best from training)
        probs, conf = pilot.net.predict(state, 0)
        action = np.argmax(probs)
        
        # Track actions
        action_names = ['Short', 'Neutral', 'Long']
        actions_taken[action_names[action]] += 1
        
        state, reward, terminated, _, _ = env.step(action)
        equity *= np.exp(reward)
        equity_curve.append(equity)
        returns.append(reward)
        steps += 1
        
    env.close()
    
    # Calculate metrics
    print("✅ Validation complete!\n")
    print("="*70)
    print("📈 PERFORMANCE METRICS (2024 Out-of-Sample)")
    print("="*70 + "\n")
    
    returns = np.array(returns)
    total_return = (equity - 1.0) * 100
    
    # Sharpe & Sortino
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)
    downside = returns[returns < 0]
    down_std = np.std(downside) if len(downside) > 1 else 1e-6
    sortino = (np.mean(returns) / (down_std + 1e-9)) * np.sqrt(252)
    
    # Max Drawdown
    peak = 1.0
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
            
    # Win Rate
    wins = sum(1 for r in returns if r > 0)
    win_rate = (wins / len(returns)) * 100 if len(returns) > 0 else 0
    
    print(f"💰 Total Return:     {total_return:+.2f}%")
    print(f"📊 Sharpe Ratio:     {sharpe:.2f}")
    print(f"🎯 Sortino Ratio:    {sortino:.2f}")
    print(f"📉 Max Drawdown:     {max_dd:.1%}")
    print(f"✅ Win Rate:         {win_rate:.1f}%")
    print(f"🔄 Total Trades:     {steps}")
    
    print("\n📌 Action Distribution:")
    for action, count in actions_taken.items():
        pct = (count / steps) * 100 if steps > 0 else 0
        print(f"   {action:8s}: {count:4d} ({pct:5.1f}%)")
        
    # Investment Grade Assessment
    print("\n" + "="*70)
    print("🏆 INVESTMENT GRADE ASSESSMENT")
    print("="*70 + "\n")
    
    grade = "⭐⭐⭐⭐⭐ EXCELLENT"
    if sortino < 1.5:
        grade = "⭐ NEEDS IMPROVEMENT"
    elif sortino < 2.5:
        grade = "⭐⭐⭐ GOOD"
    elif sortino < 3.5:
        grade = "⭐⭐⭐⭐ VERY GOOD"
        
    print(f"Grade: {grade}")
    
    if sortino >= 2.5 and max_dd < 0.20:
        print("\n✅ PITCH READY!")
        print("   - Sortino > 2.5 ✓")
        print("   - Drawdown < 20% ✓")
        print("\n📝 Recommended Pitch Points:")
        print("   1. 'Achieved Sortino of {:.2f} on out-of-sample data'".format(sortino))
        print("   2. 'Controlled drawdown to {:.1%} during validation'".format(max_dd))
        print("   3. 'Trained on 6 years of data with 200+ generations'")
    else:
        print("\n⚠️  NEEDS MORE TRAINING")
        print("   Recommendations:")
        if sortino < 2.5:
            print("   - Increase generations to 500+")
        if max_dd >= 0.20:
            print("   - Increase drawdown penalty in fitness function")
        print("   - Consider upgrading to hourly data")
        
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_colab_brain.py ultimate_brain_colab.pkl")
        sys.exit(1)
        
    brain_path = sys.argv[1]
    validate_brain(brain_path)
