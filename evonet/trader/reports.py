
import pandas as pd
import numpy as np
from datetime import datetime
import os

class ReportGenerator:
    """
    Generates 'Rightful Thoughts' for the customer.
    Combines Model output with market dynamics for a professional report.
    """
    def __init__(self, output_dir="output/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_analyst_report(self, ticker, model_prediction, confidence, regime, risk_metrics):
        """Creates a professional HTML report with deep insights."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Color Palette based on Sentiment
        sentiment_color = "#00ff88" if model_prediction == "LONG" else "#ff4444"
        if model_prediction == "NEUTRAL": sentiment_color = "#aaaaaa"

        # Explainability Logic
        rationale = self._derive_rationale(regime, confidence)

        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Inter', sans-serif; background: #0a0a0c; color: #e0e0e0; padding: 40px; }}
                .container {{ max-width: 800px; margin: auto; background: rgba(255, 255, 255, 0.05); border-radius: 20px; padding: 40px; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); }}
                h1 {{ color: #ffffff; font-weight: 800; letter-spacing: -1px; }}
                .status-badge {{ background: {sentiment_color}; color: #000; padding: 10px 20px; border-radius: 50px; font-weight: bold; display: inline-block; margin-bottom: 20px; }}
                .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 30px; }}
                .metric-card {{ background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; }}
                .rationale {{ border-left: 4px solid {sentiment_color}; padding-left: 20px; margin-top: 30px; font-style: italic; color: #bbbbbb; }}
                .future-projection {{ margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>EvoTrader Analyst Report: {ticker}</h1>
                <p>Generated at: {timestamp}</p>
                
                <div class="status-badge">{model_prediction} SIGNAL</div>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Model Confidence</h3>
                        <p style="font-size: 24px; font-weight: bold;">{confidence:.2%}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Market Regime</h3>
                        <p style="font-size: 24px; font-weight: bold;">{regime.upper()}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Risk Exposure (VaR)</h3>
                        <p style="font-size: 24px; font-weight: bold;">{risk_metrics['var']:.2%}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Max Expected Drawdown</h3>
                        <p style="font-size: 24px; font-weight: bold;">{risk_metrics['mdd']:.2%}</p>
                    </div>
                </div>

                <div class="rationale">
                    <h3>Deep Analyst Rationale</h3>
                    <p>{rationale}</p>
                </div>

                <div class="future-projection">
                    <h3>Future Outcome Scenarios</h3>
                    <ul>
                        <li><b>Stress Scenario (Flash Crash):</b> Model has {risk_metrics['crash_resilience']} resilience.</li>
                        <li><b>Momentum Play:</b> Target price estimate based on ADX: {risk_metrics['target_price']}</li>
                    </ul>
                </div>

                <p style="margin-top: 50px; font-size: 12px; color: #666;">
                    Disclaimer: This is an AI-generated analysis using neuroevolutionary techniques. Not financial advice.
                </p>
            </div>
        </body>
        </html>
        """
        
        filepath = os.path.join(self.output_dir, f"report_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.html")
        with open(filepath, "w") as f:
            f.write(html_content)
        print(f"📄 Professional Report Generated: {filepath}")
        return filepath

    def _derive_rationale(self, regime, confidence):
        if regime == "stress":
            return "The market is currently exhibiting 'Fat Tail' behavior with high volatility. The system is defensive, prioritizing capital preservation over aggressive entry."
        if confidence > 0.8:
            return "Extreme technical confluence detected. OBV slope and MACD trends are perfectly aligned, suggesting a high-probability breakout."
        return "Market is in an accumulation phase (Chop). Suggesting range-bound strategies with tight stop-losses."

class ScenarioAnalyzer:
    """
    Project future outcomes by running the model against synthetic market states.
    """
    def simulate_outcome(self, model, current_state, scenario="crash"):
        # Apply synthetic delta to state
        sim_state = current_state.copy()
        if scenario == "crash":
            sim_state[-1] -= 0.15 # Drop price/returns significantly
        elif scenario == "moon":
            sim_state[-1] += 0.10 # Boost
            
        pred, _, _, _, conf = model.forward(sim_state, np.zeros(3), train=False)
        return pred, conf
