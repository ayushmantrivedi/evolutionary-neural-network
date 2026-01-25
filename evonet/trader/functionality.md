# EvoTrader System Architecture

This document outlines the internal architecture of the **EvoTrader** system and explains why our **"Evolutionary" approach** outperforms traditional Deep Learning in financial markets.

---

## 1. The Workflow (The Pipeline)

**How does a raw Bitcoin price become a Buy/Sell decision?**

### Step 1: The "Eyes" (Data & Features)
- **Raw Data:** We pull raw OHLCV (Open, High, Low, Close, Volume) data from the market.  
- **Feature Engineering:** Usage of Log-Returns and Volatility.  
- **Traditional AI:** Looks at price (e.g., `$65,000`). This confuses it when price drops to `$20,000`.  
- **EvoTrader:** Looks at **Change (+2%)**. This pattern looks the same in 2020 and 2026. This is called **Stationarity**.  

### Step 2: The "Brain Switch" (Meta-Controller)
- **Regime Detection:** Before trading, the system asks: *"What is the weather?"* (Bullish? Bearish? Choppy?).  
- **Memory Retrieval:**  
  - If Bull → injects `memory("bull")` vector into the neural network.  
  - If Bear → injects `memory("bear")` vector.  
- **Result:** The AI instantly changes its personality from *Aggressive Buyer* to *Short Seller*.  

### Step 3: The "Pilot" (Neural Network)
- **Input:** Normalized market data (Window of 10 days).  
- **Processing:** A Dense Neural Network (**EvoNet**) processes the pattern.  
- **Action:** Outputs a probability: **Buy, Sell, or Hold**.  

---

## 2. EvoTrader vs. Traditional AI

**Why did we build it this way?**

| Feature       | Traditional AI (LSTM/Transformer) | EvoTrader (Neuroevolution) |
|---------------|-----------------------------------|-----------------------------|
| **Objective** | Predicts Price (e.g., "Tomorrow will be $60,001"). | Maximizes Profit (e.g., "Buying here makes money"). |
| **Math**      | Uses Calculus (Backpropagation). Requires smooth errors. | Uses Darwinism (Selection). Can optimize jagged rewards like "Profit". |
| **The Trap**  | Overfitting: Memorizes the past perfectly but fails today. | Generalization: Only robust strategies survive the "Evolutionary Tournament". |
| **Adaptability** | Static: Once trained, it is frozen. If the market changes, it fails. | Dynamic: Swaps strategies instantly using "Directional Memory". |

---

## 3. Why Our "Novel" Algorithms Are Better

You asked about the specific **Novel Evolutionary Algorithms** we used. Here is why they shine in Finance:

### A. Non-Differentiable Rewards (The "Profit" Problem)
- **Problem:** In real trading, you want to optimize for **Sharpe Ratio (Risk-Adjusted Return)**.  
  There is no calculus formula for this. Traditional AI cannot train correctly for it.  
- **Evo Solution:** Evolution doesn’t care about calculus. It just asks:  
  *"Did this mutant make money with low risk?"*  
  - If yes → it breeds.  
  - If no → it dies.  
  This allows us to optimize for **Real World Goals** directly.  

### B. The "Memory Vector" (Anti-Fragility)
- **Problem:** Markets are **Non-Stationary** (the rules change). A strategy that works in 2021 kills you in 2022.  
- **Evo Solution:** We don’t force one brain to learn everything. We isolate specific skills into Vectors:  
  ```python
  vector_bull = "How to Buy"
  vector_bear = "How to Short"
