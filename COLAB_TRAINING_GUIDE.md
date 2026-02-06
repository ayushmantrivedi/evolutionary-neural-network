# 🚀 COLAB GPU TRAINING - QUICK START GUIDE

## 📋 Prerequisites
1. Google account with Colab access
2. Your `evonet` folder (entire project)

## 🎯 Step-by-Step Instructions

### Option A: Upload Project Files (Recommended for Quick Start)

1. **Open Google Colab**
   - Go to: https://colab.research.google.com
   - Create a new notebook

2. **Enable GPU Runtime**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU (T4)
   ```

3. **Upload Project Files**
   - Click the 📁 folder icon on the left sidebar
   - Upload your entire `evonet` folder
   - Upload `colab_ultimate_training.py`

4. **Run Training**
   ```python
   # In a Colab cell:
   !python colab_ultimate_training.py
   ```

5. **Download Results** (after 2-4 hours)
   - `ultimate_brain_colab.pkl` ← **This is your trained brain!**
   - `training_report.txt` ← Performance metrics for pitch deck

---

### Option B: GitHub Integration (Best for Team Collaboration)

1. **Push to GitHub** (on your local machine)
   ```bash
   cd evolutionary-neural-network
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/evo-trader.git
   git push -u origin main
   ```

2. **Clone in Colab** (in a Colab cell)
   ```python
   !git clone https://github.com/YOUR_USERNAME/evo-trader.git
   %cd evo-trader
   !python colab_ultimate_training.py
   ```

---

## ⚙️ Training Configuration

### Current Settings (Optimized for Colab Free Tier)
- **Generations**: 200 (2-4 hours on T4 GPU)
- **Population**: 100 genomes
- **Data**: BTC-USD 2018-2024 (Daily)
- **Checkpoints**: Every 25 generations

### Premium Upgrade (For Colab Pro)
If you have Colab Pro (longer runtime), edit these in `colab_ultimate_training.py`:

```python
TOTAL_GENERATIONS = 500     # More depth
POPULATION_SIZE = 200       # More diversity
# Upgrade to hourly data for better granularity:
# (Requires modifying DataFetcher to pull 1h interval)
```

---

## 📊 Expected Performance Metrics

**Target for Finance Pitch:**
- ✅ Sortino Ratio: **> 3.0** (excellent risk-adjusted returns)
- ✅ Max Drawdown: **< 15%** (capital preservation)
- ✅ Fitness: **> 30** (composite score)

**What You'll Get:**
- Trained brain file ready for production
- Performance report for investor presentations
- Checkpoint files (in case of interruption)

---

## 🎬 After Training

### 1. Validate Locally
```bash
# On your machine, after downloading ultimate_brain_colab.pkl:
python tests/deep_backtest.py
```

### 2. Prepare Pitch Deck
Key talking points from `training_report.txt`:
- "Trained on 6+ years of BTC data (2018-2024)"
- "200 generations of evolutionary optimization"
- "Achieved Sortino ratio of X.XX"
- "Drawdown controlled to X.X%"

### 3. Next Steps for Production
- Upgrade to 1-minute data
- Implement walk-forward validation
- Paper trading on Binance testnet
- Scale with paid GPU clusters

---

## ⚠️ Troubleshooting

### "Runtime disconnected"
- Colab free tier has 12-hour limit
- Solution: Use checkpoints to resume training
- Or upgrade to Colab Pro

### "Out of Memory"
- Reduce `POPULATION_SIZE` to 50
- Reduce `MAX_EPISODE_STEPS` to 500

### "Module not found: evonet"
- Ensure the `evonet` folder is uploaded
- Check that all subfolders (core, trader, api) are present

---

## 💡 Pro Tips

1. **Monitor Progress**: Colab shows generation updates in real-time
2. **Download Early**: Download checkpoint files periodically (don't wait for the end)
3. **GPU Check**: Script auto-detects GPU - look for "✅ GPU Detected" message
4. **Data Quality**: For premium results, upgrade to 1-hour or 1-minute candles

---

## 🏆 Success Criteria

**Good Enough for Pitch:**
- Sortino > 2.5
- Max DD < 20%
- 200+ generations completed

**Investor-Grade:**
- Sortino > 3.5
- Max DD < 12%
- 500+ generations
- Walk-forward validation

---

Ready to train? Just paste `colab_ultimate_training.py` into Google Colab and run! 🚀
