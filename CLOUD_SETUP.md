# EvoTrader Cloud Setup — Complete Guide

## What You Just Built

3 files were added to your repo:
- `cloud_signal.py` — the brain that runs daily
- `.github/workflows/daily_signal.yml` — the scheduler  
- `get_chat_id.py` — one-time helper to get your Telegram chat ID

The system runs automatically every weekday at **3:37 PM IST** on GitHub's free servers.

---

## Setup Steps (One-Time, ~5 Minutes)

### Step 1 — Get Your Telegram Chat ID

1. Open Telegram → search for your bot → send it **any message** (e.g. `/start`)
2. Run on your laptop:
   ```bash
   python get_chat_id.py
   ```
3. Copy the number it prints (looks like `1234567890`)

### Step 2 — Add GitHub Secrets

1. Go to your repo on GitHub
2. **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
3. Add these two secrets:

   | Secret Name | Value |
   |-------------|-------|
   | `TELEGRAM_TOKEN` | `8784948027:AAEAqpKe0j_zxy4SM7zew1oZBtum7hLDQgA` |
   | `TELEGRAM_CHAT_ID` | *(the number from Step 1)* |

### Step 3 — Push the New Files to GitHub

```bash
git add cloud_signal.py get_chat_id.py .github/workflows/daily_signal.yml paper_trades.json
git commit -m "feat: add cloud paper trading system"
git push
```

### Step 4 — Test It Right Now (Don't Wait for Tomorrow)

1. Go to GitHub → **Actions** tab
2. Click **"🧠 EvoTrader Daily Signal"**
3. Click **"Run workflow"** → **Run workflow**
4. Watch it run live. You'll get a Telegram message in seconds.

---

## What Happens Every Day

```
3:37 PM IST — GitHub Actions wakes up
      ↓
Downloads the latest NIFTY50 price from Yahoo Finance
      ↓
Loads nifty50_brain_validated.pkl from your repo
      ↓
Runs the AI → gets LONG / SHORT / NEUTRAL signal
      ↓
Updates paper_trades.json (auto-committed back to GitHub)
      ↓
Sends Telegram message to your phone
      ↓
Goes back to sleep (costs you nothing)
```

---

## What the Telegram Message Looks Like

```
🧠 EVOTRADER AI — DAILY SIGNAL
📅 Date: 2026-02-24

┌────────────────────────────┐
│  Signal:  🟢 LONG          │
│  Action:  BUY / LONG       │
│  NIFTY:   ₹22,847.50       │
└────────────────────────────┘

📈 Portfolio Performance
  Capital:    ₹9,97,832.00
  P&L:       -0.22%  (₹-2,168)
  vs B&H:    +0.63% edge over buy-and-hold

📊 Stats
  Days tracked: 31
  Trades made:  2
  🟢 LONG streak: 8 days

Next signal: Tomorrow after market close (3:35 PM IST)
EvoTrader AI v1.1 · github.com/ayushmantrivedi/...
```

---

## Viewing Your Trade History

The `paper_trades.json` in your GitHub repo is updated daily.
You can always view it at:
`https://github.com/ayushmantrivedi/evolutionary-neural-network/blob/main/paper_trades.json`

---

## Cost Breakdown

| Service | Cost |
|---------|------|
| GitHub Actions | **Free** (unlimited for public repos) |
| Telegram Bot API | **Free** forever |
| Yahoo Finance data | **Free** (yfinance) |
| GitHub storage | **Free** |
| **Total** | **₹0 / month** |
