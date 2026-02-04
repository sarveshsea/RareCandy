# Rare Candy 💎

**A modular, deterministic crypto trading engine.**
Built for stability, safety, and 24/7 operation on the cloud.

---

## 🚀 Quick Start (Cloud Deployment)

The easiest way to run Rare Candy is on a **Digital Ocean Droplet** (a cheap cloud server). We have included an interactive script that sets up everything for you (Docker, Permissions, Paper Trading) in one step.

### 1. Get a Server
- Create a **Digital Ocean Droplet** (Ubuntu 22.04+).
- Recommended Size: **2GB RAM / 1 CPU** (~$12/mo).

### 2. Upload & Run
Run these commands from your local terminal to deploy the bot:

```bash
# 1. Upload code to your server
rsync -avz --exclude '.venv' --exclude 'dashboard' ./RareCandy root@<YOUR_DROPLET_IP>:~/

# 2. SSH into your server
ssh root@<YOUR_DROPLET_IP>

# 3. Running the Setup Wizard
cd RareCandy
chmod +x setup_droplet.sh
./setup_droplet.sh
```

**The wizard will ask you:**
-   To install Docker (Say Yes).
-   For your **Coinbase API Keys** (API Key & Secret).
-   Which **Mode** to run:
    -   💸 **Paper Mode** (Fake money, Real data) - *Highly Recommended for starting*.
    -   🚀 **Live Mode** (Real money).

**[See Full Deployment Guide (DEPLOY.md)](DEPLOY.md)** for details on Tailscale security and monitoring.

---

## 💻 Local Development

If you want to modify the strategy or logic locally:

1.  **Environment**:
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Copy `.env.example` to `.env` and fill in your keys.

3.  **Run**:
    ```bash
    python3 main.py
    ```

## 🏗 Architecture

-   **Core**: Pure logic. `strategy/`, `risk/`, `regime/`.
-   **Data**: `data/feed.py` (Public Data Feed, no Auth required).
-   **Execution**: `execution/exchange.py` (Order Management).
-   **Operator**: `main.py` (The Event Loop).

## 📈 Strategy

Currently configured with `TrendPullbackStrategy`:
-   **HTF (1h)**: EMA Trend-following.
-   **LTF (15m)**: Pullback to EMA band + Reversal candle.
-   **Risk**: Dynamic sizing based on equity and conservative profile.

## 🧪 Adaptive Research & Monitoring

For adaptive Pine-derived research runs:

```bash
# Rebuild adaptive regime results + Pine template
python3 analysis/adaptive_regime_selector.py

# One-shot health check (writes analysis/results/adaptive_monitor_status.json)
python3 analysis/monitor_adaptive_regime.py

# Continuous monitor loop (every 15m)
python3 analysis/monitor_adaptive_regime.py --loop --interval-sec 900
```

Monitor thresholds can be tuned:

```bash
python3 analysis/monitor_adaptive_regime.py \
  --min-avg-return 0 \
  --max-avg-dd 12 \
  --min-avg-sharpe 0
```

---

*Verified Working 2026. MIT License.*
