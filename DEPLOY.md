# Deployment Guide 🚀

## Prerequisites
-   A **Digital Ocean Droplet** (Ubuntu 22.04 or newer recommended).
-   A **Digital Ocean Droplet** (Ubuntu 22.04 or newer recommended).
    -   **Minimum**: Basic Droplet, **2GB RAM / 1 CPU** ($12/mo).
    -   **Performance (Recommended)**: Basic Droplet, **4GB RAM / 2 CPU** (~$24/mo).
    -   *Why? Faster builds, better stability for Python + Docker overhead.*
-   **Tailscale** (optional but recommended for secure access to the dashboard).
-   **Coinbase API Keys** (API Key & Secret).

## Automated Setup (Recommended)

1.  **Connect to your Droplet** via SSH.
2.  **Upload the Code**:
    ```bash
    # From your local machine
    rsync -avz --exclude '.venv' --exclude 'dashboard' ./RareCandy root@your_droplet_ip:~/
    ```
3.  **Run the Setup Script**:
    ```bash
    cd ~/RareCandy
    chmod +x setup_droplet.sh
    ./setup_droplet.sh
    ```
    This script is **interactive**. It will:
    -   Install Docker & Docker Compose for you.
    -   Ask for your **Coinbase API Keys**.
    -   Ask which **Trading Mode** you want (Paper, Sandbox, or Live).
    -   Create the `.env` file automatically.
    -   Launch the bot.

## Configuration

The setup script handles `.env` creation. If you need to change it later:
1.  Edit `.env`: `nano .env`
2.  Restart: `docker compose up -d --build`

Recommended runtime/calibration cost settings (keep aligned):
```bash
TRADING_FEE_RATE=0.006
TRADING_SLIPPAGE_RATE=0.0005
TRADING_COST_PER_SIDE=0.006
DATA_HISTORY_LIMIT=300
```

## Start the Bot

Run with Docker Compose:
```bash
docker compose up -d
```

## Safe Update Procedure (Recommended Before Every Deploy)

Use this exact flow to ensure your droplet is synced to `origin/main` and calibration guard is evaluated before new entries are allowed.

1. **Hard-sync repository to remote main**
   ```bash
   cd ~/RareCandy
   git remote set-url origin https://github.com/sarveshsea/RareCandy.git
   git config --replace-all remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
   git fetch --prune origin
   git checkout -B main origin/main
   git log --oneline -n 5
   ```

2. **Run predeploy checks**
   ```bash
   ./scripts/predeploy_check.sh
   ```

3. **Rebuild and restart**
   ```bash
   docker compose build
   docker compose up -d
   ```

4. **(Optional) run synthetic smoke test inside container**
   ```bash
   docker exec -w /app rare_candy_bot python3 generate_example.py
   docker exec -w /app rare_candy_bot python3 unit_check.py
   ```
   > This confirms wiring only. Synthetic exports are **not** valid deployment gate inputs.

5. **Write/refresh real-export manifest inside container**
   ```bash
   docker exec -w /app rare_candy_bot python3 scripts/write_export_manifest.py \
     --exports-dir /app/exports/live \
     --stem rarecandy_export \
     --data-origin live
   ```

6. **Run real-export calibration gate inside container**
   ```bash
   docker exec -w /app \
     -e REAL_EXPORTS_DIR=/app/exports/live \
     -e REAL_EXPORT_STEM=rarecandy_export \
     -e MAX_EXPORT_AGE_HOURS=24 \
     rare_candy_bot ./scripts/run_calibration_pipeline.sh || true
   ```

7. **Check guard status + manifest**
   ```bash
   docker exec -w /app rare_candy_bot cat analysis/artifacts/calibration/calibration_alert_status.json
   docker exec -w /app rare_candy_bot cat analysis/artifacts/calibration/manifest.json
   docker exec -w /app rare_candy_bot ls -la ops/deployment_pause_calibration.json
   ```
   - If `pause_deployment: true`, runtime stays in **no-new-entries** mode (expected fail-safe).
   - Remove pause only when calibration gate clears.

## Trading Modes

### 1. Paper Mode (Recommended First)
-   **Data**: Live Real-time from Coinbase.
-   **Execution**: Simulated internally.
-   **Physics**:
    -   **Fees**: 0.6% Taker Fee simulated per trade.
    -   **Slippage**: 0.05% slippage applied to mimicking real liquidity cost.
-   **Persistence**: Your balance and positions are saved to `paper_state.json`.

### 2. Sandbox Mode
-   **Data**: Fake/Stale data from Coinbase Sandbox.
-   **Execution**: Fake orders to Coinbase Sandbox.
-   *Note: Data is often unreliable.*

### 3. Live Mode
-   **Real Money. Real Risks.**

## Optional: Secure Access with Tailscale (Recommended)

To securely view your bot's dashboard without exposing it to the open internet:

1.  **Run the Tailscale Setup Script**:
    ```bash
    ./setup_tailscale.sh
    ```
2.  **Authenticate**: Click the link provided in the terminal to connect your Droplet to your Tailscale network.
3.  **View Dashboard**: The script will confirm your secure IP (e.g., `100.x.y.z`). You can now visit:
    `http://100.x.y.z:8000/status.json`
4.  **Quant Dashboard (candles + trade markers)**:
    `http://100.x.y.z:8000/quant`
5.  **View Live Terminal Logs (auto-refresh)**:
    `http://100.x.y.z:8000/logs`
6.  **View Raw Log File**:
    `http://100.x.y.z:8000/terminal.log`
7.  **SSH Terminal Over Tailscale**:
    `tailscale ssh <USER>@<HOSTNAME>`

## Monitoring

-   **Logs**: `docker compose logs -f`
-   **Dashboard**: `http://<DROPLET_IP>:8000/status.json` (or via Tailscale IP if configured).

## Updates

To update the bot code:
1.  Upload new files.
2.  Rebuild:
    ```bash
    docker compose up -d --build
    ```
