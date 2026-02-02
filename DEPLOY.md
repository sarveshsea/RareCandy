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

## Start the Bot

Run with Docker Compose:
```bash
docker compose up -d
```

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
