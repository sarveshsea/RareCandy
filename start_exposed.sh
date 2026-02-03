#!/bin/bash
set -euo pipefail

echo "💎 Starting Rare Candy + Telemetry Server..."

mkdir -p dashboard logs
touch dashboard/terminal.log

# 1. Expose telemetry and terminal log file through the dashboard web server.
echo "📊 Telemetry:      http://[TAILSCALE_IP]:8000/status.json"
echo "🖥️ Terminal logs:  http://[TAILSCALE_IP]:8000/terminal.log"
python3 -m http.server 8000 --directory dashboard > logs/http_server.log 2>&1 &
SERVER_PID=$!

# 2. Start trading engine and stream stdout/stderr into dashboard/terminal.log.
echo "🚀 Bot Engine Started."
python3 -u main.py >> dashboard/terminal.log 2>&1 &
BOT_PID=$!

cleanup() {
  kill "$SERVER_PID" "$BOT_PID" 2>/dev/null || true
}

trap cleanup SIGINT SIGTERM

# Keep execution alive.
wait
