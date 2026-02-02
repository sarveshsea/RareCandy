#!/bin/bash

# Kill exiting python processes if needed
# pkill -f rare_candy.main
# pkill -f http.server

echo "💎 Starting Rare Candy + Telemetry Server..."

# 1. Start the simple HTTP server to expose the 'dashboard' folder on port 8000
# This allows remote agents to fetch http://tailscale-ip:8000/status.json
echo "📊 Telemetry available at http://[TAILSCALE_IP]:8000/status.json"
mkdir -p dashboard
python3 -m http.server 8000 --directory dashboard &
SERVER_PID=$!

# 2. Start the Trading Bot
echo "🚀 Bot Engine Started."
python3 -m rare_candy.main &
BOT_PID=$!

# Cleanup on exit
trap "kill $SERVER_PID $BOT_PID; exit" SIGINT SIGTERM

# Keep execution alive
wait
