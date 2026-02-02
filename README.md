# Rare Candy 💎

A modular, deterministic crypto trading engine.

## Installation

1. **Clone/Copy**: Move this entire directory to a new location.
   ```bash
   cp -r rare_candy ~/Desktop/RareCandy
   cd ~/Desktop/RareCandy
   ```

2. **Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configuration**:
   Copy `.env.example` to `.env` (or create it) with:
   ```env
   COINBASE_API_KEY="your_key"
   COINBASE_API_SECRET="your_secret"
   SANDBOX_MODE=True
   ```
   *(Note: You need to create this file)*

## Usage

**Run the Operator (Wurmple):**
```bash
python3 main.py
```
This starts the loop: Fetch Data -> Strategy -> Risk -> Execution.

**Run Verification:**
```bash
python3 rare_candy/verify_core.py
```

## Architecture

- **Core**: Pure logic. `strategy/`, `risk/`, `regime/`.
- **Data**: `data/feed.py` (CCXT wrapper).
- **Execution**: `execution/exchange.py` (CCXT wrapper).
- **Operator**: `main.py` (The loop).

## Strategy

Currently configured with `TrendPullbackStrategy`:
- **HTF (1h)**: EMA Trend-following.
- **LTF (15m)**: Pullback to EMA band + Reversal candle.
- **Risk**: Dynamic sizing based on equity and conservative profile.
