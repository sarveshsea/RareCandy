import os
import time
import logging
import asyncio
from dotenv import load_dotenv

# Import Core Modules
from rare_candy.core.types import Signal, SignalType, OrderSide
from rare_candy.core.strategy.trend_pullback import TrendPullbackStrategy
from rare_candy.core.risk.manager import RiskManager
from rare_candy.ops.telemetry import TelemetryWriter

# Import System Modules
from rare_candy.execution.exchange import ExchangeAdapter
from rare_candy.data.feed import DataPipeline

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RareCandy")

# Configuration
load_dotenv()
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")
SANDBOX_MODE = os.getenv("SANDBOX_MODE", "True").lower() == "true"
SYMBOLS = ["BTC/USD", "ETH/USD"] # Target symbols

class WurmpleCallback:
    """Operator that runs the bot loop."""
    
    def __init__(self):
        logger.info(f"💎 Rare Candy (Wurmple) Starting... Sandbox={SANDBOX_MODE}")
        
        self.telemetry = TelemetryWriter(output_dir="dashboard")
        
        if not API_KEY or not API_SECRET:
            logger.error("Missing API Keys in .env")
            self.telemetry.log_event("ERROR", "Missing API Keys")
            exit(1)
            
        # Initialize Components
        self.exchange = ExchangeAdapter(API_KEY, API_SECRET, sandbox=SANDBOX_MODE)
        self.data = DataPipeline(self.exchange.client)
        
        # Initialize Core Logic
        self.strategy = TrendPullbackStrategy()
        
        # Initialize Risk (Fetch initial equity)
        initial_equity = self.exchange.fetch_balance()
        logger.info(f"Initial Equity: ${initial_equity:.2f}")
        
        self.risk = RiskManager(
            equity=initial_equity,
            profile="conservative",
            long_only=True # Safety first
        )

    def run_once(self):
        """Single Tick Execution."""
        logger.info("--- Tick ---")
        
        # 0. Update Account State
        current_equity = self.exchange.fetch_balance()
        if current_equity > 0:
            self.risk.update_equity(current_equity) # Sync equity
        
        open_positions = self.exchange.get_positions()
        logger.info(f"Equity: ${self.risk.equity:.2f} | Positions: {open_positions}")

        # Update Telemetry (Heartbeat)
        self.telemetry.update_state(self.risk.equity, open_positions, active_signal=None)

        for symbol in SYMBOLS:
            try:
                # 1. Pipeline: Data
                candles = self.data.get_latest(symbol)
                c_1h = candles.get('1h', [])
                c_15m = candles.get('15m', [])
                
                if not c_1h or not c_15m:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # 2. Pipeline: Strategy
                signal = self.strategy.evaluate(symbol, c_1h, c_15m)
                
                if signal:
                    logger.info(f"🚨 SIGNAL: {signal.type} {symbol} @ {signal.price}")
                    self.telemetry.log_event("SIGNAL", f"{signal.type} {symbol} {signal.reason}")
                    
                    # 3. Pipeline: Risk
                    decision = self.risk.evaluate(signal, open_positions)
                    
                    if decision.approved:
                        logger.info(f"✅ RISK APPROVED: {decision.quantity} units")
                        self.telemetry.log_event("TRADE_APPROVED", f"{symbol} {decision.quantity} units")
                        
                        # 4. Pipeline: Execution
                        # Determine side based on signal type
                        side = OrderSide.BUY if signal.type == SignalType.ENTRY_LONG else OrderSide.SELL
                        if signal.type in [SignalType.ENTRY_SHORT]: side = OrderSide.SELL # Short entry
                        
                        self.exchange.execute_order(decision, symbol, side)
                    else:
                        logger.info(f"🛑 RISK BLOCKED: {decision.reason}")
                        self.telemetry.log_event("RISK_BLOCK", f"{symbol} {decision.reason}")
                else:
                    # No Signal - Quiet
                    pass
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.telemetry.log_event("ERROR", str(e))

    def start_loop(self, interval_seconds=60):
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Stopping...")
                break
            except Exception as e:
                logger.error(f"Loop Crash: {e}")
            
            time.sleep(interval_seconds)

if __name__ == "__main__":
    bot = WurmpleCallback()
    bot.start_loop(interval_seconds=60 * 15) # Run every 15 mins (candle close)
