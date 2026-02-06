import os
import time
import logging
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Import Core Modules
from core.types import Bias, RiskDecision, Signal, SignalType, OrderSide
from core.strategy.trend_pullback import TrendPullbackStrategy
from core.risk.manager import RiskManager
from ops.telemetry import TelemetryWriter
from ops.pause_guard import is_entry_signal, read_pause_guard, should_block_new_entry

# Import System Modules
from execution.exchange import ExchangeAdapter
from data.feed import DataPipeline

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
DEPLOYMENT_PAUSE_FLAG = Path(os.getenv("DEPLOYMENT_PAUSE_FLAG", "ops/deployment_pause_calibration.json"))

class WurmpleCallback:
    """Operator that runs the bot loop."""
    
    def __init__(self):
        logger.info(f"💎 Rare Candy (Wurmple) Starting... Sandbox={SANDBOX_MODE}")
        
        self.telemetry = TelemetryWriter(output_dir="dashboard")
        self.pause_flag_path = DEPLOYMENT_PAUSE_FLAG
        self.pause_entries_on_flag = os.getenv("PAUSE_ENTRIES_ON_CALIBRATION_ALERT", "true").lower() == "true"
        self._last_pause_state = None
        self._last_pause_reason = ""
        self.position_guards = {}
        self._missing_guard_logged = set()
        
        if not API_KEY or not API_SECRET:
            logger.error("Missing API Keys in .env")
            self.telemetry.log_event("ERROR", "Missing API Keys")
            exit(1)
            
        # Initialize Components
        self.exchange = ExchangeAdapter(
            API_KEY, 
            API_SECRET, 
            sandbox=SANDBOX_MODE,
            paper_mode=os.getenv("PAPER_MODE", "False").lower() == "true"
        )
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

    def _entry_signal(self, signal_type: SignalType) -> bool:
        return is_entry_signal(signal_type)

    def _read_pause_guard(self):
        return read_pause_guard(self.pause_flag_path, enabled=self.pause_entries_on_flag)

    def _log_pause_state(self, paused: bool, reason: str):
        if self._last_pause_state == paused and self._last_pause_reason == reason:
            return
        self._last_pause_state = paused
        self._last_pause_reason = reason
        if paused:
            logger.warning(f"⏸️ Deployment guard active. New entries paused. Reason: {reason}")
            self.telemetry.log_event("DEPLOYMENT_PAUSED", reason)
        else:
            logger.info("▶️ Deployment guard cleared. Entry deployment resumed.")
            self.telemetry.log_event("DEPLOYMENT_RESUMED", "Calibration guard cleared")

    def _sync_position_guard(self, symbol: str, quantity: float):
        if quantity <= 0 and symbol in self.position_guards:
            self.position_guards.pop(symbol, None)
        if quantity <= 0:
            self._missing_guard_logged.discard(symbol)

    def _maybe_process_guard_exit(self, symbol: str, quantity: float) -> bool:
        """
        Enforce stop-loss/take-profit exits for active spot-long positions.
        Returns True when an exit was executed and symbol processing should stop for this tick.
        """
        if quantity <= 0:
            return False

        guard = self.position_guards.get(symbol)
        if not guard:
            if symbol not in self._missing_guard_logged:
                self._missing_guard_logged.add(symbol)
                self.telemetry.log_event("MISSING_POSITION_GUARD", json.dumps({"symbol": symbol}))
                logger.warning(f"🛡️ {symbol} has open exposure without an active guard")
            return False

        ticker = self.exchange.get_ticker(symbol)
        price = float(ticker.get("price", 0.0) or 0.0)
        if price <= 0:
            return False

        stop_loss = guard.get("stop_loss")
        take_profit = guard.get("take_profit")
        hit_stop = stop_loss is not None and price <= float(stop_loss)
        hit_take = take_profit is not None and price >= float(take_profit)
        if not (hit_stop or hit_take):
            return False

        exit_reason = "STOP_LOSS_HIT" if hit_stop else "TAKE_PROFIT_HIT"
        decision = RiskDecision(
            approved=True,
            reason=exit_reason,
            quantity=float(quantity),
            notional=float(quantity) * price,
        )
        result = self.exchange.execute_order(decision, symbol, OrderSide.SELL)
        if result:
            payload = {
                "symbol": symbol,
                "reason": exit_reason,
                "price": price,
                "quantity": quantity,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.telemetry.log_event("GUARD_EXIT_EXECUTED", json.dumps(payload))
            self.position_guards.pop(symbol, None)
            logger.info(f"🛡️ {symbol} {exit_reason} executed at {price:.2f}")
            return True

        self.telemetry.log_event("GUARD_EXIT_FAILED", json.dumps({"symbol": symbol, "reason": exit_reason}))
        logger.warning(f"🛡️ {symbol} {exit_reason} failed to execute")
        return False

    def run_once(self):
        """Single Tick Execution."""
        logger.info("--- Tick ---")
        
        # 0. Update Account State
        current_equity = self.exchange.fetch_balance()
        if current_equity > 0:
            self.risk.update_equity(current_equity) # Sync equity
        
        open_positions = self.exchange.get_positions()
        logger.info(f"Equity: ${self.risk.equity:.2f} | Positions: {open_positions}")

        paused, pause_reason = self._read_pause_guard()
        self._log_pause_state(paused, pause_reason)

        # Update Telemetry (Heartbeat)
        self.telemetry.update_state(
            self.risk.equity,
            open_positions,
            active_signal=(f"PAUSED: {pause_reason}" if paused else None),
        )

        for symbol in SYMBOLS:
            try:
                position_qty = float(open_positions.get(symbol, 0.0) or 0.0)
                has_position = position_qty > 0
                self._sync_position_guard(symbol, position_qty)
                if self._maybe_process_guard_exit(symbol, position_qty):
                    continue

                if paused and not has_position:
                    # When paused, skip fresh scans for symbols without active exposure.
                    continue

                # 1. Pipeline: Data
                candles = self.data.get_latest(symbol)
                c_1h = candles.get('1h', [])
                c_15m = candles.get('15m', [])
                
                if not c_1h or not c_15m:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # 2. Pipeline: Strategy
                bias = Bias.LONG if self.risk.long_only else None
                signal = self.strategy.evaluate(symbol, c_1h, c_15m, bias=bias)
                
                if signal:
                    logger.info(f"🚨 SIGNAL: {signal.type} {symbol} @ {signal.price}")
                    self.telemetry.log_event("SIGNAL", f"{signal.type} {symbol} {signal.reason}")

                    # Deployment safety guard: block fresh entries before risk side-effects.
                    if should_block_new_entry(paused, signal.type):
                        msg = f"{symbol} {signal.type} blocked by deployment pause guard ({pause_reason})"
                        logger.warning(f"⛔ {msg}")
                        payload = {
                            "signal_type": signal.type.value if hasattr(signal.type, "value") else str(signal.type),
                            "reason": pause_reason,
                            "pause_source": str(self.pause_flag_path),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "symbol": symbol,
                        }
                        self.telemetry.log_event("DEPLOYMENT_GUARD_BLOCK", json.dumps(payload))
                        continue

                    if self._entry_signal(signal.type) and (signal.stop_loss is None or signal.take_profit is None):
                        self.telemetry.log_event(
                            "RISK_BLOCK",
                            f"{symbol} missing stop/take-profit contract for {signal.type}",
                        )
                        logger.warning(f"🛑 {symbol} blocked: missing stop/take-profit on entry signal")
                        continue
                    
                    # 3. Pipeline: Risk
                    decision = self.risk.evaluate(signal, open_positions)
                    
                    if decision.approved:
                        logger.info(f"✅ RISK APPROVED: {decision.quantity} units")
                        self.telemetry.log_event("TRADE_APPROVED", f"{symbol} {decision.quantity} units")
                        
                        # 4. Pipeline: Execution
                        # Determine side based on signal type
                        side = OrderSide.BUY if signal.type == SignalType.ENTRY_LONG else OrderSide.SELL
                        
                        order_result = self.exchange.execute_order(decision, symbol, side)
                        if order_result:
                            if self._entry_signal(signal.type):
                                self.risk.record_trade_execution()
                                self.position_guards[symbol] = {
                                    "stop_loss": decision.stop_loss,
                                    "take_profit": decision.take_profit,
                                    "created_at": datetime.now(timezone.utc).isoformat(),
                                }
                                self._missing_guard_logged.discard(symbol)
                            self.telemetry.log_event("TRADE_FILLED", f"{symbol} {signal.type} {decision.quantity}")
                        else:
                            self.telemetry.log_event("TRADE_FAILED", f"{symbol} {signal.type} execution failed")
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
