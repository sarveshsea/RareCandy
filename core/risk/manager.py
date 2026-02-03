from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Any
from enum import Enum
from core.types import Signal, RiskDecision, SignalType

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    RISKY = "risky"

class RiskManager:
    """
    Central Risk Management Module.
    Evaluates Signals against account state and rules.
    """
    def __init__(
        self,
        equity: float,
        profile: RiskProfile = RiskProfile.CONSERVATIVE,
        max_positions: int = 5,
        target_volatility: float = 0.02,
        long_only: bool = False,
        daily_trade_cap: int = 30,
        loss_cooldown_minutes: int = 5,
        enable_futures: bool = False,
    ):
        self.equity = equity
        self.profile = profile
        self.max_positions = max_positions
        self.target_volatility = target_volatility
        self.long_only = long_only
        self.enable_futures = enable_futures
        
        # Profile Settings
        if self.profile == RiskProfile.RISKY:
            self.risk_per_trade_pct = 0.01
            self.stop_loss_pct = 0.03
            self.take_profit_pct = 0.06
        else:
            self.risk_per_trade_pct = 0.005
            self.stop_loss_pct = 0.02
            self.take_profit_pct = 0.04

        # State
        self.daily_pnl = 0.0
        self.daily_start_equity = equity
        self.peak_equity = equity
        self.daily_start_time = datetime.now(timezone.utc).date()
        self.daily_trades = 0
        self.daily_trade_cap = daily_trade_cap
        self.last_loss_time: Optional[datetime] = None
        self.loss_cooldown = timedelta(minutes=loss_cooldown_minutes)

    def evaluate(
        self, 
        signal: Signal, 
        open_positions: Dict[str, float], 
        is_crash_mode: bool = False
    ) -> RiskDecision:
        self._reset_daily_if_needed()
        
        # 1. CRASH MODE
        if is_crash_mode:
            is_closing_long = (signal.type == SignalType.ENTRY_SHORT and open_positions.get(signal.symbol, 0) > 0)
            # Simplified: SignalType currently assumes ENTRY. 
            # If we had EXIT signals they would pass. 
            # Assuming ENTRY_SHORT is used to close LONGs in some systems, but strict typing separates them.
            # Here we just block entries.
            if True: # Strict block for now unless we implement proper Exits
                return RiskDecision(False, "Crash Mode Active", 0.0, 0.0)

        # 2. Daily Loss Limit
        if self.daily_pnl < -(self.daily_start_equity * 0.05):
            return RiskDecision(False, f"Daily Loss Limit Hit ({self.daily_pnl})", 0.0, 0.0)

        # 3. Daily Cap
        if self.daily_trades >= self.daily_trade_cap:
            return RiskDecision(False, "Daily Trade Cap Reached", 0.0, 0.0)

        # 4. Cooldown
        if self.last_loss_time:
            now = datetime.now(timezone.utc)
            if now - self.last_loss_time < self.loss_cooldown:
                return RiskDecision(False, "Cooldown Active", 0.0, 0.0)

        # 5. Position Limits (For new entries)
        current_qty = open_positions.get(signal.symbol, 0.0)
        if current_qty == 0:
            active_count = sum(1 for q in open_positions.values() if abs(q) > 0)
            if active_count >= self.max_positions:
                return RiskDecision(False, "Max Positions Reached", 0.0, 0.0)
        
        # 6. Sizing
        distance = abs(signal.price - (signal.stop_loss or (signal.price * 0.98)))
        if distance == 0: distance = signal.price * 0.01
        
        risk_amt = self.equity * self.risk_per_trade_pct
        
        # Scale by confidence
        if signal.confidence:
            risk_amt *= (0.5 + signal.confidence)
            
        units = risk_amt / distance
        
        # Max notional check (20% equity)
        max_notional = self.equity * 0.20
        if units * signal.price > max_notional:
            units = max_notional / signal.price
            
        # Min size ($5)
        if units * signal.price < 5.0:
            return RiskDecision(False, "Position Too Small", 0.0, 0.0)
            
        self.daily_trades += 1
        return RiskDecision(
            approved=True,
            reason=f"Approved ({self.profile.name})",
            quantity=round(units, 6),
            notional=units * signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )

    def update_equity(self, new_equity: float):
        self._reset_daily_if_needed()
        diff = new_equity - self.equity
        self.equity = new_equity
        self.daily_pnl += diff
        self.peak_equity = max(self.peak_equity, new_equity)

    def record_loss(self):
        self.last_loss_time = datetime.now(timezone.utc)

    def _reset_daily_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_time:
            self.daily_start_time = today
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_start_equity = self.equity
            self.last_loss_time = None
