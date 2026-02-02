from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    LONG = "LONG"  # For futures
    SHORT = "SHORT" # For futures

class Candle(BaseModel):
    """OHLCV Candle Data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    class Config:
        frozen = True # Immutable

class SignalType(str, Enum):
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_ALL = "EXIT_ALL"
    ADJUST_SL = "ADJUST_SL"

class Signal(BaseModel):
    """
    A typed trading decision from a Strategy.
    """
    type: SignalType
    symbol: str
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str
    strategy_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class BlockerSeverity(str, Enum):
    INFO = "INFO"       # Just for awareness
    WARNING = "WARNING" # Prevent new entries, manage existing
    CRITICAL = "CRITICAL" # Emergency halt / close all

class Blocker(BaseModel):
    """
    Represents a reason NOT to trade.
    """
    id: str
    reason: str
    severity: BlockerSeverity
    owner: str # e.g., "RiskManager", "CircuitBreaker"
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Bias(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class Trend(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class RiskDecision(BaseModel):
    approved: bool
    reason: str
    quantity: float
    notional: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
