import math
from typing import List, Optional
from core.types import Candle, Signal, SignalType, Bias
from core.strategy.base import Strategy
from core.indicators import ema, rsi

class TrendPullbackStrategy(Strategy):
    """
    Trend-continuation pullback strategy.
    - Determine trend on 1h EMAs.
    - Enter on 15m pullback into EMA band.
    """
    
    def __init__(self):
        self.trend_fast = 50
        self.trend_slow = 200
        self.pullback_fast = 20
        self.pullback_slow = 50
        self.buffer_pct = 0.003
        self.rsi_period = 14
        self.min_spread = 0.001

    def evaluate(self, symbol: str, candles_htf: List[Candle], candles_ltf: List[Candle], bias: Optional[Bias] = None) -> Optional[Signal]:
        # candles_htf = 1h, candles_ltf = 15m
        if len(candles_htf) < (self.trend_slow + 5) or len(candles_ltf) < 50:
            return None
            
        # 1. Trend Filter (HTF)
        ema_fast = self._get_last(ema(candles_htf, self.trend_fast))
        ema_slow = self._get_last(ema(candles_htf, self.trend_slow))
        
        if not ema_fast or not ema_slow or ema_slow <= 0:
            return None
            
        spread = (ema_fast - ema_slow) / ema_slow
        if abs(spread) < self.min_spread:
            return None
            
        trend = Bias.LONG if spread > 0 else Bias.SHORT
        if bias is not None and trend != bias:
            return None
        
        # 2. Pullback Check (LTF)
        price = candles_ltf[-1].close
        pb_fast = self._get_last(ema(candles_ltf, self.pullback_fast))
        pb_slow = self._get_last(ema(candles_ltf, self.pullback_slow))
        
        if not pb_fast or not pb_slow:
            return None
            
        top = max(pb_fast, pb_slow) * (1 + self.buffer_pct)
        bottom = min(pb_fast, pb_slow) * (1 - self.buffer_pct)
        
        # Must be in band
        if not (bottom <= price <= top):
            return None
            
        # 3. RSI Filter
        rsi_val = self._get_last(rsi(candles_ltf, self.rsi_period))
        if not rsi_val: return None
        
        if trend == Bias.LONG:
            if rsi_val > 70: return None # Overbought, don't buy
        else:
            if rsi_val < 30: return None # Oversold, don't sell
            
        # 4. Trigger (Reversal Candle)
        last = candles_ltf[-1]
        prev = candles_ltf[-2]
        
        triggered = False
        if trend == Bias.LONG:
            # Bullish reversal: Green + Close > Prev Close
            if last.close > last.open and last.close > prev.close:
                triggered = True
        else:
            # Bearish reversal: Red + Close < Prev Close
            if last.close < last.open and last.close < prev.close:
                triggered = True
                
        if not triggered:
            return None
            
        # 5. Signal Construction
        # Calculate SL/TP
        if trend == Bias.LONG:
            sl = bottom * (1 - self.buffer_pct)
            risk = price - sl
            tp = price + (risk * 2.0)
            sig_type = SignalType.ENTRY_LONG
        else:
            sl = top * (1 + self.buffer_pct)
            risk = sl - price
            tp = price - (risk * 2.0)
            sig_type = SignalType.ENTRY_SHORT
            
        confidence = min(1.0, max(0.0, 0.75 + (abs(spread) * 5)))  # Clamp to Signal schema bounds.

        return Signal(
            type=sig_type,
            symbol=symbol,
            price=price,
            stop_loss=sl,
            take_profit=tp,
            reason=f"Trend Pullback ({trend.value}) at EMA Band",
            strategy_id="trend_pullback_v2",
            confidence=confidence,
            metadata={
                "spread": spread,
                "rsi": rsi_val
            }
        )

    def _get_last(self, values: List[Optional[float]]) -> Optional[float]:
        for v in reversed(values):
            if v is not None and not math.isnan(v):
                return v
        return None
