from typing import List, Dict, Any, Optional
from datetime import datetime
from rare_candy.core.types import Candle

def find_swings(candles: List[Candle], left: int = 2, right: int = 2) -> List[Dict[str, Any]]:
    """
    Identifies swing points (fractals).
    """
    swings = []
    if len(candles) < left + right + 1:
        return swings

    for i in range(left, len(candles) - right):
        # Check High
        is_high = True
        for j in range(1, left + 1):
            if candles[i-j].high > candles[i].high or candles[i+j].high > candles[i].high:
                is_high = False
                break
        if is_high:
            swings.append({
                'type': 'high', 
                'price': candles[i].high, 
                'index': i, 
                'time': candles[i].timestamp
            })
            
        # Check Low
        is_low = True
        for j in range(1, left + 1):
            if candles[i-j].low < candles[i].low or candles[i+j].low < candles[i].low:
                is_low = False
                break
        if is_low:
            swings.append({
                'type': 'low', 
                'price': candles[i].low, 
                'index': i, 
                'time': candles[i].timestamp
            })
            
    return swings

def resample_candles(candles: List[Candle], timeframe_minutes: int, source_timeframe_minutes: int = 5) -> List[Candle]:
    """
    Resamples candles to a higher timeframe.
    Assumes source candles are contiguous.
    """
    if not candles: return []
    
    factor = timeframe_minutes // source_timeframe_minutes
    if factor <= 1: return candles
    
    resampled = []
    current_batch = []
    
    for candle in candles:
        current_batch.append(candle)
        if len(current_batch) == factor:
            # Aggregate
            first = current_batch[0]
            last = current_batch[-1]
            agg_candle = Candle(
                timestamp=first.timestamp,
                open=first.open,
                high=max(c.high for c in current_batch),
                low=min(c.low for c in current_batch),
                close=last.close,
                volume=sum(c.volume for c in current_batch)
            )
            resampled.append(agg_candle)
            current_batch = []
            
    return resampled
