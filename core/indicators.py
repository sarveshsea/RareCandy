"""
Technical Indicators Module for Rare Candy.
Stateless transformations of Candle lists into indicator values.
Wraps pandas_ta for reliability.
"""
from typing import List, Optional, Tuple
import pandas as pd
import pandas_ta as ta
from rare_candy.core.types import Candle

def _candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    """Convert Candle list to pandas DataFrame."""
    # Assuming candles are sorted by time? We should probably enforce that or sort here.
    # For now, simplistic conversion.
    return pd.DataFrame({
        'open': [c.open for c in candles],
        'high': [c.high for c in candles],
        'low': [c.low for c in candles],
        'close': [c.close for c in candles],
        'volume': [c.volume for c in candles]
    })

def atr(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    """Average True Range."""
    if len(candles) < period:
        return [None] * len(candles)
    df = _candles_to_df(candles)
    atr_series = ta.atr(df['high'], df['low'], df['close'], length=period)
    return atr_series.tolist() if atr_series is not None else [None] * len(candles)

def ema(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    """Exponential Moving Average."""
    if len(candles) < period:
        return [None] * len(candles)
    df = _candles_to_df(candles)
    ema_series = ta.ema(df['close'], length=period)
    return ema_series.tolist() if ema_series is not None else [None] * len(candles)

def sma(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    """Simple Moving Average."""
    if len(candles) < period:
        return [None] * len(candles)
    df = _candles_to_df(candles)
    sma_series = ta.sma(df['close'], length=period)
    return sma_series.tolist() if sma_series is not None else [None] * len(candles)

def rsi(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    """Relative Strength Index."""
    if len(candles) < period + 1:
        return [None] * len(candles)
    df = _candles_to_df(candles)
    rsi_series = ta.rsi(df['close'], length=period)
    return rsi_series.tolist() if rsi_series is not None else [None] * len(candles)

def adx(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    """Average Directional Index."""
    if len(candles) < period * 2:
        return [None] * len(candles)
    df = _candles_to_df(candles)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=period)
    if adx_df is not None and f'ADX_{period}' in adx_df.columns:
        return adx_df[f'ADX_{period}'].tolist()
    return [None] * len(candles)

def bollinger_bands(
    candles: List[Candle],
    period: int = 20,
    stddev: float = 2.0
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Bollinger Bands (Lower, Middle, Upper)."""
    if len(candles) < period:
        none_series = [None] * len(candles)
        return none_series, none_series, none_series
    
    df = _candles_to_df(candles)
    bb = ta.bbands(df['close'], length=period, std=stddev)
    
    if bb is None or bb.empty:
        none_series = [None] * len(candles)
        return none_series, none_series, none_series
        
    # Pandas TA column naming: BBL_length_std, BBM_length_std, BBU_length_std
    # We'll use prefix matching to be safe
    lower, mid, upper = None, None, None
    
    for col in bb.columns:
        if col.startswith("BBL_"): lower = bb[col].tolist()
        elif col.startswith("BBM_"): mid = bb[col].tolist()
        elif col.startswith("BBU_"): upper = bb[col].tolist()
        
    return (
        lower if lower else [None]*len(candles), 
        mid if mid else [None]*len(candles), 
        upper if upper else [None]*len(candles)
    )
