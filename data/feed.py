import ccxt
import time
from datetime import datetime
from typing import List, Dict
from core.types import Candle

class DataPipeline:
    """
    Fetches and manages candle data for strategies.
    Default: 1h and 15m candles.
    """
    def __init__(self, exchange_client=None):
        # Use a fresh, unauthenticated client for Data to avoid 401s on public endpoints
        self.exchange = ccxt.coinbase() 
        self.cache: Dict[str, Dict[str, List[Candle]]] = {} # {symbol: {tf: [Candles]}}

    def fetch_recent_candles(self, symbol: str, timeframe: str, limit: int = 200) -> List[Candle]:
        """Fetch historical candles from exchange."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            candles = []
            for d in ohlcv:
                # [timestamp, open, high, low, close, volume]
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(d[0]/1000),
                    open=float(d[1]),
                    high=float(d[2]),
                    low=float(d[3]),
                    close=float(d[4]),
                    volume=float(d[5])
                ))
            
            # Update cache
            if symbol not in self.cache: self.cache[symbol] = {}
            self.cache[symbol][timeframe] = candles
            
            return candles
        except Exception as e:
            print(f"[Data] Fetch Error ({symbol} {timeframe}): {e}")
            # Return cached if available
            return self.cache.get(symbol, {}).get(timeframe, [])

    def get_latest(self, symbol: str) -> Dict[str, List[Candle]]:
        """Get latest 1h and 15m candles for a symbol."""
        c_1h = self.fetch_recent_candles(symbol, '1h', limit=100)
        c_15m = self.fetch_recent_candles(symbol, '15m', limit=100)
        return {
            '1h': c_1h,
            '15m': c_15m
        }
