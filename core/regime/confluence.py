from typing import List, Optional
from enum import Enum
from dataclasses import dataclass, field
from core.types import Candle, Bias
from core.utils import find_swings

class ConfluenceType(str, Enum):
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    ORDER_BLOCK = "ORDER_BLOCK"
    FVG = "FVG"
    NONE = "NONE"

@dataclass
class ConfluenceResult:
    ready: bool
    confluence_type: ConfluenceType
    level_price: float
    reason: str
    meta: dict = field(default_factory=dict)

class ConfluenceEngine:
    """
    Module 2: High-Timeframe Confluence Engine
    Decides when the bot is allowed to look for entries.
    """
    
    def analyze(self, candles: List[Candle], bias: Bias) -> ConfluenceResult:
        """
        Analyze a single timeframe (usually HTF like 1H) for confluence with bias.
        """
        if bias == Bias.NEUTRAL:
            return ConfluenceResult(False, ConfluenceType.NONE, 0.0, "Bias is Neutral")
            
        current_price = candles[-1].close
        
        # 1. Liquidity Sweeps
        sweep = self._check_liquidity_sweep(candles, bias)
        if sweep:
            return sweep
            
        # 2. Order Blocks
        ob = self._check_order_block(candles, bias, current_price)
        if ob:
            return ob
            
        # 3. FVGs
        fvg = self._check_fvg(candles, bias, current_price)
        if fvg:
            return fvg
            
        return ConfluenceResult(False, ConfluenceType.NONE, 0.0, "No Confluence")

    def _check_liquidity_sweep(self, candles: List[Candle], bias: Bias) -> Optional[ConfluenceResult]:
        """
        Detects if price swept a recent High/Low and closed back inside.
        """
        swings = find_swings(candles, right=5) # Look for established swings
        if not swings:
            return None
            
        last_index = len(candles) - 1
        lookback = min(3, last_index)  # Check recent candles for a sweep

        if bias == Bias.LONG:
            # Look for sweep of Sell-Side Liquidity (Lows)
            recent_lows = [s for s in swings if s['type'] == 'low']
            for idx in range(last_index, max(-1, last_index - lookback) - 1, -1):
                candle = candles[idx]
                for low in recent_lows[-3:]:
                    if candle.low < low['price'] and candle.close > low['price']:
                        return ConfluenceResult(
                            True,
                            ConfluenceType.LIQUIDITY_SWEEP,
                            low['price'],
                            f"Swept Low at {low['price']}"
                        )
                    
        elif bias == Bias.SHORT:
            # Look for sweep of Buy-Side Liquidity (Highs)
            recent_highs = [s for s in swings if s['type'] == 'high']
            for idx in range(last_index, max(-1, last_index - lookback) - 1, -1):
                candle = candles[idx]
                for high in recent_highs[-3:]:
                    if candle.high > high['price'] and candle.close < high['price']:
                        return ConfluenceResult(
                            True,
                            ConfluenceType.LIQUIDITY_SWEEP,
                            high['price'],
                            f"Swept High at {high['price']}"
                        )
        return None

    def _check_order_block(self, candles: List[Candle], bias: Bias, current_price: float) -> Optional[ConfluenceResult]:
        """
        Checks if price is inside a valid Order Block.
        """
        if len(candles) < 3:
            return None

        start = len(candles) - 2
        end = max(1, len(candles) - 20)

        for i in range(start, end - 1, -1):
            if bias == Bias.LONG:
                # Bullish OB Candidate: Red candle followed by Green that breaks high
                if candles[i].close < candles[i].open: # Red
                    if candles[i+1].close > candles[i].high: # Next candle broke high
                        ob_high = candles[i].high
                        ob_low = candles[i].low
                        if ob_low <= current_price <= ob_high:
                             return ConfluenceResult(True, ConfluenceType.ORDER_BLOCK, ob_high, f"Inside Bullish OB {ob_low}-{ob_high}")
                             
            elif bias == Bias.SHORT:
                # Bearish OB Candidate: Green candle followed by Red that breaks low
                if candles[i].close > candles[i].open: # Green
                    if candles[i+1].close < candles[i].low: # Next candle broke low
                        ob_high = candles[i].high
                        ob_low = candles[i].low
                        if ob_low <= current_price <= ob_high:
                             return ConfluenceResult(True, ConfluenceType.ORDER_BLOCK, ob_low, f"Inside Bearish OB {ob_low}-{ob_high}")
                             
        return None

    def _check_fvg(self, candles: List[Candle], bias: Bias, current_price: float) -> Optional[ConfluenceResult]:
        """
        Checks if price is inside a Fair Value Gap.
        """
        if len(candles) < 3:
            return None

        start = len(candles) - 2
        end = max(2, len(candles) - 20)

        for i in range(start, end - 1, -1):
            if i < 2:
                continue
            
            if bias == Bias.LONG:
                # Bullish FVG: Candle 1 High < Candle 3 Low
                c1 = candles[i-2]
                c3 = candles[i]
                if c1.high < c3.low:
                    fvg_low = c1.high
                    fvg_high = c3.low
                    if fvg_low <= current_price <= fvg_high:
                        return ConfluenceResult(True, ConfluenceType.FVG, fvg_high, f"Inside Bullish FVG {fvg_low}-{fvg_high}")
                        
            elif bias == Bias.SHORT:
                # Bearish FVG: Candle 1 Low > Candle 3 High
                c1 = candles[i-2]
                c3 = candles[i]
                if c1.low > c3.high:
                    fvg_high = c1.low
                    fvg_low = c3.high
                    if fvg_low <= current_price <= fvg_high:
                        return ConfluenceResult(True, ConfluenceType.FVG, fvg_low, f"Inside Bearish FVG {fvg_low}-{fvg_high}")
                        
        return None
