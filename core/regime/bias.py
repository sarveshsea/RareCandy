from typing import List
from dataclasses import dataclass
from core.types import Candle, Bias, Trend
from core.utils import find_swings

@dataclass
class BiasResult:
    bias: Bias
    htf_trend: Trend # 4H / HTF
    ltf_trend: Trend # 1H / LTF
    reason: str

class BiasEngine:
    """
    Module 1: Market Bias Framework
    Determines the directional bias (LONG/SHORT/NEUTRAL) based on HTF and LTF market structure.
    """
    
    def analyze(self, candles_htf: List[Candle], candles_ltf: List[Candle]) -> BiasResult:
        # Minimum requirements
        if len(candles_htf) < 2 or len(candles_ltf) < 10:
            return BiasResult(
                Bias.NEUTRAL, 
                Trend.NEUTRAL, 
                Trend.NEUTRAL, 
                f"Insufficient Data (HTF: {len(candles_htf)}, LTF: {len(candles_ltf)})"
            )

        # 1. Determine Trends
        trend_htf = self._determine_trend(candles_htf)
        trend_ltf = self._determine_trend(candles_ltf)
        
        # 2. Combine to form Master Bias
        master_bias = Bias.NEUTRAL
        reason = f"HTF: {trend_htf.value}, LTF: {trend_ltf.value}"
        
        if trend_htf == Trend.NEUTRAL and trend_ltf == Trend.NEUTRAL:
            return BiasResult(Bias.NEUTRAL, trend_htf, trend_ltf, f"Neutral Structure ({reason})")
        
        # Rule 1: Full Alignment
        if trend_htf == Trend.BULLISH and trend_ltf == Trend.BULLISH:
            master_bias = Bias.LONG
            reason = "Full Alignment (Bullish)"
        elif trend_htf == Trend.BEARISH and trend_ltf == Trend.BEARISH:
            master_bias = Bias.SHORT
            reason = "Full Alignment (Bearish)"
            
        # Rule 2: HTF Neutral, LTF Directional (Weak)
        elif trend_htf == Trend.NEUTRAL and trend_ltf == Trend.BULLISH:
            master_bias = Bias.LONG
            reason = "Weak Bias: LTF Bullish (HTF Neutral)"
        elif trend_htf == Trend.NEUTRAL and trend_ltf == Trend.BEARISH:
            master_bias = Bias.SHORT
            reason = "Weak Bias: LTF Bearish (HTF Neutral)"
            
        # Rule 3: HTF Directional, LTF Neutral (Weak)
        elif trend_ltf == Trend.NEUTRAL:
            if trend_htf == Trend.BULLISH:
                master_bias = Bias.LONG
                reason = "Weak Bias: HTF Bullish (LTF Neutral)"
            elif trend_htf == Trend.BEARISH:
                master_bias = Bias.SHORT
                reason = "Weak Bias: HTF Bearish (LTF Neutral)"
            
        # Rule 4: Conflict (Follow HTF)
        elif trend_htf == Trend.BULLISH and trend_ltf == Trend.BEARISH:
            master_bias = Bias.LONG
            reason = "Conflict: Following HTF Bullish (Ignoring LTF pullback)"
        elif trend_htf == Trend.BEARISH and trend_ltf == Trend.BULLISH:
            master_bias = Bias.SHORT
            reason = "Conflict: Following HTF Bearish (Ignoring LTF pullback)"
            
        return BiasResult(master_bias, trend_htf, trend_ltf, reason)

    def _determine_trend(self, candles: List[Candle]) -> Trend:
        """
        Determines trend based on Swing Highs and Swing Lows.
        """
        try:
            if not candles:
                return Trend.NEUTRAL

            swings = find_swings(candles)

            highs = [p for p in swings if p.get("type") == "high"]
            lows = [p for p in swings if p.get("type") == "low"]

            if len(highs) < 1 or len(lows) < 1:
                return Trend.NEUTRAL

            current_price = candles[-1].close

            # Heuristic if not enough swings
            if len(highs) < 2 or len(lows) < 2:
                last_high = float(highs[-1]["price"])
                last_low = float(lows[-1]["price"])
                if current_price > last_high:
                    return Trend.BULLISH
                if current_price < last_low:
                    return Trend.BEARISH
                return Trend.NEUTRAL

            last_high = float(highs[-1]["price"])
            prev_high = float(highs[-2]["price"])
            last_low = float(lows[-1]["price"])
            prev_low = float(lows[-2]["price"])

            # HH + HL = Bullish
            if last_high > prev_high and last_low > prev_low:
                return Trend.BULLISH

            # LH + LL = Bearish
            if last_high < prev_high and last_low < prev_low:
                return Trend.BEARISH

            return Trend.NEUTRAL
        except Exception:
            return Trend.NEUTRAL
