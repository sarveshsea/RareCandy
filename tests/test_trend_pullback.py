from __future__ import annotations

from datetime import datetime, timedelta, timezone

import core.strategy.trend_pullback as trend_mod
from core.strategy.trend_pullback import TrendPullbackStrategy
from core.types import Bias, Candle, SignalType


def _candles(n: int, base: float = 100.0) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles: list[Candle] = []
    for i in range(n):
        px = base + i * 0.01
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * i),
                open=px,
                high=px + 0.2,
                low=px - 0.2,
                close=px,
                volume=1000.0,
            )
        )
    return candles


def test_evaluate_requires_enough_history_for_slow_trend() -> None:
    strategy = TrendPullbackStrategy()
    htf = _candles(100)
    ltf = _candles(60)
    assert strategy.evaluate("BTC/USD", htf, ltf) is None


def test_bias_filter_and_confidence_cap(monkeypatch) -> None:
    strategy = TrendPullbackStrategy()
    htf = _candles(210, base=100.0)
    ltf = _candles(60, base=105.0)

    # Force last two candles to satisfy bullish trigger.
    ltf[-2] = ltf[-2].model_copy(update={"open": 104.8, "close": 104.9})
    ltf[-1] = ltf[-1].model_copy(update={"open": 104.9, "close": 105.0})

    def fake_ema(candles, period):
        n = len(candles)
        out = [None] * n
        if n == len(htf):
            if period == strategy.trend_fast:
                out[-1] = 120.0
            elif period == strategy.trend_slow:
                out[-1] = 100.0
        else:
            if period == strategy.pullback_fast:
                out[-1] = 105.1
            elif period == strategy.pullback_slow:
                out[-1] = 104.8
        return out

    def fake_rsi(candles, period):
        out = [None] * len(candles)
        out[-1] = 55.0
        return out

    monkeypatch.setattr(trend_mod, "ema", fake_ema)
    monkeypatch.setattr(trend_mod, "rsi", fake_rsi)

    long_signal = strategy.evaluate("BTC/USD", htf, ltf, bias=Bias.LONG)
    assert long_signal is not None
    assert long_signal.type == SignalType.ENTRY_LONG
    assert long_signal.confidence == 1.0  # 0.75 + spread boost is clamped at schema max.

    short_bias_blocked = strategy.evaluate("BTC/USD", htf, ltf, bias=Bias.SHORT)
    assert short_bias_blocked is None
