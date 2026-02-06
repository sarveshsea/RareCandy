from __future__ import annotations

from core.risk.manager import RiskManager
from core.types import Signal, SignalType


def _signal(signal_type: SignalType) -> Signal:
    return Signal(
        type=signal_type,
        symbol="BTC/USD",
        price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        reason="test",
        strategy_id="test",
        confidence=1.0,
    )


def test_long_only_blocks_short_entries() -> None:
    rm = RiskManager(equity=10_000.0, long_only=True)
    decision = rm.evaluate(_signal(SignalType.ENTRY_SHORT), open_positions={})
    assert decision.approved is False
    assert "Long-only" in decision.reason
    assert rm.daily_trades == 0


def test_daily_trades_count_only_on_recorded_execution() -> None:
    rm = RiskManager(equity=10_000.0, long_only=True, daily_trade_cap=1)

    decision = rm.evaluate(_signal(SignalType.ENTRY_LONG), open_positions={})
    assert decision.approved is True
    # Approval should not consume daily cap before order is filled.
    assert rm.daily_trades == 0

    rm.record_trade_execution()
    assert rm.daily_trades == 1

    blocked = rm.evaluate(_signal(SignalType.ENTRY_LONG), open_positions={})
    assert blocked.approved is False
    assert "Daily Trade Cap" in blocked.reason
