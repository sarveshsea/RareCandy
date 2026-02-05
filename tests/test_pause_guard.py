from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from core.types import RiskDecision, Signal, SignalType

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
from main import WurmpleCallback
from ops.pause_guard import read_pause_guard, should_block_new_entry


class FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.states: list[dict] = []

    def update_state(self, equity, positions, active_signal=None):
        self.states.append({"equity": equity, "positions": positions, "active_signal": active_signal})

    def log_event(self, event_type: str, message: str):
        self.events.append((event_type, message))


class FakeExchange:
    def __init__(self, positions):
        self._positions = positions
        self.execute_calls = 0

    def fetch_balance(self):
        return 10_000.0

    def get_positions(self):
        return self._positions

    def execute_order(self, decision, symbol, side):
        self.execute_calls += 1
        return None


class FakeData:
    def get_latest(self, symbol):
        return {"1h": [1], "15m": [1]}


class FakeStrategy:
    def __init__(self, signal):
        self._signal = signal

    def evaluate(self, symbol, c_1h, c_15m):
        return self._signal if symbol == "BTC/USD" else None


class FakeRisk:
    def __init__(self, approved=False):
        self.equity = 10_000.0
        self.approved = approved
        self.evaluate_calls = 0

    def update_equity(self, new_equity):
        self.equity = new_equity

    def evaluate(self, signal, open_positions):
        self.evaluate_calls += 1
        return RiskDecision(
            approved=self.approved,
            reason="approved" if self.approved else "blocked",
            quantity=1.0 if self.approved else 0.0,
            notional=100.0 if self.approved else 0.0,
        )


def _make_signal(signal_type: SignalType) -> Signal:
    return Signal(
        type=signal_type,
        symbol="BTC/USD",
        price=100.0,
        reason="test",
        strategy_id="test_strategy",
    )


def _build_bot(signal: Signal, positions, risk_approved=False, paused=True):
    bot = WurmpleCallback.__new__(WurmpleCallback)
    bot.telemetry = FakeTelemetry()
    bot.exchange = FakeExchange(positions)
    bot.data = FakeData()
    bot.strategy = FakeStrategy(signal)
    bot.risk = FakeRisk(approved=risk_approved)
    bot.pause_entries_on_flag = True
    bot.pause_flag_path = Path("ops/deployment_pause_calibration.json")
    bot._last_pause_state = None
    bot._last_pause_reason = ""
    bot._read_pause_guard = (lambda: (paused, "test pause"))
    bot._log_pause_state = (lambda paused_state, reason: None)
    return bot


def test_pause_guard_absent_file_returns_not_paused() -> None:
    with TemporaryDirectory() as tmp_dir:
        pause_file = Path(tmp_dir) / "missing_pause_file.json"
        paused, reason = read_pause_guard(pause_file, enabled=True)
        assert paused is False
        assert reason == ""


def test_pause_guard_malformed_file_fails_closed() -> None:
    with TemporaryDirectory() as tmp_dir:
        pause_file = Path(tmp_dir) / "bad_pause.json"
        pause_file.write_text("{not-json", encoding="utf-8")
        paused, reason = read_pause_guard(pause_file, enabled=True)
        assert paused is True
        assert "parse error" in reason


def test_pause_guard_with_breaches_returns_reason() -> None:
    with TemporaryDirectory() as tmp_dir:
        pause_file = Path(tmp_dir) / "pause.json"
        pause_file.write_text(
            '{"pause_deployment": true, "breaches": ["ev_ok", "signals_ok"]}',
            encoding="utf-8",
        )
        paused, reason = read_pause_guard(pause_file, enabled=True)
        assert paused is True
        assert "ev_ok" in reason
        assert "signals_ok" in reason


def test_entry_signal_is_blocked_before_risk_eval_when_paused() -> None:
    bot = _build_bot(_make_signal(SignalType.ENTRY_LONG), {"BTC/USD": 0.1}, risk_approved=True, paused=True)
    bot.run_once()
    assert bot.risk.evaluate_calls == 0
    assert bot.exchange.execute_calls == 0
    block_events = [msg for event, msg in bot.telemetry.events if event == "DEPLOYMENT_GUARD_BLOCK"]
    assert block_events
    payload = json.loads(block_events[0])
    assert payload["signal_type"] == SignalType.ENTRY_LONG.value
    assert payload["reason"] == "test pause"
    assert "timestamp" in payload


def test_non_entry_signal_allowed_to_reach_risk_when_paused() -> None:
    bot = _build_bot(_make_signal(SignalType.EXIT_ALL), {"BTC/USD": 0.1}, risk_approved=False, paused=True)
    bot.run_once()
    assert bot.risk.evaluate_calls == 1
    assert not any(event == "DEPLOYMENT_GUARD_BLOCK" for event, _ in bot.telemetry.events)
    assert should_block_new_entry(True, SignalType.EXIT_ALL) is False
