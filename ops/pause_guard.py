from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from core.types import SignalType


ENTRY_SIGNAL_TYPES = {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}


def is_entry_signal(signal_type: SignalType) -> bool:
    return signal_type in ENTRY_SIGNAL_TYPES


def should_block_new_entry(paused: bool, signal_type: SignalType) -> bool:
    return paused and is_entry_signal(signal_type)


def read_pause_guard(flag_path: Path, enabled: bool = True) -> Tuple[bool, str]:
    """
    Read deployment pause guard and return (paused, reason).
    If enabled and file is malformed, fail closed (paused=True).
    """
    if not enabled:
        return False, ""

    if not flag_path.exists():
        return False, ""

    try:
        payload = json.loads(flag_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return True, f"pause flag parse error: {exc}"

    if isinstance(payload, dict):
        paused = bool(payload.get("pause_deployment", True))
        if not paused:
            return False, ""

        breaches = payload.get("breaches", [])
        if isinstance(breaches, list) and breaches:
            return True, "calibration guard breach: " + ", ".join(str(b) for b in breaches)
        return True, str(payload.get("reason", "deployment paused by calibration guard"))

    # Non-dict payload still indicates an explicit operator-controlled guard.
    return True, "deployment paused by calibration guard file"

