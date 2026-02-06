#!/usr/bin/env bash
set -euo pipefail

python3 -m pytest -q \
  tests/test_metrics.py \
  tests/test_calibration_utils.py \
  tests/test_pause_guard.py \
  tests/test_calibration_alerts.py \
  tests/test_risk_manager.py \
  tests/test_trend_pullback.py
