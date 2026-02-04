#!/usr/bin/env bash
set -euo pipefail

python3 -m pytest -q tests/test_metrics.py tests/test_calibration_utils.py tests/test_pause_guard.py
