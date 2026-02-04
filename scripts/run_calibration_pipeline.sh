#!/usr/bin/env bash
set -euo pipefail

python3 analysis/calibration_report.py \
  --exports-dir exports \
  --stem rarecandy_export \
  --artifact-dir analysis/artifacts/calibration

python3 analysis/check_calibration_alerts.py \
  --report analysis/artifacts/calibration/calibration_report.json \
  --rules ops/calibration_alert_rules.json \
  --status-out analysis/artifacts/calibration/calibration_alert_status.json \
  --pause-flag ops/deployment_pause_calibration.json
