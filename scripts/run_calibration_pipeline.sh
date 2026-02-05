#!/usr/bin/env bash
set -euo pipefail

REAL_EXPORTS_DIR="${REAL_EXPORTS_DIR:-exports/live}"
REAL_EXPORT_STEM="${REAL_EXPORT_STEM:-rarecandy_export}"
MAX_EXPORT_AGE_HOURS="${MAX_EXPORT_AGE_HOURS:-24}"

echo "[calibration] exports_dir=${REAL_EXPORTS_DIR} stem=${REAL_EXPORT_STEM} max_age_h=${MAX_EXPORT_AGE_HOURS}"

python3 analysis/calibration_report.py \
  --exports-dir "${REAL_EXPORTS_DIR}" \
  --stem "${REAL_EXPORT_STEM}" \
  --artifact-dir analysis/artifacts/calibration \
  --require-real-export \
  --max-export-age-hours "${MAX_EXPORT_AGE_HOURS}"

python3 analysis/check_calibration_alerts.py \
  --report analysis/artifacts/calibration/calibration_report.json \
  --rules ops/calibration_alert_rules.json \
  --status-out analysis/artifacts/calibration/calibration_alert_status.json \
  --pause-flag ops/deployment_pause_calibration.json
