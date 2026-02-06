#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="${EXPECTED_BRANCH:-main}"
STATUS_FILE="${CALIBRATION_STATUS_FILE:-analysis/artifacts/calibration/calibration_alert_status.json}"
MANIFEST_FILE="${CALIBRATION_MANIFEST_FILE:-analysis/artifacts/calibration/manifest.json}"
MAX_STATUS_AGE_MINUTES="${MAX_STATUS_AGE_MINUTES:-180}"
MIN_TRADES="${MIN_TRADES:-250}"
MAX_ECE="${MAX_ECE:-0.03}"
MIN_EV_CI_LOW="${MIN_EV_CI_LOW:-0}"

echo "[predeploy] Starting checks..."

fail() {
  local code="$1"
  local cls="$2"
  shift 2
  echo "[predeploy:${cls}] ERROR: $*" >&2
  exit "${code}"
}

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  fail 11 "not_repo" "Not inside a git repository."
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${branch}" != "${EXPECTED_BRANCH}" ]]; then
  fail 12 "branch_mismatch" "Current branch is '${branch}', expected '${EXPECTED_BRANCH}'."
fi

# Refresh remote refs so branch/HEAD check is deterministic.
git fetch --prune origin >/dev/null 2>&1 || true
remote_ref="refs/remotes/origin/${EXPECTED_BRANCH}"
if ! git show-ref --verify --quiet "${remote_ref}"; then
  fail 13 "remote_missing" "Missing remote branch origin/${EXPECTED_BRANCH}. Run git fetch."
fi

local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "origin/${EXPECTED_BRANCH}")"
if [[ "${local_head}" != "${remote_head}" ]]; then
  echo "[predeploy] Hint: git pull --ff-only origin ${EXPECTED_BRANCH}" >&2
  fail 14 "head_mismatch" "Local HEAD (${local_head}) != origin/${EXPECTED_BRANCH} (${remote_head})."
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  fail 15 "dirty_tree" "Working tree has tracked changes. Commit/stash before deploy."
fi

required_files=(
  "generate_example.py"
  "unit_check.py"
  "scripts/write_export_manifest.py"
  "scripts/run_calibration_pipeline.sh"
)
for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    fail 16 "missing_file" "Missing required file: ${file}"
  fi
done

echo "[predeploy] Running syntax checks..."
py_files=()
while IFS= read -r py_file; do
  py_files+=("${py_file}")
done < <(git ls-files "main.py" "analysis/*.py")
if [[ "${#py_files[@]}" -eq 0 ]]; then
  fail 17 "syntax_check" "No Python files found for syntax check."
fi
for py_file in "${py_files[@]}"; do
  python3 -m py_compile "${py_file}"
done

if [[ ! -f "${STATUS_FILE}" ]]; then
  fail 18 "missing_status" "Missing calibration alert status: ${STATUS_FILE}"
fi

if [[ ! -f "${MANIFEST_FILE}" ]]; then
  fail 19 "missing_manifest" "Missing calibration manifest: ${MANIFEST_FILE}"
fi

if ! python3 - "${STATUS_FILE}" "${MANIFEST_FILE}" "${MAX_STATUS_AGE_MINUTES}" "${MIN_TRADES}" "${MAX_ECE}" "${MIN_EV_CI_LOW}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
max_age_minutes = float(sys.argv[3])
min_trades = int(sys.argv[4])
max_ece = float(sys.argv[5])
min_ev_ci_low = float(sys.argv[6])

payload = json.loads(status_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

required = {
    "pause_deployment",
    "reason",
    "window_trade_count",
    "ev_mean",
    "ev_ci_low_95",
    "ece",
    "generated_at",
}
missing = sorted(required.difference(payload.keys()))
if missing:
    raise SystemExit(f"[predeploy:status_contract] ERROR: missing keys {missing}")

if "gate_result" not in manifest:
    raise SystemExit("[predeploy:manifest_contract] ERROR: manifest missing gate_result")
if "input_exports" not in manifest:
    raise SystemExit("[predeploy:manifest_contract] ERROR: manifest missing input_exports")
manifest_pause = bool(manifest.get("gate_result", {}).get("pause_deployment", False))
status_pause = bool(payload["pause_deployment"])
if manifest_pause != status_pause:
    raise SystemExit("[predeploy:manifest_contract] ERROR: manifest/status pause mismatch")

generated_at = datetime.fromisoformat(str(payload["generated_at"]).replace("Z", "+00:00"))
if generated_at.tzinfo is None:
    generated_at = generated_at.replace(tzinfo=timezone.utc)
age_min = (datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)).total_seconds() / 60.0
if age_min > max_age_minutes:
    raise SystemExit(
        f"[predeploy:status_stale] ERROR: status age {age_min:.1f}m exceeds {max_age_minutes:.1f}m"
    )

if status_pause:
    raise SystemExit("[predeploy:gate_failed] ERROR: deployment is paused by calibration guard")

if int(payload["window_trade_count"]) < min_trades:
    raise SystemExit(
        f"[predeploy:gate_failed] ERROR: window_trade_count={payload['window_trade_count']} < {min_trades}"
    )
if float(payload["ev_ci_low_95"]) <= min_ev_ci_low:
    raise SystemExit(
        f"[predeploy:gate_failed] ERROR: ev_ci_low_95={payload['ev_ci_low_95']} <= {min_ev_ci_low}"
    )
if float(payload["ece"]) > max_ece:
    raise SystemExit(
        f"[predeploy:gate_failed] ERROR: ece={payload['ece']} > {max_ece}"
    )

print("[predeploy] Calibration status is clear.")
PY
then
  exit 20
fi

echo "[predeploy] PASS: deploy preflight checks completed."
