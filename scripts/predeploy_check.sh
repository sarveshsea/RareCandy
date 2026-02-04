#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="${EXPECTED_BRANCH:-main}"
STATUS_FILE="${CALIBRATION_STATUS_FILE:-analysis/artifacts/calibration/calibration_alert_status.json}"

echo "[predeploy] Starting checks..."

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "[predeploy] ERROR: Not inside a git repository." >&2
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${branch}" != "${EXPECTED_BRANCH}" ]]; then
  echo "[predeploy] ERROR: Current branch is '${branch}', expected '${EXPECTED_BRANCH}'." >&2
  exit 1
fi

# Refresh remote refs so branch/HEAD check is deterministic.
git fetch --prune origin >/dev/null 2>&1 || true
remote_ref="refs/remotes/origin/${EXPECTED_BRANCH}"
if ! git show-ref --verify --quiet "${remote_ref}"; then
  echo "[predeploy] ERROR: Missing remote branch origin/${EXPECTED_BRANCH}. Run git fetch." >&2
  exit 1
fi

local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "origin/${EXPECTED_BRANCH}")"
if [[ "${local_head}" != "${remote_head}" ]]; then
  echo "[predeploy] ERROR: Local HEAD (${local_head}) != origin/${EXPECTED_BRANCH} (${remote_head})." >&2
  echo "[predeploy] Hint: git pull --ff-only origin ${EXPECTED_BRANCH}" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[predeploy] ERROR: Working tree has tracked changes. Commit/stash before deploy." >&2
  exit 1
fi

required_files=(
  "generate_example.py"
  "unit_check.py"
  "scripts/run_calibration_pipeline.sh"
)
for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "[predeploy] ERROR: Missing required file: ${file}" >&2
    exit 1
  fi
done

echo "[predeploy] Running syntax checks..."
mapfile -t py_files < <(git ls-files "main.py" "analysis/*.py")
if [[ "${#py_files[@]}" -eq 0 ]]; then
  echo "[predeploy] ERROR: No Python files found for syntax check." >&2
  exit 1
fi
for py_file in "${py_files[@]}"; do
  python3 -m py_compile "${py_file}"
done

if [[ ! -f "${STATUS_FILE}" ]]; then
  echo "[predeploy] ERROR: Missing calibration alert status: ${STATUS_FILE}" >&2
  exit 1
fi

if ! python3 - "${STATUS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))

if "pause_deployment" not in payload:
    raise SystemExit("[predeploy] ERROR: status missing 'pause_deployment'")
if "checks" not in payload:
    raise SystemExit("[predeploy] ERROR: status missing 'checks'")

if bool(payload["pause_deployment"]):
    raise SystemExit("[predeploy] ERROR: deployment is paused by calibration guard")

print("[predeploy] Calibration status is clear.")
PY
then
  exit 1
fi

echo "[predeploy] PASS: deploy preflight checks completed."
