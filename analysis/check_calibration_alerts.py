#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibration report against alert rules.")
    parser.add_argument("--report", default="analysis/artifacts/calibration/calibration_report.json")
    parser.add_argument("--rules", default="ops/calibration_alert_rules.json")
    parser.add_argument("--status-out", default="analysis/artifacts/calibration/calibration_alert_status.json")
    parser.add_argument("--pause-flag", default="ops/deployment_pause_calibration.json")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))

    selected = report["selected_model_metrics"]
    best_thr = report.get("best_threshold", {})
    source = report.get("source_export", {})

    checks = {
        "rows_test_ok": int(source.get("rows_test", 0)) >= int(rules["min_rows_test"]),
        "ece_ok": float(selected.get("ece", 1.0)) <= float(rules["max_ece"]),
        "brier_ok": float(selected.get("brier", 1.0)) <= float(rules["max_brier"]),
        "ev_ok": float(best_thr.get("ev_per_deployed_dollar") or -1.0) >= float(rules["min_ev_per_deployed_dollar"]),
        "signals_ok": int(best_thr.get("signals") or 0) >= int(rules["min_threshold_signals"]),
    }
    breaches = [k for k, v in checks.items() if not v]
    pause = len(breaches) > 0

    status = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pause_deployment": pause,
        "breaches": breaches,
        "checks": checks,
        "selected_model": report.get("selected_model"),
        "selected_metrics": selected,
        "best_threshold": best_thr,
        "report_path": str(Path(args.report).resolve()),
        "rules_path": str(Path(args.rules).resolve()),
    }

    status_path = Path(args.status_out)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    pause_path = Path(args.pause_flag)
    if pause:
        pause_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    elif pause_path.exists():
        pause_path.unlink()

    print(json.dumps(status, indent=2))
    print(f"\nWrote alert status: {status_path}")
    if pause:
        print(f"Deployment pause flag set: {pause_path}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
