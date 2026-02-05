#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


REQUIRED_STATUS_FIELDS = {
    "pause_deployment",
    "reason",
    "window_trade_count",
    "ev_mean",
    "ev_ci_low_95",
    "ece",
    "generated_at",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "unknown"


def evaluate_alerts(report: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[Dict[str, bool], list[str], Dict[str, Any]]:
    selected = report.get("selected_model_metrics", {})
    best_thr = report.get("best_threshold", {})
    source = report.get("source_export", {})

    rows_test = _safe_int(source.get("rows_test"), 0)
    ece = _safe_float(selected.get("ece"), 1.0)
    brier = _safe_float(selected.get("brier"), 1.0)
    ev_mean = _safe_float(best_thr.get("ev_per_deployed_dollar"), -1.0)
    ev_ci_low = _safe_float(best_thr.get("ev_ci_low_95"), -1.0)
    signals = _safe_int(best_thr.get("signals"), 0)
    data_origin = str(source.get("data_origin", "unknown")).strip().lower()
    allowed_origins = {str(v).lower() for v in rules.get("allowed_data_origins", [])}

    checks = {
        "rows_test_ok": rows_test >= _safe_int(rules.get("min_rows_test", 250), 250),
        "ece_ok": ece <= _safe_float(rules.get("max_ece", 0.03), 0.03),
        "brier_ok": brier <= _safe_float(rules.get("max_brier", 0.26), 0.26),
        "ev_ok": ev_mean >= _safe_float(rules.get("min_ev_per_deployed_dollar", 0.0), 0.0),
        "ev_ci_low_ok": ev_ci_low > _safe_float(rules.get("min_ev_ci_low_95", 0.0), 0.0),
        "signals_ok": signals >= _safe_int(rules.get("min_threshold_signals", 25), 25),
        "real_export_ok": (not allowed_origins) or (data_origin in allowed_origins),
    }
    breaches = [k for k, v in checks.items() if not v]
    summary = {
        "rows_test": rows_test,
        "ece": ece,
        "ev_mean": ev_mean,
        "ev_ci_low_95": ev_ci_low,
        "signals": signals,
        "data_origin": data_origin,
    }
    return checks, breaches, summary


def build_status(report: Dict[str, Any], checks: Dict[str, bool], breaches: list[str], summary: Dict[str, Any]) -> Dict[str, Any]:
    selected = report.get("selected_model_metrics", {})
    best_thr = report.get("best_threshold", {})
    pause = len(breaches) > 0

    reason = "calibration gate clear"
    if pause:
        reason = "calibration gate breach: " + ", ".join(breaches)

    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pause_deployment": pause,
        "reason": reason,
        "window_trade_count": summary["rows_test"],
        "ev_mean": summary["ev_mean"],
        "ev_ci_low_95": summary["ev_ci_low_95"],
        "ece": summary["ece"],
        "breaches": breaches,
        "checks": checks,
        "selected_model": report.get("selected_model"),
        "selected_metrics": selected,
        "best_threshold": best_thr,
        "report_path": "",
        "rules_path": "",
    }

    missing = sorted(REQUIRED_STATUS_FIELDS.difference(status.keys()))
    if missing:
        raise RuntimeError(f"Status contract missing required fields: {missing}")

    return status


def build_manifest(
    report: Dict[str, Any], status: Dict[str, Any], rules_path: Path, report_path: Path, status_path: Path
) -> Dict[str, Any]:
    source = report.get("source_export", {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "input_exports": {
            "exports_dir": source.get("exports_dir"),
            "stem": source.get("stem"),
            "export_file": source.get("export_file"),
            "manifest_file": source.get("manifest_file"),
            "data_origin": source.get("data_origin"),
            "export_latest_ts_utc": source.get("export_latest_ts_utc"),
            "export_age_hours": source.get("export_age_hours"),
        },
        "model_version": report.get("selected_model"),
        "calibrator_type": report.get("selected_model"),
        "threshold_set": report.get("best_threshold"),
        "gate_result": {
            "pause_deployment": status.get("pause_deployment"),
            "reason": status.get("reason"),
            "breaches": status.get("breaches", []),
        },
        "artifacts": report.get("artifact_paths", {}),
        "status_path": str(status_path.resolve()),
        "report_path": str(report_path.resolve()),
        "rules_path": str(rules_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibration report against alert rules.")
    parser.add_argument("--report", default="analysis/artifacts/calibration/calibration_report.json")
    parser.add_argument("--rules", default="ops/calibration_alert_rules.json")
    parser.add_argument("--status-out", default="analysis/artifacts/calibration/calibration_alert_status.json")
    parser.add_argument("--pause-flag", default="ops/deployment_pause_calibration.json")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))

    checks, breaches, summary = evaluate_alerts(report, rules)
    status = build_status(report, checks, breaches, summary)
    status["report_path"] = str(Path(args.report).resolve())
    status["rules_path"] = str(Path(args.rules).resolve())
    pause = bool(status["pause_deployment"])

    status_path = Path(args.status_out)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    pause_path = Path(args.pause_flag)
    if pause:
        pause_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    elif pause_path.exists():
        pause_path.unlink()

    manifest = build_manifest(report, status, Path(args.rules), Path(args.report), status_path)
    manifest_path = status_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(status, indent=2))
    print(f"\nWrote alert status: {status_path}")
    print(f"Wrote deployment manifest: {manifest_path}")
    if pause:
        print(f"Deployment pause flag set: {pause_path}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
