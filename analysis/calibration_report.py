#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from calibration_utils import (
    apply_calibrators,
    build_calibration_dataset,
    calibration_curve_table,
    calibration_metrics,
    fit_calibrators,
    load_export,
    per_bin_pnl_table,
    split_train_test,
    threshold_sweep,
)


ALLOWED_REAL_EXPORT_ORIGINS = {"live", "paper_live", "production"}


def _resolve_export_file(exports_dir: Path, stem: str) -> Path:
    csv_path = exports_dir / f"{stem}.csv"
    parquet_path = exports_dir / f"{stem}.parquet"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Missing export for stem={stem}: {csv_path} and {parquet_path}")


def _load_export_manifest(exports_dir: Path, stem: str) -> Tuple[Path, dict]:
    manifest_path = exports_dir / f"{stem}.manifest.json"
    if not manifest_path.exists():
        return manifest_path, {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid export manifest at {manifest_path}: expected object payload")
    return manifest_path, payload


def _parse_latest_timestamp(df: pd.DataFrame) -> datetime:
    if "timestamp" not in df.columns:
        raise ValueError("export is missing required 'timestamp' column for staleness checks")
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
    if ts.empty:
        raise ValueError("failed to parse any timestamps from export for staleness checks")
    latest = ts.max()
    return latest.to_pydatetime()


def _write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>\n"
    svg += "<rect width='100%' height='100%' fill='white'/>\n" + body + "\n</svg>"
    path.write_text(svg, encoding="utf-8")


def save_calibration_svg(path: Path, curves: Dict[str, pd.DataFrame], width: int = 900, height: int = 520) -> None:
    margin = 60
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    colors = {"raw": "#2563eb", "isotonic": "#16a34a", "logistic": "#dc2626"}

    body = [
        "<text x='450' y='32' text-anchor='middle' font-size='20' fill='#111827'>Calibration Curve</text>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#6b7280' stroke-width='1'/>",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#6b7280' stroke-width='1'/>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{margin}' stroke='#9ca3af' stroke-width='1.5' stroke-dasharray='6 6'/>",
        f"<text x='{width//2}' y='{height - 14}' text-anchor='middle' font-size='12' fill='#374151'>Mean predicted probability</text>",
        f"<text x='16' y='{height//2}' transform='rotate(-90 16,{height//2})' text-anchor='middle' font-size='12' fill='#374151'>Empirical positive rate</text>",
    ]

    for name, curve in curves.items():
        if curve.empty:
            continue
        points = []
        for _, row in curve.iterrows():
            x_val = float(row["mean_pred_prob"])
            y_val = float(row["empirical_pos_rate"])
            x = margin + x_val * plot_w
            y = margin + (1.0 - y_val) * plot_h
            points.append((x, y))
        if len(points) < 2:
            continue
        color = colors.get(name, "#111827")
        pt_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in points])
        body.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{pt_str}'/>")
        for x, y in points:
            body.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.2' fill='{color}'/>")

    legends = [("raw", "Raw"), ("isotonic", "Isotonic"), ("logistic", "Logistic")]
    for i, (key, lbl) in enumerate(legends):
        y = margin + 20 + i * 20
        c = colors[key]
        body.append(f"<rect x='{width - 190}' y='{y - 10}' width='10' height='10' fill='{c}'/>")
        body.append(f"<text x='{width - 174}' y='{y}' font-size='12' fill='#111827'>{lbl}</text>")

    _write_svg(path, width, height, "\n".join(body))


def save_bin_pnl_svg(path: Path, bin_df: pd.DataFrame, width: int = 900, height: int = 420) -> None:
    if bin_df.empty:
        _write_svg(path, width, height, "<text x='20' y='30' fill='#111827'>No bin data available</text>")
        return

    margin = 60
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    n = len(bin_df)
    max_abs = float(max(abs(bin_df["avg_pnl"].min()), abs(bin_df["avg_pnl"].max()), 1e-9))

    body = [
        "<text x='450' y='30' text-anchor='middle' font-size='20' fill='#111827'>Per-Bin P&L (Selected Model)</text>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#6b7280' stroke-width='1'/>",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#6b7280' stroke-width='1'/>",
    ]

    bar_w = plot_w / max(1, n)
    zero_y = margin + (1.0 - ((0.0 + max_abs) / (2 * max_abs))) * plot_h
    body.append(f"<line x1='{margin}' y1='{zero_y:.2f}' x2='{width - margin}' y2='{zero_y:.2f}' stroke='#9ca3af' stroke-width='1' stroke-dasharray='4 4'/>")

    for i, row in bin_df.reset_index(drop=True).iterrows():
        val = float(row["avg_pnl"])
        x_left = margin + i * bar_w + bar_w * 0.1
        x_right = margin + (i + 1) * bar_w - bar_w * 0.1
        y_val = margin + (1.0 - ((val + max_abs) / (2 * max_abs))) * plot_h
        y_top = min(y_val, zero_y)
        y_bot = max(y_val, zero_y)
        color = "#16a34a" if val >= 0 else "#dc2626"
        body.append(
            f"<rect x='{x_left:.2f}' y='{y_top:.2f}' width='{max(1.0, x_right - x_left):.2f}' "
            f"height='{max(1.0, y_bot - y_top):.2f}' fill='{color}' opacity='0.75'/>"
        )
        label = f"{row['bin_lower']:.2f}-{row['bin_upper']:.2f}"
        body.append(f"<text x='{(x_left + x_right)/2:.2f}' y='{height - margin + 14}' font-size='10' text-anchor='middle' fill='#374151'>{label}</text>")

    _write_svg(path, width, height, "\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibration report and calibrated artifacts for RareCandy exports.")
    parser.add_argument("--exports-dir", default="exports")
    parser.add_argument("--stem", default="rarecandy_export")
    parser.add_argument("--artifact-dir", default="analysis/artifacts/calibration")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--trading-cost", type=float, default=0.0006)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-signals", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--require-real-export", action="store_true")
    parser.add_argument("--max-export-age-hours", type=float, default=24.0)
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir)
    out_dir = Path(args.artifact_dir)
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    export_file = _resolve_export_file(exports_dir, args.stem)
    manifest_path, export_manifest = _load_export_manifest(exports_dir, args.stem)
    data_origin = str(export_manifest.get("data_origin", "unknown")).strip().lower()

    if args.require_real_export:
        if not export_manifest:
            raise SystemExit(
                f"Missing export manifest: {manifest_path}. "
                "Real-export deployment gate requires a manifest with data_origin."
            )
        if data_origin not in ALLOWED_REAL_EXPORT_ORIGINS:
            raise SystemExit(
                f"Export manifest origin '{data_origin}' is not eligible for deployment gate. "
                f"Allowed origins: {sorted(ALLOWED_REAL_EXPORT_ORIGINS)}"
            )

    df = load_export(exports_dir, args.stem)
    latest_ts = _parse_latest_timestamp(df)
    now_utc = datetime.now(timezone.utc)
    age_hours = (now_utc - latest_ts.astimezone(timezone.utc)).total_seconds() / 3600.0
    if args.max_export_age_hours > 0 and age_hours > args.max_export_age_hours:
        raise SystemExit(
            f"Export is stale: latest timestamp is {latest_ts.isoformat()} "
            f"({age_hours:.2f}h old, max allowed={args.max_export_age_hours}h)."
        )

    dataset = build_calibration_dataset(df, horizon=args.horizon, trading_cost=args.trading_cost)
    train_df, test_df = split_train_test(dataset, train_frac=args.train_frac)

    calibrators = fit_calibrators(train_df)
    scored = apply_calibrators(test_df, calibrators)

    y_true = scored["label"].to_numpy(dtype=float)
    model_metrics = {
        "raw": calibration_metrics(y_true, scored["prob_raw"].to_numpy(dtype=float), n_bins=args.bins),
        "isotonic": calibration_metrics(y_true, scored["prob_isotonic"].to_numpy(dtype=float), n_bins=args.bins),
        "logistic": calibration_metrics(y_true, scored["prob_logistic"].to_numpy(dtype=float), n_bins=args.bins),
    }

    selected_model = min(model_metrics.items(), key=lambda kv: (kv[1]["ece"], kv[1]["brier"]))[0]
    selected_prob_col = {"raw": "prob_raw", "isotonic": "prob_isotonic", "logistic": "prob_logistic"}[selected_model]

    curves = {
        "raw": calibration_curve_table(y_true, scored["prob_raw"].to_numpy(dtype=float), n_bins=args.bins),
        "isotonic": calibration_curve_table(y_true, scored["prob_isotonic"].to_numpy(dtype=float), n_bins=args.bins),
        "logistic": calibration_curve_table(y_true, scored["prob_logistic"].to_numpy(dtype=float), n_bins=args.bins),
    }
    for name, table in curves.items():
        table.to_csv(out_dir / f"calibration_curve_{name}.csv", index=False)

    bin_pnl = per_bin_pnl_table(scored, selected_prob_col, n_bins=args.bins)
    bin_pnl.to_csv(out_dir / "per_bin_pnl.csv", index=False)

    sweep = threshold_sweep(
        scored,
        selected_prob_col,
        threshold_min=0.50,
        threshold_max=0.95,
        threshold_step=0.01,
        min_signals=args.min_signals,
        bootstrap_samples=args.bootstrap_samples,
    )
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)

    if sweep.empty:
        best = {
            "threshold": None,
            "signals": 0,
            "ev_per_deployed_dollar": None,
            "ev_ci_low_95": None,
            "ev_ci_high_95": None,
            "expected_total_pnl": None,
            "win_rate": None,
        }
    else:
        top = sweep.iloc[0].to_dict()
        best = {
            "threshold": float(top["threshold"]),
            "signals": int(top["signals"]),
            "ev_per_deployed_dollar": float(top["ev_per_deployed_dollar"]),
            "ev_ci_low_95": float(top["ev_ci_low_95"]),
            "ev_ci_high_95": float(top["ev_ci_high_95"]),
            "expected_total_pnl": float(top["expected_total_pnl"]),
            "win_rate": float(top["win_rate"]),
        }

    save_calibration_svg(out_dir / "calibration_curve.svg", curves)
    save_bin_pnl_svg(out_dir / "per_bin_pnl.svg", bin_pnl)

    # Save calibrator artifacts
    joblib.dump(calibrators["isotonic"], model_dir / "isotonic_calibrator.joblib")
    joblib.dump(calibrators["logistic"], model_dir / "logistic_calibrator.joblib")
    (model_dir / "selected_model.txt").write_text(selected_model + "\n", encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_export": {
            "exports_dir": str(exports_dir.resolve()),
            "stem": args.stem,
            "export_file": str(export_file.resolve()),
            "manifest_file": str(manifest_path.resolve()) if manifest_path.exists() else None,
            "data_origin": data_origin,
            "export_latest_ts_utc": latest_ts.astimezone(timezone.utc).isoformat(),
            "export_age_hours": age_hours,
            "max_export_age_hours": args.max_export_age_hours,
            "rows_total": int(len(df)),
            "rows_used": int(len(dataset.frame)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
        },
        "settings": {
            "horizon": args.horizon,
            "trading_cost": args.trading_cost,
            "train_frac": args.train_frac,
            "bins": args.bins,
            "min_signals": args.min_signals,
            "bootstrap_samples": args.bootstrap_samples,
            "require_real_export": bool(args.require_real_export),
        },
        "model_metrics": model_metrics,
        "selected_model": selected_model,
        "selected_model_metrics": model_metrics[selected_model],
        "best_threshold": best,
        "artifact_paths": {
            "curve_svg": str((out_dir / "calibration_curve.svg").resolve()),
            "bin_pnl_svg": str((out_dir / "per_bin_pnl.svg").resolve()),
            "per_bin_pnl_csv": str((out_dir / "per_bin_pnl.csv").resolve()),
            "threshold_sweep_csv": str((out_dir / "threshold_sweep.csv").resolve()),
            "selected_model_txt": str((model_dir / "selected_model.txt").resolve()),
            "isotonic_model": str((model_dir / "isotonic_calibrator.joblib").resolve()),
            "logistic_model": str((model_dir / "logistic_calibrator.joblib").resolve()),
        },
    }

    report_path = out_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote calibration report: {report_path}")


if __name__ == "__main__":
    main()
