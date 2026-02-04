#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import compute_metrics


REQUIRED_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "position",
    "signal",
    "equity",
]


def _assert_columns(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _compare_csv_vs_parquet(csv_df: pd.DataFrame, pq_df: pd.DataFrame, name: str) -> dict:
    if len(csv_df) != len(pq_df):
        raise ValueError(f"{name}: row mismatch CSV={len(csv_df)} PARQUET={len(pq_df)}")

    key_cols = ["open", "high", "low", "close", "volume", "position", "equity"]
    max_abs_diff = 0.0
    for col in key_cols:
        a = csv_df[col].astype(float).to_numpy()
        b = pq_df[col].astype(float).to_numpy()
        diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 1e-8:
            raise ValueError(f"{name}: {col} differs between CSV/PARQUET (max abs diff={diff})")

    return {"rows": int(len(csv_df)), "max_abs_diff": max_abs_diff}


def _load_export_pair(exports_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = exports_dir / f"{stem}.csv"
    pq_path = exports_dir / f"{stem}.parquet"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    if not pq_path.exists():
        raise FileNotFoundError(f"Missing file: {pq_path}")
    return pd.read_csv(csv_path), pd.read_parquet(pq_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RareCandy CSV/Parquet exports and metrics.")
    parser.add_argument("--exports-dir", default="exports", help="Directory containing export files.")
    parser.add_argument(
        "--out-report",
        default="exports/unit_check_report.json",
        help="Where to write unit check report JSON.",
    )
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir)
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    strategy_csv, strategy_pq = _load_export_pair(exports_dir, "rarecandy_export")
    baseline_csv, baseline_pq = _load_export_pair(exports_dir, "baseline_export")

    _assert_columns(strategy_csv, "strategy_csv")
    _assert_columns(strategy_pq, "strategy_parquet")
    _assert_columns(baseline_csv, "baseline_csv")
    _assert_columns(baseline_pq, "baseline_parquet")

    strategy_compare = _compare_csv_vs_parquet(strategy_csv, strategy_pq, "strategy")
    baseline_compare = _compare_csv_vs_parquet(baseline_csv, baseline_pq, "baseline")

    strategy_metrics = compute_metrics(strategy_csv)
    baseline_metrics = compute_metrics(baseline_csv)

    report = {
        "ok": True,
        "exports_dir": str(exports_dir.resolve()),
        "strategy": strategy_compare,
        "baseline": baseline_compare,
        "strategy_metrics": strategy_metrics,
        "baseline_metrics": baseline_metrics,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
