#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def resolve_export_file(exports_dir: Path, stem: str) -> Path:
    parquet_path = exports_dir / f"{stem}.parquet"
    csv_path = exports_dir / f"{stem}.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"missing export: {parquet_path} or {csv_path}")


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write export manifest metadata used by deployment gate.")
    parser.add_argument("--exports-dir", default="exports/live")
    parser.add_argument("--stem", default="rarecandy_export")
    parser.add_argument("--data-origin", default="live", choices=["live", "paper_live", "production", "synthetic"])
    parser.add_argument("--generator", default="runtime_exporter")
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)
    export_file = resolve_export_file(exports_dir, args.stem)
    frame = load_frame(export_file)
    if "timestamp" not in frame.columns:
        raise SystemExit("export is missing required timestamp column")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dropna()
    if timestamps.empty:
        raise SystemExit("unable to parse timestamps from export")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_origin": args.data_origin,
        "generator": args.generator,
        "export_file": str(export_file.resolve()),
        "rows": int(len(frame)),
        "timestamp_start": timestamps.min().to_pydatetime().isoformat(),
        "timestamp_end": timestamps.max().to_pydatetime().isoformat(),
    }
    manifest_path = exports_dir / f"{args.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
