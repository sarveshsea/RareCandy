#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from metrics import compute_equity_from_positions, compute_trade_returns


def build_strategy_export(
    *,
    seed: int = 7,
    n: int = 2500,
    timeframe_minutes: int = 15,
    symbol: str = "BTC/USD",
    initial_price: float = 45000.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq=f"{timeframe_minutes}min", tz="UTC")

    # Synthetic market with mild drift and clustered noise.
    noise = rng.normal(0.0, 0.003, n)
    drift = 0.00015
    log_ret = drift + noise
    close = initial_price * np.exp(np.cumsum(log_ret))

    open_ = np.empty(n)
    open_[0] = close[0] / (1.0 + noise[0])
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0.0012, 0.0006, n))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=7.6, sigma=0.4, size=n)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Simple trend-following signal.
    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_slow = df["close"].ewm(span=60, adjust=False).mean()
    spread = (ema_fast - ema_slow) / df["close"]
    spread_scale = spread.rolling(200, min_periods=20).std().replace(0.0, np.nan)
    spread_scale = spread_scale.fillna(spread_scale.median() if np.isfinite(spread_scale.median()) else 1e-6)
    score_z = (spread / spread_scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    confidence = 1.0 / (1.0 + np.exp(-score_z))
    raw_position = np.where(ema_fast > ema_slow, 1.0, 0.0)
    position = pd.Series(raw_position, index=df.index).shift(1).fillna(0.0)  # trade next bar
    signal = np.where((position > 0) & (position.shift(1).fillna(0) == 0), "ENTRY_LONG",
                      np.where((position == 0) & (position.shift(1).fillna(0) > 0), "EXIT_LONG", "HOLD"))

    df["position"] = position
    df["signal"] = signal
    df["raw_score"] = score_z
    df["confidence"] = confidence.clip(0.001, 0.999)
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df["equity"] = compute_equity_from_positions(df, initial_equity=10000.0, cost_per_side=0.0006)
    trades = compute_trade_returns(df, cost_per_side=0.0006)
    df["trade_return"] = np.nan
    # Stamp closed-trade returns at corresponding exit transitions.
    exits = (df["position"] == 0) & (df["position"].shift(1).fillna(0) > 0)
    exit_idx = df.index[exits]
    for i, tr in enumerate(trades.tolist()):
        if i < len(exit_idx):
            df.loc[exit_idx[i], "trade_return"] = tr

    return df


def build_baseline_export(strategy_df: pd.DataFrame) -> pd.DataFrame:
    b = strategy_df.copy()
    b["position"] = 1.0
    b["signal"] = "BUY_HOLD"
    b["equity"] = compute_equity_from_positions(b, initial_equity=10000.0, cost_per_side=0.0000)
    b["trade_return"] = np.nan
    return b


def main() -> None:
    out_dir = Path("exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = build_strategy_export()
    baseline = build_baseline_export(strategy)

    strategy_csv = out_dir / "rarecandy_export.csv"
    strategy_parquet = out_dir / "rarecandy_export.parquet"
    baseline_csv = out_dir / "baseline_export.csv"
    baseline_parquet = out_dir / "baseline_export.parquet"

    strategy.to_csv(strategy_csv, index=False)
    strategy.to_parquet(strategy_parquet, index=False)
    baseline.to_csv(baseline_csv, index=False)
    baseline.to_parquet(baseline_parquet, index=False)

    export_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_origin": "synthetic",
        "generator": "generate_example.py",
        "rows": int(len(strategy)),
        "timestamp_start": str(strategy["timestamp"].iloc[0]),
        "timestamp_end": str(strategy["timestamp"].iloc[-1]),
    }
    (out_dir / "rarecandy_export.manifest.json").write_text(
        json.dumps(export_manifest, indent=2), encoding="utf-8"
    )

    print(f"Wrote {strategy_csv} ({len(strategy)} rows)")
    print(f"Wrote {strategy_parquet} ({len(strategy)} rows)")
    print(f"Wrote {out_dir / 'rarecandy_export.manifest.json'}")
    print(f"Wrote {baseline_csv} ({len(baseline)} rows)")
    print(f"Wrote {baseline_parquet} ({len(baseline)} rows)")


if __name__ == "__main__":
    main()
