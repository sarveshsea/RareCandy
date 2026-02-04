from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.calibration_utils import (
    build_calibration_dataset,
    calibration_metrics,
    fit_calibrators,
    split_train_test,
    threshold_sweep,
)


def _make_df(n: int = 300) -> pd.DataFrame:
    close = 100 * np.cumprod(1 + np.random.default_rng(42).normal(0.0005, 0.01, n))
    return pd.DataFrame({"close": close})


def test_build_calibration_dataset_has_required_columns() -> None:
    ds = build_calibration_dataset(_make_df(), horizon=1, trading_cost=0.0)
    assert len(ds.frame) > 0
    assert {"raw_prob", "label", "pnl"}.issubset(ds.frame.columns)
    assert ds.raw_prob.between(0.0, 1.0).all()


def test_fit_calibrators_and_sweep() -> None:
    ds = build_calibration_dataset(_make_df(), horizon=1, trading_cost=0.0)
    train_df, test_df = split_train_test(ds, train_frac=0.7)
    cal = fit_calibrators(train_df)
    assert "isotonic" in cal and "logistic" in cal

    # Build a synthetic scored frame for threshold sweep
    scored = test_df.copy()
    scored["prob_logistic"] = np.clip(scored["raw_prob"], 0.001, 0.999)
    sweep = threshold_sweep(scored, "prob_logistic", threshold_min=0.5, threshold_max=0.9, threshold_step=0.1, min_signals=5)
    assert not sweep.empty
    assert "ev_per_deployed_dollar" in sweep.columns


def test_calibration_metrics_outputs() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    y_prob = np.array([0.1, 0.9, 0.3, 0.7, 0.8, 0.2], dtype=float)
    m = calibration_metrics(y_true, y_prob, n_bins=5)
    assert m["brier"] >= 0
    assert m["log_loss"] >= 0
    assert 0 <= m["ece"] <= 1
