from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import (
    bootstrap_confidence_interval,
    compare_to_baseline,
    compute_equity_from_positions,
    compute_metrics,
    compute_trade_returns,
    deployment_recommendation,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
            "position": [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        }
    )


def test_compute_equity_from_positions_increases_on_uptrend() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0], "position": [1.0, 1.0, 1.0, 1.0]})
    eq = compute_equity_from_positions(df, initial_equity=1000.0, cost_per_side=0.0)
    assert len(eq) == len(df)
    assert eq.iloc[-1] > eq.iloc[0]


def test_compute_trade_returns_detects_closed_trades() -> None:
    df = _sample_df()
    tr = compute_trade_returns(df, cost_per_side=0.0)
    assert len(tr) == 2
    assert np.isfinite(tr).all()


def test_compute_metrics_returns_expected_keys() -> None:
    df = _sample_df()
    df["equity"] = compute_equity_from_positions(df, initial_equity=10000.0, cost_per_side=0.0)
    m = compute_metrics(df)
    expected = {
        "total_return_pct",
        "annual_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "profit_factor",
        "num_trades",
    }
    assert expected.issubset(m.keys())
    assert m["rows"] == len(df)


def test_bootstrap_ci_bounds() -> None:
    values = [0.1, 0.2, 0.15, 0.3, 0.05]
    ci = bootstrap_confidence_interval(values, n_boot=400, ci=0.90, seed=1)
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
    assert ci["n"] == len(values)


def test_deployment_recommendation_full_when_all_gates_pass() -> None:
    strategy = {
        "total_return_pct": 25.0,
        "max_drawdown_pct": 8.0,
        "sharpe": 1.4,
        "profit_factor": 1.5,
    }
    baseline = {
        "total_return_pct": 10.0,
        "max_drawdown_pct": 12.0,
        "sharpe": 0.8,
        "profit_factor": 1.1,
    }
    sharpe_ci = {"ci_lower": 0.2, "ci_upper": 0.5}
    rec = deployment_recommendation(strategy, baseline, sharpe_ci)
    assert rec["mode"] == "full"
    assert len(rec["failed"]) == 0


def test_compare_to_baseline_shapes() -> None:
    s = {"total_return_pct": 8.0, "sharpe": 0.7, "max_drawdown_pct": 5.0, "profit_factor": 1.2}
    b = {"total_return_pct": 6.0, "sharpe": 0.5, "max_drawdown_pct": 7.0, "profit_factor": 1.0}
    d = compare_to_baseline(s, b)
    assert np.isclose(d["delta_total_return_pct"], 2.0)
    assert np.isclose(d["delta_sharpe"], 0.2)
    assert np.isclose(d["delta_profit_factor"], 0.2)
