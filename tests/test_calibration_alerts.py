from __future__ import annotations

from analysis.check_calibration_alerts import build_status, evaluate_alerts


def _report(*, rows_test: int, ece: float, ev: float, ev_ci_low: float, signals: int, data_origin: str = "live"):
    return {
        "selected_model": "isotonic",
        "selected_model_metrics": {"ece": ece, "brier": 0.12},
        "best_threshold": {
            "threshold": 0.72,
            "signals": signals,
            "ev_per_deployed_dollar": ev,
            "ev_ci_low_95": ev_ci_low,
        },
        "source_export": {
            "rows_test": rows_test,
            "data_origin": data_origin,
        },
    }


def _rules():
    return {
        "min_rows_test": 250,
        "max_ece": 0.03,
        "max_brier": 0.26,
        "min_ev_per_deployed_dollar": 0.0,
        "min_ev_ci_low_95": 0.0,
        "min_threshold_signals": 25,
        "allowed_data_origins": ["live", "paper_live", "production"],
    }


def test_gate_fails_low_sample_size() -> None:
    checks, breaches, _ = evaluate_alerts(
        _report(rows_test=120, ece=0.01, ev=0.002, ev_ci_low=0.001, signals=40),
        _rules(),
    )
    assert checks["rows_test_ok"] is False
    assert "rows_test_ok" in breaches


def test_gate_fails_non_positive_ev_ci_low() -> None:
    checks, breaches, _ = evaluate_alerts(
        _report(rows_test=300, ece=0.01, ev=0.002, ev_ci_low=0.0, signals=40),
        _rules(),
    )
    assert checks["ev_ci_low_ok"] is False
    assert "ev_ci_low_ok" in breaches


def test_gate_fails_high_ece() -> None:
    checks, breaches, _ = evaluate_alerts(
        _report(rows_test=300, ece=0.06, ev=0.002, ev_ci_low=0.001, signals=40),
        _rules(),
    )
    assert checks["ece_ok"] is False
    assert "ece_ok" in breaches


def test_gate_passes_when_all_checks_clear() -> None:
    checks, breaches, summary = evaluate_alerts(
        _report(rows_test=320, ece=0.02, ev=0.0015, ev_ci_low=0.0007, signals=60),
        _rules(),
    )
    assert breaches == []
    status = build_status(
        _report(rows_test=320, ece=0.02, ev=0.0015, ev_ci_low=0.0007, signals=60),
        checks,
        breaches,
        summary,
    )
    assert status["pause_deployment"] is False
    for key in ["generated_at", "reason", "window_trade_count", "ev_mean", "ev_ci_low_95", "ece"]:
        assert key in status


def test_gate_rejects_synthetic_export_origin() -> None:
    checks, breaches, _ = evaluate_alerts(
        _report(rows_test=320, ece=0.02, ev=0.0015, ev_ci_low=0.0007, signals=60, data_origin="synthetic"),
        _rules(),
    )
    assert checks["real_export_ok"] is False
    assert "real_export_ok" in breaches
