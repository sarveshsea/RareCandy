from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


@dataclass
class CalibrationDataset:
    frame: pd.DataFrame
    raw_prob: pd.Series
    label: pd.Series
    pnl: pd.Series


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_export(exports_dir: Path, stem: str) -> pd.DataFrame:
    csv_path = exports_dir / f"{stem}.csv"
    parquet_path = exports_dir / f"{stem}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing export for stem={stem}: {csv_path} and {parquet_path}")


def _infer_probability(df: pd.DataFrame) -> pd.Series:
    candidates = ["pred_prob", "prediction_prob", "prob", "confidence", "score_prob"]
    for col in candidates:
        if col in df.columns:
            prob = pd.to_numeric(df[col], errors="coerce").astype(float)
            return prob.clip(0.001, 0.999)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=60, adjust=False).mean()
    spread = (ema_fast - ema_slow) / close.replace(0.0, np.nan)
    scale = spread.rolling(200, min_periods=20).std().replace(0.0, np.nan)
    scale = scale.fillna(scale.median() if np.isfinite(scale.median()) else 1e-6)
    z = (spread / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return pd.Series(sigmoid(z), index=df.index).clip(0.001, 0.999)


def _infer_label_and_pnl(df: pd.DataFrame, *, horizon: int = 1, trading_cost: float = 0.0006) -> Tuple[pd.Series, pd.Series]:
    if "target_up" in df.columns:
        label = pd.to_numeric(df["target_up"], errors="coerce").fillna(0.0).astype(int).clip(0, 1)
    else:
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        fwd_ret = close.shift(-horizon) / close - 1.0
        label = (fwd_ret > 0).astype(int)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    fwd_ret = close.shift(-horizon) / close - 1.0
    pnl = fwd_ret - trading_cost
    return label.astype(int), pnl.astype(float)


def build_calibration_dataset(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    trading_cost: float = 0.0006,
) -> CalibrationDataset:
    prob = _infer_probability(df)
    label, pnl = _infer_label_and_pnl(df, horizon=horizon, trading_cost=trading_cost)

    work = df.copy()
    work["raw_prob"] = prob
    work["label"] = label
    work["pnl"] = pnl
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["raw_prob", "label", "pnl"])
    work["raw_prob"] = work["raw_prob"].clip(0.001, 0.999)
    work["label"] = work["label"].astype(int)
    work["pnl"] = work["pnl"].astype(float)

    return CalibrationDataset(
        frame=work,
        raw_prob=work["raw_prob"],
        label=work["label"],
        pnl=work["pnl"],
    )


def split_train_test(dataset: CalibrationDataset, train_frac: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(dataset.frame)
    cut = max(1, int(n * train_frac))
    cut = min(cut, n - 1) if n > 1 else 1
    train_df = dataset.frame.iloc[:cut].copy()
    test_df = dataset.frame.iloc[cut:].copy()
    if test_df.empty:
        test_df = dataset.frame.iloc[-1:].copy()
    return train_df, test_df


def fit_calibrators(train_df: pd.DataFrame) -> Dict[str, object]:
    x = train_df["raw_prob"].to_numpy(dtype=float)
    y = train_df["label"].to_numpy(dtype=int)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y)

    logit = LogisticRegression(max_iter=200, random_state=42)
    logit.fit(x.reshape(-1, 1), y)

    return {"isotonic": iso, "logistic": logit}


def apply_calibrators(test_df: pd.DataFrame, calibrators: Dict[str, object]) -> pd.DataFrame:
    out = test_df.copy()
    raw = out["raw_prob"].to_numpy(dtype=float)
    out["prob_raw"] = raw
    out["prob_isotonic"] = calibrators["isotonic"].predict(raw)
    out["prob_logistic"] = calibrators["logistic"].predict_proba(raw.reshape(-1, 1))[:, 1]
    for col in ["prob_raw", "prob_isotonic", "prob_logistic"]:
        out[col] = out[col].clip(0.001, 0.999)
    return out


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if np.any(mask):
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float).clip(0.001, 0.999)
    brier = float(np.mean((y_prob - y_true) ** 2))
    ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    ece = compute_ece(y_true, y_prob, n_bins=n_bins)
    return {"brier": brier, "log_loss": ll, "ece": ece}


def calibration_curve_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_pred_prob": prob_pred, "empirical_pos_rate": prob_true})


def per_bin_pnl_table(frame: pd.DataFrame, prob_col: str, n_bins: int = 10) -> pd.DataFrame:
    work = frame[[prob_col, "label", "pnl"]].copy()
    work = work.dropna()
    work["bin"] = pd.qcut(work[prob_col], q=min(n_bins, work[prob_col].nunique()), duplicates="drop")
    grouped = work.groupby("bin", observed=True)

    rows = []
    for interval, g in grouped:
        rows.append(
            {
                "bin_lower": float(interval.left),
                "bin_upper": float(interval.right),
                "count": int(len(g)),
                "avg_confidence": float(g[prob_col].mean()),
                "win_rate": float(g["label"].mean()),
                "avg_pnl": float(g["pnl"].mean()),
                "edge_per_deployed_dollar": float(g["pnl"].mean()),
                "expected_pnl_total": float(g["pnl"].mean() * len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("bin_lower").reset_index(drop=True)


def threshold_sweep(
    frame: pd.DataFrame,
    prob_col: str,
    *,
    threshold_min: float = 0.50,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
    min_signals: int = 20,
) -> pd.DataFrame:
    rows: List[dict] = []
    fixed = np.arange(threshold_min, threshold_max + 1e-9, threshold_step)
    probs = frame[prob_col].dropna().to_numpy(dtype=float)
    if probs.size == 0:
        return pd.DataFrame(
            columns=[
                "threshold",
                "signals",
                "deployed_dollars",
                "win_rate",
                "ev_per_deployed_dollar",
                "expected_total_pnl",
            ]
        )
    quantile_grid = np.linspace(0.50, 0.98, 25)
    dynamic = np.quantile(probs, quantile_grid)
    thresholds = np.unique(np.concatenate([fixed, dynamic]))
    for thr in thresholds:
        deploy = frame[frame[prob_col] >= thr]
        n = len(deploy)
        if n < min_signals:
            continue
        pnl = deploy["pnl"].to_numpy(dtype=float)
        ev = float(np.mean(pnl))
        win_rate = float(np.mean(deploy["label"].to_numpy(dtype=float)))
        deployed_dollars = float(n)
        expected_total = ev * deployed_dollars
        rows.append(
            {
                "threshold": float(thr),
                "signals": int(n),
                "deployed_dollars": deployed_dollars,
                "win_rate": win_rate,
                "ev_per_deployed_dollar": ev,
                "expected_total_pnl": expected_total,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "threshold",
                "signals",
                "deployed_dollars",
                "win_rate",
                "ev_per_deployed_dollar",
                "expected_total_pnl",
            ]
        )
    return pd.DataFrame(rows).sort_values(["ev_per_deployed_dollar", "signals"], ascending=[False, False]).reset_index(drop=True)
