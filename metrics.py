from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_BARS_PER_YEAR = 365 * 24 * 4  # 15m bars


def _safe_std(x: pd.Series) -> float:
    v = float(x.std())
    return v if np.isfinite(v) else 0.0


def compute_equity_from_positions(
    df: pd.DataFrame,
    *,
    initial_equity: float = 10000.0,
    price_col: str = "close",
    position_col: str = "position",
    cost_per_side: float = 0.0006,
) -> pd.Series:
    """
    Build an equity curve from close prices and signed position {-1,0,1}.
    Costs are charged on absolute position changes (per side).
    """
    if price_col not in df.columns:
        raise ValueError(f"Missing required column: {price_col}")
    if position_col not in df.columns:
        raise ValueError(f"Missing required column: {position_col}")

    close = df[price_col].astype(float).to_numpy()
    pos = df[position_col].fillna(0.0).astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float, index=df.index, name="equity")

    equity = np.full(n, np.nan, dtype=float)
    equity[0] = float(initial_equity)

    for i in range(1, n):
        prev_eq = equity[i - 1]
        r = (close[i] / close[i - 1] - 1.0) if close[i - 1] > 0 else 0.0
        eq = prev_eq * (1.0 + pos[i - 1] * r)
        delta = abs(pos[i] - pos[i - 1])
        if delta > 1e-12:
            eq *= (1.0 - cost_per_side * delta)
        equity[i] = eq

    return pd.Series(equity, index=df.index, name="equity")


def compute_trade_returns(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    position_col: str = "position",
    cost_per_side: float = 0.0006,
) -> pd.Series:
    """
    Approximate per-trade returns from position transitions.
    """
    if price_col not in df.columns or position_col not in df.columns:
        return pd.Series([], dtype=float, name="trade_return")

    close = df[price_col].astype(float).to_numpy()
    pos = df[position_col].fillna(0.0).astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float, name="trade_return")

    equity = 1.0
    entry_eq = np.nan
    trades = []

    for i in range(1, n):
        r = (close[i] / close[i - 1] - 1.0) if close[i - 1] > 0 else 0.0
        if abs(pos[i - 1]) > 1e-12:
            equity *= (1.0 + pos[i - 1] * r)

        delta = abs(pos[i] - pos[i - 1])
        if delta > 1e-12:
            equity *= (1.0 - cost_per_side * delta)
            if abs(pos[i - 1]) < 1e-12 and abs(pos[i]) > 1e-12:
                entry_eq = equity
            elif abs(pos[i - 1]) > 1e-12 and abs(pos[i]) < 1e-12 and not np.isnan(entry_eq):
                trades.append(equity / entry_eq - 1.0)
                entry_eq = np.nan
            elif pos[i - 1] * pos[i] < 0 and not np.isnan(entry_eq):
                trades.append(equity / entry_eq - 1.0)
                entry_eq = equity

    if abs(pos[-1]) > 1e-12 and not np.isnan(entry_eq):
        trades.append(equity / entry_eq - 1.0)

    return pd.Series(trades, name="trade_return", dtype=float)


def compute_metrics(
    df: pd.DataFrame,
    *,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    equity_col: str = "equity",
    price_col: str = "close",
    position_col: str = "position",
    initial_equity: float = 10000.0,
    cost_per_side: float = 0.0006,
) -> Dict[str, float]:
    """
    Compute headline strategy metrics.
    """
    if equity_col in df.columns:
        equity = df[equity_col].astype(float).copy()
    else:
        equity = compute_equity_from_positions(
            df,
            initial_equity=initial_equity,
            price_col=price_col,
            position_col=position_col,
            cost_per_side=cost_per_side,
        )

    if equity.empty:
        raise ValueError("No rows available to compute metrics")

    ret = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_periods = max(len(equity) - 1, 1)
    annual_return = float((1.0 + total_return) ** (bars_per_year / n_periods) - 1.0)

    volatility = _safe_std(ret) * np.sqrt(bars_per_year)
    sharpe = float((ret.mean() / ret.std()) * np.sqrt(bars_per_year)) if _safe_std(ret) > 1e-12 else 0.0

    downside = ret.clip(upper=0.0)
    downside_std = _safe_std(downside)
    sortino = float((ret.mean() / downside_std) * np.sqrt(bars_per_year)) if downside_std > 1e-12 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(abs(drawdown.min()))
    calmar = float(annual_return / max_dd) if max_dd > 1e-12 else 0.0

    trade_returns = (
        df["trade_return"].dropna().astype(float)
        if "trade_return" in df.columns and df["trade_return"].notna().any()
        else compute_trade_returns(df, price_col=price_col, position_col=position_col, cost_per_side=cost_per_side)
    )
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    win_rate = float((len(wins) / len(trade_returns)) * 100.0) if len(trade_returns) else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-12 else (99.0 if gross_profit > 0 else 0.0)
    expectancy = float(trade_returns.mean()) if len(trade_returns) else 0.0

    return {
        "rows": int(len(df)),
        "bars_per_year": int(bars_per_year),
        "total_return_pct": total_return * 100.0,
        "annual_return_pct": annual_return * 100.0,
        "volatility_pct": volatility * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100.0,
        "calmar": calmar,
        "num_trades": int(len(trade_returns)),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "expectancy_pct": expectancy * 100.0,
    }


def bootstrap_confidence_interval(
    values: Iterable[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap CI for the mean of a vector.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("bootstrap_confidence_interval received no finite values")

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return {
        "mean": float(arr.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "confidence": float(ci),
        "n": int(n),
        "n_boot": int(n_boot),
    }


def compare_to_baseline(strategy_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "delta_total_return_pct": float(strategy_metrics["total_return_pct"] - baseline_metrics["total_return_pct"]),
        "delta_sharpe": float(strategy_metrics["sharpe"] - baseline_metrics["sharpe"]),
        "delta_max_drawdown_pct": float(strategy_metrics["max_drawdown_pct"] - baseline_metrics["max_drawdown_pct"]),
        "delta_profit_factor": float(strategy_metrics["profit_factor"] - baseline_metrics["profit_factor"]),
    }


def deployment_recommendation(
    strategy_metrics: Dict[str, float],
    baseline_metrics: Optional[Dict[str, float]] = None,
    sharpe_ci: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Emit deployment recommendation:
    - full: strong positive and better than baseline.
    - canary: positive but mixed confidence.
    - no-go: weak/negative risk-adjusted performance.
    """
    gates = {
        "positive_return": strategy_metrics["total_return_pct"] > 0,
        "acceptable_drawdown": strategy_metrics["max_drawdown_pct"] <= 20.0,
        "positive_sharpe": strategy_metrics["sharpe"] > 0.0,
        "profit_factor_gt_1": strategy_metrics["profit_factor"] > 1.0,
    }

    if baseline_metrics is not None:
        gates["beats_baseline_return"] = (
            strategy_metrics["total_return_pct"] >= baseline_metrics["total_return_pct"]
        )
        gates["beats_baseline_sharpe"] = strategy_metrics["sharpe"] >= baseline_metrics["sharpe"]

    if sharpe_ci is not None:
        gates["sharpe_ci_above_zero"] = sharpe_ci["ci_lower"] > 0.0

    passed = [k for k, v in gates.items() if v]
    failed = [k for k, v in gates.items() if not v]

    if all(gates.values()):
        mode = "full"
    elif gates["positive_return"] and gates["acceptable_drawdown"] and gates["profit_factor_gt_1"]:
        mode = "canary"
    else:
        mode = "no-go"

    return {
        "mode": mode,
        "gates": gates,
        "passed": passed,
        "failed": failed,
        "summary": (
            "Promote to full deployment."
            if mode == "full"
            else "Run canary deployment with tighter monitoring."
            if mode == "canary"
            else "Do not deploy; iterate strategy."
        ),
    }
