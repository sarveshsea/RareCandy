#!/usr/bin/env python3
"""
Rebalance-based adaptive selector:
- Every N bars, pick the best expert from recent history.
- Hold that expert's position stream until next rebalance.
- This sharply reduces turnover vs per-bar voting.
"""

from __future__ import annotations

import datetime as dt
import itertools
import os
from dataclasses import dataclass
from typing import Dict, List

import ccxt
import numpy as np
import pandas as pd

import backtest_pine_batch as base
import backtest_pine_walkforward as wf
import adaptive_online_ensemble as aoe


TIMEFRAME = "15m"
START_UTC = dt.datetime(2025, 5, 1, tzinfo=dt.timezone.utc)
END_UTC = dt.datetime(2026, 2, 3, tzinfo=dt.timezone.utc)
COST_PER_SIDE = 0.0006
SYMBOLS = ["BTC/USD", "ETH/USD"]
BOS_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "XRP/USD", "ADA/USD", "DOGE/USD", "BCH/USD"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ohlcv_cache")


@dataclass
class SymbolCache:
    index: pd.Index
    close: np.ndarray
    returns: np.ndarray
    ema200: np.ndarray
    expert_pos: np.ndarray  # (n, m)
    reward_csum: np.ndarray  # (n, m)
    bos_long_ok: np.ndarray
    bos_short_ok: np.ndarray
    expert_names: List[str]


def load_or_fetch_all() -> Dict[str, pd.DataFrame]:
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(END_UTC.timestamp() * 1000)
    ex = ccxt.coinbase({"enableRateLimit": True})
    all_syms = sorted(set(SYMBOLS + BOS_SYMBOLS))
    os.makedirs(CACHE_DIR, exist_ok=True)

    def cpath(sym: str) -> str:
        tag = sym.replace("/", "_")
        return os.path.join(CACHE_DIR, f"{tag}_{TIMEFRAME}_{START_UTC.date()}_{END_UTC.date()}.csv")

    out: Dict[str, pd.DataFrame] = {}
    for sym in all_syms:
        p = cpath(sym)
        if os.path.exists(p):
            d = pd.read_csv(p, index_col=0, parse_dates=True)
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            out[sym] = d
        else:
            d = base.clean_df(base.fetch_ohlcv(ex, sym, TIMEFRAME, start_ms, end_ms))
            d.to_csv(p)
            out[sym] = d
    return out


def build_cache(df: pd.DataFrame, bos_data: Dict[str, pd.DataFrame]) -> SymbolCache:
    idx = df.index
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    rets = np.zeros(n, dtype=float)
    rets[1:] = close[1:] / close[:-1] - 1.0
    ema200 = pd.Series(close, index=idx).ewm(span=200, adjust=False).mean().to_numpy(dtype=float)

    cols: List[pd.Series] = []
    names: List[str] = []
    for name, fn in wf.STRATEGIES.items():
        for variant in ["fast", "base", "slow"]:
            sig = fn(df, variant)
            pos = aoe.build_position_state(df, sig)
            col_name = f"{name}:{variant}"
            cols.append(pos.rename(col_name))
            names.append(col_name)
    ep = pd.concat(cols, axis=1).fillna(0.0)
    M = ep.to_numpy(dtype=float)
    dpos = np.abs(np.diff(M, axis=0, prepend=np.zeros((1, M.shape[1]))))
    rew = M * np.expand_dims(rets, 1) - dpos * COST_PER_SIDE
    csum = np.cumsum(rew, axis=0)

    long_ok, short_ok, _ = wf.bos_filter(idx, bos_data)

    return SymbolCache(
        index=idx,
        close=close,
        returns=rets,
        ema200=ema200,
        expert_pos=M,
        reward_csum=csum,
        bos_long_ok=long_ok.to_numpy(dtype=bool),
        bos_short_ok=short_ok.to_numpy(dtype=bool),
        expert_names=names,
    )


def simulate(
    c: SymbolCache,
    lookback: int,
    rebalance_bars: int,
    min_score: float,
    use_bos: bool,
    long_only: bool,
    trend_filter: bool,
) -> dict:
    n, m = c.expert_pos.shape
    eq = 1.0
    pos = 0.0
    entry_eq = np.nan
    trades: List[float] = []
    curve = np.ones(n, dtype=float)
    chosen = -1

    for i in range(1, n):
        eq *= 1.0 + pos * c.returns[i]

        # Rebalance expert selection periodically.
        if (i % rebalance_bars == 0) or chosen < 0:
            if i > lookback:
                score_vec = c.reward_csum[i] - c.reward_csum[i - lookback]
            else:
                score_vec = c.reward_csum[i]
            best_idx = int(np.argmax(score_vec))
            if score_vec[best_idx] >= min_score:
                chosen = best_idx
            else:
                chosen = -1

        desired = float(c.expert_pos[i, chosen]) if chosen >= 0 else 0.0

        if long_only and desired < 0:
            desired = 0.0

        if trend_filter:
            if c.close[i] < c.ema200[i] and desired > 0:
                desired = 0.0
            if c.close[i] > c.ema200[i] and desired < 0:
                desired = 0.0

        if use_bos:
            if desired > 0 and not c.bos_long_ok[i]:
                desired = 0.0
            if desired < 0 and not c.bos_short_ok[i]:
                desired = 0.0

        delta = abs(desired - pos)
        if delta > 1e-12:
            eq *= 1.0 - COST_PER_SIDE * delta
            if abs(pos) < 1e-12 and abs(desired) > 1e-12:
                entry_eq = eq
            elif abs(pos) > 1e-12 and abs(desired) < 1e-12 and not np.isnan(entry_eq):
                trades.append(eq / entry_eq - 1.0)
                entry_eq = np.nan
            elif pos * desired < 0 and not np.isnan(entry_eq):
                trades.append(eq / entry_eq - 1.0)
                entry_eq = eq

        pos = desired
        curve[i] = eq

    if abs(pos) > 1e-12 and not np.isnan(entry_eq):
        trades.append(eq / entry_eq - 1.0)

    curve_s = pd.Series(curve, index=c.index)
    ret_pct = float((curve_s.iloc[-1] - 1.0) * 100.0)
    dd_pct = float(abs(((curve_s / curve_s.cummax()) - 1.0).min() * 100.0))
    rs = curve_s.pct_change().fillna(0.0)
    std = float(rs.std())
    bars_per_year = base.BARS_PER_YEAR.get(TIMEFRAME, 365 * 24 * 4)
    sharpe = float((rs.mean() / std) * np.sqrt(bars_per_year)) if std > 1e-12 else 0.0
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    win_rate = float((len(wins) / len(trades)) * 100.0) if trades else 0.0
    gp = float(sum(wins))
    gl = float(abs(sum(losses)))
    pf = float(gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    return {
        "ret_pct": ret_pct,
        "dd_pct": dd_pct,
        "sharpe": sharpe,
        "trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": pf,
    }


def main() -> None:
    print("Loading market data...")
    data = load_or_fetch_all()
    bos_data = {s: data[s] for s in BOS_SYMBOLS}

    print("Preparing expert caches...")
    caches: Dict[str, SymbolCache] = {}
    for sym in SYMBOLS:
        caches[sym] = build_cache(data[sym], bos_data)
        print(f"- {sym}: bars={len(caches[sym].index)} experts={caches[sym].expert_pos.shape[1]}")

    grid = {
        "lookback": [288, 576, 960, 1440, 2016],
        "rebalance_bars": [24, 48, 96, 192],
        "min_score": [0.0, 0.002, 0.006, 0.012],
        "use_bos": [True, False],
        "long_only": [True, False],
        "trend_filter": [True, False],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Sweeping {len(combos)} combos...")

    rows = []
    for combo in combos:
        p = dict(zip(keys, combo))
        per = []
        for sym in SYMBOLS:
            m = simulate(caches[sym], **p)
            per.append((sym, m))

        avg_ret = float(np.mean([x[1]["ret_pct"] for x in per]))
        avg_dd = float(np.mean([x[1]["dd_pct"] for x in per]))
        avg_sh = float(np.mean([x[1]["sharpe"] for x in per]))
        avg_pf = float(np.mean([x[1]["profit_factor"] for x in per]))
        avg_wr = float(np.mean([x[1]["win_rate"] for x in per]))
        avg_tr = float(np.mean([x[1]["trades"] for x in per]))

        score = avg_ret + avg_sh * 8.0 + avg_pf * 2.0 - avg_dd * 0.6
        row = {**p, "avg_ret_pct": avg_ret, "avg_dd_pct": avg_dd, "avg_sharpe": avg_sh, "avg_profit_factor": avg_pf, "avg_win_rate": avg_wr, "avg_trades": avg_tr, "score": score}
        for sym, m in per:
            tag = sym.replace("/", "_")
            row[f"{tag}_ret_pct"] = m["ret_pct"]
            row[f"{tag}_dd_pct"] = m["dd_pct"]
            row[f"{tag}_sharpe"] = m["sharpe"]
            row[f"{tag}_trades"] = m["trades"]
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adaptive_rebalance_sweep.csv")
    out.to_csv(out_path, index=False)

    print("\nTop 20:")
    print(out.head(20).to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
