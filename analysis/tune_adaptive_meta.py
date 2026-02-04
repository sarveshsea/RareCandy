#!/usr/bin/env python3
"""
Fast parameter sweep for an online adaptive expert-switching meta strategy.

Goal:
- Find settings that materially improve over the first adaptive ensemble pass.
- Keep logic realistic: bar-close execution and per-side trading cost.
"""

from __future__ import annotations

import datetime as dt
import itertools
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    expert_pos: np.ndarray  # shape: (n_bars, n_experts)
    reward_csum: np.ndarray  # shape: (n_bars, n_experts)
    bos_long_ok: np.ndarray
    bos_short_ok: np.ndarray


def compute_symbol_cache(df: pd.DataFrame, bos_data: Dict[str, pd.DataFrame]) -> SymbolCache:
    idx = df.index
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    cols: List[pd.Series] = []
    for name, fn in wf.STRATEGIES.items():
        for variant in ["fast", "base", "slow"]:
            sig = fn(df, variant)
            pos = aoe.build_position_state(df, sig)
            cols.append(pos.rename(f"{name}:{variant}"))

    ep = pd.concat(cols, axis=1).fillna(0.0)
    expert_pos = ep.to_numpy(dtype=float)

    rets = np.zeros(n, dtype=float)
    rets[1:] = close[1:] / close[:-1] - 1.0

    # Cost-aware "expert reward" stream.
    dpos = np.abs(np.diff(expert_pos, axis=0, prepend=np.zeros((1, expert_pos.shape[1]))))
    rew = expert_pos * np.expand_dims(rets, 1) - dpos * COST_PER_SIDE
    reward_csum = np.cumsum(rew, axis=0)

    bos_long_ok, bos_short_ok, _ = wf.bos_filter(idx, bos_data)
    ema200 = pd.Series(close, index=idx).ewm(span=200, adjust=False).mean().to_numpy(dtype=float)

    return SymbolCache(
        index=idx,
        close=close,
        returns=rets,
        ema200=ema200,
        expert_pos=expert_pos,
        reward_csum=reward_csum,
        bos_long_ok=bos_long_ok.to_numpy(dtype=bool),
        bos_short_ok=bos_short_ok.to_numpy(dtype=bool),
    )


def simulate(
    c: SymbolCache,
    lookback: int,
    top_k: int,
    min_score: float,
    vote_threshold: float,
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

    for i in range(1, n):
        eq *= 1.0 + pos * c.returns[i]

        if i > lookback:
            win = c.reward_csum[i] - c.reward_csum[i - lookback]
        else:
            win = c.reward_csum[i]

        order = np.argsort(win)[::-1]
        chosen = order[:top_k]
        chosen_scores = win[chosen]
        weights = np.where(chosen_scores > min_score, chosen_scores, 0.0)

        if np.sum(np.abs(weights)) > 1e-12:
            vote = float(np.dot(weights, c.expert_pos[i, chosen]) / np.sum(np.abs(weights)))
        else:
            vote = float(np.mean(c.expert_pos[i, chosen]))

        desired = 0.0
        if vote > vote_threshold:
            desired = 1.0
        elif vote < -vote_threshold and not long_only:
            desired = -1.0

        if trend_filter:
            # Trend filter to reduce chop:
            # - no longs below EMA200
            # - no shorts above EMA200
            if c.close[i] < c.ema200[i]:
                desired = min(desired, 0.0)
            if c.close[i] > c.ema200[i]:
                desired = max(desired, 0.0) if not long_only else desired

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

    bars_per_year = base.BARS_PER_YEAR.get(TIMEFRAME, 365 * 24 * 4)
    rs = curve_s.pct_change().fillna(0.0)
    std = float(rs.std())
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
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(END_UTC.timestamp() * 1000)

    ex = ccxt.coinbase({"enableRateLimit": True})
    all_syms = sorted(set(SYMBOLS + BOS_SYMBOLS))

    os.makedirs(CACHE_DIR, exist_ok=True)

    def cache_path(sym: str) -> str:
        tag = sym.replace("/", "_")
        return os.path.join(CACHE_DIR, f"{tag}_{TIMEFRAME}_{START_UTC.date()}_{END_UTC.date()}.csv")

    def load_or_fetch(sym: str) -> pd.DataFrame:
        p = cache_path(sym)
        if os.path.exists(p):
            cdf = pd.read_csv(p, index_col=0, parse_dates=True)
            if cdf.index.tz is None:
                cdf.index = cdf.index.tz_localize("UTC")
            return cdf

        last_err = None
        for attempt in range(1, 6):
            try:
                df = base.clean_df(base.fetch_ohlcv(ex, sym, TIMEFRAME, start_ms, end_ms))
                df.to_csv(p)
                return df
            except Exception as e:
                last_err = e
                wait_s = min(12, 2 * attempt)
                print(f"fetch retry {attempt}/5 for {sym}: {type(e).__name__} ({wait_s}s)")
                time.sleep(wait_s)
        raise RuntimeError(f"Failed to fetch {sym} after retries: {last_err}")

    print("Fetching OHLCV...")
    data: Dict[str, pd.DataFrame] = {}
    for sym in all_syms:
        data[sym] = load_or_fetch(sym)
    bos_data = {s: data[s] for s in BOS_SYMBOLS}

    print("Building expert caches...")
    caches: Dict[str, SymbolCache] = {}
    for sym in SYMBOLS:
        caches[sym] = compute_symbol_cache(data[sym], bos_data)
        print(f"- {sym}: bars={len(caches[sym].index)} experts={caches[sym].expert_pos.shape[1]}")

    grid = {
        "lookback": [288, 576, 960, 1440],
        "top_k": [1, 2, 3, 5],
        "min_score": [0.0, 0.0015, 0.004],
        "vote_threshold": [0.08, 0.12, 0.18, 0.24],
        "use_bos": [True, False],
        "long_only": [True, False],
        "trend_filter": [True, False],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Sweeping {len(combos)} parameter combos...")

    rows = []
    for combo in combos:
        p = dict(zip(keys, combo))
        sym_rows = []
        for sym in SYMBOLS:
            m = simulate(caches[sym], **p)
            sym_rows.append((sym, m))

        avg_ret = float(np.mean([x[1]["ret_pct"] for x in sym_rows]))
        avg_dd = float(np.mean([x[1]["dd_pct"] for x in sym_rows]))
        avg_sh = float(np.mean([x[1]["sharpe"] for x in sym_rows]))
        avg_pf = float(np.mean([x[1]["profit_factor"] for x in sym_rows]))
        avg_wr = float(np.mean([x[1]["win_rate"] for x in sym_rows]))
        avg_tr = float(np.mean([x[1]["trades"] for x in sym_rows]))

        score = avg_ret + avg_sh * 8.0 + avg_pf * 2.0 - avg_dd * 0.6

        row = {
            **p,
            "avg_ret_pct": avg_ret,
            "avg_dd_pct": avg_dd,
            "avg_sharpe": avg_sh,
            "avg_profit_factor": avg_pf,
            "avg_win_rate": avg_wr,
            "avg_trades": avg_tr,
            "score": score,
        }
        for sym, m in sym_rows:
            tag = sym.replace("/", "_")
            row[f"{tag}_ret_pct"] = m["ret_pct"]
            row[f"{tag}_dd_pct"] = m["dd_pct"]
            row[f"{tag}_sharpe"] = m["sharpe"]
            row[f"{tag}_trades"] = m["trades"]
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adaptive_meta_sweep.csv")
    out.to_csv(out_path, index=False)

    print("\nTop 20:")
    print(out.head(20).to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
