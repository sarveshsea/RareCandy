#!/usr/bin/env python3
"""
Adaptive online ensemble for Pine-derived strategy experts.

What it does:
1) Builds 33 experts (11 strategies x fast/base/slow variants).
2) Converts each expert's entry/exit signals to bar-wise position states.
3) Runs an online scoring model (exponential reward update + trade penalty).
4) Selects top-K positive-score experts each bar and votes net direction.
5) Applies BOS regime filter + volatility targeting + drawdown kill-switch.
6) Exports performance and expert diagnostics.

This is an adaptive research framework. It cannot guarantee future profitability.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import ccxt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import backtest_pine_batch as base
import backtest_pine_walkforward as wf


TIMEFRAME = "15m"
START_UTC = dt.datetime(2025, 5, 1, tzinfo=dt.timezone.utc)
END_UTC = dt.datetime(2026, 2, 3, tzinfo=dt.timezone.utc)
COST_PER_SIDE = 0.0006
PRIMARY_SYMBOLS = ["BTC/USD", "ETH/USD"]
BOS_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "XRP/USD", "ADA/USD", "DOGE/USD", "BCH/USD"]
VARIANTS = ["fast", "base", "slow"]


@dataclass
class EnsembleMetrics:
    symbol: str
    trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    profit_factor: float


def build_position_state(df: pd.DataFrame, sig: Dict[str, pd.Series]) -> pd.Series:
    """Convert entry/exit booleans to persistent position state {-1,0,1}."""
    idx = df.index
    n = len(df)

    le = sig["long"].reindex(idx).fillna(False).to_numpy(dtype=bool)
    se = sig["short"].reindex(idx).fillna(False).to_numpy(dtype=bool)
    lx = sig.get("long_exit", pd.Series(False, index=idx)).reindex(idx).fillna(False).to_numpy(dtype=bool)
    sx = sig.get("short_exit", pd.Series(False, index=idx)).reindex(idx).fillna(False).to_numpy(dtype=bool)

    pos = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        p = int(pos[i - 1])
        if p == 1:
            if lx[i] or se[i]:
                p = 0
                if se[i] and not lx[i]:
                    p = -1
        elif p == -1:
            if sx[i] or le[i]:
                p = 0
                if le[i] and not sx[i]:
                    p = 1
        else:
            if le[i] and not se[i]:
                p = 1
            elif se[i] and not le[i]:
                p = -1
        pos[i] = p

    return pd.Series(pos, index=idx)


def compute_metrics(equity_curve: pd.Series, trade_returns: List[float], timeframe: str = TIMEFRAME) -> Tuple[float, float, float, float, float]:
    total_return_pct = (float(equity_curve.iloc[-1]) - 1.0) * 100.0

    running_max = equity_curve.cummax()
    dd = (equity_curve / running_max - 1.0).min()
    max_dd_pct = abs(float(dd) * 100.0)

    ret_s = equity_curve.pct_change().fillna(0.0)
    bars_per_year = base.BARS_PER_YEAR.get(timeframe, 365 * 24 * 4)
    std = float(ret_s.std())
    sharpe = float((ret_s.mean() / std) * math.sqrt(bars_per_year)) if std > 1e-12 else 0.0

    wins = [x for x in trade_returns if x > 0]
    losses = [x for x in trade_returns if x < 0]
    win_rate = (len(wins) / len(trade_returns) * 100.0) if trade_returns else 0.0

    gp = sum(wins)
    gl = abs(sum(losses))
    profit_factor = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)

    return total_return_pct, max_dd_pct, sharpe, win_rate, profit_factor


def run_adaptive_ensemble(
    df: pd.DataFrame,
    expert_positions: pd.DataFrame,
    bos_long_ok: pd.Series,
    bos_short_ok: pd.Series,
    reward_scale: float = 220.0,
    score_decay: float = 0.985,
    top_k: int = 6,
    min_confidence: float = 0.16,
    trade_penalty: float = COST_PER_SIDE,
    vol_window: int = 96,
    target_bar_vol: float = 0.0022,
    max_leverage: float = 1.0,
    dd_kill_threshold: float = 0.14,
    dd_cooldown_bars: int = 96,
) -> Tuple[EnsembleMetrics, pd.DataFrame, pd.DataFrame]:
    idx = df.index
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    experts = list(expert_positions.columns)
    m = len(experts)
    pos_matrix = expert_positions.to_numpy(dtype=float).T  # shape: (m, n)

    scores = np.zeros(m, dtype=float)
    reward_sums = np.zeros(m, dtype=float)

    equity = 1.0
    peak = 1.0
    cooldown = 0
    position = 0.0
    entry_equity = np.nan
    trade_returns: List[float] = []

    curve = np.full(n, np.nan)
    curve[0] = equity
    ensemble_pos = np.zeros(n, dtype=float)
    ensemble_vote = np.zeros(n, dtype=float)
    ensemble_lev = np.zeros(n, dtype=float)
    selected_count = np.zeros(n, dtype=int)

    for i in range(1, n):
        # Mark-to-market for previous position over current bar.
        if close[i - 1] > 0:
            r_bar = close[i] / close[i - 1] - 1.0
        else:
            r_bar = 0.0

        if abs(position) > 0:
            equity *= (1.0 + position * r_bar)

        # Online score update using realized reward at this bar.
        for j in range(m):
            p_prev = pos_matrix[j, i - 1]
            p_curr = pos_matrix[j, i]
            switch_penalty = abs(p_curr - p_prev) * trade_penalty
            reward = p_prev * r_bar - switch_penalty
            reward_sums[j] += reward
            scores[j] = scores[j] * score_decay + reward * reward_scale
            scores[j] = float(np.clip(scores[j], -80.0, 80.0))

        # Build desired position.
        desired = 0.0
        vote = 0.0
        lev = 1.0
        sel_cnt = 0

        # Drawdown kill-switch.
        peak = max(peak, equity)
        dd_now = 1.0 - (equity / peak if peak > 0 else 1.0)
        if dd_now > dd_kill_threshold and cooldown == 0:
            cooldown = dd_cooldown_bars

        if cooldown > 0:
            cooldown -= 1
        else:
            pos_idx = np.where(scores > 0.0)[0]
            if pos_idx.size > 0:
                ranked = pos_idx[np.argsort(scores[pos_idx])[::-1]]
                chosen = ranked[:top_k]
                weights = scores[chosen]
                signals = pos_matrix[chosen, i]
                denom = np.sum(np.abs(weights))
                vote = float(np.dot(weights, signals) / denom) if denom > 1e-12 else 0.0
                sel_cnt = int(len(chosen))

                if vote > min_confidence:
                    desired = 1.0
                elif vote < -min_confidence:
                    desired = -1.0

        # BOS overlay regime filter.
        if desired > 0 and not bool(bos_long_ok.iloc[i]):
            desired = 0.0
        if desired < 0 and not bool(bos_short_ok.iloc[i]):
            desired = 0.0

        # Volatility targeting.
        if i >= vol_window:
            logret = np.diff(np.log(close[i - vol_window : i + 1]))
            rv = float(np.nanstd(logret))
            if rv > 1e-10:
                lev = float(min(max_leverage, target_bar_vol / rv))
            else:
                lev = 1.0
        else:
            lev = 1.0

        desired *= lev

        # Apply transaction cost on position change.
        delta = abs(desired - position)
        if delta > 1e-12:
            equity *= (1.0 - trade_penalty * delta)

            # trade accounting by round-trip proxy
            if abs(position) < 1e-12 and abs(desired) > 1e-12:
                entry_equity = equity
            elif abs(position) > 1e-12 and abs(desired) < 1e-12 and not np.isnan(entry_equity):
                trade_returns.append(equity / entry_equity - 1.0)
                entry_equity = np.nan
            elif (position > 0 > desired) or (position < 0 < desired):
                if not np.isnan(entry_equity):
                    trade_returns.append(equity / entry_equity - 1.0)
                entry_equity = equity

        position = desired
        curve[i] = equity
        ensemble_pos[i] = position
        ensemble_vote[i] = vote
        ensemble_lev[i] = lev
        selected_count[i] = sel_cnt

    # force close performance bookkeeping
    if abs(position) > 1e-12 and not np.isnan(entry_equity):
        trade_returns.append(equity / entry_equity - 1.0)

    curve_s = pd.Series(curve, index=idx).ffill().bfill()

    total_return_pct, max_dd_pct, sharpe, win_rate, profit_factor = compute_metrics(curve_s, trade_returns, timeframe=TIMEFRAME)

    metrics = EnsembleMetrics(
        symbol="",
        trades=len(trade_returns),
        win_rate=win_rate,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_dd_pct,
        sharpe=sharpe,
        profit_factor=profit_factor,
    )

    debug = pd.DataFrame(
        {
            "equity": curve_s,
            "ensemble_pos": ensemble_pos,
            "vote": ensemble_vote,
            "leverage": ensemble_lev,
            "selected_experts": selected_count,
        },
        index=idx,
    )

    expert_diag = pd.DataFrame(
        {
            "expert": experts,
            "final_score": scores,
            "reward_sum": reward_sums,
        }
    ).sort_values("final_score", ascending=False)

    return metrics, debug, expert_diag


def run_single_expert_baselines(
    df: pd.DataFrame,
    bos_long_ok: pd.Series,
    bos_short_ok: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, fn in wf.STRATEGIES.items():
        for variant in VARIANTS:
            sig = fn(df, variant)
            long_e = sig["long"].reindex(df.index).fillna(False) & bos_long_ok
            short_e = sig["short"].reindex(df.index).fillna(False) & bos_short_ok

            bt = base.run_backtest(
                df,
                long_e,
                short_e,
                long_exit=sig.get("long_exit"),
                short_exit=sig.get("short_exit"),
                cost_per_side=COST_PER_SIDE,
                timeframe=TIMEFRAME,
            )

            rows.append(
                {
                    "strategy": name,
                    "variant": variant,
                    "trades": bt.trades,
                    "win_rate": bt.win_rate,
                    "return_pct": bt.total_return_pct,
                    "max_dd_pct": bt.max_drawdown_pct,
                    "sharpe": bt.sharpe,
                    "profit_factor": bt.profit_factor,
                }
            )

    out = pd.DataFrame(rows)
    out["score"] = out["return_pct"] + (out["sharpe"] * 10.0) + (out["profit_factor"] * 2.0) - (out["max_dd_pct"] * 0.6)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def write_adaptive_pine(path: str) -> None:
    pine = """// Generated by adaptive_online_ensemble.py
// Adaptive Ensemble Strategy (research template)
// Uses online score updates for multiple sub-signals and weighted voting.

//@version=6
strategy("Adaptive Ensemble (Research)", overlay=true, initial_capital=10000, commission_type=strategy.commission.percent, commission_value=0.06, pyramiding=0, calc_on_bar_close=true)

// ===== Inputs =====
decay        = input.float(0.985, "Score Decay", step=0.001)
rewardScale  = input.float(220.0, "Reward Scale", step=1.0)
entryThresh  = input.float(0.16, "Vote Threshold", step=0.01)
killDDPct    = input.float(14.0, "Kill DD %", step=0.1)
stopATRMult  = input.float(2.2, "ATR Stop Mult", step=0.1)

// ===== Module 1: Jurik-like Breakout =====
jLen = input.int(9, "Jurik Length")
pivL = input.int(4, "Pivot Length")
phase = 0.1
beta = 0.45 * (jLen - 1) / (0.45 * (jLen - 1) + 2)
alpha = math.pow(beta, phase)
var float jma = na
jma := na(jma[1]) ? close : (1 - alpha) * close + alpha * jma[1]
trendJ = jma >= jma[3]
ph = ta.pivothigh(high, pivL, pivL)
pl = ta.pivotlow(low, pivL, pivL)
atr200 = ta.atr(200)
var float H = na
var int Hi = 0
var int brkUp = 0
var float L = na
var int Li = 0
var int brkDn = 0
var float upperLvl = na
var float lowerLvl = na
var bool upperOn = false
var bool lowerOn = false
if trendJ != trendJ[1]
    upperOn := false
    lowerOn := false
if trendJ and not upperOn and not na(ph) and Hi > brkUp and not na(H) and math.abs(ph - H) < atr200
    upperOn := true
    upperLvl := ph
if trendJ and not na(ph)
    H := ph
    Hi := bar_index - pivL
m1Long = upperOn and close > upperLvl
if m1Long
    upperOn := false
    brkUp := bar_index
if not trendJ and not lowerOn and not na(pl) and Li > brkDn and not na(L) and math.abs(pl - L) < atr200
    lowerOn := true
    lowerLvl := pl
if not trendJ and not na(pl)
    L := pl
    Li := bar_index - pivL
m1Short = lowerOn and close < lowerLvl
if m1Short
    lowerOn := false
    brkDn := bar_index
var int m1Pos = 0
m1Pos := m1Long ? 1 : m1Short ? -1 : nz(m1Pos[1], 0)

// ===== Module 2: Extreme HMA ATR =====
lenH = input.int(33, "Extreme HMA Length")
atrMulH = input.float(0.7, "Extreme HMA ATR Mult", step=0.05)
sqrtlen = math.round(math.sqrt(lenH))
halflen = math.round(lenH / 2)
hma = ta.ema(close, sqrtlen)
h = ta.highest(hma, lenH)
l = ta.lowest(hma, lenH)
hh = ta.lowest(h, halflen)
ll = ta.highest(l, halflen)
mid = (hh + ll) / 2
atrH = ta.atr(30) * atrMulH
upH = mid + atrH
dnH = mid - atrH
Lh = close > upH
Sh = close < dnH
var int m2Pos = 0
if Lh
    m2Pos := 1
else if Sh
    m2Pos := -1
else
    m2Pos := nz(m2Pos[1], 0)

// ===== Module 3: GK Ribbon =====
lenGK = input.int(70, "GK Length")
multGK = input.float(1.2, "GK ATR Mult", step=0.1)
lag = math.max(int(math.floor((lenGK - 1) / 2)), 0)
zl = ta.ema(lag > 0 ? close + (close - close[lag]) : close, lenGK)
atrGK = ta.atr(14)
upGK = zl + atrGK * multGK
dnGK = zl - atrGK * multGK
bullGK = (close > upGK and close[1] > upGK[1] and close[2] > upGK[2]) and zl > zl[1]
bearGK = (close < dnGK and close[1] < dnGK[1] and close[2] < dnGK[2]) and zl < zl[1]
var int trGK = 0
trGK := bullGK ? 1 : bearGK ? -1 : nz(trGK[1], 0)
var int m3Pos = 0
if trGK != trGK[1] and trGK != 0
    m3Pos := trGK
else
    m3Pos := nz(m3Pos[1], 0)

// ===== Online scoring =====
ret = close[1] != 0 ? (close / close[1] - 1.0) : 0.0
var float s1 = 0.0
var float s2 = 0.0
var float s3 = 0.0
s1 := nz(s1[1]) * decay + (nz(m1Pos[1]) * ret) * rewardScale
s2 := nz(s2[1]) * decay + (nz(m2Pos[1]) * ret) * rewardScale
s3 := nz(s3[1]) * decay + (nz(m3Pos[1]) * ret) * rewardScale

w1 = math.max(s1, 0)
w2 = math.max(s2, 0)
w3 = math.max(s3, 0)
wd = w1 + w2 + w3
vote = wd > 0 ? (w1 * m1Pos + w2 * m2Pos + w3 * m3Pos) / wd : 0

wantLong  = vote > entryThresh
wantShort = vote < -entryThresh

// ===== Risk control =====
atrRisk = ta.atr(14)
longStop = close - atrRisk * stopATRMult
shortStop = close + atrRisk * stopATRMult

var float eqPeak = na
eqPeak := na(eqPeak) ? strategy.equity : math.max(eqPeak, strategy.equity)
ddPct = eqPeak > 0 ? (1 - strategy.equity / eqPeak) * 100 : 0
kill = ddPct > killDDPct

if kill
    strategy.close_all("DD kill")
else
    if wantLong and strategy.position_size <= 0
        strategy.entry("L", strategy.long)
    if wantShort and strategy.position_size >= 0
        strategy.entry("S", strategy.short)

if strategy.position_size > 0
    strategy.exit("Lx", "L", stop=longStop)
if strategy.position_size < 0
    strategy.exit("Sx", "S", stop=shortStop)

plot(vote, "Ensemble Vote", color=color.new(color.aqua, 0), linewidth=2, display=display.pane)
plot(0, "Zero", color=color.new(color.gray, 70), display=display.pane)
plot(zl, "GK Base", color=trGK == 1 ? color.lime : trGK == -1 ? color.red : color.gray)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(pine)


def main() -> None:
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(END_UTC.timestamp() * 1000)

    ex = ccxt.coinbase({"enableRateLimit": True})
    all_syms = sorted(set(PRIMARY_SYMBOLS + BOS_SYMBOLS))

    data: Dict[str, pd.DataFrame] = {}
    for sym in all_syms:
        data[sym] = base.clean_df(base.fetch_ohlcv(ex, sym, TIMEFRAME, start_ms, end_ms))

    bos_data = {s: data[s] for s in BOS_SYMBOLS}

    all_metrics = []

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    for sym in PRIMARY_SYMBOLS:
        df = data[sym].copy()

        long_ok, short_ok, score = wf.bos_filter(df.index, bos_data)

        # Build experts matrix.
        expert_cols = []
        pos_cols = []
        for name, fn in wf.STRATEGIES.items():
            for variant in VARIANTS:
                sig = fn(df, variant)
                pos = build_position_state(df, sig)
                col = f"{name}:{variant}"
                expert_cols.append(col)
                pos_cols.append(pos.rename(col))

        expert_positions = pd.concat(pos_cols, axis=1).fillna(0.0)

        # Single-expert baselines with BOS filter.
        baseline = run_single_expert_baselines(df, long_ok, short_ok)
        baseline_path = os.path.join(out_dir, f"adaptive_baseline_{sym.replace('/', '_')}.csv")
        baseline.to_csv(baseline_path, index=False)

        # Adaptive ensemble run.
        metrics, debug, diag = run_adaptive_ensemble(df, expert_positions, long_ok, short_ok)
        metrics.symbol = sym

        all_metrics.append(
            {
                "symbol": sym,
                "trades": metrics.trades,
                "win_rate": metrics.win_rate,
                "total_return_pct": metrics.total_return_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "sharpe": metrics.sharpe,
                "profit_factor": metrics.profit_factor,
            }
        )

        debug_out = os.path.join(out_dir, f"adaptive_debug_{sym.replace('/', '_')}.csv")
        diag_out = os.path.join(out_dir, f"adaptive_expert_diag_{sym.replace('/', '_')}.csv")
        debug.to_csv(debug_out)
        diag.to_csv(diag_out, index=False)

    met_df = pd.DataFrame(all_metrics)
    summary = {
        "timeframe": TIMEFRAME,
        "start_utc": START_UTC.isoformat(),
        "end_utc": END_UTC.isoformat(),
        "symbols": PRIMARY_SYMBOLS,
        "cost_per_side": COST_PER_SIDE,
        "avg_return_pct": float(met_df["total_return_pct"].mean()),
        "avg_max_dd_pct": float(met_df["max_drawdown_pct"].mean()),
        "avg_sharpe": float(met_df["sharpe"].mean()),
        "avg_win_rate": float(met_df["win_rate"].mean()),
        "avg_trades": float(met_df["trades"].mean()),
        "per_symbol": met_df.to_dict(orient="records"),
        "disclaimer": "Adaptive logic improves responsiveness but does not guarantee future profitability.",
    }

    met_path = os.path.join(out_dir, "adaptive_ensemble_metrics.csv")
    sum_path = os.path.join(out_dir, "adaptive_ensemble_summary.json")
    met_df.to_csv(met_path, index=False)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pine_path = os.path.join(SCRIPT_DIR, "generated_adaptive_ensemble.pine")
    write_adaptive_pine(pine_path)

    print("=== Adaptive Ensemble Metrics ===")
    print(met_df.to_string(index=False))
    print("\nWrote:")
    print(f"- {met_path}")
    print(f"- {sum_path}")
    print(f"- {pine_path}")


if __name__ == "__main__":
    main()
