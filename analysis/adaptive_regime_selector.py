#!/usr/bin/env python3
"""
Adaptive regime selector (low-churn online bandit).

Design goals:
- Keep a small, high-signal expert set.
- Adapt weights online from realized returns.
- Use macro regime + trend gate + drawdown kill-switch.
- Prefer low turnover by selecting top-1 expert each bar within regime pool.

This is a research tool. It is not a guarantee of future profitability.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
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

# Tuned config (from randomized search over this dataset).
CFG = {
    "decay": 0.99,
    "reward_scale": 120.0,
    "vote_threshold": 0.28,
    "top_k": 1,
    "dd_kill_threshold": 0.08,
    "dd_cooldown_bars": 48,
    "target_bar_vol": 0.0025,
    "max_leverage": 1.2,
    "vol_window": 96,
    "risk_on_threshold": 3.0,
    "risk_off_threshold": 2.0,
    "allow_short": False,
    "trend_filter": True,
    "neutral_trend_mode": True,
}

# Ordered expert set (used by regime pools).
EXPERTS = [
    ("Jurik_Breakouts", "fast"),            # idx 0
    ("GK_Trend_Ribbon", "base"),            # idx 1
    ("BackQuant_VolSkew", "base"),          # idx 2
    ("DAFE_Bands_SmartComposite", "base"),  # idx 3
]

REGIME_POOLS = {
    "risk_on": [0, 1, 3],
    "risk_off": [1, 2, 3],
    "neutral_trend": [0, 1],
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ohlcv_cache")


@dataclass
class SymbolResult:
    symbol: str
    trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    profit_factor: float


def load_or_fetch_all() -> Dict[str, pd.DataFrame]:
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(END_UTC.timestamp() * 1000)

    os.makedirs(CACHE_DIR, exist_ok=True)
    ex = ccxt.coinbase({"enableRateLimit": True})
    all_syms = sorted(set(SYMBOLS + BOS_SYMBOLS))

    def cache_path(sym: str) -> str:
        tag = sym.replace("/", "_")
        return os.path.join(CACHE_DIR, f"{tag}_{TIMEFRAME}_{START_UTC.date()}_{END_UTC.date()}.csv")

    data: Dict[str, pd.DataFrame] = {}
    for sym in all_syms:
        p = cache_path(sym)
        if os.path.exists(p):
            d = pd.read_csv(p, index_col=0, parse_dates=True)
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            data[sym] = d
            continue

        last_err = None
        for attempt in range(1, 6):
            try:
                d = base.clean_df(base.fetch_ohlcv(ex, sym, TIMEFRAME, start_ms, end_ms))
                d.to_csv(p)
                data[sym] = d
                last_err = None
                break
            except Exception as e:
                last_err = e
                wait_s = min(12, 2 * attempt)
                print(f"fetch retry {attempt}/5 for {sym}: {type(e).__name__} ({wait_s}s)")
                time.sleep(wait_s)
        if last_err is not None:
            raise RuntimeError(f"Failed to fetch {sym} after retries: {last_err}")

    return data


def compute_metrics(curve: pd.Series, trades: List[float], timeframe: str = TIMEFRAME) -> Dict[str, float]:
    total_return_pct = float((curve.iloc[-1] - 1.0) * 100.0)
    max_dd_pct = float(abs(((curve / curve.cummax()) - 1.0).min()) * 100.0)

    rs = curve.pct_change().fillna(0.0)
    std = float(rs.std())
    bars_per_year = base.BARS_PER_YEAR.get(timeframe, 365 * 24 * 4)
    sharpe = float((rs.mean() / std) * np.sqrt(bars_per_year)) if std > 1e-12 else 0.0

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x < 0]
    win_rate = float((len(wins) / len(trades)) * 100.0) if trades else 0.0
    gp = float(sum(wins))
    gl = float(abs(sum(losses)))
    pf = float(gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "sharpe": sharpe,
        "profit_factor": pf,
    }


def run_selector_for_symbol(
    df: pd.DataFrame,
    bos_score: pd.Series,
    cfg: dict = CFG,
) -> tuple[SymbolResult, pd.DataFrame, pd.DataFrame]:
    idx = df.index
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    # Build small expert matrix.
    pos_cols: List[pd.Series] = []
    expert_names: List[str] = []
    for name, variant in EXPERTS:
        sig = wf.STRATEGIES[name](df, variant)
        pos = aoe.build_position_state(df, sig)
        col_name = f"{name}:{variant}"
        pos_cols.append(pos.rename(col_name))
        expert_names.append(col_name)

    M = pd.concat(pos_cols, axis=1).fillna(0.0).to_numpy(dtype=float).T  # (m, n)
    m = M.shape[0]

    ema200 = pd.Series(close, index=idx).ewm(span=200, adjust=False).mean().to_numpy(dtype=float)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    rv = pd.Series(logret, index=idx).rolling(cfg["vol_window"]).std().to_numpy(dtype=float)
    score_arr = bos_score.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    scores = np.zeros(m, dtype=float)
    equity = 1.0
    peak = 1.0
    cooldown = 0
    position = 0.0
    entry_eq = np.nan
    trade_returns: List[float] = []

    curve = np.full(n, np.nan)
    curve[0] = equity
    vote_s = np.zeros(n, dtype=float)
    lev_s = np.ones(n, dtype=float)
    pos_s = np.zeros(n, dtype=float)
    regime_s = np.array(["neutral"] * n, dtype=object)
    chosen_s = np.array(["none"] * n, dtype=object)

    for i in range(1, n):
        # Mark-to-market previous position.
        r = (close[i] / close[i - 1] - 1.0) if close[i - 1] > 0 else 0.0
        if abs(position) > 1e-12:
            equity *= 1.0 + position * r

        # Online score update.
        p_prev = M[:, i - 1]
        p_curr = M[:, i]
        reward = p_prev * r - np.abs(p_curr - p_prev) * COST_PER_SIDE
        scores = scores * cfg["decay"] + reward * cfg["reward_scale"]
        scores = np.clip(scores, -120.0, 120.0)

        peak = max(peak, equity)
        dd_now = 1.0 - (equity / peak if peak > 0 else 1.0)
        if dd_now > cfg["dd_kill_threshold"] and cooldown == 0:
            cooldown = cfg["dd_cooldown_bars"]

        desired = 0.0
        vote = 0.0
        chosen_name = "none"
        regime = "neutral"

        if cooldown > 0:
            cooldown -= 1
        else:
            sc = score_arr[i]
            if sc >= cfg["risk_on_threshold"]:
                candidate = np.array(REGIME_POOLS["risk_on"], dtype=int)
                regime = "risk_on"
            elif sc <= -cfg["risk_off_threshold"]:
                candidate = np.array(REGIME_POOLS["risk_off"], dtype=int)
                regime = "risk_off"
            else:
                if cfg["neutral_trend_mode"] and close[i] > ema200[i]:
                    candidate = np.array(REGIME_POOLS["neutral_trend"], dtype=int)
                    regime = "neutral_trend"
                else:
                    candidate = np.array([], dtype=int)
                    regime = "neutral"

            if candidate.size > 0:
                # top-k by score within regime
                local_scores = scores[candidate]
                ord_idx = np.argsort(local_scores)[::-1][: min(cfg["top_k"], candidate.size)]
                chosen = candidate[ord_idx]

                w = np.where(scores[chosen] > 0.0, scores[chosen], 0.0)
                sig = M[chosen, i]
                denom = np.sum(np.abs(w))
                vote = float(np.dot(w, sig) / denom) if denom > 1e-12 else float(np.mean(sig))

                if len(chosen) > 0:
                    chosen_name = expert_names[int(chosen[0])]

                if vote > cfg["vote_threshold"]:
                    desired = 1.0
                elif vote < -cfg["vote_threshold"] and cfg["allow_short"]:
                    desired = -1.0

        if cfg["trend_filter"]:
            if close[i] < ema200[i] and desired > 0:
                desired = 0.0
            if close[i] > ema200[i] and desired < 0:
                desired = 0.0

        # Volatility target.
        if not np.isnan(rv[i]) and rv[i] > 1e-10:
            lev = float(min(cfg["max_leverage"], cfg["target_bar_vol"] / rv[i]))
        else:
            lev = 1.0
        desired *= lev

        # Trading cost on position change.
        delta = abs(desired - position)
        if delta > 1e-12:
            equity *= 1.0 - COST_PER_SIDE * delta
            if abs(position) < 1e-12 and abs(desired) > 1e-12:
                entry_eq = equity
            elif abs(position) > 1e-12 and abs(desired) < 1e-12 and not np.isnan(entry_eq):
                trade_returns.append(equity / entry_eq - 1.0)
                entry_eq = np.nan
            elif position * desired < 0 and not np.isnan(entry_eq):
                trade_returns.append(equity / entry_eq - 1.0)
                entry_eq = equity

        position = desired
        curve[i] = equity
        vote_s[i] = vote
        lev_s[i] = lev
        pos_s[i] = position
        regime_s[i] = regime
        chosen_s[i] = chosen_name

    if abs(position) > 1e-12 and not np.isnan(entry_eq):
        trade_returns.append(equity / entry_eq - 1.0)

    curve_s = pd.Series(curve, index=idx).ffill().bfill()
    met = compute_metrics(curve_s, trade_returns, timeframe=TIMEFRAME)

    res = SymbolResult(
        symbol="",
        trades=int(met["trades"]),
        win_rate=float(met["win_rate"]),
        total_return_pct=float(met["total_return_pct"]),
        max_drawdown_pct=float(met["max_drawdown_pct"]),
        sharpe=float(met["sharpe"]),
        profit_factor=float(met["profit_factor"]),
    )

    debug = pd.DataFrame(
        {
            "equity": curve_s,
            "vote": vote_s,
            "position": pos_s,
            "leverage": lev_s,
            "regime": regime_s,
            "chosen_expert": chosen_s,
            "bos_score": score_arr,
        },
        index=idx,
    )

    diag = pd.DataFrame(
        {
            "expert": expert_names,
            "final_score": scores,
        }
    ).sort_values("final_score", ascending=False)

    return res, debug, diag


def write_pine(path: str) -> None:
    pine = """// Generated by adaptive_regime_selector.py
// Adaptive Regime Selector (research template)
//@version=6
strategy("Adaptive Regime Selector (Research)", overlay=true, initial_capital=10000, commission_type=strategy.commission.percent, commission_value=0.06, pyramiding=0, calc_on_bar_close=true)

// Tuned defaults from offline search
decay          = input.float(0.99, "Score Decay", step=0.001)
rewardScale    = input.float(120.0, "Reward Scale", step=1.0)
voteThreshold  = input.float(0.28, "Vote Threshold", step=0.01)
ddKillPct      = input.float(8.0, "DD Kill %", step=0.1)
cooldownBars   = input.int(48, "Cooldown Bars", minval=1)
targetBarVol   = input.float(0.0025, "Target Bar Vol", step=0.0001)
maxLeverage    = input.float(1.2, "Max Leverage", step=0.1)
allowShort     = input.bool(false, "Allow Shorts")

// --- Expert 1: Jurik breakout (compact approximation)
jLen = input.int(9, "Jurik Len")
phase = 0.1
beta = 0.45 * (jLen - 1) / (0.45 * (jLen - 1) + 2)
alpha = math.pow(beta, phase)
var float jma = na
jma := na(jma[1]) ? close : (1 - alpha) * close + alpha * jma[1]
jTrend = jma >= jma[3]
e1 = jTrend ? 1.0 : -1.0

// --- Expert 2: GK ribbon trend
lenGK = input.int(70, "GK Len")
multGK = input.float(1.2, "GK Mult", step=0.1)
lag = math.max(int(math.floor((lenGK - 1) / 2)), 0)
zl = ta.ema(lag > 0 ? close + (close - close[lag]) : close, lenGK)
atrGK = ta.atr(14)
upGK = zl + atrGK * multGK
dnGK = zl - atrGK * multGK
bullGK = (close > upGK and close[1] > upGK[1] and close[2] > upGK[2]) and zl > zl[1]
bearGK = (close < dnGK and close[1] < dnGK[1] and close[2] < dnGK[2]) and zl < zl[1]
var float e2 = 0.0
if bullGK
    e2 := 1.0
else if bearGK
    e2 := -1.0

// --- Expert 3: volume-skew style oscillator sign
sk = ta.ema(close - ta.vwma(close, 100), 5)
e3 = sk > 0 ? 1.0 : -1.0

// --- Expert 4: DAFE-style bandwidth/position composite
basis = ta.sma(close, 20)
dev = ta.stdev(close, 20)
upper = basis + 2.0 * dev
lower = basis - 2.0 * dev
pctb = (close - lower) / math.max(upper - lower, syminfo.mintick)
e4 = pctb > 0.55 ? 1.0 : pctb < 0.45 ? -1.0 : 0.0

// Macro regime proxy (BOS-like): risk score from broad market internals.
// In Pine we approximate with BTC, ETH, DXY, VIX direction spreads.
btcUp = request.security("COINBASE:BTCUSD", timeframe.period, close > ta.ema(close, 50) ? 1 : 0)
ethUp = request.security("COINBASE:ETHUSD", timeframe.period, close > ta.ema(close, 50) ? 1 : 0)
dxyUp = request.security("TVC:DXY", timeframe.period, close > ta.ema(close, 50) ? 1 : 0)
vixUp = request.security("TVC:VIX", timeframe.period, close > ta.ema(close, 50) ? 1 : 0)
riskScore = (btcUp + ethUp) - (dxyUp + vixUp)

// Online expert scoring
ret = close[1] != 0 ? (close / close[1] - 1.0) : 0.0
var float s1 = 0.0
var float s2 = 0.0
var float s3 = 0.0
var float s4 = 0.0
s1 := nz(s1[1]) * decay + (nz(e1[1]) * ret) * rewardScale
s2 := nz(s2[1]) * decay + (nz(e2[1]) * ret) * rewardScale
s3 := nz(s3[1]) * decay + (nz(e3[1]) * ret) * rewardScale
s4 := nz(s4[1]) * decay + (nz(e4[1]) * ret) * rewardScale

ema200 = ta.ema(close, 200)

float vote = 0.0
if riskScore >= 1
    // risk-on pool: e1,e2,e4
    w1 = math.max(s1, 0), w2 = math.max(s2, 0), w4 = math.max(s4, 0)
    den = w1 + w2 + w4
    vote := den > 0 ? (w1 * e1 + w2 * e2 + w4 * e4) / den : 0
else if riskScore <= -1
    // risk-off pool: e2,e3,e4
    w2 = math.max(s2, 0), w3 = math.max(s3, 0), w4 = math.max(s4, 0)
    den = w2 + w3 + w4
    vote := den > 0 ? (w2 * e2 + w3 * e3 + w4 * e4) / den : 0
else
    // neutral: trend-follow only when above EMA200
    if close > ema200
        w1 = math.max(s1, 0), w2 = math.max(s2, 0)
        den = w1 + w2
        vote := den > 0 ? (w1 * e1 + w2 * e2) / den : 0
    else
        vote := 0

wantLong = vote > voteThreshold
wantShort = allowShort and vote < -voteThreshold

// Trend gate
if close < ema200 and wantLong
    wantLong := false
if close > ema200 and wantShort
    wantShort := false

// Vol target leverage approximation
rv = ta.stdev(math.log(close / close[1]), 96)
lev = rv > 0 ? math.min(maxLeverage, targetBarVol / rv) : 1.0

// DD kill-switch
var float eqPeak = na
eqPeak := na(eqPeak) ? strategy.equity : math.max(eqPeak, strategy.equity)
ddPct = eqPeak > 0 ? (1 - strategy.equity / eqPeak) * 100 : 0
var int cool = 0
if ddPct > ddKillPct and cool == 0
    cool := cooldownBars
if cool > 0
    cool -= 1

if cool > 0
    strategy.close_all("DD Kill")
else
    if wantLong and strategy.position_size <= 0
        strategy.entry("L", strategy.long, qty=lev)
    if wantShort and strategy.position_size >= 0
        strategy.entry("S", strategy.short, qty=lev)
    if not wantLong and strategy.position_size > 0
        strategy.close("L")
    if not wantShort and strategy.position_size < 0
        strategy.close("S")

plot(vote, "Vote", color=color.aqua, linewidth=2, display=display.pane)
plot(0, "Zero", color=color.new(color.gray, 70), display=display.pane)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(pine)


def main() -> None:
    data = load_or_fetch_all()
    bos_data = {s: data[s] for s in BOS_SYMBOLS}

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for sym in SYMBOLS:
        df = data[sym]
        _, _, score = wf.bos_filter(df.index, bos_data)
        res, debug, diag = run_selector_for_symbol(df, score, CFG)
        res.symbol = sym
        rows.append(
            {
                "symbol": sym,
                "trades": res.trades,
                "win_rate": res.win_rate,
                "total_return_pct": res.total_return_pct,
                "max_drawdown_pct": res.max_drawdown_pct,
                "sharpe": res.sharpe,
                "profit_factor": res.profit_factor,
            }
        )

        debug.to_csv(os.path.join(out_dir, f"adaptive_regime_debug_{sym.replace('/', '_')}.csv"))
        diag.to_csv(os.path.join(out_dir, f"adaptive_regime_expert_diag_{sym.replace('/', '_')}.csv"), index=False)

    met_df = pd.DataFrame(rows)
    met_path = os.path.join(out_dir, "adaptive_regime_metrics.csv")
    met_df.to_csv(met_path, index=False)

    summary = {
        "timeframe": TIMEFRAME,
        "start_utc": START_UTC.isoformat(),
        "end_utc": END_UTC.isoformat(),
        "symbols": SYMBOLS,
        "cost_per_side": COST_PER_SIDE,
        "config": CFG,
        "avg_return_pct": float(met_df["total_return_pct"].mean()),
        "avg_max_dd_pct": float(met_df["max_drawdown_pct"].mean()),
        "avg_sharpe": float(met_df["sharpe"].mean()),
        "avg_win_rate": float(met_df["win_rate"].mean()),
        "avg_profit_factor": float(met_df["profit_factor"].mean()),
        "per_symbol": met_df.to_dict(orient="records"),
        "disclaimer": "Backtest fit on a fixed window; use forward-testing before live deployment.",
    }
    sum_path = os.path.join(out_dir, "adaptive_regime_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_regime_selector.pine")
    write_pine(pine_path)

    print("=== Adaptive Regime Selector Metrics ===")
    print(met_df.to_string(index=False))
    print("\nWrote:")
    print(f"- {met_path}")
    print(f"- {sum_path}")
    print(f"- {pine_path}")


if __name__ == "__main__":
    main()
