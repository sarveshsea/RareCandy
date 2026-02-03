#!/usr/bin/env python3
"""
Walk-forward + coarse parameter sweep for user-provided Pine indicators.

Assumptions (defaults, because user said DONE without overrides):
- Symbols: BTC/USD, ETH/USD
- Timeframe: 15m
- Range: 2025-05-01 to 2026-02-03 (UTC)
- Costs: 0.06% per side
- BOSWaves used as a regime filter overlay for all entry strategies.

Sweep style:
- Each strategy evaluates 3 variants: fast / base / slow.
- For each walk-forward fold, best variant is selected on train, then evaluated on test.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import ccxt
import numpy as np
import pandas as pd

import backtest_pine_batch as base


TIMEFRAME = "15m"
START_UTC = dt.datetime(2025, 5, 1, tzinfo=dt.timezone.utc)
END_UTC = dt.datetime(2026, 2, 3, tzinfo=dt.timezone.utc)
COST_PER_SIDE = 0.0006
PRIMARY_SYMBOLS = ["BTC/USD", "ETH/USD"]
BOS_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "XRP/USD", "ADA/USD", "DOGE/USD", "BCH/USD"]
VARIANTS = ["fast", "base", "slow"]


def s_int(v: int, variant: str, lo: int = 1) -> int:
    if variant == "fast":
        return max(lo, int(round(v * 0.7)))
    if variant == "slow":
        return max(lo, int(round(v * 1.4)))
    return max(lo, int(v))


def v_th(base_v: float, variant: str, delta: float) -> float:
    if variant == "fast":
        return base_v - delta
    if variant == "slow":
        return base_v + delta
    return base_v


# --------------------------
# Parameterized strategy signals
# --------------------------

def sig_jurik(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    n = len(d)

    length = s_int(9, variant, 2)
    pivot_len = s_int(4, variant, 2)

    src = d["close"].to_numpy(dtype=float)
    phase = 0.1
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** phase

    j = np.full(n, np.nan)
    j[0] = src[0]
    for i in range(1, n):
        j[i] = (1.0 - alpha) * src[i] + alpha * j[i - 1]
    d["j"] = j

    trend = d["j"] >= d["j"].shift(3)
    ph = base.pivot_high(d["high"], pivot_len, pivot_len)
    pl = base.pivot_low(d["low"], pivot_len, pivot_len)
    a200 = base.atr(d, 200)

    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)

    H = np.nan
    Hi = -1
    break_up = -1
    upper_active = False
    upper_level = np.nan

    L = np.nan
    Li = -1
    break_dn = -1
    lower_active = False
    lower_level = np.nan

    for i in range(n):
        tr = bool(trend.iloc[i]) if not pd.isna(trend.iloc[i]) else False

        if i > 0 and trend.iloc[i] != trend.iloc[i - 1]:
            upper_active = False
            lower_active = False

        phv = ph.iloc[i]
        plv = pl.iloc[i]
        atrv = a200.iloc[i]

        if tr and not upper_active:
            if (not pd.isna(phv)) and (Hi > break_up) and (not pd.isna(H)) and (not pd.isna(atrv)):
                if abs(phv - H) < atrv:
                    upper_active = True
                    upper_level = phv

        if tr and not pd.isna(phv):
            H = phv
            Hi = i - pivot_len

        if upper_active and d["close"].iloc[i] > upper_level:
            long_sig[i] = True
            upper_active = False
            break_up = i

        if (not tr) and (not lower_active):
            if (not pd.isna(plv)) and (Li > break_dn) and (not pd.isna(L)) and (not pd.isna(atrv)):
                if abs(plv - L) < atrv:
                    lower_active = True
                    lower_level = plv

        if (not tr) and (not pd.isna(plv)):
            L = plv
            Li = i - pivot_len

        if lower_active and d["close"].iloc[i] < lower_level:
            short_sig[i] = True
            lower_active = False
            break_dn = i

    return {"long": pd.Series(long_sig, index=d.index), "short": pd.Series(short_sig, index=d.index)}


def sig_luxalgo(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    p = s_int(7, variant, 2)
    zthr = v_th(0.5, variant, 0.15)

    pc = d["close"].diff()
    avg = base.sma(pc, 50)
    std = pc.rolling(50, min_periods=50).std()
    z = (pc - avg) / std.replace(0, np.nan)

    ph = base.pivot_high(d["high"], p, p)
    pl = base.pivot_low(d["low"], p, p)

    long_sig = np.zeros(len(d), dtype=bool)
    short_sig = np.zeros(len(d), dtype=bool)

    last_ph = np.nan
    last_pl = np.nan

    for i in range(len(d)):
        if not pd.isna(ph.iloc[i]):
            last_ph = ph.iloc[i]
        if not pd.isna(pl.iloc[i]):
            last_pl = pl.iloc[i]
        if i == 0:
            continue

        bull = (
            (not pd.isna(last_ph))
            and d["close"].iloc[i] > last_ph
            and d["close"].iloc[i - 1] <= last_ph
            and (z.iloc[i] > zthr if not pd.isna(z.iloc[i]) else False)
        )
        bear = (
            (not pd.isna(last_pl))
            and d["close"].iloc[i] < last_pl
            and d["close"].iloc[i - 1] >= last_pl
            and (z.iloc[i] < -zthr if not pd.isna(z.iloc[i]) else False)
        )

        if bull:
            long_sig[i] = True
            last_ph = np.nan
        if bear:
            short_sig[i] = True
            last_pl = np.nan

    return {"long": pd.Series(long_sig, index=d.index), "short": pd.Series(short_sig, index=d.index)}


def sig_backquant(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    lookback = s_int(100, variant, 20)
    smooth = s_int(5, variant, 1)
    thr = v_th(0.25, variant, 0.05)

    c = d["close"].to_numpy(dtype=float)
    v = d["volume"].to_numpy(dtype=float)
    n = len(d)
    sk = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        cw = c[i - lookback + 1 : i + 1]
        vw = v[i - lookback + 1 : i + 1]
        sv = float(np.sum(vw))
        if sv <= 0:
            continue
        m = float(np.sum(cw * vw) / sv)
        diff = cw - m
        var = float(np.sum(vw * diff * diff) / sv)
        if var <= 0:
            continue
        sd = math.sqrt(var)
        sk[i] = float(np.sum(vw * diff * diff * diff) / sv / (sd ** 3))

    s = pd.Series(sk, index=d.index).ewm(span=smooth, adjust=False).mean()
    long_sig = base.crossover(s, thr)
    short_sig = base.crossunder(s, -thr)
    return {"long": long_sig.fillna(False), "short": short_sig.fillna(False)}


def sig_reh_rel(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    p = s_int(2, variant, 1)
    tol = v_th(2.0, variant, -0.5 if variant == "slow" else 0.5)

    ph = base.pivot_high(d["high"], p, p)
    pl = base.pivot_low(d["low"], p, p)

    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    active_reh = np.nan
    active_rel = np.nan

    long_sig = np.zeros(len(d), dtype=bool)
    short_sig = np.zeros(len(d), dtype=bool)

    for i in range(len(d)):
        if not pd.isna(ph.iloc[i]):
            highs.append((i - p, float(ph.iloc[i])))
            highs = highs[-200:]
            if len(highs) >= 3:
                (_, h1), (_, h2), (_, h3) = highs[-3], highs[-2], highs[-1]
                if h1 >= h2 >= h3 and (max(h1, h2, h3) - min(h1, h2, h3) <= tol):
                    active_reh = h1

        if not pd.isna(pl.iloc[i]):
            lows.append((i - p, float(pl.iloc[i])))
            lows = lows[-200:]
            if len(lows) >= 3:
                (_, l1), (_, l2), (_, l3) = lows[-3], lows[-2], lows[-1]
                if l1 <= l2 <= l3 and (max(l1, l2, l3) - min(l1, l2, l3) <= tol):
                    active_rel = l1

        c = d["close"].iloc[i]
        if not pd.isna(active_reh) and c > active_reh:
            long_sig[i] = True
            active_reh = np.nan
        if not pd.isna(active_rel) and c < active_rel:
            short_sig[i] = True
            active_rel = np.nan

    return {"long": pd.Series(long_sig, index=d.index), "short": pd.Series(short_sig, index=d.index)}


def sig_dafe(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    length = s_int(20, variant, 5)
    mult = v_th(2.0, variant, 0.4)
    sq_len = s_int(100, variant, 20)
    sq_pct = v_th(20.0, variant, -5.0 if variant == "slow" else 5.0)

    basis = base.sma(d["close"], length)
    dev = d["close"].rolling(length, min_periods=length).std()
    upper = basis + dev * mult
    lower = basis - dev * mult
    upper_mid = basis + dev * mult * 0.5
    lower_mid = basis - dev * mult * 0.5

    bandwidth = upper - lower
    pct_b = (d["close"] - lower) / (upper - lower).replace(0, np.nan)

    bw_rank = base.rolling_percentrank(bandwidth, sq_len)
    is_squeeze = bw_rank <= sq_pct

    basis_slope = basis - basis.shift(1)
    exp_bull = is_squeeze.shift(1).fillna(False) & (~is_squeeze.fillna(False)) & (d["close"] > basis) & (basis_slope > 0)
    exp_bear = is_squeeze.shift(1).fillna(False) & (~is_squeeze.fillna(False)) & (d["close"] < basis) & (basis_slope < 0)

    walk_bars = 2 if variant == "fast" else (3 if variant == "base" else 4)
    n = len(d)
    walk_up = np.zeros(n, dtype=bool)
    walk_dn = np.zeros(n, dtype=bool)
    for i in range(walk_bars - 1, n):
        ok_u, ok_d = True, True
        for j in range(walk_bars):
            if pd.isna(upper_mid.iloc[i - j]) or d["close"].iloc[i - j] <= upper_mid.iloc[i - j]:
                ok_u = False
            if pd.isna(lower_mid.iloc[i - j]) or d["close"].iloc[i - j] >= lower_mid.iloc[i - j]:
                ok_d = False
        walk_up[i], walk_dn[i] = ok_u, ok_d

    p_tr = base.ema(d["close"], 10) - base.ema(d["close"], 20)
    b_tr = base.ema(bandwidth, 10) - base.ema(bandwidth, 20)
    div_th = 0.3 if variant != "slow" else 0.25
    bull_div = (p_tr < 0) & (b_tr < 0) & (pct_b < div_th)
    bear_div = (p_tr > 0) & (b_tr < 0) & (pct_b > (1 - div_th))

    vol_avg = base.sma(d["volume"], 20)
    vol_ratio = d["volume"] / vol_avg.replace(0, np.nan)

    buy_score = np.zeros(n)
    sell_score = np.zeros(n)
    buy_score += np.where(pct_b < 0.2, 2, 0)
    sell_score += np.where(pct_b > 0.8, 2, 0)
    buy_score += np.where(exp_bull, 3, 0)
    sell_score += np.where(exp_bear, 3, 0)
    buy_score += np.where(bull_div, 2, 0)
    sell_score += np.where(bear_div, 2, 0)
    buy_score += np.where(walk_up, 1, 0)
    sell_score += np.where(walk_dn, 1, 0)
    strong_vol = vol_ratio > 1.5
    buy_score += np.where(strong_vol, 1, 0)
    sell_score += np.where(strong_vol, 1, 0)

    score_th = 4 if variant != "fast" else 3
    buy_raw = buy_score >= score_th
    sell_raw = sell_score >= score_th

    adx_val, _, _ = base.adx(d, 14)
    vol_filter = d["volume"] >= (vol_avg * 1.2)
    adx_filter = adx_val >= (18 if variant == "fast" else 20)

    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    bars_since = 999
    gap = 2 if variant == "fast" else 3
    for i in range(n):
        filt = bool(vol_filter.iloc[i]) and bool(adx_filter.iloc[i]) and (bars_since >= gap)
        b, s = bool(buy_raw[i]) and filt, bool(sell_raw[i]) and filt
        if b:
            buy_sig[i] = True
            bars_since = 0
        elif s:
            sell_sig[i] = True
            bars_since = 0
        else:
            bars_since += 1

    return {"long": pd.Series(buy_sig, index=d.index), "short": pd.Series(sell_sig, index=d.index)}


def sig_prime(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    ema_n = 34 if variant == "fast" else (50 if variant == "base" else 80)
    e = base.ema(d["close"], ema_n)
    ema_up = e > e.shift(1)
    ema_dn = e < e.shift(1)
    hrs = d.index.hour
    in_kill = (hrs >= 1) & (hrs < 23)

    buy_logic = (d["close"] > (d["high"] + d["low"]) / 2.0) & (d["low"] < d["low"].shift(1))
    sell_logic = (d["close"] < (d["high"] + d["low"]) / 2.0) & (d["high"] > d["high"].shift(1))

    long_sig = buy_logic & in_kill & ema_up
    short_sig = sell_logic & in_kill & ema_dn
    return {"long": long_sig.fillna(False), "short": short_sig.fillna(False)}


def sig_extreme_hma(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    length = 24 if variant == "fast" else (33 if variant == "base" else 48)
    atr_mult = 0.6 if variant == "fast" else (0.7 if variant == "base" else 0.9)

    sqrtlen = max(1, int(round(math.sqrt(length))))
    halflen = max(1, int(round(length / 2)))

    hma = base.ema(d["close"], sqrtlen)
    h = hma.rolling(length, min_periods=length).max()
    l = hma.rolling(length, min_periods=length).min()
    hh = h.rolling(halflen, min_periods=halflen).min()
    ll = l.rolling(halflen, min_periods=halflen).max()

    mid = (hh + ll) / 2.0
    a = base.atr(d, 30) * atr_mult
    upper, lower = mid + a, mid - a

    L = d["close"] > upper
    S = d["close"] < lower

    tr = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        tr[i] = tr[i - 1]
        if bool(L.iloc[i]):
            tr[i] = 1
        elif bool(S.iloc[i]):
            tr[i] = -1

    tr_s = pd.Series(tr, index=d.index)
    return {
        "long": ((tr_s == 1) & (tr_s.shift(1) != 1)).fillna(False),
        "short": ((tr_s == -1) & (tr_s.shift(1) != -1)).fillna(False),
    }


def sig_gk(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    length = 50 if variant == "fast" else (70 if variant == "base" else 100)
    mult = 1.0 if variant == "fast" else (1.2 if variant == "base" else 1.6)

    lag = max(int(math.floor((length - 1) / 2)), 0)
    zsrc = d["close"] + (d["close"] - d["close"].shift(lag))
    zl = base.ema(zsrc, length)
    a = base.atr(d, 14)
    up = zl + a * mult
    dn = zl - a * mult

    bull = (d["close"] > up) & (d["close"].shift(1) > up.shift(1)) & (d["close"].shift(2) > up.shift(2)) & (zl > zl.shift(1))
    bear = (d["close"] < dn) & (d["close"].shift(1) < dn.shift(1)) & (d["close"].shift(2) < dn.shift(2)) & (zl < zl.shift(1))

    tr = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        tr[i] = tr[i - 1]
        if bool(bull.iloc[i]):
            tr[i] = 1
        elif bool(bear.iloc[i]):
            tr[i] = -1

    tr_s = pd.Series(tr, index=d.index)
    flip = (tr_s != tr_s.shift(1)) & (tr_s != 0)
    return {"long": (flip & (tr_s == 1)).fillna(False), "short": (flip & (tr_s == -1)).fillna(False)}


def sig_abcd(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    pivot = 4 if variant == "fast" else (5 if variant == "base" else 7)
    tol = 0.05 if variant == "fast" else (0.035 if variant == "base" else 0.025)

    ph = base.pivot_high(d["high"], pivot, pivot)
    pl = base.pivot_low(d["low"], pivot, pivot)

    pivots: List[Tuple[int, float, str]] = []
    active: List[dict] = []

    long_sig = np.zeros(len(d), dtype=bool)
    short_sig = np.zeros(len(d), dtype=bool)

    for i in range(len(d)):
        npiv = None
        if not pd.isna(ph.iloc[i]):
            npiv = (i - pivot, float(ph.iloc[i]), "H")
        if not pd.isna(pl.iloc[i]):
            if npiv is None:
                npiv = (i - pivot, float(pl.iloc[i]), "L")
            else:
                if abs(float(pl.iloc[i]) - d["close"].iloc[i]) > abs(npiv[1] - d["close"].iloc[i]):
                    npiv = (i - pivot, float(pl.iloc[i]), "L")

        if npiv is not None:
            if pivots and pivots[-1][2] == npiv[2]:
                if (npiv[2] == "H" and npiv[1] > pivots[-1][1]) or (npiv[2] == "L" and npiv[1] < pivots[-1][1]):
                    pivots[-1] = npiv
            else:
                pivots.append(npiv)
                pivots = pivots[-30:]

            if len(pivots) >= 3:
                A, B, C = pivots[-3], pivots[-2], pivots[-1]
                AB = B[1] - A[1]
                BC = C[1] - B[1]
                if AB != 0:
                    ratio = abs(BC / AB)
                    if 0.5 <= ratio <= 0.868:
                        t1 = C[1] + AB * 1.0
                        t2 = C[1] + AB * 1.272
                        t3 = C[1] + AB * 1.618
                        z1 = (min(t1 * (1 - tol), t1 * (1 + tol)), max(t1 * (1 - tol), t1 * (1 + tol)))
                        z2 = (min(t2 * (1 - tol), t2 * (1 + tol)), max(t2 * (1 - tol), t2 * (1 + tol)))
                        z3 = (min(t3 * (1 - tol), t3 * (1 + tol)), max(t3 * (1 - tol), t3 * (1 + tol)))
                        active.append({"AB": AB, "C": C[1], "dir": 1 if AB > 0 else -1, "z1": z1, "z2": z2, "z3": z3})
                        active = active[-8:]

        hi, lo = d["high"].iloc[i], d["low"].iloc[i]
        keep = []
        for p in active:
            hit = False
            zones = [p["z1"], p["z2"], p["z3"]]
            if p["dir"] == 1:
                if lo < p["C"]:
                    continue
                if any((hi >= z[0] and hi <= z[1]) for z in zones):
                    short_sig[i] = True
                    hit = True
                elif hi > p["z3"][1] * 1.01:
                    hit = True
            else:
                if hi > p["C"]:
                    continue
                if any((lo >= z[0] and lo <= z[1]) for z in zones):
                    long_sig[i] = True
                    hit = True
                elif lo < p["z3"][0] * 0.99:
                    hit = True

            if not hit:
                keep.append(p)
        active = keep

    return {"long": pd.Series(long_sig, index=d.index), "short": pd.Series(short_sig, index=d.index)}


def sig_gaussian(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    sample = 30 if variant == "fast" else (50 if variant == "base" else 80)
    min_slope = 1.0 if variant == "fast" else (1.5 if variant == "base" else 2.2)

    base_h = 0.5 + (sample * 0.04)
    base_atr = 0.02 + (sample * 0.0016)
    final_h = base_h * 0.85
    final_atr_mult = base_atr * 0.9 * 0.8

    yhat = base.gaussian_regression_max(d["close"], final_h)
    diff = yhat.diff()
    raw_slope = (diff / yhat.replace(0, np.nan)) * 1000.0

    vol_factor = base.atr(d, 20) / base.alma(d["close"], 20, 0.7, 4).replace(0, np.nan)
    adaptive_min = min_slope * (vol_factor * 100.0)
    slope_ok = raw_slope.abs() > adaptive_min

    alma_vol = base.alma(d["volume"], 20, 0.7, 4)
    vol_ratio = d["volume"] / alma_vol.replace(0, np.nan)
    prog = (1.0 / vol_ratio).pow(0.5)

    dyn_lim = base.atr(d, 5) * (final_atr_mult * prog)
    up = diff > dyn_lim
    dn = diff < -dyn_lim

    tr = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        tr[i] = tr[i - 1]
        if bool(up.iloc[i]) and bool(slope_ok.iloc[i]):
            tr[i] = 1
        elif bool(dn.iloc[i]) and bool(slope_ok.iloc[i]):
            tr[i] = -1

    trs = pd.Series(tr, index=d.index)
    return {"long": ((trs == 1) & (trs.shift(1) != 1)).fillna(False), "short": ((trs == -1) & (trs.shift(1) != -1)).fillna(False)}


def sig_safezone(df: pd.DataFrame, variant: str) -> Dict[str, pd.Series]:
    d = df.copy()
    trend_n = 16 if variant == "fast" else (22 if variant == "base" else 34)
    lookback = 7 if variant == "fast" else (10 if variant == "base" else 15)
    mult = 2.0 if variant == "fast" else (2.5 if variant == "base" else 3.2)

    ema_tr = base.ema(d["close"], trend_n)
    uptrend = d["close"] > ema_tr
    sw_up = uptrend & (~uptrend.shift(1).fillna(False))
    sw_dn = (~uptrend) & (uptrend.shift(1).fillna(False))

    dn_pen = np.maximum(d["low"].shift(1) - d["low"], 0.0)
    up_pen = np.maximum(d["high"] - d["high"].shift(1), 0.0)
    noise_dn = base.sma(pd.Series(dn_pen, index=d.index), lookback)
    noise_up = base.sma(pd.Series(up_pen, index=d.index), lookback)

    raw_long = d["low"].shift(1) - noise_dn * mult
    raw_short = d["high"].shift(1) + noise_up * mult

    stop_l = np.full(len(d), np.nan)
    stop_s = np.full(len(d), np.nan)

    for i in range(1, len(d)):
        if bool(sw_up.iloc[i]):
            stop_l[i] = raw_long.iloc[i]
        elif bool(uptrend.iloc[i]):
            p0 = stop_l[i - 1] if not np.isnan(stop_l[i - 1]) else raw_long.iloc[i]
            p1 = stop_l[i - 2] if i >= 2 and not np.isnan(stop_l[i - 2]) else raw_long.iloc[i]
            stop_l[i] = np.nanmax([raw_long.iloc[i], p0, p1])

        if bool(sw_dn.iloc[i]):
            stop_s[i] = raw_short.iloc[i]
        elif not bool(uptrend.iloc[i]):
            p0 = stop_s[i - 1] if not np.isnan(stop_s[i - 1]) else raw_short.iloc[i]
            p1 = stop_s[i - 2] if i >= 2 and not np.isnan(stop_s[i - 2]) else raw_short.iloc[i]
            vals = [raw_short.iloc[i], p0, p1]
            vals = [v for v in vals if not np.isnan(v)]
            stop_s[i] = np.min(vals) if vals else np.nan

    stop_l_s = pd.Series(stop_l, index=d.index)
    stop_s_s = pd.Series(stop_s, index=d.index)

    long_exit = uptrend & (d["close"] < stop_l_s)
    short_exit = (~uptrend) & (d["close"] > stop_s_s)

    return {
        "long": sw_up.fillna(False),
        "short": sw_dn.fillna(False),
        "long_exit": long_exit.fillna(False),
        "short_exit": short_exit.fillna(False),
    }


# BOS filter helpers

def bos_filter(main_index: pd.Index, bos_data: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ys = {}
    for sym, d in bos_data.items():
        d2 = d.reindex(main_index).ffill()
        _, pdi, mdi = base.adx(d2, 14)
        ys[sym] = (pdi - mdi).clip(-50, 50)

    y1, y2, y3, y4, y5, y6, y7, y8 = [ys[s] for s in BOS_SYMBOLS]
    risk_on = (y1 > 0).astype(int) + (y2 > 0).astype(int) + (y4 > 0).astype(int) + (y5 > 0).astype(int) + (y6 > 0).astype(int)
    risk_off = (y3 > 0).astype(int) + (y7 > 0).astype(int) + (y8 > 0).astype(int)
    score = risk_on - risk_off

    long_ok = (score >= 1)
    short_ok = (score <= -1)
    return long_ok.fillna(False), short_ok.fillna(False), score.fillna(0)


StrategyFn = Callable[[pd.DataFrame, str], Dict[str, pd.Series]]

STRATEGIES: Dict[str, StrategyFn] = {
    "Jurik_Breakouts": sig_jurik,
    "LuxAlgo_MSB_OB": sig_luxalgo,
    "BackQuant_VolSkew": sig_backquant,
    "REH_REL_Breakout": sig_reh_rel,
    "DAFE_Bands_SmartComposite": sig_dafe,
    "PrimeUltimate_Sniper": sig_prime,
    "Extreme_HMA_ATR": sig_extreme_hma,
    "GK_Trend_Ribbon": sig_gk,
    "KTY_ABCD": sig_abcd,
    "Gaussian_MA": sig_gaussian,
    "SmartSafeZone": sig_safezone,
}


@dataclass
class FoldWindow:
    fold: int
    train_start: dt.datetime
    train_end: dt.datetime
    test_start: dt.datetime
    test_end: dt.datetime


def metric_score(avg_ret: float, avg_dd: float, avg_sharpe: float, avg_pf: float) -> float:
    return avg_ret + (avg_sharpe * 10.0) + (avg_pf * 2.5) - (avg_dd * 0.6)


def run_strategy_on_window(
    name: str,
    fn: StrategyFn,
    variant: str,
    data: Dict[str, pd.DataFrame],
    bos_data: Dict[str, pd.DataFrame],
    start: dt.datetime,
    end: dt.datetime,
) -> dict:
    rows = []
    for sym in PRIMARY_SYMBOLS:
        df = data[sym]
        seg = df[(df.index >= start) & (df.index <= end)].copy()
        if len(seg) < 200:
            continue

        sig = fn(seg, variant)
        long_ok, short_ok, _ = bos_filter(seg.index, bos_data)

        long_e = sig["long"].reindex(seg.index).fillna(False) & long_ok
        short_e = sig["short"].reindex(seg.index).fillna(False) & short_ok

        bt = base.run_backtest(
            seg,
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
                "symbol": sym,
                "trades": bt.trades,
                "win_rate": bt.win_rate,
                "return_pct": bt.total_return_pct,
                "max_dd_pct": bt.max_drawdown_pct,
                "sharpe": bt.sharpe,
                "profit_factor": bt.profit_factor,
            }
        )

    if not rows:
        return {
            "strategy": name,
            "variant": variant,
            "symbols": 0,
            "avg_trades": 0.0,
            "avg_win_rate": 0.0,
            "avg_return_pct": -999.0,
            "avg_max_dd_pct": 99.0,
            "avg_sharpe": -99.0,
            "avg_profit_factor": 0.0,
            "score": -999.0,
        }

    rdf = pd.DataFrame(rows)
    out = {
        "strategy": name,
        "variant": variant,
        "symbols": rdf["symbol"].nunique(),
        "avg_trades": float(rdf["trades"].mean()),
        "avg_win_rate": float(rdf["win_rate"].mean()),
        "avg_return_pct": float(rdf["return_pct"].mean()),
        "avg_max_dd_pct": float(rdf["max_dd_pct"].mean()),
        "avg_sharpe": float(rdf["sharpe"].mean()),
        "avg_profit_factor": float(rdf["profit_factor"].mean()),
    }
    out["score"] = metric_score(out["avg_return_pct"], out["avg_max_dd_pct"], out["avg_sharpe"], out["avg_profit_factor"])
    return out


def build_folds(start: dt.datetime, end: dt.datetime) -> List[FoldWindow]:
    total = (end - start).total_seconds()
    p25 = start + dt.timedelta(seconds=total * 0.25)
    p50 = start + dt.timedelta(seconds=total * 0.50)
    p75 = start + dt.timedelta(seconds=total * 0.75)
    return [
        FoldWindow(1, start, p25, p25, p50),
        FoldWindow(2, start, p50, p50, p75),
        FoldWindow(3, start, p75, p75, end),
    ]


def main() -> None:
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(END_UTC.timestamp() * 1000)

    ex = ccxt.coinbase({"enableRateLimit": True})
    all_symbols = sorted(set(PRIMARY_SYMBOLS + BOS_SYMBOLS))

    data: Dict[str, pd.DataFrame] = {}
    missing = []
    for sym in all_symbols:
        try:
            data[sym] = base.clean_df(base.fetch_ohlcv(ex, sym, TIMEFRAME, start_ms, end_ms))
        except Exception:
            missing.append(sym)

    if missing:
        raise RuntimeError(f"Missing symbols from exchange feed: {missing}")

    bos_data = {s: data[s] for s in BOS_SYMBOLS}
    folds = build_folds(START_UTC, END_UTC)

    fold_rows = []
    test_rows = []

    for fw in folds:
        for strat_name, fn in STRATEGIES.items():
            train_eval = []
            for variant in VARIANTS:
                met = run_strategy_on_window(
                    strat_name, fn, variant, data, bos_data, fw.train_start, fw.train_end
                )
                train_eval.append(met)

            tdf = pd.DataFrame(train_eval).sort_values("score", ascending=False)
            best = tdf.iloc[0].to_dict()

            test_met = run_strategy_on_window(
                strat_name,
                fn,
                str(best["variant"]),
                data,
                bos_data,
                fw.test_start,
                fw.test_end,
            )

            fold_rows.extend(
                [
                    {
                        "fold": fw.fold,
                        "window": "train",
                        "strategy": strat_name,
                        **m,
                        "train_start": fw.train_start.isoformat(),
                        "train_end": fw.train_end.isoformat(),
                        "test_start": fw.test_start.isoformat(),
                        "test_end": fw.test_end.isoformat(),
                    }
                    for m in train_eval
                ]
            )

            test_rows.append(
                {
                    "fold": fw.fold,
                    "strategy": strat_name,
                    "selected_variant": best["variant"],
                    "train_score": best["score"],
                    "test_avg_return_pct": test_met["avg_return_pct"],
                    "test_avg_max_dd_pct": test_met["avg_max_dd_pct"],
                    "test_avg_sharpe": test_met["avg_sharpe"],
                    "test_avg_profit_factor": test_met["avg_profit_factor"],
                    "test_avg_trades": test_met["avg_trades"],
                    "test_avg_win_rate": test_met["avg_win_rate"],
                    "test_score": test_met["score"],
                    "train_start": fw.train_start.isoformat(),
                    "train_end": fw.train_end.isoformat(),
                    "test_start": fw.test_start.isoformat(),
                    "test_end": fw.test_end.isoformat(),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    test_df = pd.DataFrame(test_rows)

    summary = (
        test_df.groupby("strategy", as_index=False)
        .agg(
            folds=("fold", "count"),
            chosen_variants=("selected_variant", lambda s: ",".join(s.tolist())),
            avg_test_return_pct=("test_avg_return_pct", "mean"),
            avg_test_max_dd_pct=("test_avg_max_dd_pct", "mean"),
            avg_test_sharpe=("test_avg_sharpe", "mean"),
            avg_test_profit_factor=("test_avg_profit_factor", "mean"),
            avg_test_trades=("test_avg_trades", "mean"),
            avg_test_win_rate=("test_avg_win_rate", "mean"),
            avg_test_score=("test_score", "mean"),
        )
        .sort_values("avg_test_score", ascending=False)
        .reset_index(drop=True)
    )

    out_dir = "analysis/results"
    os.makedirs(out_dir, exist_ok=True)

    fold_path = f"{out_dir}/walkforward_train_sweep.csv"
    fold_df.to_csv(fold_path, index=False)

    test_path = f"{out_dir}/walkforward_fold_results.csv"
    test_df.to_csv(test_path, index=False)

    summary_path = f"{out_dir}/walkforward_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Also run full-period sweep snapshot with BOS overlay.
    full_rows = []
    for strat_name, fn in STRATEGIES.items():
        for variant in VARIANTS:
            full_rows.append(
                run_strategy_on_window(strat_name, fn, variant, data, bos_data, START_UTC, END_UTC)
            )
    full_df = pd.DataFrame(full_rows).sort_values("score", ascending=False)
    full_path = f"{out_dir}/sweep_fullperiod_bos_overlay.csv"
    full_df.to_csv(full_path, index=False)

    meta = {
        "timeframe": TIMEFRAME,
        "start_utc": START_UTC.isoformat(),
        "end_utc": END_UTC.isoformat(),
        "primary_symbols": PRIMARY_SYMBOLS,
        "bos_filter_symbols": BOS_SYMBOLS,
        "cost_per_side": COST_PER_SIDE,
        "variants": VARIANTS,
        "folds": [f"{f.fold}:{f.train_start.isoformat()}->{f.train_end.isoformat()} train, {f.test_start.isoformat()}->{f.test_end.isoformat()} test" for f in folds],
    }
    with open(f"{out_dir}/walkforward_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=== Walk-Forward Leaderboard (BOS overlay) ===")
    print(summary.head(10).to_string(index=False))
    print("\nWrote:")
    print(f"- {fold_path}")
    print(f"- {test_path}")
    print(f"- {summary_path}")
    print(f"- {full_path}")
    print(f"- {out_dir}/walkforward_meta.json")


if __name__ == "__main__":
    main()
