#!/usr/bin/env python3
"""
Batch backtest harness for user-provided Pine indicators.

Defaults:
- Symbols: BTC/USD, ETH/USD
- Timeframe: 15m
- Date range: 2025-05-01 to 2026-02-03 (UTC)
- Costs: 0.06% per side (fee+slippage)
- Positioning: long+short, 1x notional, full allocation
"""

from __future__ import annotations

import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import ccxt


# -----------------------------
# Config
# -----------------------------
TIMEFRAME = "15m"
START_UTC = dt.datetime(2025, 5, 1, tzinfo=dt.timezone.utc)
END_UTC = dt.datetime(2026, 2, 3, tzinfo=dt.timezone.utc)
TRADE_COST_PER_SIDE = 0.0006  # 0.06% per side
PRIMARY_SYMBOLS = ["BTC/USD", "ETH/USD"]
BOSWAVES_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD",
    "XRP/USD", "ADA/USD", "DOGE/USD", "BCH/USD",
]

BARS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


# -----------------------------
# Helpers
# -----------------------------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def alma(s: pd.Series, window: int, offset: float, sigma: float) -> pd.Series:
    arr = s.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    m = offset * (window - 1)
    sd = window / sigma
    w = np.array([math.exp(-((i - m) ** 2) / (2 * sd * sd)) for i in range(window)], dtype=float)
    w /= w.sum()
    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        if np.isnan(chunk).any():
            continue
        out[i] = float(np.dot(chunk, w))
    return pd.Series(out, index=s.index)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def adx(df: pd.DataFrame, n: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_ = rma(tr, n)

    plus_di = 100.0 * rma(plus_dm, n) / atr_.replace(0, np.nan)
    minus_di = 100.0 * rma(minus_dm, n) / atr_.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = rma(dx, n)
    return adx_, plus_di, minus_di


def crossover(a: pd.Series, b) -> pd.Series:
    if np.isscalar(b):
        b = pd.Series(float(b), index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b) -> pd.Series:
    if np.isscalar(b):
        b = pd.Series(float(b), index=a.index)
    return (a < b) & (a.shift(1) >= b.shift(1))


def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    arr = high.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        v = arr[i]
        if np.isnan(v):
            continue
        w = arr[i - left : i + right + 1]
        if np.isnan(w).any():
            continue
        if v == np.max(w):
            out[i + right] = v
    return pd.Series(out, index=high.index)


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    arr = low.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        v = arr[i]
        if np.isnan(v):
            continue
        w = arr[i - left : i + right + 1]
        if np.isnan(w).any():
            continue
        if v == np.min(w):
            out[i + right] = v
    return pd.Series(out, index=low.index)


def rolling_percentrank(s: pd.Series, length: int) -> pd.Series:
    arr = s.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(length - 1, len(arr)):
        w = arr[i - length + 1 : i + 1]
        if np.isnan(w).any():
            continue
        rank = np.sum(w <= arr[i])
        out[i] = 100.0 * (rank - 1) / max(length - 1, 1)
    return pd.Series(out, index=s.index)


def gaussian_regression_max(src: pd.Series, h: float) -> pd.Series:
    arr = src.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    l = int(math.ceil(h * 6))
    d = 2.0 * (h ** 2)
    ws = np.array([math.exp(-(i ** 2) / d) for i in range(l + 1)], dtype=float)
    for i in range(len(arr)):
        if i < l:
            continue
        chunk = arr[i - l : i + 1][::-1]  # i..i-l
        if np.isnan(chunk).any():
            continue
        out[i] = float(np.dot(chunk, ws) / ws.sum())
    return pd.Series(out, index=src.index)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


# -----------------------------
# Backtest Engine
# -----------------------------
@dataclass
class BtResult:
    strategy: str
    symbol: str
    trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    profit_factor: float


def run_backtest(
    df: pd.DataFrame,
    long_entry: pd.Series,
    short_entry: pd.Series,
    long_exit: Optional[pd.Series] = None,
    short_exit: Optional[pd.Series] = None,
    cost_per_side: float = TRADE_COST_PER_SIDE,
    timeframe: str = TIMEFRAME,
) -> BtResult:
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    le = long_entry.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    se = short_entry.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    lx = (
        long_exit.reindex(df.index).fillna(False).to_numpy(dtype=bool)
        if long_exit is not None
        else np.zeros(n, dtype=bool)
    )
    sx = (
        short_exit.reindex(df.index).fillna(False).to_numpy(dtype=bool)
        if short_exit is not None
        else np.zeros(n, dtype=bool)
    )

    equity = 1.0
    position = 0  # 1 long, -1 short
    entry_equity = np.nan
    trades: List[float] = []
    curve = np.full(n, np.nan)
    curve[0] = equity

    for i in range(1, n):
        # Mark-to-market for open position across bar i-1 -> i.
        if position != 0 and close[i - 1] > 0:
            r = (close[i] / close[i - 1] - 1.0) * position
            equity *= (1.0 + r)

        # Signal processing at bar close i.
        if position == 1:
            must_exit = lx[i] or se[i]
            if must_exit:
                equity *= (1.0 - cost_per_side)
                if not np.isnan(entry_equity) and entry_equity > 0:
                    trades.append(equity / entry_equity - 1.0)
                position = 0
                entry_equity = np.nan
                if se[i] and not lx[i]:
                    position = -1
                    equity *= (1.0 - cost_per_side)
                    entry_equity = equity
        elif position == -1:
            must_exit = sx[i] or le[i]
            if must_exit:
                equity *= (1.0 - cost_per_side)
                if not np.isnan(entry_equity) and entry_equity > 0:
                    trades.append(equity / entry_equity - 1.0)
                position = 0
                entry_equity = np.nan
                if le[i] and not sx[i]:
                    position = 1
                    equity *= (1.0 - cost_per_side)
                    entry_equity = equity
        else:
            if le[i] and not se[i]:
                position = 1
                equity *= (1.0 - cost_per_side)
                entry_equity = equity
            elif se[i] and not le[i]:
                position = -1
                equity *= (1.0 - cost_per_side)
                entry_equity = equity

        curve[i] = equity

    # Force-close last position.
    if position != 0 and not np.isnan(entry_equity):
        equity *= (1.0 - cost_per_side)
        trades.append(equity / entry_equity - 1.0)
        curve[-1] = equity

    curve_s = pd.Series(curve, index=df.index).ffill().bfill()
    ret_s = curve_s.pct_change().fillna(0.0)
    bars_per_year = BARS_PER_YEAR.get(timeframe, 365 * 24 * 4)

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]

    total_return_pct = (curve_s.iloc[-1] - 1.0) * 100.0
    running_max = curve_s.cummax()
    dd = (curve_s / running_max - 1.0).min()
    max_drawdown_pct = abs(float(dd) * 100.0)

    std = float(ret_s.std())
    sharpe = float((ret_s.mean() / std) * math.sqrt(bars_per_year)) if std > 1e-12 else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    win_rate = float((len(wins) / len(trades)) * 100.0) if trades else 0.0

    return BtResult(
        strategy="",
        symbol="",
        trades=len(trades),
        win_rate=win_rate,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe=sharpe,
        profit_factor=profit_factor,
    )


# -----------------------------
# Strategy Signal Builders
# -----------------------------

def signals_jurik(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    n = len(d)
    src = d["close"].to_numpy(dtype=float)
    length = 9
    phase = 0.1
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** phase
    j = np.full(n, np.nan)
    j[0] = src[0]
    for i in range(1, n):
        j[i] = (1.0 - alpha) * src[i] + alpha * j[i - 1]
    d["j"] = j
    trend = d["j"] >= d["j"].shift(3)

    pivot_len = 4
    ph = pivot_high(d["high"], pivot_len, pivot_len)
    pl = pivot_low(d["low"], pivot_len, pivot_len)
    a200 = atr(d, 200)

    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)

    H = np.nan
    Hi = -1
    BreakUp = -1
    upper_active = False
    upper_level = np.nan

    L = np.nan
    Li = -1
    BreakDn = -1
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

        if tr and (not upper_active):
            if (not pd.isna(phv)) and (Hi > BreakUp) and (not pd.isna(H)) and (not pd.isna(atrv)):
                if abs(phv - H) < atrv:
                    upper_active = True
                    upper_level = phv

        if tr and (not pd.isna(phv)):
            H = phv
            Hi = i - pivot_len

        if upper_active and (d["close"].iloc[i] > upper_level):
            long_sig[i] = True
            upper_active = False
            BreakUp = i

        if (not tr) and (not lower_active):
            if (not pd.isna(plv)) and (Li > BreakDn) and (not pd.isna(L)) and (not pd.isna(atrv)):
                if abs(plv - L) < atrv:
                    lower_active = True
                    lower_level = plv

        if (not tr) and (not pd.isna(plv)):
            L = plv
            Li = i - pivot_len

        if lower_active and (d["close"].iloc[i] < lower_level):
            short_sig[i] = True
            lower_active = False
            BreakDn = i

    return {
        "long": pd.Series(long_sig, index=d.index),
        "short": pd.Series(short_sig, index=d.index),
    }


def signals_luxalgo(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    pc = d["close"].diff()
    avg = sma(pc, 50)
    std = pc.rolling(50, min_periods=50).std()
    z = (pc - avg) / std.replace(0, np.nan)

    p = 7
    ph = pivot_high(d["high"], p, p)
    pl = pivot_low(d["low"], p, p)

    long_sig = np.zeros(len(d), dtype=bool)
    short_sig = np.zeros(len(d), dtype=bool)
    lastPh = np.nan
    lastPl = np.nan

    for i in range(len(d)):
        if not pd.isna(ph.iloc[i]):
            lastPh = ph.iloc[i]
        if not pd.isna(pl.iloc[i]):
            lastPl = pl.iloc[i]

        if i == 0:
            continue

        bull = (
            (not pd.isna(lastPh))
            and d["close"].iloc[i] > lastPh
            and d["close"].iloc[i - 1] <= lastPh
            and (z.iloc[i] > 0.5 if not pd.isna(z.iloc[i]) else False)
        )
        bear = (
            (not pd.isna(lastPl))
            and d["close"].iloc[i] < lastPl
            and d["close"].iloc[i - 1] >= lastPl
            and (z.iloc[i] < -0.5 if not pd.isna(z.iloc[i]) else False)
        )

        if bull:
            long_sig[i] = True
            lastPh = np.nan
        if bear:
            short_sig[i] = True
            lastPl = np.nan

    return {"long": pd.Series(long_sig, index=d.index), "short": pd.Series(short_sig, index=d.index)}


def signals_volume_skew(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    lookback = 100
    c = d["close"].to_numpy(dtype=float)
    v = d["volume"].to_numpy(dtype=float)
    n = len(d)
    sk = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        cw = c[i - lookback + 1 : i + 1]
        vw = v[i - lookback + 1 : i + 1]
        sw = float(np.sum(vw))
        if sw <= 0:
            continue
        mean = float(np.sum(cw * vw) / sw)
        diff = cw - mean
        var = float(np.sum(vw * diff * diff) / sw)
        if var <= 0:
            continue
        sd = math.sqrt(var)
        sk[i] = float(np.sum(vw * diff * diff * diff) / sw / (sd ** 3))

    s = pd.Series(sk, index=d.index).ewm(span=5, adjust=False).mean()
    long_sig = crossover(s, 0.25)
    short_sig = crossunder(s, -0.25)
    return {"long": long_sig.fillna(False), "short": short_sig.fillna(False)}


def signals_reh_rel(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    p = 2
    tol = 2.0
    ph = pivot_high(d["high"], p, p)
    pl = pivot_low(d["low"], p, p)

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
                (t1, h1), (t2, h2), (t3, h3) = highs[-3], highs[-2], highs[-1]
                if h1 >= h2 >= h3 and (max(h1, h2, h3) - min(h1, h2, h3) <= tol) and (t2 > t1) and (t3 > t2):
                    active_reh = h1

        if not pd.isna(pl.iloc[i]):
            lows.append((i - p, float(pl.iloc[i])))
            lows = lows[-200:]
            if len(lows) >= 3:
                (t1, l1), (t2, l2), (t3, l3) = lows[-3], lows[-2], lows[-1]
                if l1 <= l2 <= l3 and (max(l1, l2, l3) - min(l1, l2, l3) <= tol) and (t2 > t1) and (t3 > t2):
                    active_rel = l1

        c = d["close"].iloc[i]
        if not pd.isna(active_reh) and c > active_reh:
            long_sig[i] = True
            active_reh = np.nan
        if not pd.isna(active_rel) and c < active_rel:
            short_sig[i] = True
            active_rel = np.nan

    return {"long": pd.Series(long_sig, index=d.index), "short": pd.Series(short_sig, index=d.index)}


def signals_dafe(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    length = 20
    mult = 2.0

    basis = sma(d["close"], length)
    dev = d["close"].rolling(length, min_periods=length).std()
    upper = basis + dev * mult
    lower = basis - dev * mult
    upper_mid = basis + dev * mult * 0.5
    lower_mid = basis - dev * mult * 0.5

    bandwidth = upper - lower
    pct_b = (d["close"] - lower) / (upper - lower).replace(0, np.nan)

    bw_pct_rank = rolling_percentrank(bandwidth, 100)
    is_squeeze = bw_pct_rank <= 20

    basis_slope = basis - basis.shift(1)
    expansion_breakout_bull = is_squeeze.shift(1).fillna(False) & (~is_squeeze.fillna(False)) & (d["close"] > basis) & (basis_slope > 0)
    expansion_breakout_bear = is_squeeze.shift(1).fillna(False) & (~is_squeeze.fillna(False)) & (d["close"] < basis) & (basis_slope < 0)

    walk_bars = int(3 / 1.5)
    n = len(d)
    walking_upper = np.zeros(n, dtype=bool)
    walking_lower = np.zeros(n, dtype=bool)
    for i in range(walk_bars - 1, n):
        ok_u = True
        ok_l = True
        for j in range(walk_bars):
            if pd.isna(upper_mid.iloc[i - j]) or d["close"].iloc[i - j] <= upper_mid.iloc[i - j]:
                ok_u = False
            if pd.isna(lower_mid.iloc[i - j]) or d["close"].iloc[i - j] >= lower_mid.iloc[i - j]:
                ok_l = False
        walking_upper[i] = ok_u
        walking_lower[i] = ok_l

    price_trend = ema(d["close"], 10) - ema(d["close"], 20)
    band_trend = ema(bandwidth, 10) - ema(bandwidth, 20)
    div_th = 0.3 / 1.5
    bull_div = (price_trend < 0) & (band_trend < 0) & (pct_b < div_th)
    bear_div = (price_trend > 0) & (band_trend < 0) & (pct_b > (1 - div_th))

    vol_avg = sma(d["volume"], 20)
    vol_ratio = d["volume"] / vol_avg.replace(0, np.nan)

    buy_score = np.zeros(n)
    sell_score = np.zeros(n)
    buy_score += np.where(pct_b < 0.2, 2, 0)
    sell_score += np.where(pct_b > 0.8, 2, 0)
    buy_score += np.where(expansion_breakout_bull, 3, 0)
    sell_score += np.where(expansion_breakout_bear, 3, 0)
    buy_score += np.where(bull_div, 2, 0)
    sell_score += np.where(bear_div, 2, 0)
    buy_score += np.where(walking_upper, 1, 0)
    sell_score += np.where(walking_lower, 1, 0)
    both = vol_ratio > 1.5
    buy_score += np.where(both, 1, 0)
    sell_score += np.where(both, 1, 0)

    buy_raw = buy_score >= 4
    sell_raw = sell_score >= 4

    adx_val, _, _ = adx(d, 14)
    vol_filter = d["volume"] >= (vol_avg * 1.2)
    adx_filter = adx_val >= 20

    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    bars_since_signal = 999
    for i in range(n):
        filt = bool(vol_filter.iloc[i]) and bool(adx_filter.iloc[i]) and (bars_since_signal >= 3)
        b = bool(buy_raw[i]) and filt
        s = bool(sell_raw[i]) and filt
        if b:
            buy_sig[i] = True
            bars_since_signal = 0
        elif s:
            sell_sig[i] = True
            bars_since_signal = 0
        else:
            bars_since_signal += 1

    return {"long": pd.Series(buy_sig, index=d.index), "short": pd.Series(sell_sig, index=d.index)}


def signals_boswaves(main_df: pd.DataFrame, aux: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    ys: Dict[str, pd.Series] = {}
    for sym, d in aux.items():
        d2 = d.reindex(main_df.index).ffill()
        adx_v, plus_di, minus_di = adx(d2, 14)
        y = (plus_di - minus_di).clip(-50, 50)
        ys[sym] = y

    # Position mapping 1..8 from script defaults.
    s = BOSWAVES_SYMBOLS
    y1, y2, y3, y4, y5, y6, y7, y8 = [ys[k] for k in s]

    risk_on_bull = (y1 > 0).astype(int) + (y2 > 0).astype(int) + (y4 > 0).astype(int) + (y5 > 0).astype(int) + (y6 > 0).astype(int)
    risk_off_bull = (y3 > 0).astype(int) + (y7 > 0).astype(int) + (y8 > 0).astype(int)
    score = risk_on_bull - risk_off_bull

    long_sig = (score >= 1) & (score.shift(1) < 1)
    short_sig = (score <= -1) & (score.shift(1) > -1)
    long_exit = score < 1
    short_exit = score > -1

    return {
        "long": long_sig.fillna(False),
        "short": short_sig.fillna(False),
        "long_exit": long_exit.fillna(False),
        "short_exit": short_exit.fillna(False),
    }


def signals_prime(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    ema50 = ema(d["close"], 50)
    ema_up = ema50 > ema50.shift(1)
    ema_dn = ema50 < ema50.shift(1)

    hrs = d.index.hour
    in_kill = (hrs >= 1) & (hrs < 23)

    buy_logic = (d["close"] > (d["high"] + d["low"]) / 2.0) & (d["low"] < d["low"].shift(1))
    sell_logic = (d["close"] < (d["high"] + d["low"]) / 2.0) & (d["high"] > d["high"].shift(1))

    long_sig = buy_logic & in_kill & ema_up
    short_sig = sell_logic & in_kill & ema_dn
    return {"long": long_sig.fillna(False), "short": short_sig.fillna(False)}


def signals_extreme_hma(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    length = 33
    sqrtlen = int(round(math.sqrt(length)))
    halflen = int(round(length / 2))

    hma = ema(d["close"], sqrtlen)
    h = hma.rolling(length, min_periods=length).max()
    l = hma.rolling(length, min_periods=length).min()
    hh = h.rolling(halflen, min_periods=halflen).min()
    ll = l.rolling(halflen, min_periods=halflen).max()

    mid = (hh + ll) / 2.0
    a = atr(d, 30) * 0.7
    upper = mid + a
    lower = mid - a

    L = d["close"] > upper
    S = d["close"] < lower

    trend = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        trend[i] = trend[i - 1]
        if bool(L.iloc[i]):
            trend[i] = 1
        elif bool(S.iloc[i]):
            trend[i] = -1

    long_sig = (pd.Series(trend, index=d.index) == 1) & (pd.Series(trend, index=d.index).shift(1) != 1)
    short_sig = (pd.Series(trend, index=d.index) == -1) & (pd.Series(trend, index=d.index).shift(1) != -1)

    return {"long": long_sig.fillna(False), "short": short_sig.fillna(False)}


def signals_gk(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    length = 70
    lag = max(int(math.floor((length - 1) / 2)), 0)
    zsrc = d["close"] + (d["close"] - d["close"].shift(lag))
    zl = ema(zsrc, length)
    a = atr(d, 14)
    up = zl + a * 1.2
    dn = zl - a * 1.2

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
    long_sig = flip & (tr_s == 1)
    short_sig = flip & (tr_s == -1)
    return {"long": long_sig.fillna(False), "short": short_sig.fillna(False)}


def signals_abcd(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()
    ph = pivot_high(d["high"], 5, 5)
    pl = pivot_low(d["low"], 5, 5)

    pivots: List[Tuple[int, float, str]] = []
    active_patterns: List[dict] = []

    long_sig = np.zeros(len(d), dtype=bool)
    short_sig = np.zeros(len(d), dtype=bool)

    tol = 0.035

    for i in range(len(d)):
        # Build alternating pivots.
        new_pivot = None
        if not pd.isna(ph.iloc[i]):
            new_pivot = (i - 5, float(ph.iloc[i]), "H")
        if not pd.isna(pl.iloc[i]):
            if new_pivot is None:
                new_pivot = (i - 5, float(pl.iloc[i]), "L")
            else:
                # Keep the one with larger absolute excursion from prior close.
                if abs(float(pl.iloc[i]) - d["close"].iloc[i]) > abs(float(new_pivot[1]) - d["close"].iloc[i]):
                    new_pivot = (i - 5, float(pl.iloc[i]), "L")

        if new_pivot is not None:
            if pivots and pivots[-1][2] == new_pivot[2]:
                # replace with more extreme pivot
                if (new_pivot[2] == "H" and new_pivot[1] > pivots[-1][1]) or (new_pivot[2] == "L" and new_pivot[1] < pivots[-1][1]):
                    pivots[-1] = new_pivot
            else:
                pivots.append(new_pivot)
                pivots = pivots[-30:]

            if len(pivots) >= 3:
                A, B, C = pivots[-3], pivots[-2], pivots[-1]
                AB = B[1] - A[1]
                BC = C[1] - B[1]
                if AB != 0:
                    ratio = abs(BC / AB)
                    if 0.5 <= ratio <= 0.868:
                        # valid pattern, expect D in direction of AB, then reversal
                        t1 = C[1] + AB * 1.0
                        t2 = C[1] + AB * 1.272
                        t3 = C[1] + AB * 1.618
                        z1 = (min(t1 * (1 - tol), t1 * (1 + tol)), max(t1 * (1 - tol), t1 * (1 + tol)))
                        z2 = (min(t2 * (1 - tol), t2 * (1 + tol)), max(t2 * (1 - tol), t2 * (1 + tol)))
                        z3 = (min(t3 * (1 - tol), t3 * (1 + tol)), max(t3 * (1 - tol), t3 * (1 + tol)))
                        active_patterns.append(
                            {
                                "AB": AB,
                                "C": C[1],
                                "dir": 1 if AB > 0 else -1,
                                "z1": z1,
                                "z2": z2,
                                "z3": z3,
                            }
                        )
                        active_patterns = active_patterns[-8:]

        # Monitor active patterns.
        hi = d["high"].iloc[i]
        lo = d["low"].iloc[i]
        next_patterns = []
        for p in active_patterns:
            hit = False
            if p["dir"] == 1:
                # AB up => D up => SHORT setup
                invalid = lo < p["C"]
                if invalid:
                    continue
                zones = [p["z1"], p["z2"], p["z3"]]
                in_zone = any((hi >= z[0] and hi <= z[1]) for z in zones)
                if in_zone:
                    short_sig[i] = True
                    hit = True
                elif hi > p["z3"][1] * 1.01:
                    hit = True
            else:
                # AB down => D down => LONG setup
                invalid = hi > p["C"]
                if invalid:
                    continue
                zones = [p["z1"], p["z2"], p["z3"]]
                in_zone = any((lo >= z[0] and lo <= z[1]) for z in zones)
                if in_zone:
                    long_sig[i] = True
                    hit = True
                elif lo < p["z3"][0] * 0.99:
                    hit = True

            if not hit:
                next_patterns.append(p)

        active_patterns = next_patterns

    return {"long": pd.Series(long_sig, index=d.index), "short": pd.Series(short_sig, index=d.index)}


def signals_gaussian(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()

    sample_size = 50
    min_slope = 1.5
    vol_sens_adj = 1.0

    base_h = 0.5 + (sample_size * 0.04)
    base_atr = 0.02 + (sample_size * 0.0016)
    final_h = base_h * 0.85
    final_atr_mult = base_atr * 0.9 * 0.8

    yhat = gaussian_regression_max(d["close"], final_h)
    diff = yhat.diff()
    raw_slope = (diff / yhat.replace(0, np.nan)) * 1000.0

    vol_factor = atr(d, 20) / alma(d["close"], 20, 0.7, 4).replace(0, np.nan)
    adaptive_min = min_slope * (vol_factor * 100.0 * vol_sens_adj)
    slope_ok = raw_slope.abs() > adaptive_min

    alma_vol = alma(d["volume"], 20, 0.7, 4)
    vol_ratio = d["volume"] / alma_vol.replace(0, np.nan)
    prog_factor = (1.0 / vol_ratio).pow(0.5)

    dyn_atr_lim = atr(d, 5) * (final_atr_mult * prog_factor)

    trigger_up = diff > dyn_atr_lim
    trigger_dn = diff < -dyn_atr_lim

    tr = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        tr[i] = tr[i - 1]
        if bool(trigger_up.iloc[i]) and bool(slope_ok.iloc[i]):
            tr[i] = 1
        elif bool(trigger_dn.iloc[i]) and bool(slope_ok.iloc[i]):
            tr[i] = -1

    tr_s = pd.Series(tr, index=d.index)
    long_sig = (tr_s == 1) & (tr_s.shift(1) != 1)
    short_sig = (tr_s == -1) & (tr_s.shift(1) != -1)
    return {"long": long_sig.fillna(False), "short": short_sig.fillna(False)}


def signals_safezone(df: pd.DataFrame) -> Dict[str, pd.Series]:
    d = df.copy()

    ema_trend = ema(d["close"], 22)
    uptrend = d["close"] > ema_trend

    trend_switch_up = uptrend & (~uptrend.shift(1).fillna(False))
    trend_switch_dn = (~uptrend) & (uptrend.shift(1).fillna(False))

    downside_pen = np.maximum(d["low"].shift(1) - d["low"], 0.0)
    upside_pen = np.maximum(d["high"] - d["high"].shift(1), 0.0)

    noise_dn = sma(pd.Series(downside_pen, index=d.index), 10)
    noise_up = sma(pd.Series(upside_pen, index=d.index), 10)

    raw_long = d["low"].shift(1) - noise_dn * 2.5
    raw_short = d["high"].shift(1) + noise_up * 2.5

    stop_long = np.full(len(d), np.nan)
    stop_short = np.full(len(d), np.nan)

    for i in range(len(d)):
        if i == 0:
            continue
        if bool(trend_switch_up.iloc[i]):
            stop_long[i] = raw_long.iloc[i]
        elif bool(uptrend.iloc[i]):
            p0 = stop_long[i - 1] if not np.isnan(stop_long[i - 1]) else raw_long.iloc[i]
            p1 = stop_long[i - 2] if i - 2 >= 0 and not np.isnan(stop_long[i - 2]) else raw_long.iloc[i]
            stop_long[i] = np.nanmax([raw_long.iloc[i], p0, p1])
        else:
            stop_long[i] = np.nan

        if bool(trend_switch_dn.iloc[i]):
            stop_short[i] = raw_short.iloc[i]
        elif not bool(uptrend.iloc[i]):
            p0 = stop_short[i - 1] if not np.isnan(stop_short[i - 1]) else raw_short.iloc[i]
            p1 = stop_short[i - 2] if i - 2 >= 0 and not np.isnan(stop_short[i - 2]) else raw_short.iloc[i]
            stop_short[i] = np.nanmin([raw_short.iloc[i], p0, p1])
        else:
            stop_short[i] = np.nan

    stop_long_s = pd.Series(stop_long, index=d.index)
    stop_short_s = pd.Series(stop_short, index=d.index)

    long_exit = uptrend & (d["close"] < stop_long_s)
    short_exit = (~uptrend) & (d["close"] > stop_short_s)

    return {
        "long": trend_switch_up.fillna(False),
        "short": trend_switch_dn.fillna(False),
        "long_exit": long_exit.fillna(False),
        "short_exit": short_exit.fillna(False),
    }


# -----------------------------
# Data Fetch
# -----------------------------

def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows: List[List[float]] = []
    cursor = since_ms
    while cursor < end_ms:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

    if not all_rows:
        raise RuntimeError(f"No OHLCV for {symbol}")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] <= end_ms)].copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# -----------------------------
# Runner
# -----------------------------

def main() -> None:
    start_ms = int(START_UTC.timestamp() * 1000)
    end_ms = int(END_UTC.timestamp() * 1000)

    ex = ccxt.coinbase({"enableRateLimit": True})

    all_symbols = sorted(set(PRIMARY_SYMBOLS + BOSWAVES_SYMBOLS))
    market_data: Dict[str, pd.DataFrame] = {}
    for sym in all_symbols:
        market_data[sym] = clean_df(fetch_ohlcv(ex, sym, TIMEFRAME, start_ms, end_ms))

    strategies = {
        "Jurik_Breakouts": lambda df: signals_jurik(df),
        "LuxAlgo_MSB_OB": lambda df: signals_luxalgo(df),
        "BackQuant_VolSkew": lambda df: signals_volume_skew(df),
        "REH_REL_Breakout": lambda df: signals_reh_rel(df),
        "DAFE_Bands_SmartComposite": lambda df: signals_dafe(df),
        "PrimeUltimate_Sniper": lambda df: signals_prime(df),
        "Extreme_HMA_ATR": lambda df: signals_extreme_hma(df),
        "GK_Trend_Ribbon": lambda df: signals_gk(df),
        "KTY_ABCD": lambda df: signals_abcd(df),
        "Gaussian_MA": lambda df: signals_gaussian(df),
        "SmartSafeZone": lambda df: signals_safezone(df),
    }

    results: List[dict] = []

    for sym in PRIMARY_SYMBOLS:
        df = market_data[sym]

        # Regular single-market strategies.
        for name, fn in strategies.items():
            sig = fn(df)
            bt = run_backtest(
                df,
                sig["long"],
                sig["short"],
                long_exit=sig.get("long_exit"),
                short_exit=sig.get("short_exit"),
                cost_per_side=TRADE_COST_PER_SIDE,
                timeframe=TIMEFRAME,
            )
            results.append(
                {
                    "strategy": name,
                    "symbol": sym,
                    "trades": bt.trades,
                    "win_rate": bt.win_rate,
                    "total_return_pct": bt.total_return_pct,
                    "max_drawdown_pct": bt.max_drawdown_pct,
                    "sharpe": bt.sharpe,
                    "profit_factor": bt.profit_factor,
                }
            )

        # BOSWaves standalone (cross-market).
        sig_bos = signals_boswaves(df, {k: market_data[k] for k in BOSWAVES_SYMBOLS})
        bt_bos = run_backtest(
            df,
            sig_bos["long"],
            sig_bos["short"],
            long_exit=sig_bos.get("long_exit"),
            short_exit=sig_bos.get("short_exit"),
            cost_per_side=TRADE_COST_PER_SIDE,
            timeframe=TIMEFRAME,
        )
        results.append(
            {
                "strategy": "BOSWaves_Standalone",
                "symbol": sym,
                "trades": bt_bos.trades,
                "win_rate": bt_bos.win_rate,
                "total_return_pct": bt_bos.total_return_pct,
                "max_drawdown_pct": bt_bos.max_drawdown_pct,
                "sharpe": bt_bos.sharpe,
                "profit_factor": bt_bos.profit_factor,
            }
        )

    res_df = pd.DataFrame(results)

    summary = (
        res_df.groupby("strategy", as_index=False)
        .agg(
            symbols=("symbol", "nunique"),
            avg_trades=("trades", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_return_pct=("total_return_pct", "mean"),
            avg_max_dd_pct=("max_drawdown_pct", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
        )
    )

    summary["robust_score"] = (
        summary["avg_return_pct"]
        + (summary["avg_sharpe"] * 12.0)
        + (summary["avg_profit_factor"] * 4.0)
        - (summary["avg_max_dd_pct"] * 0.75)
    )

    summary = summary.sort_values("robust_score", ascending=False).reset_index(drop=True)

    out_dir = "analysis/results"
    pd.Series(dtype=float)  # no-op to keep lint quiet in simple runtime
    import os

    os.makedirs(out_dir, exist_ok=True)
    res_path = f"{out_dir}/per_symbol_results.csv"
    sum_path = f"{out_dir}/summary_ranked.csv"
    meta_path = f"{out_dir}/run_meta.json"

    res_df.to_csv(res_path, index=False)
    summary.to_csv(sum_path, index=False)

    meta = {
        "timeframe": TIMEFRAME,
        "start_utc": START_UTC.isoformat(),
        "end_utc": END_UTC.isoformat(),
        "symbols": PRIMARY_SYMBOLS,
        "boswaves_symbols": BOSWAVES_SYMBOLS,
        "cost_per_side": TRADE_COST_PER_SIDE,
        "note": "Signal mappings are deterministic approximations of indicator logic for cross-strategy comparability.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=== Ranked Summary (Top 8) ===")
    print(summary.head(8).to_string(index=False))
    print("\nWrote:")
    print(f"- {res_path}")
    print(f"- {sum_path}")
    print(f"- {meta_path}")


if __name__ == "__main__":
    main()
