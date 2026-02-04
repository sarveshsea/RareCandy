#!/usr/bin/env python3
"""
Continuous monitor for adaptive regime performance.

Runs the adaptive selector, evaluates headline KPIs, and writes a status JSON.
Can run once or loop forever (for server monitoring).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Dict, List

import pandas as pd

import adaptive_regime_selector as ars
import backtest_pine_walkforward as wf


def evaluate_once() -> Dict:
    data = ars.load_or_fetch_all()
    bos_data = {s: data[s] for s in ars.BOS_SYMBOLS}

    rows: List[Dict] = []
    for sym in ars.SYMBOLS:
        df = data[sym]
        _, _, score = wf.bos_filter(df.index, bos_data)
        res, _, _ = ars.run_selector_for_symbol(df, score, ars.CFG)
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

    mdf = pd.DataFrame(rows)
    return {
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": ars.CFG,
        "per_symbol": rows,
        "avg_return_pct": float(mdf["total_return_pct"].mean()),
        "avg_max_dd_pct": float(mdf["max_drawdown_pct"].mean()),
        "avg_sharpe": float(mdf["sharpe"].mean()),
        "avg_profit_factor": float(mdf["profit_factor"].mean()),
        "avg_win_rate": float(mdf["win_rate"].mean()),
        "avg_trades": float(mdf["trades"].mean()),
    }


def check_thresholds(report: Dict, min_avg_return: float, max_avg_dd: float, min_avg_sharpe: float) -> Dict:
    checks = {
        "avg_return_ok": report["avg_return_pct"] >= min_avg_return,
        "avg_dd_ok": report["avg_max_dd_pct"] <= max_avg_dd,
        "avg_sharpe_ok": report["avg_sharpe"] >= min_avg_sharpe,
    }
    checks["all_ok"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor adaptive regime strategy health.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument("--interval-sec", type=int, default=900, help="Loop interval in seconds.")
    parser.add_argument("--min-avg-return", type=float, default=0.0, help="Minimum acceptable avg return %%")
    parser.add_argument("--max-avg-dd", type=float, default=12.0, help="Maximum acceptable avg drawdown %%")
    parser.add_argument("--min-avg-sharpe", type=float, default=0.0, help="Minimum acceptable avg Sharpe")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "adaptive_monitor_status.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    while True:
        report = evaluate_once()
        checks = check_thresholds(report, args.min_avg_return, args.max_avg_dd, args.min_avg_sharpe)
        payload = {"report": report, "checks": checks}

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        status = "PASS" if checks["all_ok"] else "FAIL"
        print(
            f"[{report['run_utc']}] {status} | "
            f"ret={report['avg_return_pct']:.2f}% dd={report['avg_max_dd_pct']:.2f}% "
            f"sh={report['avg_sharpe']:.2f} pf={report['avg_profit_factor']:.2f}"
        )

        if not args.loop:
            break
        time.sleep(max(1, args.interval_sec))


if __name__ == "__main__":
    main()
