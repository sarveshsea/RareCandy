#!/usr/bin/env python3
"""
Rare Candy dashboard server.

Features:
- No-cache responses (mobile browsers keep stale text otherwise).
- Live logs page.
- Quant dashboard with TradingView-style candle chart + trade markers.
- Lightweight JSON APIs for status, candles, trades, events, and performance.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import ccxt


DASHBOARD_DIR = Path(os.getenv("DASHBOARD_DIR", "dashboard")).resolve()
PORT = int(os.getenv("DASHBOARD_PORT", "8000"))
MAX_LOG_BYTES = int(os.getenv("DASHBOARD_MAX_LOG_BYTES", "300000"))
PAPER_STATE_FILE = Path(os.getenv("PAPER_STATE_FILE", "paper_state.json")).resolve()
PAPER_START_EQUITY = float(os.getenv("PAPER_START_EQUITY", "10000"))

EX = ccxt.coinbase({"enableRateLimit": True})
_CANDLE_CACHE: dict[tuple[str, str, int], tuple[float, list[dict]]] = {}


def _to_epoch_seconds(ts: str) -> Optional[int]:
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return int(t.timestamp())
    except Exception:
        return None


def _read_text_tail(path: Path, max_bytes: int = MAX_LOG_BYTES) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
        first_nl = data.find(b"\n")
        if first_nl != -1:
            data = data[first_nl + 1 :]
    return data.decode("utf-8", errors="replace")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path, fallback: dict | list) -> dict | list:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _fetch_candles(symbol: str, timeframe: str, limit: int) -> list[dict]:
    key = (symbol, timeframe, limit)
    now = time.time()
    cached = _CANDLE_CACHE.get(key)
    if cached and now - cached[0] < 8:
        return cached[1]

    rows = EX.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    out = [
        {
            "time": int(r[0] / 1000),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        }
        for r in rows
    ]
    _CANDLE_CACHE[key] = (now, out)
    return out


def _paper_performance() -> dict:
    state = _read_json(PAPER_STATE_FILE, {"balance": {"USD": PAPER_START_EQUITY}, "positions": {}, "history": []})
    bal = state.get("balance", {}) if isinstance(state, dict) else {}
    positions = state.get("positions", {}) if isinstance(state, dict) else {}
    cash = float(bal.get("USD", 0.0))

    mark_value = 0.0
    marks = {}
    for sym, qty in positions.items():
        q = float(qty)
        if q <= 0:
            continue
        price = None
        try:
            t = EX.fetch_ticker(sym)
            price = float(t.get("last") or 0.0)
        except Exception:
            price = 0.0
        mark_value += q * price
        marks[sym] = {"qty": q, "mark": price, "value": q * price}

    equity = cash + mark_value
    pnl = equity - PAPER_START_EQUITY
    pnl_pct = (pnl / PAPER_START_EQUITY * 100.0) if PAPER_START_EQUITY > 0 else 0.0

    return {
        "cash_usd": cash,
        "positions": positions,
        "marks": marks,
        "equity_estimate": equity,
        "start_equity": PAPER_START_EQUITY,
        "pnl_usd": pnl,
        "pnl_pct": pnl_pct,
    }


def _events(limit: int) -> list[dict]:
    lines = _read_text_tail(DASHBOARD_DIR / "events.log").splitlines()
    out = []
    for line in lines[-limit:]:
        # [timestamp] [TYPE] message
        if not line.startswith("["):
            continue
        try:
            p1 = line.find("]")
            ts = line[1:p1]
            rem = line[p1 + 2 :]
            p2 = rem.find("]")
            ev_type = rem[1:p2]
            msg = rem[p2 + 2 :]
            out.append({"timestamp": ts, "type": ev_type, "message": msg})
        except Exception:
            out.append({"timestamp": "", "type": "RAW", "message": line})
    return out


def _trades(symbol: str, limit: int) -> list[dict]:
    state = _read_json(PAPER_STATE_FILE, {"history": []})
    hist = state.get("history", []) if isinstance(state, dict) else []
    out = []
    for t in hist:
        if not isinstance(t, dict):
            continue
        if symbol and t.get("symbol") != symbol:
            continue
        ts = t.get("timestamp", "")
        epoch = _to_epoch_seconds(ts)
        out.append(
            {
                "timestamp": ts,
                "time": epoch,
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "quantity": t.get("quantity"),
                "price": t.get("price"),
                "fee": t.get("fee"),
            }
        )
    return out[-limit:]


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RareCandyDashboard/2.0"

    def _send_headers(self, code: int, content_type: str, content_len: Optional[int] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        if content_len is not None:
            self.send_header("Content-Length", str(content_len))
        self.end_headers()

    def _send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", code: int = 200) -> None:
        body = text.encode("utf-8", errors="replace")
        self._send_headers(code, content_type, len(body))
        self.wfile.write(body)

    def _send_json(self, obj: dict | list, code: int = 200) -> None:
        self._send_text(json.dumps(obj, indent=2), "application/json; charset=utf-8", code)

    def _logs_page(self) -> str:
        return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rare Candy Logs</title>
  <style>
    body { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #0b0f14; color: #d9e2ec; }
    .top { position: sticky; top: 0; background: #111827; padding: 10px 12px; border-bottom: 1px solid #1f2937; }
    .top a { color: #93c5fd; text-decoration: none; margin-right: 12px; }
    .meta { color: #9ca3af; font-size: 12px; }
    pre { margin: 0; padding: 12px; white-space: pre-wrap; word-break: break-word; line-height: 1.35; }
  </style>
</head>
<body>
  <div class="top">
    <a href="/quant">quant dashboard</a>
    <a href="/status.json">status.json</a>
    <a href="/terminal.log">terminal.log</a>
    <a href="/events.log">events.log</a>
    <span class="meta" id="meta">loading...</span>
  </div>
  <pre id="log">Loading logs...</pre>
  <script>
    const el = document.getElementById('log');
    const meta = document.getElementById('meta');
    let autoScroll = true;
    window.addEventListener('scroll', () => {
      const nearBottom = (window.innerHeight + window.scrollY) >= (document.body.offsetHeight - 120);
      autoScroll = nearBottom;
    });
    async function tick() {
      const t = Date.now();
      try {
        const [logRes, stRes] = await Promise.all([
          fetch('/terminal.log?ts=' + t, { cache: 'no-store' }),
          fetch('/status.json?ts=' + t, { cache: 'no-store' }),
        ]);
        const logText = await logRes.text();
        el.textContent = logText || '(terminal.log is empty)';
        if (stRes.ok) {
          const st = await stRes.json();
          meta.textContent = `last_update=${st.last_update || 'n/a'} healthy=${st.healthy}`;
        } else {
          meta.textContent = 'status unavailable';
        }
        if (autoScroll) window.scrollTo(0, document.body.scrollHeight);
      } catch (e) {
        meta.textContent = 'fetch error';
      }
    }
    tick();
    setInterval(tick, 2000);
  </script>
</body>
</html>
"""

    def _quant_page(self) -> str:
        return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rare Candy Quant Dashboard</title>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {
      --bg: #0b0f14;
      --panel: #111827;
      --panel2: #0f172a;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --green: #00c17c;
      --red: #ff4d6d;
      --line: #1f2937;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 14px; }
    .top { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    .top a { color: #93c5fd; text-decoration: none; margin-right: 8px; font-size: 13px; }
    .select, .btn { background: var(--panel); color: var(--text); border: 1px solid var(--line); padding: 8px 10px; border-radius: 8px; }
    .grid { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; margin-bottom: 12px; }
    .card { background: linear-gradient(180deg, var(--panel), var(--panel2)); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-size: 18px; font-weight: 700; margin-top: 4px; }
    #chart { height: 460px; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    .bottom { margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .tblWrap { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: var(--panel2); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 8px; border-bottom: 1px solid #1a2433; text-align: left; }
    th { color: var(--muted); font-weight: 600; background: #0d1523; position: sticky; top: 0; }
    tbody tr:hover { background: #111b2b; }
    .g { color: var(--green); } .r { color: var(--red); }
    @media (max-width: 980px) { .grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); } .bottom { grid-template-columns: 1fr; } #chart { height: 360px; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <select id="symbol" class="select">
        <option>BTC/USD</option>
        <option>ETH/USD</option>
      </select>
      <select id="timeframe" class="select">
        <option value="15m" selected>15m</option>
        <option value="1h">1h</option>
        <option value="5m">5m</option>
      </select>
      <button id="refresh" class="btn">Refresh</button>
      <a href="/logs">logs</a>
      <a href="/status.json">status.json</a>
      <a href="/terminal.log">terminal.log</a>
      <span id="meta" style="color:var(--muted);font-size:12px"></span>
    </div>

    <div class="grid">
      <div class="card"><div class="label">Estimated Equity</div><div class="value" id="eq">$-</div></div>
      <div class="card"><div class="label">PnL</div><div class="value" id="pnl">$-</div></div>
      <div class="card"><div class="label">PnL %</div><div class="value" id="pnlp">-%</div></div>
      <div class="card"><div class="label">Open Positions</div><div class="value" id="pos">-</div></div>
      <div class="card"><div class="label">Healthy</div><div class="value" id="healthy">-</div></div>
    </div>

    <div id="chart"></div>

    <div class="bottom">
      <div class="tblWrap">
        <table>
          <thead><tr><th colspan="6">Recent Trades</th></tr><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Fee</th></tr></thead>
          <tbody id="tradesBody"></tbody>
        </table>
      </div>
      <div class="tblWrap">
        <table>
          <thead><tr><th colspan="3">Recent Events</th></tr><tr><th>Time</th><th>Type</th><th>Message</th></tr></thead>
          <tbody id="eventsBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const symbolSel = document.getElementById('symbol');
    const tfSel = document.getElementById('timeframe');
    const refreshBtn = document.getElementById('refresh');
    const meta = document.getElementById('meta');
    const chartEl = document.getElementById('chart');
    const tradesBody = document.getElementById('tradesBody');
    const eventsBody = document.getElementById('eventsBody');

    const chart = LightweightCharts.createChart(chartEl, {
      layout: { background: { color: '#0b0f14' }, textColor: '#d1d5db' },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      rightPriceScale: { borderColor: '#1f2937' },
      timeScale: { borderColor: '#1f2937', timeVisible: true, secondsVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    const candles = chart.addCandlestickSeries({
      upColor: '#00c17c', downColor: '#ff4d6d', borderVisible: false,
      wickUpColor: '#00c17c', wickDownColor: '#ff4d6d'
    });
    const vol = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: '',
      scaleMargins: { top: 0.8, bottom: 0 },
      color: '#334155'
    });

    function money(v) { return '$' + Number(v || 0).toLocaleString(undefined, {maximumFractionDigits: 2}); }
    function pct(v) { return Number(v || 0).toFixed(2) + '%'; }
    function clsBySign(v) { return v >= 0 ? 'g' : 'r'; }

    async function load() {
      const sym = symbolSel.value;
      const tf = tfSel.value;
      const ts = Date.now();
      meta.textContent = 'loading...';
      try {
        const [cRes, tRes, sRes, pRes, eRes] = await Promise.all([
          fetch(`/api/candles?symbol=${encodeURIComponent(sym)}&timeframe=${encodeURIComponent(tf)}&limit=350&ts=${ts}`, {cache: 'no-store'}),
          fetch(`/api/trades?symbol=${encodeURIComponent(sym)}&limit=120&ts=${ts}`, {cache: 'no-store'}),
          fetch(`/status.json?ts=${ts}`, {cache: 'no-store'}),
          fetch(`/api/performance?ts=${ts}`, {cache: 'no-store'}),
          fetch(`/api/events?limit=80&ts=${ts}`, {cache: 'no-store'}),
        ]);

        const cObj = await cRes.json();
        const tObj = await tRes.json();
        const sObj = await sRes.json();
        const pObj = await pRes.json();
        const eObj = await eRes.json();

        const cData = cObj.candles || [];
        candles.setData(cData.map(x => ({time:x.time, open:x.open, high:x.high, low:x.low, close:x.close})));
        vol.setData(cData.map(x => ({time:x.time, value:x.volume, color: x.close >= x.open ? 'rgba(0,193,124,0.4)' : 'rgba(255,77,109,0.4)'})));

        const trades = tObj.trades || [];
        const markers = [];
        for (const t of trades) {
          if (!t.time) continue;
          markers.push({
            time: t.time,
            position: t.side === 'BUY' ? 'belowBar' : 'aboveBar',
            color: t.side === 'BUY' ? '#00c17c' : '#ff4d6d',
            shape: t.side === 'BUY' ? 'arrowUp' : 'arrowDown',
            text: `${t.side} ${Number(t.quantity).toFixed(4)}`
          });
        }
        candles.setMarkers(markers);
        chart.timeScale().fitContent();

        document.getElementById('eq').textContent = money(pObj.equity_estimate);
        document.getElementById('pnl').textContent = money(pObj.pnl_usd);
        document.getElementById('pnl').className = 'value ' + clsBySign(pObj.pnl_usd || 0);
        document.getElementById('pnlp').textContent = pct(pObj.pnl_pct);
        document.getElementById('pnlp').className = 'value ' + clsBySign(pObj.pnl_pct || 0);
        document.getElementById('pos').textContent = Object.keys((pObj.positions || {})).length;
        document.getElementById('healthy').textContent = String(sObj.healthy);
        document.getElementById('healthy').className = 'value ' + (sObj.healthy ? 'g' : 'r');

        tradesBody.innerHTML = '';
        for (const t of trades.slice().reverse().slice(0, 40)) {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${t.timestamp || ''}</td><td>${t.symbol || ''}</td><td class='${t.side === 'BUY' ? 'g' : 'r'}'>${t.side || ''}</td><td>${Number(t.quantity || 0).toFixed(6)}</td><td>${Number(t.price || 0).toFixed(2)}</td><td>${Number(t.fee || 0).toFixed(2)}</td>`;
          tradesBody.appendChild(tr);
        }

        eventsBody.innerHTML = '';
        for (const ev of (eObj.events || []).slice().reverse().slice(0, 60)) {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${ev.timestamp || ''}</td><td>${ev.type || ''}</td><td>${ev.message || ''}</td>`;
          eventsBody.appendChild(tr);
        }

        meta.textContent = `updated ${sObj.last_update || ''}`;
      } catch (err) {
        meta.textContent = 'load error';
      }
    }

    refreshBtn.addEventListener('click', load);
    symbolSel.addEventListener('change', load);
    tfSel.addEventListener('change', load);
    window.addEventListener('resize', () => chart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight }));
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
"""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)

        if path in ("/", "/index.html", "/quant"):
            self._send_text(self._quant_page(), "text/html; charset=utf-8")
            return

        if path in ("/logs", "/logs.html"):
            self._send_text(self._logs_page(), "text/html; charset=utf-8")
            return

        if path == "/status.json":
            p = DASHBOARD_DIR / "status.json"
            if not p.exists():
                self._send_json({"healthy": False, "error": "status.json not found"}, 404)
                return
            self._send_text(_read_text(p), "application/json; charset=utf-8")
            return

        if path == "/terminal.log":
            self._send_text(_read_text_tail(DASHBOARD_DIR / "terminal.log"))
            return

        if path == "/events.log":
            self._send_text(_read_text_tail(DASHBOARD_DIR / "events.log"))
            return

        if path == "/api/candles":
            symbol = q.get("symbol", ["BTC/USD"])[0]
            timeframe = q.get("timeframe", ["15m"])[0]
            limit = int(q.get("limit", ["300"])[0])
            limit = max(50, min(limit, 1500))
            try:
                rows = _fetch_candles(symbol, timeframe, limit)
                self._send_json({"symbol": symbol, "timeframe": timeframe, "candles": rows})
            except Exception as e:
                self._send_json({"error": f"candle fetch failed: {type(e).__name__}"}, 500)
            return

        if path == "/api/trades":
            symbol = q.get("symbol", [""])[0]
            limit = int(q.get("limit", ["200"])[0])
            limit = max(1, min(limit, 1000))
            self._send_json({"symbol": symbol, "trades": _trades(symbol, limit)})
            return

        if path == "/api/events":
            limit = int(q.get("limit", ["200"])[0])
            limit = max(1, min(limit, 1000))
            self._send_json({"events": _events(limit)})
            return

        if path == "/api/performance":
            self._send_json(_paper_performance())
            return

        if path == "/health":
            self._send_json({"ok": True}, 200)
            return

        self._send_text("Not Found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"Dashboard server running on :{PORT}, dir={DASHBOARD_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
