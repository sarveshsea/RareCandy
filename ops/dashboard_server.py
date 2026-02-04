#!/usr/bin/env python3
"""
Tiny dashboard server with no-cache headers and live log page.

Why this exists:
- `python -m http.server` can be aggressively cached by mobile browsers.
- This server disables caching and provides a `/logs` page that auto-refreshes.
"""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


DASHBOARD_DIR = Path(os.getenv("DASHBOARD_DIR", "dashboard")).resolve()
PORT = int(os.getenv("DASHBOARD_PORT", "8000"))
MAX_LOG_BYTES = int(os.getenv("DASHBOARD_MAX_LOG_BYTES", "250000"))


def _read_text_tail(path: Path, max_bytes: int = MAX_LOG_BYTES) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
        # avoid returning a partial first line
        first_nl = data.find(b"\n")
        if first_nl != -1:
            data = data[first_nl + 1 :]
    return data.decode("utf-8", errors="replace")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RareCandyDashboard/1.0"

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

    def _send_json(self, obj: dict, code: int = 200) -> None:
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

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            html = (
                "<html><body style='font-family:system-ui;padding:16px'>"
                "<h3>Rare Candy Dashboard</h3>"
                "<ul>"
                "<li><a href='/status.json'>status.json</a></li>"
                "<li><a href='/logs'>logs (live)</a></li>"
                "<li><a href='/terminal.log'>terminal.log (raw)</a></li>"
                "<li><a href='/events.log'>events.log (raw)</a></li>"
                "</ul>"
                "</body></html>"
            )
            self._send_text(html, "text/html; charset=utf-8")
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

        if path == "/health":
            self._send_json({"ok": True}, 200)
            return

        self._send_text("Not Found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        # Keep stdout quiet; container log already has bot activity.
        return


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"Dashboard server running on :{PORT}, dir={DASHBOARD_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
