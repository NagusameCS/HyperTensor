"""
Live web server for ht-repro — serves an Anthropic-style dashboard
with real-time test execution, streaming output, and result history.

Uses only stdlib (http.server) — no Flask, no dependencies.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

from .catalog import load_catalog, find_test, EXPECTED_OUTPUTS
from .runner import load_results, save_results, has_gpu
from .setup_wizard import run_setup as get_env_report

ROOT = Path(__file__).resolve().parent.parent.parent
THIS_DIR = Path(__file__).resolve().parent

# ── SSE Event Bus ──────────────────────────────────────────────────
_event_queues: list = []  # List of queue.Queue for connected clients

def broadcast_event(event: str, data: dict):
    """Send an SSE event to all connected clients."""
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for q in _event_queues:
        try:
            q.put_nowait(payload)
        except queue.Full:
            dead.append(q)
    for q in dead:
        _event_queues.remove(q)

# ── Live Test Runner (background thread) ───────────────────────────
def run_test_live(test_id: str):
    """Run a test in background, streaming output line-by-line via SSE."""
    test = find_test(test_id)
    if not test:
        broadcast_event("error", {"test_id": test_id, "msg": "Test not found"})
        return

    broadcast_event("test_start", {"test_id": test_id, "name": test["name"]})

    script_path = ROOT / test["script"]
    if not script_path.exists():
        broadcast_event("test_error", {"test_id": test_id, "msg": f"Script not found: {script_path}"})
        return

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        output_lines = []
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            line = line.rstrip()
            output_lines.append(line)
            broadcast_event("test_output", {"test_id": test_id, "line": line})

        proc.wait()
        elapsed = time.time() - t0
        output = "\n".join(output_lines)
        passed = proc.returncode == 0

        # Check expected
        expected = test.get("desc", "")
        if expected and expected.lower() in output.lower():
            passed = True

        broadcast_event("test_done", {
            "test_id": test_id,
            "passed": passed,
            "time": round(elapsed, 2),
            "output": output[-3000:],
        })

        # Save to results
        results = load_results()
        run_record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tests": {test_id: {"status": "pass" if passed else "fail", "time": elapsed}},
            "passed": 1 if passed else 0, "failed": 0 if passed else 1,
            "skipped": 0, "total_time": elapsed,
        }
        results["runs"].append(run_record)
        save_results(results)

    except Exception as e:
        elapsed = time.time() - t0
        broadcast_event("test_error", {"test_id": test_id, "msg": str(e), "time": elapsed})

# ── Server ─────────────────────────────────────────────────────────

_INDEX_HTML = None

def get_index_html() -> str:
    global _INDEX_HTML
    if _INDEX_HTML is None:
        path = THIS_DIR / "server_ui.html"
        if path.exists():
            _INDEX_HTML = path.read_text(encoding="utf-8")
        else:
            _INDEX_HTML = _FALLBACK_HTML
    return _INDEX_HTML

class Handler(BaseHTTPRequestHandler):
    """HTTP handler for ht-repro server."""

    def log_message(self, format, *args):
        pass  # Silent

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html(self, html, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        path = self.path.split("?")[0]

        # ── API Routes ──
        if path == "/api/catalog":
            self._send_json(load_catalog())
            return

        if path == "/api/env":
            report = get_env_report(interactive=False)
            self._send_json(report)
            return

        if path == "/api/results":
            results = load_results()
            self._send_json(results)
            return

        if path == "/api/status":
            results = load_results()
            runs = results.get("runs", [])
            last = runs[-1] if runs else None
            passed = sum(r["passed"] for r in runs) if runs else 0
            failed = sum(r["failed"] for r in runs) if runs else 0
            skipped = sum(r["skipped"] for r in runs) if runs else 0
            self._send_json({
                "total_runs": len(runs),
                "total_passed": passed,
                "total_failed": failed,
                "total_skipped": skipped,
                "last_run": last,
            })
            return

        if path == "/api/stream":
            # SSE endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = queue.Queue(maxsize=64)
            _event_queues.append(q)
            try:
                # Send initial connected event
                self.wfile.write(f"event: connected\ndata: {{}}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                    except queue.Empty:
                        # Send keepalive
                        self.wfile.write(f": keepalive\n\n".encode())
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if q in _event_queues:
                    _event_queues.remove(q)
            return

        # ── Action Routes ──
        if path.startswith("/api/run/"):
            test_id = path.split("/api/run/")[1]
            t = threading.Thread(target=run_test_live, args=(test_id,), daemon=True)
            t.start()
            self._send_json({"status": "started", "test_id": test_id})
            return

        if path == "/api/run-all":
            tier = self.path.split("tier=")[-1] if "tier=" in self.path else "T1"
            catalog = [t for t in load_catalog() if t["tier"] == tier]
            def run_all():
                for t in catalog:
                    run_test_live(t["id"])
                    time.sleep(0.2)
            t = threading.Thread(target=run_all, daemon=True)
            t.start()
            self._send_json({"status": "started", "count": len(catalog), "tier": tier})
            return

        # ── Static UI ──
        if path == "/" or path == "/index.html":
            self._send_html(get_index_html())
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

def start_server(port: int = 8765, open_browser: bool = True):
    """Start the ht-repro localhost server."""
    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"

    print(f"\n  {'─'*50}")
    print(f"  ht-repro server running at {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"  {'─'*50}\n")

    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()

# ── Fallback HTML (in case server_ui.html is missing) ──────────────
_FALLBACK_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>ht-repro</title></head>
<body style="background:#141413;color:#e8e6e3;font-family:system-ui;padding:40px">
<h1>ht-repro Server</h1><p>UI file not found. API available at /api/catalog</p>
</body></html>"""
