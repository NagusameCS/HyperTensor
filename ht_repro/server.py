"""
ht-repro localhost server — API / SSE / test queue / GPU config / stop.
Pure stdlib. No Flask, no dependencies.
"""
import json, os, queue, subprocess, sys, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from .catalog import load_catalog, find_test
from .runner import load_results, save_results

ROOT = Path(__file__).resolve().parent.parent.parent
THIS_DIR = Path(__file__).resolve().parent

# State
_queues = []
_proc = None
_proc_id = None
_lock = threading.Lock()
_gpu = {"device": "auto", "vram_limit_gb": 0, "warn_baseline_pct": 75.0}

def broadcast(event, data):
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead = []; _ = [dead.append(q) if q.full() else q.put_nowait(payload) for q in _queues]
    for q in dead: _queues.remove(q)

def gpu_check():
    try:
        import pynvml; pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h); pynvml.nvmlShutdown()
        p = (m.used/m.total)*100
        if u.gpu > _gpu["warn_baseline_pct"] or p > 50:
            return {"warn": True, "gpu_pct": u.gpu, "vram_pct": round(p,1),
                    "msg": f"GPU at {u.gpu}% util / {p:.0f}% VRAM — background usage may affect benchmark accuracy."}
        return {"warn": False, "gpu_pct": u.gpu, "vram_pct": round(p,1)}
    except: return {"warn": False, "gpu_pct": 0, "vram_pct": 0}

def run_test(tid):
    global _proc, _proc_id
    t = find_test(tid)
    if not t: broadcast("error", {"test_id": tid, "msg": "Not found"}); return

    g = gpu_check()
    if g.get("warn"): broadcast("gpu_warn", g)

    broadcast("test_start", {"test_id": tid, "name": t["name"], "tier": t["tier"], "paper": t["paper"]})
    sp = ROOT / t["script"]
    if not sp.exists(): broadcast("test_error", {"test_id": tid, "msg": f"Script missing: {sp}"}); return

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT); env["PYTHONUNBUFFERED"] = "1"
    if _gpu["device"] != "auto": env["CUDA_VISIBLE_DEVICES"] = str(_gpu["device"])

    try:
        proc = subprocess.Popen([sys.executable, "-u", str(sp)], cwd=str(ROOT),
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        _proc = proc; _proc_id = tid
        broadcast("status", {"running": True, "test_id": tid, "pid": proc.pid})
        lines = []
        for ln in iter(proc.stdout.readline, ""):
            if not ln: break
            l = ln.rstrip(); lines.append(l)
            broadcast("test_output", {"test_id": tid, "line": l})
        proc.wait(); elapsed = time.time() - t0
        out = "\n".join(lines)
        ok = proc.returncode == 0
        exp = t.get("desc","")
        if exp and exp.lower() in out.lower(): ok = True
        broadcast("test_done", {"test_id": tid, "passed": ok, "time": round(elapsed,2), "output": out[-3000:]})
        r = load_results(); r.setdefault("runs",[]).append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tests": {tid: {"status": "pass" if ok else "fail", "time": elapsed}},
            "passed": 1 if ok else 0, "failed": 0 if ok else 1,
            "skipped": 0, "total_time": elapsed,
        }); save_results(r)
    except Exception as e:
        broadcast("test_error", {"test_id": tid, "msg": str(e), "time": time.time()-t0})
    finally:
        _proc = None; _proc_id = None
        broadcast("status", {"running": False, "test_id": None, "pid": None})

def stop():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try: _proc.wait(timeout=3)
        except subprocess.TimeoutExpired: _proc.kill()
        broadcast("test_stopped", {"test_id": _proc_id})
        return True
    return False

# Handler
_UI = None
def ui():
    global _UI
    if _UI is None:
        p = THIS_DIR / "server_ui.html"
        _UI = p.read_text(encoding="utf-8") if p.exists() else "<h1>UI missing</h1>"
    return _UI

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _j(self, d, c=200):
        self.send_response(c); self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
        self.wfile.write(json.dumps(d).encode())
    def _h(self, h, c=200):
        self.send_response(c); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers()
        self.wfile.write(h.encode())

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/catalog": return self._j(load_catalog())
        if p == "/api/env":
            from .setup_wizard import run_setup as env_report
            r = env_report(interactive=False); r["gpu_config"] = _gpu; return self._j(r)
        if p == "/api/results": return self._j(load_results())
        if p == "/api/gpu_check": return self._j(gpu_check())
        if p == "/api/status": return self._j({"running": _proc is not None, "test_id": _proc_id})

        if p.startswith("/api/gpu_config"):
            qs = self.path.split("?")[-1] if "?" in self.path else ""
            for kv in qs.split("&"):
                if "=" in kv:
                    k,v = kv.split("=",1)
                    if k in _gpu:
                        try: _gpu[k] = float(v) if "." in v else int(v)
                        except: _gpu[k] = v
            return self._j({"ok": True, "gpu_config": _gpu})

        if p == "/api/stream":
            self.send_response(200); self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache"); self.send_header("Connection","keep-alive")
            self.send_header("Access-Control-Allow-Origin","*"); self.end_headers()
            q = queue.Queue(maxsize=64); _queues.append(q)
            try:
                self.wfile.write(b"event: connected\ndata: {}\n\n"); self.wfile.flush()
                while True:
                    try: self.wfile.write(q.get(timeout=15).encode()); self.wfile.flush()
                    except queue.Empty: self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError): pass
            finally:
                if q in _queues: _queues.remove(q)
            return

        if p.startswith("/api/run/"):
            tid = p.split("/api/run/")[1]
            if _proc and _proc.poll() is None: return self._j({"error": "already_running", "test_id": _proc_id}, 409)
            threading.Thread(target=run_test, args=(tid,), daemon=True).start()
            return self._j({"status": "started", "test_id": tid})

        if p == "/api/run_all":
            tier = "T1"
            if "?" in self.path:
                for kv in self.path.split("?")[1].split("&"):
                    if kv.startswith("tier="): tier = kv[5:]
            if _proc and _proc.poll() is None: return self._j({"error": "already_running"}, 409)
            tests = [t for t in load_catalog() if t["tier"] == tier]
            def seq():
                for tst in tests: run_test(tst["id"])
            threading.Thread(target=seq, daemon=True).start()
            return self._j({"status": "started", "count": len(tests), "tier": tier})

        if p == "/api/stop":
            return self._j({"stopped": stop()})

        if p == "/" or p == "/index.html": return self._h(ui())
        self.send_response(404); self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","*"); self.end_headers()

def start_server(port=8765, open_browser=True):
    srv = HTTPServer(("127.0.0.1", port), H)
    print(f"\n  ht-repro server — http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    if open_browser:
        import webbrowser; threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\n  Shutting down..."); stop(); srv.shutdown()
