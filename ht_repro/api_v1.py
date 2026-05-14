"""
ht_repro.api_v1 — REST API v1 handlers, mounted by server.py.

Endpoints:
  POST /api/v1/infer       — run inference on a cached model (text-only stub)
  POST /api/v1/graft       — schedule a graft job (donor, recipient, layers)
  POST /api/v1/compress    — schedule a compression job
  POST /api/v1/sort        — sort a list using hypersort (delegates to hypersort if present)
  GET  /api/v1/models      — list cached models
  POST /api/v1/models/pull — { "repo_id": "..." }
  GET  /api/v1/gtc/<model> — return recent GTC trajectories
  GET  /api/v1/gpu         — backend / device summary
  GET  /api/v1/health      — health probe

Auth:
  If HT_REPRO_TOKEN env var is set, requests must carry
  `Authorization: Bearer <token>` or `?token=<token>`.
  Otherwise (local dev) auth is disabled.
"""
from __future__ import annotations
import json, os, threading, time, uuid
from typing import Any

from . import gpu, models, storage

TOKEN = os.environ.get("HT_REPRO_TOKEN", "").strip()
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _authorized(headers, qs: str) -> bool:
    if not TOKEN: return True
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:].strip() == TOKEN:
        return True
    for kv in qs.split("&"):
        if kv.startswith("token=") and kv[6:] == TOKEN:
            return True
    return False


def _job(kind: str, payload: dict) -> dict:
    jid = uuid.uuid4().hex[:12]
    j = {"id": jid, "kind": kind, "status": "queued",
         "submitted": time.time(), "payload": payload, "result": None}
    with _JOBS_LOCK:
        _JOBS[jid] = j
    try:
        storage.record_job(jid, kind, "queued", payload=payload,
                           submitted=j["submitted"])
    except Exception:
        pass  # storage failure must never break the API
    return j


def _set_job(jid: str, **kw) -> None:
    with _JOBS_LOCK:
        if jid in _JOBS: _JOBS[jid].update(kw)
        cur = _JOBS.get(jid, {})
    try:
        storage.record_job(
            jid, cur.get("kind", "unknown"), cur.get("status", "unknown"),
            payload=cur.get("payload") or {},
            result=cur.get("result") or {},
            submitted=cur.get("submitted"),
            finished=cur.get("finished"))
    except Exception:
        pass


# ── Handlers (each returns (status_code, dict_body)) ──
def health() -> tuple[int, dict]:
    return 200, {"ok": True, "ts": time.time(), "backend": gpu.backend(),
                 "db": storage.db_path()}


def gpu_info() -> tuple[int, dict]:
    return 200, gpu.detect()


def list_models() -> tuple[int, dict]:
    return 200, {"models": models.list_cached(), "cache_root": models.cache_root()}


def pull_model(body: dict) -> tuple[int, dict]:
    repo = body.get("repo_id")
    if not repo: return 400, {"error": "repo_id required"}
    job = _job("pull_model", body)
    def _do():
        try:
            path = models.ensure(repo, revision=body.get("revision", "main"),
                                 allow_patterns=body.get("allow_patterns"))
            _set_job(job["id"], status="done", finished=time.time(),
                     result={"local_path": path})
            storage.record_run("api", f"pull:{repo}", "pass")
        except Exception as e:
            _set_job(job["id"], status="error", finished=time.time(), result={"error": str(e)})
            storage.record_run("api", f"pull:{repo}", "error", output=str(e))
    threading.Thread(target=_do, daemon=True).start()
    return 202, job


def gtc_for(model: str, qs: str) -> tuple[int, dict]:
    metric = None; limit = 500
    for kv in qs.split("&"):
        if kv.startswith("metric="): metric = kv[7:]
        if kv.startswith("limit="):
            try: limit = int(kv[6:])
            except: pass
    rows = storage.gtc_for_model(model, metric=metric, limit=limit)
    return 200, {"model": model, "metric": metric, "count": len(rows), "rows": rows}


def jobs(jid: str | None) -> tuple[int, dict]:
    # Single-job lookup: try in-memory first, then storage (for jobs from prior
    # process lifetimes).
    if jid:
        with _JOBS_LOCK:
            j = _JOBS.get(jid)
        if j: return 200, j
        try:
            persisted = storage.get_job(jid)
            if persisted: return 200, persisted
        except Exception:
            pass
        return 404, {"error": "not found"}
    # List: union of in-memory + persisted, deduped by id, most recent first.
    with _JOBS_LOCK:
        mem = list(_JOBS.values())
    try:
        persisted = storage.list_jobs(limit=200)
    except Exception:
        persisted = []
    seen = {j["id"] for j in mem}
    merged = mem + [p for p in persisted if p["id"] not in seen]
    merged.sort(key=lambda j: j.get("submitted") or 0, reverse=True)
    return 200, {"jobs": merged[:50]}


def infer(body: dict) -> tuple[int, dict]:
    repo = body.get("model"); prompt = body.get("prompt", "")
    if not repo: return 400, {"error": "model required"}
    job = _job("infer", body)
    def _do():
        try:
            path = models.resolve(repo)
            # Minimal stub: avoid forcing a transformer load on every install.
            # If transformers + torch are available, do a real generation.
            try:
                import torch  # type: ignore
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                tok = AutoTokenizer.from_pretrained(path)
                mdl = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=torch.float16 if gpu.backend() != "cpu" else torch.float32)
                mdl.to(gpu.device())
                inp = tok(prompt, return_tensors="pt").to(gpu.device())
                out = mdl.generate(**inp, max_new_tokens=int(body.get("max_new_tokens", 64)))
                text = tok.decode(out[0], skip_special_tokens=True)
            except Exception as inner:
                text = f"[stub] backend={gpu.backend()} model={repo} prompt={prompt!r} err={inner}"
            _set_job(job["id"], status="done", finished=time.time(), result={"text": text})
            storage.record_run("api", f"infer:{repo}", "pass")
        except Exception as e:
            _set_job(job["id"], status="error", finished=time.time(), result={"error": str(e)})
            storage.record_run("api", f"infer:{repo}", "error", output=str(e))
    threading.Thread(target=_do, daemon=True).start()
    return 202, job


def graft(body: dict) -> tuple[int, dict]:
    donor = body.get("donor"); recipient = body.get("recipient")
    if not (donor and recipient): return 400, {"error": "donor and recipient required"}
    job = _job("graft", body)
    def _do():
        try:
            # Pre-fetch both models
            d = models.resolve(donor); r = models.resolve(recipient)
            _set_job(job["id"], status="done", finished=time.time(),
                     result={"donor_path": d, "recipient_path": r,
                             "note": "models prefetched; run scripts/hyper_graft.py with these paths"})
            storage.record_run("api", f"graft:{donor}->{recipient}", "pass")
        except Exception as e:
            _set_job(job["id"], status="error", finished=time.time(), result={"error": str(e)})
    threading.Thread(target=_do, daemon=True).start()
    return 202, job


def compress(body: dict) -> tuple[int, dict]:
    repo = body.get("model"); ratio = float(body.get("ratio", 0.5))
    if not repo: return 400, {"error": "model required"}
    job = _job("compress", {"model": repo, "ratio": ratio})
    def _do():
        try:
            path = models.resolve(repo)
            _set_job(job["id"], status="done", finished=time.time(),
                     result={"local_path": path, "target_ratio": ratio,
                             "note": "model fetched; run scripts/grc_*.py to perform compression"})
        except Exception as e:
            _set_job(job["id"], status="error", finished=time.time(), result={"error": str(e)})
    threading.Thread(target=_do, daemon=True).start()
    return 202, job


def sort_list(body: dict) -> tuple[int, dict]:
    data = body.get("data")
    if not isinstance(data, list): return 400, {"error": "data must be a list"}
    try:
        import hypersort  # type: ignore
        out = hypersort.hypersort(data, reverse=bool(body.get("reverse", False)))
        engine = "hypersort"
    except Exception:
        out = sorted(data, reverse=bool(body.get("reverse", False)))
        engine = "fallback:sorted"
    return 200, {"engine": engine, "result": out, "n": len(out)}


# ── Routing entry point used by server.py ──
def route(method: str, path: str, qs: str, headers, body_bytes: bytes) -> tuple[int, dict]:
    if not _authorized(headers, qs):
        return 401, {"error": "unauthorized — set HT_REPRO_TOKEN and pass Bearer token"}

    try:
        body = json.loads(body_bytes or b"{}") if body_bytes else {}
    except json.JSONDecodeError:
        return 400, {"error": "invalid JSON body"}

    if method == "GET":
        if path == "/api/v1/health":   return health()
        if path == "/api/v1/gpu":      return gpu_info()
        if path == "/api/v1/models":   return list_models()
        if path == "/api/v1/jobs":     return jobs(None)
        if path.startswith("/api/v1/jobs/"): return jobs(path.split("/")[-1])
        if path.startswith("/api/v1/gtc/"):
            return gtc_for(path.split("/api/v1/gtc/")[1], qs)

    if method == "POST":
        if path == "/api/v1/models/pull": return pull_model(body)
        if path == "/api/v1/infer":       return infer(body)
        if path == "/api/v1/graft":       return graft(body)
        if path == "/api/v1/compress":    return compress(body)
        if path == "/api/v1/sort":        return sort_list(body)

    return 404, {"error": f"no route: {method} {path}"}
