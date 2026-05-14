"""
ht_repro.storage — Persistent SQLite store for GTC trajectories,
manifold state, run history, and model metadata.

Default path: ~/.ht-repro/store.db  (override with HT_REPRO_DB env var).
"""
from __future__ import annotations
import json, os, sqlite3, time
from pathlib import Path
from typing import Any, Iterable

_DB_PATH = Path(os.environ.get("HT_REPRO_DB", Path.home() / ".ht-repro" / "store.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,            -- 'test' | 'tool' | 'api'
    name TEXT NOT NULL,
    status TEXT NOT NULL,          -- 'pass' | 'fail' | 'error'
    elapsed REAL,
    output TEXT,
    meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);

CREATE TABLE IF NOT EXISTS gtc_trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model TEXT NOT NULL,
    layer INTEGER,
    sample_id TEXT,
    metric TEXT NOT NULL,          -- 'curvature' | 'geodesic' | 'residue'
    value REAL NOT NULL,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_gtc_model ON gtc_trajectories(model);
CREATE INDEX IF NOT EXISTS idx_gtc_metric ON gtc_trajectories(metric);

CREATE TABLE IF NOT EXISTS manifold_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    repo_id TEXT PRIMARY KEY,
    local_path TEXT,
    size_gb REAL,
    downloaded REAL,
    last_used REAL,
    meta_json TEXT
);
"""


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH))
    c.executescript(_SCHEMA)
    return c


# ── Run history ──
def record_run(kind: str, name: str, status: str, elapsed: float | None = None,
               output: str = "", meta: dict | None = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs(ts,kind,name,status,elapsed,output,meta_json) VALUES (?,?,?,?,?,?,?)",
            (time.time(), kind, name, status, elapsed, output[-4000:] if output else "",
             json.dumps(meta or {})))
        return cur.lastrowid


def recent_runs(limit: int = 50, kind: str | None = None) -> list[dict]:
    q = "SELECT id,ts,kind,name,status,elapsed FROM runs"
    args: tuple = ()
    if kind:
        q += " WHERE kind=?"; args = (kind,)
    q += " ORDER BY ts DESC LIMIT ?"; args = args + (limit,)
    with _conn() as c:
        return [dict(zip(["id","ts","kind","name","status","elapsed"], r))
                for r in c.execute(q, args)]


# ── GTC trajectories ──
def record_gtc(model: str, metric: str, value: float, layer: int | None = None,
               sample_id: str | None = None, payload: dict | None = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO gtc_trajectories(ts,model,layer,sample_id,metric,value,payload_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.time(), model, layer, sample_id, metric, value, json.dumps(payload or {})))
        return cur.lastrowid


def record_gtc_batch(rows: Iterable[dict]) -> int:
    n = 0
    with _conn() as c:
        for r in rows:
            c.execute(
                "INSERT INTO gtc_trajectories(ts,model,layer,sample_id,metric,value,payload_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (r.get("ts", time.time()), r["model"], r.get("layer"), r.get("sample_id"),
                 r["metric"], float(r["value"]), json.dumps(r.get("payload", {}))))
            n += 1
    return n


def gtc_for_model(model: str, metric: str | None = None, limit: int = 1000) -> list[dict]:
    q = "SELECT ts,model,layer,sample_id,metric,value FROM gtc_trajectories WHERE model=?"
    args: tuple = (model,)
    if metric:
        q += " AND metric=?"; args = args + (metric,)
    q += " ORDER BY ts DESC LIMIT ?"; args = args + (limit,)
    with _conn() as c:
        return [dict(zip(["ts","model","layer","sample_id","metric","value"], r))
                for r in c.execute(q, args)]


# ── Manifold state KV ──
def set_state(key: str, value: Any) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO manifold_state(key,value_json,updated) VALUES (?,?,?)",
                  (key, json.dumps(value), time.time()))


def get_state(key: str, default: Any = None) -> Any:
    with _conn() as c:
        r = c.execute("SELECT value_json FROM manifold_state WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default


# ── Model registry ──
def register_model(repo_id: str, local_path: str, size_gb: float = 0.0,
                   meta: dict | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO models(repo_id,local_path,size_gb,downloaded,last_used,meta_json) "
            "VALUES (?,?,?,COALESCE((SELECT downloaded FROM models WHERE repo_id=?),?),?,?)",
            (repo_id, local_path, size_gb, repo_id, time.time(), time.time(), json.dumps(meta or {})))


def get_model(repo_id: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT repo_id,local_path,size_gb,downloaded,last_used FROM models WHERE repo_id=?",
                      (repo_id,)).fetchone()
    if not r: return None
    return dict(zip(["repo_id","local_path","size_gb","downloaded","last_used"], r))


def list_models() -> list[dict]:
    with _conn() as c:
        return [dict(zip(["repo_id","local_path","size_gb","downloaded","last_used"], r))
                for r in c.execute(
                    "SELECT repo_id,local_path,size_gb,downloaded,last_used FROM models ORDER BY last_used DESC")]


def db_path() -> str:
    return str(_DB_PATH)


if __name__ == "__main__":
    print(f"DB: {db_path()}")
    with _conn() as c: pass
    print("Schema initialized.")
    print(f"Runs: {len(recent_runs(10))}  Models: {len(list_models())}")
