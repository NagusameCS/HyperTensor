"""Tests for jobs persistence: ensure submitting via API survives a process
restart by being queryable from storage."""
import importlib
import time


def test_storage_record_and_get_job(isolated_storage):
    import ht_repro.storage as s
    s.record_job("j1", "infer", "queued", payload={"prompt": "hi"})
    j = s.get_job("j1")
    assert j is not None
    assert j["status"] == "queued"
    assert j["payload"] == {"prompt": "hi"}


def test_storage_job_status_update(isolated_storage):
    import ht_repro.storage as s
    s.record_job("j2", "infer", "queued", payload={})
    s.record_job("j2", "infer", "done", payload={}, result={"text": "ok"},
                 finished=time.time())
    j = s.get_job("j2")
    assert j["status"] == "done"
    assert j["result"] == {"text": "ok"}


def test_storage_list_jobs_filter(isolated_storage):
    import ht_repro.storage as s
    s.record_job("a", "infer", "done")
    s.record_job("b", "infer", "queued")
    s.record_job("c", "graft", "done")
    assert len(s.list_jobs()) == 3
    assert len(s.list_jobs(status="done")) == 2
    assert len(s.list_jobs(status="queued")) == 1


def test_api_sort_does_not_create_job(no_auth):
    # sort is synchronous; should not pollute the jobs table
    import ht_repro.storage as s
    before = len(s.list_jobs())
    code, body = no_auth.route("POST", "/api/v1/sort", "",
                               {}, b'{"data":[3,1,2]}')
    assert code == 200
    after = len(s.list_jobs())
    assert after == before


def test_api_compress_persists_job(no_auth, monkeypatch):
    # compress() should return 202 + create a queued job recorded to SQLite.
    import ht_repro.storage as s
    # Block real model resolution
    import ht_repro.models as m
    monkeypatch.setattr(m, "resolve", lambda r: f"/fake/{r}")
    code, body = no_auth.route("POST", "/api/v1/compress", "",
                               {}, b'{"model":"org/x","ratio":0.3}')
    assert code == 202
    jid = body["id"]
    # Give the background thread a moment to finalize state
    for _ in range(20):
        j = s.get_job(jid)
        if j and j["status"] in ("done", "error"): break
        time.sleep(0.05)
    j = s.get_job(jid)
    assert j is not None
    assert j["status"] in ("queued", "done", "error")  # any persisted state OK
    assert j["kind"] == "compress"


def test_jobs_endpoint_includes_persisted_jobs(no_auth, isolated_storage):
    # Pre-seed storage with a job; the API should surface it even though it's
    # not in the in-memory _JOBS dict.
    import ht_repro.storage as s
    s.record_job("ghost-1", "infer", "done", payload={"prompt": "h"},
                 result={"text": "world"})
    code, body = no_auth.route("GET", "/api/v1/jobs", "", {}, b"")
    assert code == 200
    ids = {j["id"] for j in body["jobs"]}
    assert "ghost-1" in ids


def test_jobs_single_lookup_falls_back_to_storage(no_auth):
    import ht_repro.storage as s
    s.record_job("ghost-2", "graft", "queued")
    code, body = no_auth.route("GET", "/api/v1/jobs/ghost-2", "", {}, b"")
    assert code == 200
    assert body["id"] == "ghost-2"
    assert body["kind"] == "graft"
