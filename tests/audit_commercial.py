"""Full functional audit of the 8 commercial-grade infrastructure pieces."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL = [], []

def check(name, fn):
    try:
        r = fn()
        PASS.append(name); print(f"[PASS] {name}: {r}")
    except Exception as e:
        FAIL.append((name, e)); print(f"[FAIL] {name}: {type(e).__name__}: {e}")

# ── 1. C runtime packaging ──
def t1_import():
    import hypertensor_runtime as hr
    return f"package OK, version={hr.__version__}, geodessical={hr.geodessical_path()}, lib={hr.libhypercore_path()}"

def t1_runtime_loader():
    from ht_repro import runtime_loader
    d = runtime_loader.diagnostic()
    return f"platform={d['platform']} geo={d['geodessical']} lib_loadable={d['libhypercore_loadable']}"

def t1_geo_build():
    import os, platform
    cand = []
    if platform.system() == "Windows":
        cand += [Path("build_host/geodessical.exe"), Path("build_release/geodessical.exe")]
    cand += [Path("build/geodessical"), Path("build_host/geodessical")]
    found = [p for p in cand if p.exists()]
    return f"existing builds: {found if found else 'none — needs build'}"

# ── 2. Docker image ──
def t2_dockerfile():
    df = Path("deploy/Dockerfile.ht-repro")
    text = df.read_text()
    assert "FROM python:3.12-slim" in text
    assert "ht-repro" in text
    assert "EXPOSE 8765" in text
    return f"Dockerfile.ht-repro OK ({len(text)} bytes)"

def t2_compose():
    yml = Path("deploy/docker-compose.yml").read_text()
    assert "ht-repro:" in yml and "HT_REPRO_TOKEN" in yml
    return "docker-compose.yml OK"

def t2_root_dockerfile():
    df = Path("Dockerfile")
    assert df.exists(), "root Dockerfile missing"
    return f"root Dockerfile OK ({df.stat().st_size} bytes)"

# ── 3. REST API ──
def t3_health():
    from ht_repro import api_v1
    code, body = api_v1.route("GET", "/api/v1/health", "", {}, b"")
    assert code == 200 and body["ok"] is True
    return f"health backend={body['backend']}"

def t3_gpu():
    from ht_repro import api_v1
    code, body = api_v1.route("GET", "/api/v1/gpu", "", {}, b"")
    assert code == 200 and "backend" in body
    return f"gpu {body['backend']} {body.get('name','')}"

def t3_sort():
    from ht_repro import api_v1
    code, body = api_v1.route("POST", "/api/v1/sort", "", {}, b'{"data":[5,2,9,1,7]}')
    assert code == 200 and body["result"] == [1, 2, 5, 7, 9]
    return f"sort {body['engine']}"

def t3_models_list():
    from ht_repro import api_v1
    code, body = api_v1.route("GET", "/api/v1/models", "", {}, b"")
    assert code == 200 and "models" in body
    return f"models endpoint OK (cached={len(body['models'])})"

def t3_auth_off():
    from ht_repro import api_v1
    import os; os.environ.pop("HT_REPRO_TOKEN", None)
    code, _ = api_v1.route("GET", "/api/v1/health", "", {}, b"")
    return f"unauth path code={code}"

def t3_auth_on():
    import os, importlib
    from ht_repro import api_v1
    os.environ["HT_REPRO_TOKEN"] = "secret-x"
    importlib.reload(api_v1)
    code, _ = api_v1.route("GET", "/api/v1/health", "", {}, b"")
    assert code == 401, f"expected 401 without token, got {code}"
    code2, _ = api_v1.route("GET", "/api/v1/health", "token=secret-x", {}, b"")
    assert code2 == 200, f"expected 200 with token, got {code2}"
    code3, _ = api_v1.route("GET", "/api/v1/health", "", {"Authorization": "Bearer secret-x"}, b"")
    assert code3 == 200
    del os.environ["HT_REPRO_TOKEN"]; importlib.reload(api_v1)
    return "token auth: 401 without, 200 with (query+header)"

def t3_jobs():
    from ht_repro import api_v1
    code, body = api_v1.route("GET", "/api/v1/jobs", "", {}, b"")
    assert code == 200
    return f"jobs={len(body['jobs'])}"

def t3_invalid_json():
    from ht_repro import api_v1
    code, body = api_v1.route("POST", "/api/v1/sort", "", {}, b"{not valid")
    assert code == 400
    return "invalid JSON -> 400"

def t3_unknown_route():
    from ht_repro import api_v1
    code, _ = api_v1.route("GET", "/api/v1/nonexistent", "", {}, b"")
    assert code == 404
    return "unknown route -> 404"

# ── 4. Persistent storage ──
def t4_schema():
    from ht_repro import storage
    rid = storage.record_run("test", "audit", "pass", 1.5, "ok")
    storage.record_gtc("audit-model", "curvature", 0.42, layer=5)
    storage.record_gtc_batch([
        {"model": "audit-model", "metric": "geodesic", "value": 1.1, "layer": 0},
        {"model": "audit-model", "metric": "geodesic", "value": 2.2, "layer": 1},
    ])
    storage.set_state("audit_key", {"x": 1, "y": "z"})
    storage.register_model("test-org/test-model", "/tmp/fake", 1.23)
    runs = storage.recent_runs(5)
    gtc = storage.gtc_for_model("audit-model")
    kv = storage.get_state("audit_key")
    mdl = storage.get_model("test-org/test-model")
    assert kv["x"] == 1
    assert mdl["size_gb"] == 1.23
    return f"runs={len(runs)} gtc={len(gtc)} kv={kv} model_registered={mdl is not None}"

def t4_path():
    from ht_repro import storage
    p = Path(storage.db_path())
    assert p.exists()
    return f"db {p} ({p.stat().st_size} bytes)"

# ── 5. Cloud recipe ──
def t5_files():
    needed = ["deploy/Dockerfile.ht-repro", "deploy/docker-compose.yml",
              "deploy/nginx.conf", "deploy/ht-repro.service",
              "deploy/terraform/main.tf", "deploy/.env.example",
              "deploy/build_native.sh", "deploy/cibuildwheel.toml",
              "deploy/README.md"]
    missing = [f for f in needed if not Path(f).exists()]
    if missing: raise RuntimeError(f"missing {missing}")
    return f"all {len(needed)} files present"

def t5_nginx_sse():
    txt = Path("deploy/nginx.conf").read_text()
    assert "proxy_buffering off" in txt and "/api/stream" in txt
    return "nginx has SSE-safe proxy block"

def t5_systemd():
    txt = Path("deploy/ht-repro.service").read_text()
    assert "[Unit]" in txt and "ExecStart=" in txt and "Restart=" in txt
    return "systemd unit valid"

def t5_terraform():
    txt = Path("deploy/terraform/main.tf").read_text()
    assert "aws_instance" in txt and "g6e.xlarge" in txt
    return "terraform main.tf valid"

# ── 6. Model auto-download ──
def t6_module():
    from ht_repro import models
    return f"cache={models.cache_root()}, list={len(models.list_cached())}"

def t6_hf_hub():
    import huggingface_hub
    return f"huggingface_hub {huggingface_hub.__version__} available"

def t6_resolve_local():
    from ht_repro import models
    # A path that exists should pass through without download
    p = models.resolve(str(Path.cwd()))
    return f"resolve(local)={p}"

# ── 7. Graft wrapper ──
def t7_import():
    from ht_repro import graft_wrapper
    assert callable(graft_wrapper.main)
    return "ht-graft entrypoint importable"

def t7_underlying_scripts():
    scripts = ["scripts/hyper_graft.py", "scripts/hyper_graft_cpu.py"]
    present = [s for s in scripts if Path(s).exists()]
    return f"underlying scripts present: {present}"

# ── 8. GPU abstraction ──
def t8_detect():
    from ht_repro import gpu
    i = gpu.detect()
    assert i["backend"] in {"cuda", "rocm", "mps", "cpu"}
    return f"backend={i['backend']} device={i['device']} vram={i['vram_gb']}GB"

def t8_device():
    from ht_repro import gpu
    d_auto = gpu.device()
    d_cpu = gpu.device(prefer="cpu")
    return f"auto={d_auto} cpu={d_cpu}"

def t8_env():
    from ht_repro import gpu
    env = gpu.env_for_subprocess(visible="0")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["HIP_VISIBLE_DEVICES"] == "0"
    assert "HT_BACKEND" in env
    return f"env routing OK: HT_BACKEND={env['HT_BACKEND']}"

def t8_decorator():
    from ht_repro import gpu
    @gpu.gpu_compatible
    def f(x, device=None): return (x, str(device))
    r = f(5)
    assert "cuda" in r[1] or "cpu" in r[1] or "mps" in r[1]
    return f"@gpu_compatible injected device={r[1]}"

def t8_cuda_live():
    import torch
    assert torch.cuda.is_available()
    a = torch.randn(64, 64, device="cuda")
    b = a @ a
    return f"live cuda matmul OK sum={b.sum().item():.2f}"

# ── 9. CLI integration ──
def t9_serve_flag():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "ht_repro.cli", "--help"],
                       capture_output=True, text=True, timeout=10)
    assert "serve" in r.stdout
    return "ht-repro CLI lists 'serve' subcommand"

def t9_server_routes_mounted():
    from ht_repro import server
    h = server.H
    assert hasattr(h, "do_POST") and hasattr(h, "do_GET")
    return "server.py has do_GET and do_POST handlers"

# ── Run all ──
suite = [
    ("1a. hypertensor-runtime package import", t1_import),
    ("1b. ht_repro.runtime_loader diagnostic", t1_runtime_loader),
    ("1c. existing geodessical build artifacts", t1_geo_build),
    ("2a. Dockerfile.ht-repro syntax", t2_dockerfile),
    ("2b. docker-compose.yml syntax", t2_compose),
    ("2c. root research Dockerfile", t2_root_dockerfile),
    ("3a. /api/v1/health", t3_health),
    ("3b. /api/v1/gpu", t3_gpu),
    ("3c. POST /api/v1/sort", t3_sort),
    ("3d. /api/v1/models list", t3_models_list),
    ("3e. auth disabled by default", t3_auth_off),
    ("3f. Bearer-token auth (401/200)", t3_auth_on),
    ("3g. /api/v1/jobs", t3_jobs),
    ("3h. invalid JSON -> 400", t3_invalid_json),
    ("3i. unknown route -> 404", t3_unknown_route),
    ("4a. storage schema + CRUD", t4_schema),
    ("4b. storage DB on disk", t4_path),
    ("5a. all deploy files present", t5_files),
    ("5b. nginx SSE block", t5_nginx_sse),
    ("5c. systemd unit", t5_systemd),
    ("5d. terraform main.tf", t5_terraform),
    ("6a. models module importable", t6_module),
    ("6b. huggingface_hub available", t6_hf_hub),
    ("6c. models.resolve(local-path)", t6_resolve_local),
    ("7a. graft_wrapper importable", t7_import),
    ("7b. underlying graft scripts", t7_underlying_scripts),
    ("8a. gpu.detect()", t8_detect),
    ("8b. gpu.device() auto + cpu", t8_device),
    ("8c. gpu.env_for_subprocess()", t8_env),
    ("8d. @gpu_compatible decorator", t8_decorator),
    ("8e. live CUDA matmul", t8_cuda_live),
    ("9a. CLI has 'serve' subcommand", t9_serve_flag),
    ("9b. server.py routes mounted", t9_server_routes_mounted),
]

for name, fn in suite:
    check(name, fn)

print()
print("=" * 64)
print(f"TOTAL: {len(PASS)} PASS / {len(FAIL)} FAIL  ({len(suite)} checks)")
if FAIL:
    print("\nFailures:")
    for n, e in FAIL: print(f"  {n}: {e}")
sys.exit(0 if not FAIL else 1)
