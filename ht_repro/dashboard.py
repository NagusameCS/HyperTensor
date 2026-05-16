#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................:::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............:::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............:::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
HTML Dashboard Generator — produces a standalone results report.
"""
import json
import time
from pathlib import Path

from .catalog import load_catalog, find_test
from .runner import load_results, last_run
from .setup_wizard import run_setup

ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_FILE = ROOT / "benchmarks" / "ht_repro_dashboard.html"

def generate_dashboard() -> Path:
    """Generate a standalone HTML dashboard of all results."""
    catalog = load_catalog()
    results = load_results()
    report = run_setup(interactive=False)

    runs_html = ""
    for run in results.get("runs", []):
        passed = run["passed"]
        failed = run["failed"]
        skipped = run["skipped"]
        total = passed + failed + skipped
        bar_w = 300
        p_w = int(bar_w * passed / max(total, 1))
        f_w = int(bar_w * failed / max(total, 1))
        s_w = bar_w - p_w - f_w

        test_rows = ""
        for tid, r in run.get("tests", {}).items():
            t = find_test(tid)
            name = t["name"] if t else tid
            icon = {"pass": "✅", "fail": "❌", "skipped": "⊘"}.get(r["status"], "?")
            color = {"pass": "#34d399", "fail": "#f87171", "skipped": "#fbbf24"}.get(r["status"], "#888")
            time_str = f"{r.get('time', 0):.1f}s" if r.get("time") else "—"
            summary = r.get("summary", r.get("reason", ""))[:150]
            test_rows += f"""<tr style="border-top:1px solid #2a2a2a">
                <td style="padding:6px 10px;color:{color}">{icon}</td>
                <td style="padding:6px 10px">{name}</td>
                <td style="padding:6px 10px;font-family:monospace;font-size:.8rem">{time_str}</td>
                <td style="padding:6px 10px;font-size:.75rem;color:#888">{summary}</td>
            </tr>"""

        runs_html += f"""
        <div style="margin:16px 0;background:#1c1c1a;border:1px solid #2a2a2a;border-radius:10px;padding:16px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <span style="font-family:monospace;font-size:.8rem;color:#888">{run['timestamp']}</span>
                <span style="font-size:.8rem">✅{passed} ❌{failed} ⊘{skipped} · {run['total_time']:.1f}s</span>
            </div>
            <div style="display:flex;height:6px;border-radius:3px;overflow:hidden;margin-bottom:12px">
                <div style="width:{p_w}px;background:#34d399"></div>
                <div style="width:{f_w}px;background:#f87171"></div>
                <div style="width:{s_w}px;background:#555"></div>
            </div>
            <table style="width:100%;border-collapse:collapse">
                {test_rows}
            </table>
        </div>"""

    # Catalog table
    catalog_rows = ""
    for t in sorted(catalog, key=lambda x: ({"T1":0,"T2":1,"T3":2}[x["tier"]], x["group"], x["id"])):
        tier_color = {"T1": "#34d399", "T2": "#fbbf24", "T3": "#f87171"}[t["tier"]]
        catalog_rows += f"""<tr style="border-top:1px solid #2a2a2a">
            <td style="padding:4px 10px;font-family:monospace;font-size:.78rem">{t['id']}</td>
            <td style="padding:4px 10px;color:{tier_color}">{t['tier']}</td>
            <td style="padding:4px 10px">{t['name']}</td>
            <td style="padding:4px 10px;font-size:.75rem;color:#888">{t['desc']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ht-repro Dashboard — HyperTensor</title>
<style>
:root{{--bg:#141413;--card:#1c1c1a;--border:#2a2a2a;--text:#e8e6e3;--dim:#8b8a86;--accent:#f59e0b;--green:#34d399;--red:#f87171;--yellow:#fbbf24}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;padding:24px;max-width:900px;margin:0 auto}}
h1{{font-size:1.6rem;font-weight:500;letter-spacing:-.02em;margin-bottom:4px}}
h1 span{{color:var(--accent)}}
h2{{font-size:1rem;font-weight:500;margin:24px 0 10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px}}
.env{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px;margin:12px 0}}
.env .kv{{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:10px 14px}}
.env .k{{font-size:.65rem;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}}
.env .v{{font-size:.85rem;margin-top:2px}}
code{{font-family:'SF Mono',monospace;background:rgba(255,255,255,.05);padding:1px 5px;border-radius:3px}}
footer{{text-align:center;padding:32px 0 16px;color:var(--dim);font-size:.7rem}}
footer a{{color:var(--accent);text-decoration:none}}
</style>
</head>
<body>
<h1>ht-<span>repro</span> Dashboard</h1>
<p style="color:var(--dim);font-size:.85rem">HyperTensor Reproduction Results · {time.strftime('%Y-%m-%d %H:%M')}</p>

<h2>Environment</h2>
<div class="env">
<div class="kv"><div class="k">OS</div><div class="v">{report['os']}</div></div>
<div class="kv"><div class="k">Python</div><div class="v" style="font-size:.75rem">{report['python']}</div></div>
<div class="kv"><div class="k">GPU</div><div class="v" style="color:{'#34d399' if report['gpu']['available'] else '#888'}">{report['gpu']['name']} ({report['gpu']['vram_gb']} GB)</div></div>
<div class="kv"><div class="k">Available Tier</div><div class="v" style="color:{'#34d399' if report['tier_available']=='T1' else '#fbbf24' if report['tier_available']=='T2' else '#f87171'}">{report['tier_available']}</div></div>
<div class="kv"><div class="k">Disk Free</div><div class="v">{report['disk']['free_gb']} GB</div></div>
</div>

<h2>Run History</h2>
{runs_html if runs_html else '<p style="color:var(--dim);font-size:.85rem">No runs yet. Run <code>ht-repro smoke</code> to get started.</p>'}

<h2>Test Catalog</h2>
<table style="width:100%;border-collapse:collapse">
{catalog_rows}
</table>

<footer>Generated by <a href="https://github.com/NagusameCS/HyperTensor">HyperTensor</a> ht-repro v1.0 · <code>ht-repro dashboard</code></footer>
</body>
</html>"""

    DASHBOARD_FILE.parent.mkdir(exist_ok=True)
    DASHBOARD_FILE.write_text(html)
    return DASHBOARD_FILE
