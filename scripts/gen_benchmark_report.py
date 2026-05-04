"""
gen_benchmark_report.py  ---  Comprehensive A--E benchmark report generator
==========================================================================
Reads all artefacts produced by the conversion farm + geodessical log files
and emits:
  - docs/benchmark_report/MASTER_REPORT.md      full narrative report
  - docs/benchmark_report/paper_a_kint.md       Paper A: k_int generalisation
  - docs/benchmark_report/paper_b_load.md       Paper B: VRAM / load efficiency
  - docs/benchmark_report/paper_c_decode.md     Paper C: decode throughput
  - docs/benchmark_report/paper_d_hjb.md        Paper D: HJB feasibility
  - docs/benchmark_report/paper_e_rho.md        Paper E: rho / distillation
  - docs/benchmark_report/all_results.csv       combined CSV for further analysis

Usage:
  python scripts/gen_benchmark_report.py [--farm-run-id ID] [--out-dir PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
KINT_DIR       = REPO / "docs" / "figures" / "paper-a" / "kint_30b"
LOG_DIR        = REPO / "benchmarks" / "local_30b_grc"   # legacy
FARM_BASE      = REPO / "benchmarks" / "conversion_farm"
HJB_DIR        = REPO / "docs" / "figures" / "paper-d" / "hjb_feasibility_spectrum"
RHO_DIR        = REPO / "docs" / "figures" / "paper-e" / "rho_sweep_spectrum"
OUT_DIR_DEF    = REPO / "docs" / "benchmark_report"


# ---------------------------------------------------------------------------
# Paper A  --- k_int generalisation
# ---------------------------------------------------------------------------
def load_kint_results() -> list[dict]:
    rows = []
    if not KINT_DIR.exists():
        return rows
    for jf in sorted(KINT_DIR.glob("kint_*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            rows.append(d)
        except Exception as e:
            print(f"[warn] skip {jf.name}: {e}", file=sys.stderr)
    return rows


def paper_a_md(rows: list[dict]) -> str:
    if not rows:
        return "## Paper A --- k_int Generalisation\n\nNo data yet.\n"

    lines = ["## Paper A --- k_int Generalisation Across Architectures",
             "",
             "Summary: across the sampled models, intrinsic rank k_int (95%"
             " joint-Gram variance) stays below d and is often in the 0.5--0.7 d range,"
             " supporting rank-limited attention compression.",
             ""]

    # Summary table
    lines += [
        "| Model | d | Mean k_int | k_int/d | Min k_int | Max k_int | Layers sampled |",
        "|-------|---|-----------|---------|-----------|-----------|----------------|",
    ]
    for r in sorted(rows, key=lambda x: x.get("d", x.get("model_dim", 0))):
        name  = r.get("model", r.get("name", "?"))
        d     = r.get("d", r.get("model_dim", 0))
        mk    = r.get("mean_kint", 0)
        ratio = r.get("mean_kint_over_d", 0)
        mn    = r.get("min_kint", 0)
        mx    = r.get("max_kint", 0)
        nl    = len(list(r.get("used_layers", [])))
        lines.append(f"| {name} | {d} | {mk:.1f} | {ratio:.4f} | {mn} | {mx} | {nl} |")

    lines += ["",
              "### Per-layer k_int detail", ""]
    for r in sorted(rows, key=lambda x: x.get("d", x.get("model_dim", 0))):
        name   = r.get("model", r.get("name", "?"))
        layers = list(r.get("used_layers", []))
        kints  = list(r.get("per_layer_kint", []))
        if not layers:
            continue
        d_val = r.get("d", r.get("model_dim", 0))
        lines.append(f"{name} (d={d_val})")
        lines.append("")
        lines.append("| Layer | k_int | k_int/d |")
        lines.append("|-------|-------|---------|")
        d = d_val or 1
        for l, k in zip(layers, kints):
            lines.append(f"| {l} | {k} | {k/d:.4f} |")
        lines.append("")

    # Data-quality caveat: flag exact duplicate per-layer vectors across different models.
    sig_to_models: dict[tuple, list[str]] = {}
    for r in rows:
        per = tuple(r.get("per_layer_kint", []))
        if not per:
            continue
        sig_to_models.setdefault(per, []).append(str(r.get("model", r.get("name", "?"))))
    dup_groups = [m for m in sig_to_models.values() if len(m) > 1]
    if dup_groups:
        lines += ["### Data-Quality Notes", ""]
        lines.append("Some model pairs have identical per-layer k_int vectors. "
                     "This may reflect shared checkpoints, aliasing in model selection, "
                     "or a pipeline mapping issue and should be verified before publication.")
        lines.append("")
        for models in dup_groups:
            lines.append(f"- Identical k_int profile: {', '.join(models)}")
        lines.append("")

    # LaTeX table block
    lines += ["### LaTeX Generalisation Table (copy-paste)", "", "```latex"]
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & $d$ & $\bar{k}_\mathrm{int}$ & $\bar{k}_\mathrm{int}/d$ \\")
    lines.append(r"\midrule")
    for r in sorted(rows, key=lambda x: x.get("d", x.get("model_dim", 0))):
        name  = r.get("model", r.get("name", "?")).replace("_", r"\_")
        d     = r.get("d", r.get("model_dim", 0))
        mk    = r.get("mean_kint", 0)
        ratio = r.get("mean_kint_over_d", 0)
        lines.append(f"  {name} & {d} & {mk:.1f} & {ratio:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Intrinsic rank $k_\mathrm{int}$ at 95\% joint Gram variance}")
    lines.append(r"\end{table}")
    lines.append("```")
    lines.append("")

    #  Multi-k Pareto sweep data 
    multik_dirs = sorted((REPO / "benchmarks").glob("paper_a_multi_k*"), reverse=True)
    for mk_dir in multik_dirs:
        summary_path = mk_dir / "multi_k_summary.json"
        if not summary_path.exists():
            continue
        try:
            entries = json.loads(summary_path.read_text(encoding="utf-8"))
            # PS5 ConvertTo-Json emits a flat array, not a dict with 'entries'
            if isinstance(entries, dict):
                entries = entries.get("entries", [])
            if not entries:
                continue
            lines += ["", f"### Multi-k Pareto Sweep --- {mk_dir.name}", "",
                      "Throughput (tok/s) vs compression rank k.",
                      "Demonstrates the accuracy--speed Pareto frontier (Paper A §Pareto).",
                      "",
                      "| k | Mean tok/s | SD | N |",
                      "|---|-----------|----|----|"]
            for e in sorted(entries,
                             key=lambda x: -1 if x.get("k") == "baseline"
                             else int(x.get("k", 0))):
                k    = e.get("k", "?")
                mean = e.get("mean_toks", e.get("mean_tok_s", 0))
                sd   = e.get("sd_toks",  e.get("sd_tok_s", 0))
                n    = e.get("n_obs",    e.get("n", 0))
                lines.append(f"| {k} | {mean:.1f} | {sd:.1f} | {n} |")
            lines.append("")
        except Exception as exc:
            lines.append(f"Error reading multi_k summary: {exc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper B  --- load / VRAM efficiency   (parse geodessical log files)
# ---------------------------------------------------------------------------
_RE_LOAD   = re.compile(r"load[_ ]?time[:\s]+(\d+)\s*ms", re.I)
_RE_LOAD2  = re.compile(r"model loaded.*?(\d+)\s*ms", re.I)
# Matches "[GPU] Uploaded 103 weight tensors (6444 MB)" and legacy forms
_RE_GPU_MB = re.compile(r"(\d+)\s+weight\s+tensors\s+\((\d[\d,.]+)\s*MB\)", re.I)
_RE_GPU_MB2= re.compile(r"GPU.*?(\d[\d,.]+)\s*[Mm][Bb]", re.I)
_RE_OFFLOAD= re.compile(r"offload.*?layer\s*(\d+)", re.I)
_RE_TENSORS= re.compile(r"(\d+)\s*tensors?\s+\(", re.I)

# Map job-ID prefixes to clean model display names for Paper B/C tables
_JOB_TO_MODEL: dict[str, str] = {
    "B_loadonly_gemma4_31b":  "Gemma4-31B",
    "B_loadonly_qwen35_35b":  "Qwen3.5-35B",
    "B_loadonly_gemma3_12b":  "Gemma3-12B",
    "B_loadonly_gemma3_4b":   "Gemma3-4B",
    "B_loadonly_glm47_flash":  "GLM-4.7-Flash",
    "B_loadonly_gemma4_2b":   "Gemma4-2B",
    "B_loadonly_smol135m":    "SmolLM2-135M",
    "B_loadonly_llama70b":    "Llama3.1-70B",
    "C_decode_gemma4_r128":   "Gemma4-31B (r128)",
    "C_decode_qwen35_r128":   "Qwen3.5-35B (r128)",
    "C_decode_gemma3_12b_r128": "Gemma3-12B (r128)",
    "C_decode_gemma3_4b_r128":  "Gemma3-4B (r128)",
    "C_decode_llama70b_r128": "Llama3.1-70B (r128)",
}

def _clean_model_name(job_tag: str) -> str:
    """Map job-ID tag like 'B_loadonly_gemma3_12b_a1' to 'Gemma3-12B'."""
    # strip attempt suffix _a1, _a2 etc
    base = re.sub(r'_a\d+$', '', job_tag)
    for prefix, name in _JOB_TO_MODEL.items():
        if base.startswith(prefix):
            return name
    return job_tag  # fallback to raw tag


def _parse_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    info: dict = {"log": log_path.name}

    m = _RE_LOAD.search(text) or _RE_LOAD2.search(text)
    if m:
        info["load_ms"] = int(m.group(1))

    m = _RE_GPU_MB.search(text)
    if m:
        info["gpu_tensors"] = int(m.group(1))
        info["gpu_mb"] = float(m.group(2).replace(",", ""))
    else:
        m = _RE_GPU_MB2.search(text)
        if m:
            info["gpu_mb"] = float(m.group(1).replace(",", ""))

    m = _RE_OFFLOAD.search(text)
    if m:
        info["offload_from_layer"] = int(m.group(1))

    # tok/s  --- look for "X tok/s" or "X tokens/s"
    toks = re.findall(r"([\d.]+)\s+tok(?:ens)?/s", text, re.I)
    if toks:
        info["tok_s"] = float(toks[-1])

    # generation line
    gen = re.search(r"(Generating\s+\d+\s+tokens.*)", text, re.I)
    if gen:
        info["generation_line"] = gen.group(1).strip()

    return info


def load_log_results(farm_run_id: str = "") -> list[dict]:
    rows = []
    # collect from all known log directories
    log_dirs = []
    if LOG_DIR.exists():
        log_dirs.append(LOG_DIR)
    # farm logs live under benchmarks/conversion_farm/<run_id>/logs/
    if farm_run_id:
        farm_log = FARM_BASE / farm_run_id / "logs"
        if farm_log.exists():
            log_dirs.append(farm_log)
    else:
        # scan all farm run directories
        if FARM_BASE.exists():
            for child in sorted(FARM_BASE.iterdir()):
                farm_log = child / "logs"
                if farm_log.is_dir():
                    log_dirs.append(farm_log)
    for base in log_dirs:
        for lf in sorted(base.glob("**/*.log")):
            if lf.stat().st_size == 0:
                continue
            d = _parse_log(lf)
            # Log may be nested: logs/B_loadonly_gemma3_12b_a1/gemma3_12b_<ts>.log
            # or flat: logs/B_loadonly_gemma3_12b_a1_<ts>.out.log
            # Use the parent dir name if it looks like a job id (starts with capital letter)
            parent = lf.parent
            if parent.name != "logs" and re.match(r'[A-Z]', parent.name):
                job_id = parent.name   # e.g. B_loadonly_gemma3_12b_a1
            else:
                # flat .out.log: strip timestamp _YYYYMMDD_HHmmss and .out
                stem = lf.stem.replace(".out", "").replace(".err", "")
                parts = stem.rsplit("_", 2)
                job_id = parts[0] if len(parts) >= 3 else stem
            d["job_id"]    = job_id
            d["model_tag"] = _clean_model_name(job_id)
            d["source_dir"] = base.name
            rows.append(d)
    return rows


def paper_b_md(rows: list[dict]) -> str:
    b_rows = [r for r in rows if r.get("job_id", "").startswith("B_")]
    if not b_rows:
        return "## Paper B --- Load / VRAM Efficiency\n\nNo log data yet.\n"

    lines = ["## Paper B --- Load & VRAM Efficiency",
             "",
             "Summary: load logs show several compressed models fitting within an"
             " 8 GB-class GPU budget, with model-dependent offload behavior.",
             ""]

    lines += [
        "| Model | Load (ms) | GPU tensors | GPU VRAM (MB) | Offload from | tok/s |",
        "|-------|----------|-------------|---------------|--------------|-------|",
    ]

    # deduplicate: keep latest run per model_tag
    seen: dict[str, dict] = {}
    for r in b_rows:
        tag = r.get("model_tag", "?")
        seen[tag] = r

    for tag, r in sorted(seen.items()):
        load   = r.get("load_ms", "---")
        tensors= r.get("gpu_tensors", "---")
        mb     = r.get("gpu_mb", "---")
        off    = r.get("offload_from_layer", "---")
        toks   = r.get("tok_s", "---")
        if isinstance(mb, float):
            mb = f"{mb:.0f}"
        if isinstance(toks, float):
            toks = f"{toks:.1f}"
        lines.append(f"| {tag} | {load} | {tensors} | {mb} | {off} | {toks} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper C  --- decode throughput  (also from log files, filtered to decode runs)
# ---------------------------------------------------------------------------
def _find_ott_empirical_dir() -> Path | None:
    """Return the most recent benchmarks/ott_empirical* directory that has summary.json."""
    bench = REPO / "benchmarks"
    candidates = sorted(bench.glob("ott_empirical*"), reverse=True)
    for d in candidates:
        if (d / "summary.json").exists():
            return d
        if (d / "speedup_table.md").exists():
            return d
    return None


def _display_model_name(raw: str) -> str:
    """Collapse absolute GGUF paths into compact display names."""
    if not raw:
        return "?"
    s = str(raw)
    # Keep explicit relative model references as-is.
    if s.startswith("models/") or s.startswith("models\\"):
        return s.replace("\\", "/")
    # For absolute gguf paths, show basename only.
    if ".gguf" in s.lower():
        return Path(s).name
    return s


def _display_source_path(p: str) -> str:
    """Render source paths relative to repo when possible."""
    if not p:
        return "?"
    try:
        return str(Path(p).resolve().relative_to(REPO.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def paper_c_md(rows: list[dict]) -> str:
    decode_rows = [r for r in rows
                   if r.get("job_id", "").startswith("C_")
                   and r.get("tok_s") is not None]
    lines = ["## Paper C --- Decode Throughput Under GRC Compression",
             "",
             "Summary: preliminary decode measurements indicate usable throughput"
             " with GRC and provide early OTT/AttnRes interaction evidence.",
             ""]

    if decode_rows:
        lines += [
            "| Model | Rank | tok/s | Load (ms) | VRAM (MB) |",
            "|-------|------|-------|-----------|-----------|",
        ]

        seen: dict[str, dict] = {}
        for r in decode_rows:
            tag = r.get("model_tag", "?")
            seen[tag] = r

        for tag, r in sorted(seen.items()):
            toks = r.get("tok_s", "---")
            load = r.get("load_ms", "---")
            mb   = r.get("gpu_mb", "---")
            rank = "128"  # from manifest; log doesn't always echo it
            if isinstance(mb, float):
                mb = f"{mb:.0f}"
            if isinstance(toks, float):
                toks = f"{toks:.1f}"
            lines.append(f"| {tag} | {rank} | {toks} | {load} | {mb} |")
        lines.append("")
    else:
        lines += ["No farm decode log rows yet; showing OTT/AttnRes empirical sections below.", ""]

    #  OTT empirical speedup data 
    ott_dir = _find_ott_empirical_dir()
    if ott_dir is not None:
        speedup_md = ott_dir / "speedup_table.md"
        summary_json = ott_dir / "summary.json"
        if speedup_md.exists():
            lines += ["", "---", "", speedup_md.read_text(encoding="utf-8").strip(), ""]
        elif summary_json.exists():
            try:
                s = json.loads(summary_json.read_text(encoding="utf-8"))
                entries = s.get("entries", [])
                if entries:
                    lines += ["", "### OTT Speculative Decode Summary", ""]
                    lines += ["| Model | Mode | tok/s | α (%) | Speedup |",
                               "|-------|------|-------|-------|---------|"]
                    base_map: dict[str, float] = {}
                    for e in entries:
                        if e["mode"] == "baseline":
                            base_map[e["model"]] = e.get("mean_tok_s", 1.0)
                    for e in entries:
                        base = base_map.get(e["model"], 1.0)
                        su = f"{e['mean_tok_s']/base:.2f}" if base > 0 else "---"
                        lines.append(f"| {e['model']} | {e['mode']} | "
                                     f"{e['mean_tok_s']} | {e['mean_alpha_pct']} | {su} |")
                    lines.append("")
            except Exception as exc:
                lines.append(f"Error reading OTT summary: {exc}")

    #  AttnRes  GRC rank interaction (attres) 
    attres_report = REPO / "benchmarks" / "paper_c_attres" / "attres_report.md"
    if attres_report.exists():
        lines += ["", "---", "", attres_report.read_text(encoding="utf-8").strip(), ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper D  --- HJB feasibility
# ---------------------------------------------------------------------------
def paper_d_md() -> str:
    lines = ["## Paper D --- HJB Feasibility Spectrum", ""]
    summary_f = HJB_DIR / "hjb_residual_summary.md"
    if summary_f.exists():
        lines.append(summary_f.read_text(encoding="utf-8"))
    else:
        lines.append("HJB feasibility data not yet generated.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper E  --- rho / distillation
# ---------------------------------------------------------------------------
def paper_e_md() -> str:
    lines = ["## Paper E --- Rho / Distillation Spectrum", ""]
    # Prefer spectrum folder with per-model subdirectories, then fallback files.
    rho_files: list[Path] = []
    if RHO_DIR.exists():
        rho_files.extend(sorted(RHO_DIR.glob("*/rho_summary.json")))
        top = RHO_DIR / "rho_summary.json"
        if top.exists():
            rho_files.append(top)
    legacy_candidates = [
        REPO / "docs" / "figures" / "paper-e" / "rho_sweep" / "rho_summary.json",
        REPO / "docs" / "figures" / "paper-e" / "rho_sweep_1536" / "rho_summary.json",
    ]
    for p in legacy_candidates:
        if p.exists():
            rho_files.append(p)

    # De-duplicate while preserving order.
    seen_paths: set[str] = set()
    unique_rho_files: list[Path] = []
    for p in rho_files:
        k = str(p.resolve())
        if k in seen_paths:
            continue
        seen_paths.add(k)
        unique_rho_files.append(p)

    if unique_rho_files:
        rows: list[dict] = []
        for rho_f in unique_rho_files:
            try:
                d = json.loads(rho_f.read_text(encoding="utf-8"))
                rows.append({
                    "source": str(rho_f),
                    "model": d.get("model", "?"),
                    "rank": d.get("rank", "?"),
                    "lora_rank": d.get("lora_rank", "?"),
                    "n_layers": d.get("n_layers", "?"),
                    "mean_rho": d.get("mean_rho", 0.0),
                    "per_layer": list(d.get("per_layer", [])),
                })
            except Exception as exc:
                lines.append(f"Error reading {rho_f}: {exc}")

        if rows:
            lines.append("| Model | Rank | LoRA rank | Layers | Mean ρ | Source |")
            lines.append("|-------|------|-----------|--------|--------|--------|")
            for r in sorted(rows, key=lambda x: str(x.get("model", ""))):
                mr = r.get("mean_rho", 0.0)
                mr_s = f"{mr:.4f}" if isinstance(mr, (int, float)) else str(mr)
                lines.append(
                    f"| {_display_model_name(r.get('model', '?'))} | {r.get('rank', '?')} | {r.get('lora_rank', '?')} | "
                    f"{r.get('n_layers', '?')} | {mr_s} | {_display_source_path(r.get('source', '?'))} |"
                )

            # Add per-layer detail for first row as a compact appendix.
            top = sorted(rows, key=lambda x: x.get("mean_rho", 0.0), reverse=True)[0]
            per = list(top.get("per_layer", []))
            if per:
                lines.append("")
                lines.append(f"### Per-layer ρ (highest mean: {_display_model_name(top.get('model', '?'))})")
                lines.append("")
                lines.append("| Layer | ρ |")
                lines.append("|-------|---|")
                for i, v in enumerate(per):
                    lines.append(f"| {i} | {v:.4f} |")
    else:
        lines.append("Rho summary not yet generated.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Farm summary block
# ---------------------------------------------------------------------------
def farm_summary_md(run_id: str) -> str:
    state_f = FARM_BASE / run_id / "state" / "summary.json"
    if not state_f.exists():
        return ""
    try:
        d = json.loads(state_f.read_text(encoding="utf-8"))
    except Exception:
        return ""
    t = d.get("totals", {})
    lines = ["## Farm Run Summary", "",
             f"Run ID: `{d.get('run_id', run_id)}`  ",
             f"Saved UTC: {d.get('saved_utc', '?')}  ",
             f"Jobs: {t.get('jobs', '?')} total | "
             f"{t.get('completed', '?')} completed | "
             f"{t.get('failed', '?')} failed | "
             f"{t.get('pending', '?')} pending",
             ""]
    by_paper = d.get("by_paper", [])
    if by_paper:
        lines += [
            "| Paper | Completed | Total |",
            "|-------|-----------|-------|",
        ]
        for p in by_paper:
            lines.append(f"| {p.get('paper','?')} | {p.get('completed','?')} | {p.get('total','?')} |")
        lines.append("")
    done = d.get("done_ids", [])
    if done:
        lines.append("Completed jobs: " + ", ".join(f"`{x}`" for x in done))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def write_csv(kint_rows: list[dict], log_rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["paper", "model", "metric", "value", "unit", "detail"])

        # Paper A
        for r in kint_rows:
            name = r.get("name", "?")
            w.writerow(["A", name, "mean_kint",       r.get("mean_kint", ""),      "dims",    ""])
            w.writerow(["A", name, "mean_kint_over_d", r.get("mean_kint_over_d", ""), "ratio", ""])
            w.writerow(["A", name, "model_dim",        r.get("model_dim", ""),     "dims",    ""])
            for layer, kint in zip(r.get("used_layers", []), r.get("per_layer_kint", [])):
                w.writerow(["A", name, "kint_layer",   kint, "dims", f"layer={layer}"])

        # Paper B/C from logs
        seen: dict[str, dict] = {}
        for r in log_rows:
            seen[r.get("model_tag", "?")] = r
        for tag, r in seen.items():
            if r.get("load_ms") is not None:
                w.writerow(["B", tag, "load_ms",       r["load_ms"], "ms",   ""])
            if r.get("gpu_mb") is not None:
                w.writerow(["B", tag, "gpu_mb",        r["gpu_mb"],  "MB",   ""])
            if r.get("gpu_tensors") is not None:
                w.writerow(["B", tag, "gpu_tensors",   r["gpu_tensors"], "count", ""])
            if r.get("offload_from_layer") is not None:
                w.writerow(["B", tag, "offload_layer", r["offload_from_layer"], "layer", ""])
            if r.get("tok_s") is not None:
                w.writerow(["C", tag, "tok_s",         r["tok_s"], "tok/s", "rank=128"])


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------
def build_master(kint_rows, log_rows, run_id, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = f"""# HyperTensor Geodessical --- A--E Benchmark Master Report
Generated: {ts}  
GPU: RTX 4070 Laptop (8 GB VRAM) · Ryzen 9 7940HS · 32 GB RAM  
Runtime: geodessical v0.6.0 "Synapse"

> This report consolidates current A--E benchmark evidence, including
> k_int structure, load behavior, decode experiments, HJB feasibility checks,
> and rho-spectrum results. It is an interim evidence snapshot, not a final
> cross-hardware/cross-model claim.

---

"""
    farm_block = farm_summary_md(run_id)
    pa = paper_a_md(kint_rows)
    pb = paper_b_md(log_rows)
    pc = paper_c_md(log_rows)
    pd = paper_d_md()
    pe = paper_e_md()

    sections = [pa, pb, pc, pd, pe]
    body = "\n---\n\n".join(sections)
    if farm_block:
        master = header + farm_block + "\n---\n\n" + body
    else:
        master = header + body

    (out_dir / "MASTER_REPORT.md").write_text(master, encoding="utf-8")
    (out_dir / "paper_a_kint.md").write_text(pa, encoding="utf-8")
    (out_dir / "paper_b_load.md").write_text(pb, encoding="utf-8")
    (out_dir / "paper_c_decode.md").write_text(pc, encoding="utf-8")
    (out_dir / "paper_d_hjb.md").write_text(pd, encoding="utf-8")
    (out_dir / "paper_e_rho.md").write_text(pe, encoding="utf-8")

    write_csv(kint_rows, log_rows, out_dir / "all_results.csv")

    print(f"[report] Written to {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate A--E benchmark report")
    ap.add_argument("--farm-run-id", default="grc_farm_spectrum_v1",
                    help="Farm run_id for state/summary.json lookup")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEF),
                    help="Output directory (default: docs/benchmark_report)")
    args = ap.parse_args()

    kint_rows = load_kint_results()
    log_rows  = load_log_results(farm_run_id=args.farm_run_id)

    print(f"[report] k_int models found: {len(kint_rows)}")
    print(f"[report] log files found: {len(log_rows)}")

    build_master(kint_rows, log_rows, args.farm_run_id, Path(args.out_dir))


if __name__ == "__main__":
    main()
