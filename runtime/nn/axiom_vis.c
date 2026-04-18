/*
 * Geodessical Manifold Visualization — JSON emitter + HTML renderer writer
 *
 * Each phase emits a JSON file to the output directory:
 *   phase1_manifold.json  — PCA cloud, eigenvalues, intrinsic dim
 *   phase2_symmetry.json  — Head similarity matrix
 *   phase3_curvature.json — Metric field points, scalar curvature, Christoffel norms
 *   phase4_axioms.json    — Axiom candidates, types, confidences
 *   phase5_geodesic.json  — Trajectories, endpoints, scores
 *   index.html            — Self-contained renderer
 */

#include "runtime/nn/axiom_vis.h"
#include "runtime/nn/axiom_geo.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define VIS_MKDIR(p) _mkdir(p)
#else
#define VIS_MKDIR(p) mkdir(p, 0755)
#endif

#ifdef GEODESSICAL_HOSTED
#include "host/hal.h"
#else
#include "kernel/core/kernel.h"
#endif

/* ─── State ─── */
static char vis_dir[512];
static int  vis_active = 0;

/* Model metadata for comparison views */
static char vis_model_name[256] = "";
static char vis_model_arch[64]  = "";
static char vis_quant_type[32]  = "";
static int  vis_model_dim    = 0;
static int  vis_model_layers = 0;
static int  vis_model_vocab  = 0;
static int  vis_model_heads  = 0;

int axiom_vis_active(void) { return vis_active; }

void axiom_vis_set_model(const char *name, const char *arch,
                         int dim, int layers, int vocab, int heads,
                         const char *quant_type)
{
    if (name) snprintf(vis_model_name, sizeof(vis_model_name), "%s", name);
    if (arch) snprintf(vis_model_arch, sizeof(vis_model_arch), "%s", arch);
    if (quant_type) snprintf(vis_quant_type, sizeof(vis_quant_type), "%s", quant_type);
    vis_model_dim    = dim;
    vis_model_layers = layers;
    vis_model_vocab  = vocab;
    vis_model_heads  = heads;
}

int axiom_vis_init(const char *output_dir)
{
    if (!output_dir || !output_dir[0]) return -1;

    /* Ensure directory exists */
    VIS_MKDIR(output_dir);

    size_t len = strlen(output_dir);
    if (len >= sizeof(vis_dir) - 2) return -1;
    memcpy(vis_dir, output_dir, len);
    /* Strip trailing slash for consistent path building */
    if (vis_dir[len - 1] == '/' || vis_dir[len - 1] == '\\')
        vis_dir[len - 1] = '\0';
    else
        vis_dir[len] = '\0';

    vis_active = 1;
    kprintf("[VIS] Manifold visualization enabled → %s/\n", vis_dir);
    return 0;
}

/* Helper: build path */
static void vis_path(char *buf, int bufsize, const char *filename)
{
    snprintf(buf, bufsize, "%s/%s", vis_dir, filename);
}

/* Helper: sanitize NaN/Inf for JSON (JSON has no nan/inf literals) */
static double vis_safe(double v) { return isfinite(v) ? v : 0.0; }
static void vis_fprintf_safe(FILE *f, const char *fmt, double v)
{
    fprintf(f, fmt, vis_safe(v));
}

/* Helper: write model metadata block to JSON */
static void vis_write_model_meta(FILE *f)
{
    fprintf(f, "  \"model_name\": \"%s\",\n", vis_model_name);
    fprintf(f, "  \"model_arch\": \"%s\",\n", vis_model_arch);
    fprintf(f, "  \"model_dim\": %d,\n", vis_model_dim);
    fprintf(f, "  \"model_layers\": %d,\n", vis_model_layers);
    fprintf(f, "  \"model_vocab\": %d,\n", vis_model_vocab);
    fprintf(f, "  \"model_heads\": %d,\n", vis_model_heads);
    fprintf(f, "  \"quant_type\": \"%s\",\n", vis_quant_type);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Phase 1: PCA Embedding Cloud
 * ═══════════════════════════════════════════════════════════════════════════ */

int axiom_vis_emit_phase1(const axiom_phase1_t *p1,
                          const axpca_t *pca,
                          const double *proj_data,
                          int n_samples)
{
    if (!vis_active || !p1) return -1;

    char path[600];
    vis_path(path, sizeof(path), "phase1_manifold.json");
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int nc = p1->pca_components_kept;
    int show_dims = nc < 3 ? nc : 3; /* cap at 3 for visualization */

    fprintf(f, "{\n");
    fprintf(f, "  \"phase\": 1,\n");
    fprintf(f, "  \"title\": \"Manifold Identification\",\n");
    vis_write_model_meta(f);
    fprintf(f, "  \"intrinsic_dim\": %d,\n", p1->intrinsic_dim);
    fprintf(f, "  \"pca_components\": %d,\n", nc);
    fprintf(f, "  \"explained_ratio\": %.6f,\n", p1->explained_ratio);
    fprintf(f, "  \"total_variance\": %.6f,\n", p1->total_variance);
    fprintf(f, "  \"twonn_raw\": %.4f,\n", p1->twonn_raw);
    fprintf(f, "  \"embedding_dim\": %d,\n", p1->embedding_dim);
    fprintf(f, "  \"n_samples\": %d,\n", n_samples);

    /* Eigenvalues (scree plot data) */
    fprintf(f, "  \"eigenvalues\": [");
    if (pca && pca->eigenvalues) {
        for (int i = 0; i < nc; i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.8f", pca->eigenvalues[i]);
        }
    }
    fprintf(f, "],\n");

    /* Projected embedding cloud (first 3 PCA dims) */
    fprintf(f, "  \"cloud\": [\n");
    if (proj_data && n_samples > 0) {
        int emit_n = n_samples > 2048 ? 2048 : n_samples; /* cap for file size */
        for (int i = 0; i < emit_n; i++) {
            fprintf(f, "    [");
            for (int d = 0; d < show_dims; d++) {
                if (d > 0) fprintf(f, ", ");
                fprintf(f, "%.6f", proj_data[i * nc + d]);
            }
            fprintf(f, "]%s\n", (i < emit_n - 1) ? "," : "");
        }
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    kprintf("[VIS] Phase 1 manifold data → %s\n", path);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Phase 2: Head Similarity Heatmap
 * ═══════════════════════════════════════════════════════════════════════════ */

int axiom_vis_emit_phase2(const axiom_phase2_t *p2,
                          const double *head_sims,
                          int n_heads)
{
    if (!vis_active || !p2) return -1;

    char path[600];
    vis_path(path, sizeof(path), "phase2_symmetry.json");
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "{\n");
    fprintf(f, "  \"phase\": 2,\n");
    fprintf(f, "  \"title\": \"Symmetry Extraction\",\n");
    vis_write_model_meta(f);
    fprintf(f, "  \"symmetry_score\": %.6f,\n", p2->symmetry_score);
    fprintf(f, "  \"generators_found\": %d,\n", p2->generators_found);
    fprintf(f, "  \"head_similarity_mean\": %.6f,\n", p2->head_similarity_mean);
    fprintf(f, "  \"head_similarity_max\": %.6f,\n", p2->head_similarity_max);
    fprintf(f, "  \"n_heads\": %d,\n", n_heads);

    /* Head similarity matrix (upper triangle, row-major) */
    fprintf(f, "  \"similarity_matrix\": [\n");
    if (head_sims && n_heads > 0) {
        for (int i = 0; i < n_heads; i++) {
            fprintf(f, "    [");
            for (int j = 0; j < n_heads; j++) {
                if (j > 0) fprintf(f, ", ");
                fprintf(f, "%.4f", head_sims[i * n_heads + j]);
            }
            fprintf(f, "]%s\n", (i < n_heads - 1) ? "," : "");
        }
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    kprintf("[VIS] Phase 2 symmetry data → %s\n", path);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Phase 3: Metric Field + Curvature
 * ═══════════════════════════════════════════════════════════════════════════ */

int axiom_vis_emit_phase3(const axiom_phase3_t *p3,
                          const axgeo_metric_field_t *mf,
                          const axgeo_christoffel_t *ch,
                          const double *curv_scalar,
                          int n_points, int dim)
{
    if (!vis_active || !p3) return -1;

    char path[600];
    vis_path(path, sizeof(path), "phase3_curvature.json");
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int show_dims = dim < 3 ? dim : 3;

    fprintf(f, "{\n");
    fprintf(f, "  \"phase\": 3,\n");
    fprintf(f, "  \"title\": \"Curvature Field\",\n");
    vis_write_model_meta(f);
    fprintf(f, "  \"mean_scalar_curvature\": %.8f,\n", vis_safe(p3->mean_scalar_curvature));
    fprintf(f, "  \"max_scalar_curvature\": %.8f,\n", vis_safe(p3->max_scalar_curvature));
    fprintf(f, "  \"min_scalar_curvature\": %.8f,\n", vis_safe(p3->min_scalar_curvature));
    fprintf(f, "  \"curvature_std\": %.8f,\n", vis_safe(p3->curvature_std));
    fprintf(f, "  \"high_curvature_loci\": %d,\n", p3->high_curvature_loci);
    fprintf(f, "  \"metric_field_points\": %d,\n", n_points);
    fprintf(f, "  \"dim\": %d,\n", dim);
    fprintf(f, "  \"fisher_trace_mean\": %.8f,\n", vis_safe(p3->fisher_trace_mean));
    fprintf(f, "  \"fisher_det_log_mean\": %.8f,\n", vis_safe(p3->fisher_det_log_mean));

    /* Metric field sample points (first 3 PCA dims) + scalar curvature */
    fprintf(f, "  \"points\": [\n");
    if (mf && mf->points) {
        for (int i = 0; i < n_points; i++) {
            const double *pt = mf->points + (uint64_t)i * dim;
            fprintf(f, "    {\"pos\": [");
            for (int d = 0; d < show_dims; d++) {
                if (d > 0) fprintf(f, ", ");
                fprintf(f, "%.6f", vis_safe(pt[d]));
            }
            fprintf(f, "]");

            /* Scalar curvature at this point */
            if (curv_scalar) {
                fprintf(f, ", \"R\": %.8f", vis_safe(curv_scalar[i]));
            }

            /* Metric eigenvalues at this point (for anisotropy visualization) */
            if (mf->metrics) {
                const double *g = mf->metrics + (uint64_t)i * dim * dim;
                /* Compute diagonal of metric (approximate eigenvalues for vis) */
                fprintf(f, ", \"g_diag\": [");
                for (int d = 0; d < show_dims; d++) {
                    if (d > 0) fprintf(f, ", ");
                    fprintf(f, "%.6f", vis_safe(g[d * dim + d]));
                }
                fprintf(f, "]");
            }

            /* Christoffel symbol norm at this point */
            if (ch && ch->gamma) {
                /* Christoffel is global (n_points=1), so always use index 0 */
                int ci = (i < ch->n_points) ? i : 0;
                const double *G = ch->gamma + (uint64_t)ci * dim * dim * dim;
                double norm2 = 0.0;
                for (int a = 0; a < dim * dim * dim; a++)
                    norm2 += G[a] * G[a];
                fprintf(f, ", \"christoffel_norm\": %.8f", sqrt(norm2));
            }

            fprintf(f, "}%s\n", (i < n_points - 1) ? "," : "");
        }
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    kprintf("[VIS] Phase 3 curvature data → %s (%d points, %d-dim)\n",
            path, n_points, dim);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Phase 4: Axiom Confidence Distribution
 * ═══════════════════════════════════════════════════════════════════════════ */

int axiom_vis_emit_phase4(const axiom_phase4_t *p4)
{
    if (!vis_active || !p4) return -1;

    char path[600];
    vis_path(path, sizeof(path), "phase4_axioms.json");
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "{\n");
    fprintf(f, "  \"phase\": 4,\n");
    fprintf(f, "  \"title\": \"Axiom Formalization\",\n");
    vis_write_model_meta(f);
    fprintf(f, "  \"axiom_count\": %d,\n", p4->axiom_count);
    fprintf(f, "  \"consistency_score\": %.6f,\n", p4->consistency_score);
    fprintf(f, "  \"candidates_tested\": %d,\n", p4->candidates_tested);
    fprintf(f, "  \"candidates_accepted\": %d,\n", p4->candidates_accepted);
    fprintf(f, "  \"oracle_calls_used\": %d,\n", p4->oracle_calls_used);
    fprintf(f, "  \"information_gain\": %.6f\n", p4->information_gain);
    fprintf(f, "}\n");

    fclose(f);
    kprintf("[VIS] Phase 4 axiom data → %s\n", path);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Phase 5: Geodesic Trajectories
 * ═══════════════════════════════════════════════════════════════════════════ */

int axiom_vis_emit_phase5(const axiom_phase5_t *p5,
                          const double **trajectories,
                          const int *traj_steps,
                          const double *endpoints_true,
                          const double *cosine_sims,
                          int n_test, int dim)
{
    if (!vis_active || !p5) return -1;

    char path[600];
    vis_path(path, sizeof(path), "phase5_geodesic.json");
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int show_dims = dim < 3 ? dim : 3;

    fprintf(f, "{\n");
    fprintf(f, "  \"phase\": 5,\n");
    fprintf(f, "  \"title\": \"Geodesic Pilot\",\n");
    vis_write_model_meta(f);
    fprintf(f, "  \"cosine_similarity\": %.6f,\n", p5->geodesic_cosine_similarity);
    fprintf(f, "  \"reconstruction_error\": %.6f,\n", p5->geodesic_reconstruction_error);
    fprintf(f, "  \"path_length\": %.6f,\n", p5->geodesic_path_length);
    fprintf(f, "  \"top1_match_rate\": %.6f,\n", p5->geodesic_top1_match_rate);
    fprintf(f, "  \"target_mrr\": %.6f,\n", p5->geodesic_target_mrr);
    fprintf(f, "  \"projected_speedup\": %.2f,\n", p5->projected_speedup);
    fprintf(f, "  \"converged\": %d,\n", p5->geodesic_converged);
    fprintf(f, "  \"n_test\": %d,\n", n_test);
    fprintf(f, "  \"dim\": %d,\n", dim);

    /* Geodesic paths */
    fprintf(f, "  \"trajectories\": [\n");
    if (trajectories && traj_steps) {
        for (int t = 0; t < n_test; t++) {
            int steps = traj_steps[t];
            fprintf(f, "    {\n");
            fprintf(f, "      \"steps\": %d,\n", steps);
            if (cosine_sims) fprintf(f, "      \"cos_sim\": %.6f,\n", cosine_sims[t]);

            /* True target point */
            if (endpoints_true) {
                const double *ep = endpoints_true + t * dim;
                fprintf(f, "      \"target\": [");
                for (int d = 0; d < show_dims; d++) {
                    if (d > 0) fprintf(f, ", ");
                    fprintf(f, "%.6f", ep[d]);
                }
                fprintf(f, "],\n");
            }

            /* Trajectory path points */
            fprintf(f, "      \"path\": [\n");
            if (trajectories[t] && steps > 0) {
                for (int s = 0; s <= steps; s++) {
                    const double *pt = trajectories[t] + s * dim;
                    fprintf(f, "        [");
                    for (int d = 0; d < show_dims; d++) {
                        if (d > 0) fprintf(f, ", ");
                        fprintf(f, "%.6f", pt[d]);
                    }
                    fprintf(f, "]%s\n", (s < steps) ? "," : "");
                }
            }
            fprintf(f, "      ]\n");
            fprintf(f, "    }%s\n", (t < n_test - 1) ? "," : "");
        }
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    kprintf("[VIS] Phase 5 geodesic data → %s (%d trajectories)\n",
            path, n_test);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Finalize: Write index.html renderer
 * ═══════════════════════════════════════════════════════════════════════════ */

int axiom_vis_finalize(void)
{
    if (!vis_active) return -1;

    char path[600];
    vis_path(path, sizeof(path), "index.html");
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Self-contained HTML/JS renderer */
    fprintf(f,
"<!DOCTYPE html>\n"
"<html lang=\"en\">\n"
"<head>\n"
"<meta charset=\"UTF-8\">\n"
"<title>Geodessical — Riemannian Manifold Visualization</title>\n"
"<style>\n"
"* { margin: 0; padding: 0; box-sizing: border-box; }\n"
"body { background: #0a0a0f; color: #e0e0e0; font-family: 'JetBrains Mono', 'Fira Code', monospace; }\n"
"#header { padding: 20px; border-bottom: 1px solid #222; text-align: center; }\n"
"#header h1 { font-size: 1.6em; color: #7af; letter-spacing: 2px; }\n"
"#header .sub { font-size: 0.85em; color: #667; margin-top: 4px; }\n"
"#phases { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; padding: 12px; }\n"
".phase-btn { padding: 8px 18px; border: 1px solid #333; background: #151520; color: #8af;\n"
"  cursor: pointer; border-radius: 4px; font-family: inherit; font-size: 0.85em; }\n"
".phase-btn.active { background: #1a2a4a; border-color: #5af; color: #fff; }\n"
".phase-btn:hover { background: #1a1a2a; }\n"
"#view { padding: 16px; display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; }\n"
".panel { background: #111118; border: 1px solid #222; border-radius: 8px; padding: 16px;\n"
"  min-width: 320px; max-width: 700px; flex: 1; }\n"
".panel h2 { font-size: 1.1em; color: #7af; margin-bottom: 10px; }\n"
".stat { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1a1a2a; }\n"
".stat-label { color: #889; }\n"
".stat-value { color: #aef; font-weight: bold; }\n"
"canvas { display: block; margin: 10px auto; border: 1px solid #222; border-radius: 4px; background: #08080d; }\n"
".legend { font-size: 0.75em; color: #667; text-align: center; margin-top: 4px; }\n"
"#loading { text-align: center; padding: 80px; color: #556; font-size: 1.2em; }\n"
"</style>\n"
"</head>\n"
"<body>\n"
"<div id=\"header\">\n"
"  <h1>GEODESSICAL</h1>\n"
"  <div class=\"sub\">Riemannian Manifold Visualization — Organic Training Theory</div>\n"
"</div>\n"
"<div id=\"phases\"></div>\n"
"<div id=\"view\"><div id=\"loading\">Loading manifold data...</div></div>\n"
"\n"
"<script>\n"
"const FILES = [\n"
"  'phase1_manifold.json', 'phase2_symmetry.json',\n"
"  'phase3_curvature.json', 'phase4_axioms.json', 'phase5_geodesic.json'\n"
"];\n"
"const PHASE_LABELS = ['Phase 1: Manifold', 'Phase 2: Symmetry', 'Phase 3: Curvature',\n"
"                      'Phase 4: Axioms', 'Phase 5: Geodesic'];\n"
"let data = {};\n"
    );

    /* ─── Inline JSON data to avoid CORS issues with file:// ─── */
    {
        static const char *json_files[] = {
            "phase1_manifold.json", "phase2_symmetry.json",
            "phase3_curvature.json", "phase4_axioms.json", "phase5_geodesic.json"
        };
        for (int i = 0; i < 5; i++) {
            char jp[600];
            vis_path(jp, sizeof(jp), json_files[i]);
            FILE *jf = fopen(jp, "rb");
            if (!jf) continue;
            fprintf(f, "data['%s'] = ", json_files[i]);
            char cbuf[4096];
            size_t nr;
            while ((nr = fread(cbuf, 1, sizeof(cbuf), jf)) > 0)
                fwrite(cbuf, 1, nr, f);
            fprintf(f, ";\n");
            fclose(jf);
        }
    }

    fprintf(f,
"\n"
"function loadData() { buildUI(); }\n"
"\n"
"function buildUI() {\n"
"  const pb = document.getElementById('phases');\n"
"  pb.innerHTML = '';\n"
"  FILES.forEach((f, i) => {\n"
"    if (!data[f]) return;\n"
"    const btn = document.createElement('button');\n"
"    btn.className = 'phase-btn';\n"
"    btn.textContent = PHASE_LABELS[i];\n"
"    btn.onclick = () => showPhase(i, btn);\n"
"    pb.appendChild(btn);\n"
"  });\n"
"  /* Auto-show first available phase */\n"
"  const first = pb.querySelector('.phase-btn');\n"
"  if (first) first.click();\n"
"  else document.getElementById('view').innerHTML = '<div id=\"loading\">No manifold data found.</div>';\n"
"}\n"
"\n"
"function showPhase(idx, btn) {\n"
"  document.querySelectorAll('.phase-btn').forEach(b => b.classList.remove('active'));\n"
"  btn.classList.add('active');\n"
"  const view = document.getElementById('view');\n"
"  view.innerHTML = '';\n"
"  switch(idx) {\n"
"    case 0: renderPhase1(view); break;\n"
"    case 1: renderPhase2(view); break;\n"
"    case 2: renderPhase3(view); break;\n"
"    case 3: renderPhase4(view); break;\n"
"    case 4: renderPhase5(view); break;\n"
"  }\n"
"}\n"
"\n"
"/* ─── Helpers ─── */\n"
"function mkPanel(parent, title) {\n"
"  const p = document.createElement('div'); p.className = 'panel';\n"
"  const h = document.createElement('h2'); h.textContent = title;\n"
"  p.appendChild(h); parent.appendChild(p); return p;\n"
"}\n"
"function addStat(panel, label, value) {\n"
"  const d = document.createElement('div'); d.className = 'stat';\n"
"  d.innerHTML = `<span class=\"stat-label\">${label}</span>`\n"
"    + `<span class=\"stat-value\">${value}</span>`;\n"
"  panel.appendChild(d);\n"
"}\n"
"function mkCanvas(panel, w, h) {\n"
"  const c = document.createElement('canvas'); c.width = w; c.height = h;\n"
"  panel.appendChild(c); return c;\n"
"}\n"
"\n"
"/* Color ramps */\n"
"function curvColor(v, lo, hi) {\n"
"  let t = (hi > lo) ? (v - lo) / (hi - lo) : 0.5;\n"
"  t = Math.max(0, Math.min(1, t));\n"
"  /* Blue (low R) → Cyan → Green → Yellow → Red (high R) */\n"
"  const r = Math.floor(255 * Math.min(1, 2*t));\n"
"  const g = Math.floor(255 * (t < 0.5 ? 2*t : 2*(1-t)));\n"
"  const b = Math.floor(255 * Math.min(1, 2*(1-t)));\n"
"  return `rgb(${r},${g},${b})`;\n"
"}\n"
"function simColor(v) {\n"
"  const t = Math.max(0, Math.min(1, v));\n"
"  return `rgb(${Math.floor(30+200*t)},${Math.floor(50+150*t)},${Math.floor(200-100*t)})`;\n"
"}\n"
"\n"
"/* ─── Phase 1: PCA Cloud & Scree Plot ─── */\n"
"function renderPhase1(view) {\n"
"  const d = data['phase1_manifold.json']; if (!d) return;\n"
"  /* Stats */\n"
"  const sp = mkPanel(view, 'Manifold Identification');\n"
"  addStat(sp, 'Intrinsic Dimension (TwoNN)', d.intrinsic_dim);\n"
"  addStat(sp, 'PCA Components Kept', d.pca_components);\n"
"  addStat(sp, 'Explained Variance', (d.explained_ratio * 100).toFixed(2) + '%%');\n"
"  addStat(sp, 'Total Variance', d.total_variance.toFixed(4));\n"
"  addStat(sp, 'Embedding Dim', d.embedding_dim);\n"
"  addStat(sp, 'Samples', d.n_samples);\n"
"\n"
"  /* Scree plot */\n"
"  if (d.eigenvalues && d.eigenvalues.length > 0) {\n"
"    const cp = mkPanel(view, 'Scree Plot (PCA Eigenvalues)');\n"
"    const c = mkCanvas(cp, 600, 280);\n"
"    const ctx = c.getContext('2d');\n"
"    const ev = d.eigenvalues;\n"
"    const maxE = Math.max(...ev);\n"
"    const pad = 50;\n"
"    ctx.fillStyle = '#08080d'; ctx.fillRect(0,0,c.width,c.height);\n"
"    /* Axes */\n"
"    ctx.strokeStyle = '#333'; ctx.lineWidth = 1;\n"
"    ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, c.height-pad);\n"
"    ctx.lineTo(c.width-10, c.height-pad); ctx.stroke();\n"
"    /* Bars */\n"
"    const bw = Math.max(2, (c.width - pad - 20) / ev.length - 1);\n"
"    for (let i = 0; i < ev.length; i++) {\n"
"      const h = (ev[i] / maxE) * (c.height - pad - 20);\n"
"      const x = pad + 5 + i * (bw + 1);\n"
"      ctx.fillStyle = i < d.intrinsic_dim ? '#5af' : '#334';\n"
"      ctx.fillRect(x, c.height - pad - h, bw, h);\n"
"    }\n"
"    /* Labels */\n"
"    ctx.fillStyle = '#778'; ctx.font = '10px monospace';\n"
"    ctx.fillText('Component', c.width/2-30, c.height-5);\n"
"    ctx.save(); ctx.translate(12, c.height/2); ctx.rotate(-Math.PI/2);\n"
"    ctx.fillText('Eigenvalue', 0, 0); ctx.restore();\n"
"    const lg = document.createElement('div'); lg.className = 'legend';\n"
"    lg.textContent = `Blue = within intrinsic dim (${d.intrinsic_dim}), Gray = beyond`;\n"
"    cp.appendChild(lg);\n"
"  }\n"
"\n"
"  /* 2D/3D PCA cloud */\n"
"  if (d.cloud && d.cloud.length > 0) {\n"
"    const cp = mkPanel(view, 'Embedding Cloud (PCA Projection)');\n"
"    const c = mkCanvas(cp, 600, 500);\n"
"    drawCloud(c, d.cloud);\n"
"    const lg = document.createElement('div'); lg.className = 'legend';\n"
"    lg.textContent = `${d.cloud.length} samples in ${d.cloud[0].length}D PCA subspace`;\n"
"    cp.appendChild(lg);\n"
"  }\n"
"}\n"
"\n"
"function drawCloud(canvas, pts) {\n"
"  const ctx = canvas.getContext('2d');\n"
"  ctx.fillStyle = '#08080d'; ctx.fillRect(0,0,canvas.width,canvas.height);\n"
"  if (pts.length === 0) return;\n"
"  const nd = pts[0].length;\n"
"  /* Find bounds */\n"
"  let mins = new Array(nd).fill(Infinity), maxs = new Array(nd).fill(-Infinity);\n"
"  for (const p of pts) {\n"
"    for (let d = 0; d < nd; d++) { mins[d] = Math.min(mins[d], p[d]); maxs[d] = Math.max(maxs[d], p[d]); }\n"
"  }\n"
"  const pad = 40, w = canvas.width - 2*pad, h = canvas.height - 2*pad;\n"
"  function proj(p) {\n"
"    const x = (maxs[0] > mins[0]) ? (p[0]-mins[0])/(maxs[0]-mins[0]) : 0.5;\n"
"    const y = (nd > 1 && maxs[1] > mins[1]) ? (p[1]-mins[1])/(maxs[1]-mins[1]) : 0.5;\n"
"    return [pad + x*w, pad + (1-y)*h];\n"
"  }\n"
"  /* Grid */\n"
"  ctx.strokeStyle = '#181825'; ctx.lineWidth = 0.5;\n"
"  for (let i = 0; i <= 4; i++) {\n"
"    const gy = pad + i*h/4; ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(pad+w, gy); ctx.stroke();\n"
"    const gx = pad + i*w/4; ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, pad+h); ctx.stroke();\n"
"  }\n"
"  /* Points */\n"
"  for (const p of pts) {\n"
"    const [x, y] = proj(p);\n"
"    let depth = nd > 2 ? (maxs[2] > mins[2] ? (p[2]-mins[2])/(maxs[2]-mins[2]) : 0.5) : 0.5;\n"
"    const alpha = 0.3 + 0.5 * depth;\n"
"    const size = 1.5 + 2 * depth;\n"
"    ctx.fillStyle = `rgba(100,180,255,${alpha})`;\n"
"    ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI*2); ctx.fill();\n"
"  }\n"
"  /* Axis labels */\n"
"  ctx.fillStyle = '#556'; ctx.font = '11px monospace';\n"
"  ctx.fillText('PC1', canvas.width/2, canvas.height - 5);\n"
"  ctx.save(); ctx.translate(10, canvas.height/2); ctx.rotate(-Math.PI/2);\n"
"  ctx.fillText('PC2', 0, 0); ctx.restore();\n"
"}\n"
"\n"
"/* ─── Phase 2: Head Similarity Heatmap ─── */\n"
"function renderPhase2(view) {\n"
"  const d = data['phase2_symmetry.json']; if (!d) return;\n"
"  const sp = mkPanel(view, 'Symmetry Extraction');\n"
"  addStat(sp, 'Symmetry Score', d.symmetry_score.toFixed(4));\n"
"  addStat(sp, 'Generators Found', d.generators_found);\n"
"  addStat(sp, 'Head Similarity (mean)', d.head_similarity_mean.toFixed(4));\n"
"  addStat(sp, 'Head Similarity (max)', d.head_similarity_max.toFixed(4));\n"
"  addStat(sp, 'Total Heads', d.n_heads);\n"
"\n"
"  if (d.similarity_matrix && d.similarity_matrix.length > 0) {\n"
"    const hp = mkPanel(view, 'Head Similarity Heatmap');\n"
"    const n = d.n_heads;\n"
"    const cellSize = Math.min(16, Math.floor(500 / n));\n"
"    const sz = cellSize * n + 60;\n"
"    const c = mkCanvas(hp, sz, sz);\n"
"    const ctx = c.getContext('2d');\n"
"    ctx.fillStyle = '#08080d'; ctx.fillRect(0,0,sz,sz);\n"
"    const off = 40;\n"
"    for (let i = 0; i < n; i++) {\n"
"      for (let j = 0; j < n; j++) {\n"
"        ctx.fillStyle = simColor(d.similarity_matrix[i][j]);\n"
"        ctx.fillRect(off + j*cellSize, off + i*cellSize, cellSize-1, cellSize-1);\n"
"      }\n"
"    }\n"
"    ctx.fillStyle = '#556'; ctx.font = '10px monospace';\n"
"    ctx.fillText('Head j →', sz/2-20, sz-5);\n"
"    ctx.save(); ctx.translate(8, sz/2); ctx.rotate(-Math.PI/2);\n"
"    ctx.fillText('Head i →', 0, 0); ctx.restore();\n"
"    const lg = document.createElement('div'); lg.className = 'legend';\n"
"    lg.textContent = `${n}x${n} cosine similarity, bright = high similarity`;\n"
"    hp.appendChild(lg);\n"
"  }\n"
"}\n"
"\n"
"/* ─── Phase 3: Curvature Heatmap ─── */\n"
"function renderPhase3(view) {\n"
"  const d = data['phase3_curvature.json']; if (!d) return;\n"
"  const sp = mkPanel(view, 'Curvature Field');\n"
"  addStat(sp, 'Mean Scalar R', d.mean_scalar_curvature.toExponential(3));\n"
"  addStat(sp, 'Max Scalar R', d.max_scalar_curvature.toExponential(3));\n"
"  addStat(sp, 'Min Scalar R', d.min_scalar_curvature.toExponential(3));\n"
"  addStat(sp, 'Std Dev', d.curvature_std.toExponential(3));\n"
"  addStat(sp, 'High-Curvature Loci', d.high_curvature_loci);\n"
"  addStat(sp, 'Fisher Trace (mean)', d.fisher_trace_mean.toExponential(3));\n"
"  addStat(sp, 'Metric Field Points', d.metric_field_points);\n"
"  addStat(sp, 'Subspace Dim', d.dim);\n"
"\n"
"  if (d.points && d.points.length > 0) {\n"
"    /* Curvature scatter plot */\n"
"    const cp = mkPanel(view, 'Scalar Curvature Map');\n"
"    const c = mkCanvas(cp, 600, 500);\n"
"    drawCurvatureMap(c, d.points);\n"
"    const lg = document.createElement('div'); lg.className = 'legend';\n"
"    lg.textContent = 'Blue = low R (flat), Red = high |R| (curved). Size = ||Γ|| (connection strength)';\n"
"    cp.appendChild(lg);\n"
"\n"
"    /* Christoffel norm distribution */\n"
"    const np = mkPanel(view, 'Christoffel Norm Distribution');\n"
"    const c2 = mkCanvas(np, 600, 250);\n"
"    drawChristoffelHist(c2, d.points);\n"
"  }\n"
"}\n"
"\n"
"function drawCurvatureMap(canvas, points) {\n"
"  const ctx = canvas.getContext('2d');\n"
"  ctx.fillStyle = '#08080d'; ctx.fillRect(0,0,canvas.width,canvas.height);\n"
"  if (points.length === 0) return;\n"
"  let Rs = points.filter(p=>p.R!==undefined).map(p=>p.R);\n"
"  let absRs = Rs.map(Math.abs);\n"
"  const minR = Math.min(...Rs), maxR = Math.max(...Rs);\n"
"  const maxAbsR = Math.max(...absRs) || 1;\n"
"  let cNorms = points.filter(p=>p.christoffel_norm!==undefined).map(p=>p.christoffel_norm);\n"
"  const maxCN = cNorms.length > 0 ? Math.max(...cNorms) : 1;\n"
"  /* Bounds from position */\n"
"  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity;\n"
"  for (const p of points) {\n"
"    if (!p.pos) continue;\n"
"    minX = Math.min(minX, p.pos[0]); maxX = Math.max(maxX, p.pos[0]);\n"
"    if (p.pos.length > 1) { minY = Math.min(minY, p.pos[1]); maxY = Math.max(maxY, p.pos[1]); }\n"
"  }\n"
"  const pad = 50, w = canvas.width - 2*pad, h = canvas.height - 2*pad;\n"
"  for (const p of points) {\n"
"    if (!p.pos) continue;\n"
"    const x = pad + (maxX>minX ? (p.pos[0]-minX)/(maxX-minX) : 0.5) * w;\n"
"    const y = pad + (1 - (maxY>minY && p.pos.length>1 ? (p.pos[1]-minY)/(maxY-minY) : 0.5)) * h;\n"
"    const R = p.R || 0;\n"
"    const cn = p.christoffel_norm || 0;\n"
"    const size = 3 + 8 * (maxCN > 0 ? cn / maxCN : 0);\n"
"    ctx.fillStyle = curvColor(Math.abs(R), 0, maxAbsR);\n"
"    ctx.globalAlpha = 0.8;\n"
"    ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI*2); ctx.fill();\n"
"    ctx.globalAlpha = 1;\n"
"  }\n"
"  ctx.fillStyle = '#556'; ctx.font = '11px monospace';\n"
"  ctx.fillText('PC1', canvas.width/2, canvas.height - 5);\n"
"  ctx.save(); ctx.translate(10, canvas.height/2); ctx.rotate(-Math.PI/2);\n"
"  ctx.fillText('PC2', 0, 0); ctx.restore();\n"
"}\n"
"\n"
"function drawChristoffelHist(canvas, points) {\n"
"  const ctx = canvas.getContext('2d');\n"
"  ctx.fillStyle = '#08080d'; ctx.fillRect(0,0,canvas.width,canvas.height);\n"
"  let cns = points.filter(p=>p.christoffel_norm!==undefined).map(p=>p.christoffel_norm).sort((a,b)=>a-b);\n"
"  if (cns.length === 0) return;\n"
"  const nbins = 30;\n"
"  const lo = cns[0], hi = cns[cns.length-1];\n"
"  const binW = (hi - lo) / nbins || 1;\n"
"  let bins = new Array(nbins).fill(0);\n"
"  for (const v of cns) { const b = Math.min(nbins-1, Math.floor((v-lo)/binW)); bins[b]++; }\n"
"  const maxBin = Math.max(...bins);\n"
"  const pad = 50, w = canvas.width - pad - 20, h2 = canvas.height - pad - 20;\n"
"  const bw = w / nbins;\n"
"  ctx.strokeStyle = '#333'; ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, canvas.height-pad);\n"
"  ctx.lineTo(canvas.width-10, canvas.height-pad); ctx.stroke();\n"
"  for (let i = 0; i < nbins; i++) {\n"
"    const bh = maxBin > 0 ? (bins[i]/maxBin)*h2 : 0;\n"
"    ctx.fillStyle = '#5a8';\n"
"    ctx.fillRect(pad + i*bw, canvas.height-pad-bh, bw-1, bh);\n"
"  }\n"
"  ctx.fillStyle = '#556'; ctx.font = '10px monospace';\n"
"  ctx.fillText('||Γ||_F', canvas.width/2-15, canvas.height-5);\n"
"}\n"
"\n"
"/* ─── Phase 4: Axiom Summary ─── */\n"
"function renderPhase4(view) {\n"
"  const d = data['phase4_axioms.json']; if (!d) return;\n"
"  const sp = mkPanel(view, 'Axiom Formalization');\n"
"  addStat(sp, 'Unique Axioms', d.axiom_count);\n"
"  addStat(sp, 'Consistency Score', d.consistency_score.toFixed(4));\n"
"  addStat(sp, 'Candidates Tested', d.candidates_tested);\n"
"  addStat(sp, 'Candidates Accepted', d.candidates_accepted);\n"
"  addStat(sp, 'Oracle Calls', d.oracle_calls_used);\n"
"  addStat(sp, 'Information Gain', d.information_gain.toFixed(4));\n"
"\n"
"  /* Simple bar chart: accepted vs tested */\n"
"  const cp = mkPanel(view, 'Axiom Discovery Funnel');\n"
"  const c = mkCanvas(cp, 400, 200);\n"
"  const ctx = c.getContext('2d');\n"
"  ctx.fillStyle = '#08080d'; ctx.fillRect(0,0,c.width,c.height);\n"
"  const bars = [\n"
"    { label: 'Tested', val: d.candidates_tested, color: '#335' },\n"
"    { label: 'Accepted', val: d.candidates_accepted, color: '#358' },\n"
"    { label: 'Unique', val: d.axiom_count, color: '#5af' }\n"
"  ];\n"
"  const maxV = Math.max(...bars.map(b=>b.val)) || 1;\n"
"  const bw = 80, gap = 30, startX = 60;\n"
"  bars.forEach((b, i) => {\n"
"    const h = (b.val / maxV) * 140;\n"
"    const x = startX + i * (bw + gap);\n"
"    ctx.fillStyle = b.color;\n"
"    ctx.fillRect(x, 170 - h, bw, h);\n"
"    ctx.fillStyle = '#aaf'; ctx.font = '11px monospace';\n"
"    ctx.fillText(b.val.toString(), x + bw/2 - 8, 165 - h);\n"
"    ctx.fillStyle = '#778';\n"
"    ctx.fillText(b.label, x + 5, 190);\n"
"  });\n"
"}\n"
"\n"
"/* ─── Phase 5: Geodesic Trajectories ─── */\n"
"function renderPhase5(view) {\n"
"  const d = data['phase5_geodesic.json']; if (!d) return;\n"
"  const sp = mkPanel(view, 'Geodesic Pilot');\n"
"  addStat(sp, 'Cosine Similarity', d.cosine_similarity.toFixed(4));\n"
"  addStat(sp, 'Reconstruction Error (L2)', d.reconstruction_error.toFixed(4));\n"
"  addStat(sp, 'Path Length', d.path_length.toFixed(4));\n"
"  addStat(sp, 'Top-1 Match Rate', (d.top1_match_rate*100).toFixed(1) + '%%');\n"
"  addStat(sp, 'Target MRR', d.target_mrr.toFixed(4));\n"
"  addStat(sp, 'Projected Speedup', d.projected_speedup.toFixed(1) + 'x');\n"
"  addStat(sp, 'Converged', d.converged ? 'Yes' : 'No');\n"
"\n"
"  if (d.trajectories && d.trajectories.length > 0) {\n"
"    const tp = mkPanel(view, 'Geodesic Trajectories (PCA Subspace)');\n"
"    const c = mkCanvas(tp, 600, 500);\n"
"    drawTrajectories(c, d.trajectories);\n"
"    const lg = document.createElement('div'); lg.className = 'legend';\n"
"    lg.textContent = `${d.trajectories.length} geodesic paths. ○ = start, × = target, ● = endpoint. `\n"
"      + 'Green = high cos_sim, Red = low.';\n"
"    tp.appendChild(lg);\n"
"  }\n"
"}\n"
"\n"
"function drawTrajectories(canvas, trajs) {\n"
"  const ctx = canvas.getContext('2d');\n"
"  ctx.fillStyle = '#08080d'; ctx.fillRect(0,0,canvas.width,canvas.height);\n"
"  /* Gather all points for bounds */\n"
"  let allX = [], allY = [];\n"
"  for (const t of trajs) {\n"
"    if (!t.path) continue;\n"
"    for (const p of t.path) { allX.push(p[0]); if (p.length > 1) allY.push(p[1]); }\n"
"    if (t.target) { allX.push(t.target[0]); if (t.target.length > 1) allY.push(t.target[1]); }\n"
"  }\n"
"  if (allX.length === 0) return;\n"
"  const minX = Math.min(...allX), maxX = Math.max(...allX);\n"
"  const minY = allY.length > 0 ? Math.min(...allY) : 0;\n"
"  const maxY = allY.length > 0 ? Math.max(...allY) : 1;\n"
"  const pad = 50, w = canvas.width - 2*pad, h = canvas.height - 2*pad;\n"
"  function proj(p) {\n"
"    const x = pad + (maxX>minX ? (p[0]-minX)/(maxX-minX) : 0.5) * w;\n"
"    const y = pad + (1 - (maxY>minY && p.length>1 ? (p[1]-minY)/(maxY-minY) : 0.5)) * h;\n"
"    return [x, y];\n"
"  }\n"
"  /* Grid */\n"
"  ctx.strokeStyle = '#181825'; ctx.lineWidth = 0.5;\n"
"  for (let i = 0; i <= 4; i++) {\n"
"    const gy = pad + i*h/4; ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(pad+w, gy); ctx.stroke();\n"
"    const gx = pad + i*w/4; ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, pad+h); ctx.stroke();\n"
"  }\n"
"  /* Draw each trajectory */\n"
"  for (const t of trajs) {\n"
"    if (!t.path || t.path.length < 2) continue;\n"
"    const cs = t.cos_sim || 0;\n"
"    const r = Math.floor(255 * (1 - Math.max(0, cs)));\n"
"    const g = Math.floor(255 * Math.max(0, cs));\n"
"    const col = `rgb(${r},${g},80)`;\n"
"    /* Path line */\n"
"    ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.7;\n"
"    ctx.beginPath();\n"
"    const [sx, sy] = proj(t.path[0]); ctx.moveTo(sx, sy);\n"
"    for (let i = 1; i < t.path.length; i++) {\n"
"      const [px, py] = proj(t.path[i]); ctx.lineTo(px, py);\n"
"    }\n"
"    ctx.stroke(); ctx.globalAlpha = 1;\n"
"    /* Start marker (circle) */\n"
"    ctx.fillStyle = '#fff'; ctx.strokeStyle = col; ctx.lineWidth = 2;\n"
"    ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI*2); ctx.fill(); ctx.stroke();\n"
"    /* Endpoint marker (filled dot) */\n"
"    const lastPt = t.path[t.path.length - 1];\n"
"    const [ex, ey] = proj(lastPt);\n"
"    ctx.fillStyle = col;\n"
"    ctx.beginPath(); ctx.arc(ex, ey, 4, 0, Math.PI*2); ctx.fill();\n"
"    /* Target marker (cross) */\n"
"    if (t.target) {\n"
"      const [tx, ty] = proj(t.target);\n"
"      ctx.strokeStyle = '#fa0'; ctx.lineWidth = 2;\n"
"      ctx.beginPath(); ctx.moveTo(tx-5,ty-5); ctx.lineTo(tx+5,ty+5); ctx.stroke();\n"
"      ctx.beginPath(); ctx.moveTo(tx+5,ty-5); ctx.lineTo(tx-5,ty+5); ctx.stroke();\n"
"    }\n"
"  }\n"
"  ctx.fillStyle = '#556'; ctx.font = '11px monospace';\n"
"  ctx.fillText('PC1', canvas.width/2, canvas.height - 5);\n"
"  ctx.save(); ctx.translate(10, canvas.height/2); ctx.rotate(-Math.PI/2);\n"
"  ctx.fillText('PC2', 0, 0); ctx.restore();\n"
"}\n"
"\n"
"/* Boot */\n"
"loadData();\n"
"</script>\n"
"</body>\n"
"</html>\n"
    );

    fclose(f);
    kprintf("[VIS] Renderer written → %s\n", path);
    kprintf("[VIS] Open %s in a browser to explore manifold geometry\n", path);
    vis_active = 0;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Export: Combined JSON for external 3D viewers
 * ═══════════════════════════════════════════════════════════════════════════ */

int axiom_vis_export(const char *output_path)
{
    const char *out = output_path;
    if (!out || !out[0]) {
        /* Default: write to vis dir */
        static char default_path[600];
        if (vis_dir[0])
            snprintf(default_path, sizeof(default_path),
                     "%s/manifold_export.json", vis_dir);
        else
            snprintf(default_path, sizeof(default_path),
                     "manifold_export.json");
        out = default_path;
    }

    FILE *f = fopen(out, "wb");
    if (!f) {
        kprintf("[VIS] ERROR: Cannot write export to %s\n", out);
        return -1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"geodessical-manifold-v1\",\n");
    fprintf(f, "  \"model\": {\n");
    fprintf(f, "    \"name\": \"%s\",\n", vis_model_name);
    fprintf(f, "    \"arch\": \"%s\",\n", vis_model_arch);
    fprintf(f, "    \"dim\": %d,\n", vis_model_dim);
    fprintf(f, "    \"layers\": %d,\n", vis_model_layers);
    fprintf(f, "    \"vocab\": %d,\n", vis_model_vocab);
    fprintf(f, "    \"heads\": %d,\n", vis_model_heads);
    fprintf(f, "    \"quant\": \"%s\"\n", vis_quant_type);
    fprintf(f, "  },\n");

    /* Inline each phase JSON if it exists */
    static const char *phase_keys[] = {
        "manifold", "symmetry", "curvature", "axioms", "geodesic"
    };
    static const char *json_files[] = {
        "phase1_manifold.json", "phase2_symmetry.json",
        "phase3_curvature.json", "phase4_axioms.json",
        "phase5_geodesic.json"
    };
    for (int i = 0; i < 5; i++) {
        char jp[600];
        vis_path(jp, sizeof(jp), json_files[i]);
        FILE *jf = fopen(jp, "rb");
        if (!jf) {
            fprintf(f, "  \"%s\": null%s\n",
                    phase_keys[i], (i < 4) ? "," : "");
            continue;
        }
        fprintf(f, "  \"%s\": ", phase_keys[i]);
        char cbuf[4096];
        size_t nr;
        while ((nr = fread(cbuf, 1, sizeof(cbuf), jf)) > 0)
            fwrite(cbuf, 1, nr, f);
        fclose(jf);
        fprintf(f, "%s\n", (i < 4) ? "," : "");
    }

    fprintf(f, "}\n");
    fclose(f);

    kprintf("[VIS] 3D export → %s\n", out);
    return 0;
}
