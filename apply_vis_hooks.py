"""Apply axiom_vis hooks to axiom_beta.c on disk."""
import os

f = os.path.join(os.path.dirname(__file__), "runtime", "nn", "axiom_beta.c")
with open(f, "r", encoding="utf-8") as fp:
    c = fp.read()

edits = 0

# 1. Add axiom_vis.h include
old = '#include "runtime/nn/axiom_geo.h"\n#include "runtime/nn/llm.h"'
new = '#include "runtime/nn/axiom_geo.h"\n#include "runtime/nn/axiom_vis.h"\n#include "runtime/nn/llm.h"'
if old in c and 'axiom_vis.h' not in c:
    c = c.replace(old, new, 1)
    edits += 1
    print("1. Include added")
else:
    print("1. Include already present or pattern not found")

# 2. Phase 1 vis hook - insert before r->phase1_us at end of phase1 function
# Find the second occurrence of "r->phase1_us = hal_timer_us() - t0;" (the main one, not the skip branch)
old2 = """    r->phase1_us = hal_timer_us() - t0;
    return 0;
}

"""
# Find the second occurrence
idx1 = c.find(old2)
if idx1 >= 0:
    idx2 = c.find(old2, idx1 + 1)
    target_idx = idx2 if idx2 >= 0 else idx1
    phase1_hook = """    /* VIS: emit PCA cloud */
    if (axiom_vis_active() && phase1_pca.n_components > 0) {
        int k_vis = phase1_pca.n_components;
        float *v_emb = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
        double *v_e64 = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
        double *v_proj = (double *)tensor_alloc((uint64_t)k_vis * sizeof(double));
        double *v_cloud = (double *)tensor_alloc((uint64_t)n_samples * k_vis * sizeof(double));
        if (v_emb && v_e64 && v_proj && v_cloud) {
            for (int i = 0; i < n_samples; i++) {
                int tok = ax_rng_range(seed, 0, vocab);
                int rc = ott_get_hidden_state(tok, -1, v_emb, dim);
                if (rc != 0) rc = llm_get_embedding_vec(tok, v_emb, dim);
                if (rc == 0) {
                    for (int j = 0; j < dim; j++) v_e64[j] = (double)v_emb[j];
                    axpca_project(&phase1_pca, v_e64, v_proj);
                    memcpy(v_cloud + i * k_vis, v_proj, (uint64_t)k_vis * sizeof(double));
                } else {
                    memset(v_cloud + i * k_vis, 0, (uint64_t)k_vis * sizeof(double));
                }
            }
            axiom_vis_emit_phase1(&r->phase1, &phase1_pca, v_cloud, n_samples);
        }
        if (v_emb)   tensor_free(v_emb);
        if (v_e64)   tensor_free(v_e64);
        if (v_proj)  tensor_free(v_proj);
        if (v_cloud) tensor_free(v_cloud);
    }

"""
    c = c[:target_idx] + phase1_hook + c[target_idx:]
    edits += 1
    print("2. Phase 1 vis hook added")
else:
    print("2. Phase 1 pattern not found")

# 3. Phase 3 vis hook - insert before curvature_destroy
old3 = '    /* Curvature is temporary'
# Only insert if not already present
if 'axiom_vis_emit_phase3' not in c:
    new3 = """    /* VIS: emit metric field + curvature before destroying curvature */
    if (axiom_vis_active() && rc_curv == 0) {
        axiom_vis_emit_phase3(&r->phase3, &mf, &ch,
                              curv.scalar_curv, n_mp, sub_dim);
    }

    /* Curvature is temporary"""
    c = c.replace(old3, new3, 1)
    edits += 1
    print("3. Phase 3 vis hook added")
else:
    print("3. Phase 3 hook already present")

# 4. Phase 5 vis hooks
# 4a. Add vis buffer declarations before the geodesic loop
old4a = '    for (int t = 0; t < n_test; t++) {\n        int tok_start = ax_rng_range(seed, 0, vocab);'
if 'vis_trajs' not in c:
    new4a = """    /* VIS: trajectory capture buffers */
    int vis_on = axiom_vis_active();
    double **vis_trajs = NULL;
    int *vis_traj_steps = NULL;
    double *vis_targets = NULL;
    double *vis_cos_sims = NULL;
    if (vis_on) {
        vis_trajs = (double **)tensor_alloc((uint64_t)n_test * sizeof(double *));
        vis_traj_steps = (int *)tensor_alloc((uint64_t)n_test * sizeof(int));
        vis_targets = (double *)tensor_alloc((uint64_t)n_test * sub_dim * sizeof(double));
        vis_cos_sims = (double *)tensor_alloc((uint64_t)n_test * sizeof(double));
        if (vis_trajs) memset(vis_trajs, 0, (uint64_t)n_test * sizeof(double *));
    }

    for (int t = 0; t < n_test; t++) {
        int tok_start = ax_rng_range(seed, 0, vocab);"""
    c = c.replace(old4a, new4a, 1)
    edits += 1
    print("4a. Phase 5 vis buffers added")
else:
    print("4a. Phase 5 vis buffers already present")

# 4b. Add trajectory capture before geodesic_destroy in the loop
old4b = '        axgeo_geodesic_destroy(&geo);\n        tensor_free(v0);\n    }'
if 'vis_trajs[t]' not in c:
    new4b = """        /* VIS: capture trajectory */
        if (vis_on && vis_trajs) {
            vis_traj_steps[t] = geo.steps;
            vis_trajs[t] = (double *)tensor_alloc((uint64_t)geo.steps * sub_dim * sizeof(double));
            if (vis_trajs[t])
                memcpy(vis_trajs[t], geo.trajectory, (uint64_t)geo.steps * sub_dim * sizeof(double));
            if (vis_targets)
                memcpy(vis_targets + t * sub_dim, proj_b, (uint64_t)sub_dim * sizeof(double));
            if (vis_cos_sims)
                vis_cos_sims[t] = cos_sim;
        }

        axgeo_geodesic_destroy(&geo);
        tensor_free(v0);
    }"""
    c = c.replace(old4b, new4b, 1)
    edits += 1
    print("4b. Phase 5 trajectory capture added")
else:
    print("4b. Phase 5 trajectory capture already present")

# 4c. Add vis emit after the loop, before phase5_us
old4c = '    r->phase5_us = hal_timer_us() - t0;\n    return 0;\n}\n'
# Find the LAST occurrence (end of phase5 function)
idx = c.rfind(old4c)
if idx >= 0 and 'axiom_vis_emit_phase5' not in c:
    phase5_emit = """    /* VIS: emit trajectories */
    if (vis_on && vis_trajs && vis_traj_steps) {
        axiom_vis_emit_phase5(&r->phase5,
                              (const double **)vis_trajs, vis_traj_steps,
                              vis_targets, vis_cos_sims,
                              n_test, sub_dim);
    }
    /* VIS: cleanup trajectory buffers */
    if (vis_trajs) {
        for (int t = 0; t < n_test; t++) {
            if (vis_trajs[t]) tensor_free(vis_trajs[t]);
        }
        tensor_free(vis_trajs);
    }
    if (vis_traj_steps) tensor_free(vis_traj_steps);
    if (vis_targets) tensor_free(vis_targets);
    if (vis_cos_sims) tensor_free(vis_cos_sims);

"""
    c = c[:idx] + phase5_emit + c[idx:]
    edits += 1
    print("4c. Phase 5 vis emit added")
else:
    print("4c. Phase 5 vis emit already present or pattern not found")

# 5. Phase 2 vis emit in axiom_beta_run
old5 = '        rc = phase2_symmetry(cfg, report, &seed);\n'
if 'axiom_vis_emit_phase2' not in c:
    # Find the line after the phase2 kprintf that follows
    phase2_marker = '            (double)report->phase2_us / 1000.0);\n'
    idx5 = c.find(phase2_marker)
    if idx5 >= 0:
        insert_pos = idx5 + len(phase2_marker)
        phase2_vis = """
        /* VIS: emit Phase 2 */
        if (axiom_vis_active())
            axiom_vis_emit_phase2(&report->phase2, NULL, 0);

"""
        c = c[:insert_pos] + phase2_vis + c[insert_pos:]
        edits += 1
        print("5. Phase 2 vis emit added")
    else:
        print("5. Phase 2 marker not found")
else:
    print("5. Phase 2 vis emit already present")

# 6. Phase 4 vis emit in axiom_beta_run
old6_marker = '            (double)report->phase4_us / 1000.0);\n'
if 'axiom_vis_emit_phase4' not in c:
    idx6 = c.find(old6_marker)
    if idx6 >= 0:
        insert_pos6 = idx6 + len(old6_marker)
        phase4_vis = """
        /* VIS: emit Phase 4 */
        if (axiom_vis_active())
            axiom_vis_emit_phase4(&report->phase4);

"""
        c = c[:insert_pos6] + phase4_vis + c[insert_pos6:]
        edits += 1
        print("6. Phase 4 vis emit added")
    else:
        print("6. Phase 4 marker not found")
else:
    print("6. Phase 4 vis emit already present")

# 7. vis_finalize before final return
old7 = '    report->total_us = hal_timer_us() - t0;\n    kprintf("[AXIOM-BETA-3] Complete: %.1f ms total\\n",'
if 'axiom_vis_finalize' not in c:
    new7 = """    /* VIS: finalize */
    if (axiom_vis_active())
        axiom_vis_finalize();

    report->total_us = hal_timer_us() - t0;
    kprintf("[AXIOM-BETA-3] Complete: %.1f ms total\\n","""
    c = c.replace(old7, new7, 1)
    edits += 1
    print("7. vis_finalize added")
else:
    print("7. vis_finalize already present")

# Write back
with open(f, "w", encoding="utf-8") as fp:
    fp.write(c)
print(f"\nDone: {edits} edits applied")
