# Peer Review of `volume_extended.tex` (HyperTensor Volume)

Reviewer perspective: senior PhD reviewer, joint expertise in numerical
linear algebra / mechanistic interpretability / analytic number theory.
Date of review: 2026‑05‑09. Source file: `ARXIV_SUBMISSIONS/volume_extended.tex`
(9,701 lines, 18 papers, plus the new "Updated Evidence (May 2026)" sections at
L2186, L3011, L7097, L7866, L8392, L8633, L8968, L9239, L9443, L9581).

The volume has clearly improved since the initial drafts: most papers now
report negatives explicitly, the May‑2026 updated‑evidence blocks tighten
the empirical record, and the cross‑volume framing is much less
overclaimed than it was. Below I go paper by paper, naming the smallest
set of changes I would require before I would recommend this preprint
without reservation. I cite by line number against the file as it stands
on 2026‑05‑09.

---

## Volume‑level concerns (read these first)

### V1. The "ten‑papers‑share‑one‑story" packaging is starting to fight the science.

The volume is structured as a single arc (compression → geometry →
universality → safety → number theory → bridge), and that arc is the
thing a careful reader will judge it on. Two of the load‑bearing claims
in that arc are now on shakier ground than the framing acknowledges:

  - Paper XI's "universal basis" reading is **not supported** at the
    subspace level (you say so cleanly at L7476–7531) and is
    Bonferroni‑significantly **rejected in the opposite direction at 7B**
    (your N4, L9686 "13.6 percentage points *less* damaging than a
    Haar‑random orthonormal basis"). The *coordinate*-level reading
    survives, but it is a different and weaker claim.
  - Paper XV's behavioural‑residue invariance is **not uniform across
    layers**: the new artefact gives ratios `{2.92, 1.78, 1.81, 2.31, 0.79}`
    (L8968+, Updated Evidence). Mean 1.92 but min 0.79.

The paper bodies still occasionally lean on the stronger (rejected)
forms. The Updated‑Evidence blocks correct this locally, but the
*abstracts* of XI, XII, XIII, XIV, XV do not. A reader who skims
abstracts will get a more confident story than the data supports.

**Recommended action.** Add one or two sentences to each of the abstracts
of Papers XI–XV explicitly demoting "universal/meaningful basis" to
"coordinate‑level structure used as an engineering basis." Concretely:

> *(Suggested abstract suffix for Paper XI):* "The subspace‑level
> universality conjecture initially proposed in Sec.~\ref{sec:ugt-falsification}
> did not bear out under matched‑rank random‑basis controls; the result
> we retain is a coordinate‑level structure (per‑coordinate captures and
> per‑coordinate ablations outperform random by 25–249×, see
> Sec.~\ref{sec:ugt-coordinate-structure}) that is what downstream papers
> XII–XV actually consume."

### V2. "Updated Evidence (May 2026)" sections are great; please make them visible.

They are inserted as `\section*{...}` (unnumbered) and appear immediately
before each paper boundary, so they read as floating addenda. A reader
glancing at the table of contents will not see them. Two fixes:

  - Promote each `\section*{Updated Evidence (May 2026)}` to a numbered
    subsection of the paper it terminates (e.g.
    `\subsection{Updated Evidence (May 2026)}` inside Paper N), so it is
    in the TOC.
  - Cross‑reference each block from the parent paper's abstract or
    introduction (e.g. Paper XVI abstract should point to L9239 for the
    74,949‑zero LMFDB result).

### V3. Several papers still use "falsification/rejection" language too aggressively.

Per the standing rule, prefer "did not bear out / not supported /
rejection of conjecture." I found one residual occurrence at L9686
("Bonferroni‑significantly **falsified** in the opposite direction").
Replace with "Bonferroni‑significantly **not supported, with effect in
the opposite direction**." This is the only place I want this changed
on grammar grounds; everywhere else the wording is now appropriately
hedged.

### V4. Cross‑paper claim audit is consistency, not validation.

`bulletproof_audit.json` (51/51 OK, cited at L9582+) confirms manuscript
numbers match the JSON artefacts on disk. That is housekeeping, not
external replication. The current text reads it conservatively, which
is correct, but I would *strengthen* the conservative framing by adding
in Paper XVIII's Updated‑Evidence block one explicit sentence: "No
external party has re‑run any of these scripts on independent hardware;
the audit is a self‑consistency check, and we recommend a full
third‑party replication of at least Papers I, XI, XIV, and XVI before
acceptance."

### V5. Hardware concentration is a single‑point‑of‑failure risk.

The vast majority of throughput numbers come from one RTX 4070 Laptop
(8.585 GB, 32 MB L2, AD106). Paper I explicitly anchors the
cross‑vendor scaling formula `k* ≈ L2_MB × 48.0` on this single device.
Papers II, V, VII, VIII, IX, X cite the same machine. I cannot accept
the cache‑fit super‑baseline claim as "cross‑vendor" without at least
one AMD or Apple Silicon datapoint.

Either (a) acquire one cross‑vendor measurement before preprint
release, or (b) restate Paper I's headline as "single‑vendor (Ada
AD106) measurement; cross‑vendor scaling is a *prediction*." The body
of Paper I is honest about this (L9685, "tentative pending cross‑vendor
measurement") but the abstract still presents `48.0` as a constant.

### V6. The Riemann arc (XVI–XVIII) is the rhetorical centre of gravity, and it is still the weakest part of the volume.

I deal with this in detail under Papers XVI, XVII, XVIII below. Top
line: the explicit RH‑disclaimers in those papers are now correct, but
the volume's *front matter* (abstract at L194 onwards, top‑level
conclusion at L682) reads as if the Riemann work is the climax. If the
Riemann work cannot be tightened (and I will argue below that XVI in
particular cannot be tightened beyond what it already is), it should
not be the narrative climax.

---

## Per‑paper review

### Paper I — GRC Attention Compression (L708)

**Verdict.** The strongest empirical paper in the volume. Headline result
holds.

**Strengths.**
  - The protocol discipline is exemplary: the 30‑second pre‑run idle
    rationale at L744–751 ("we discovered the necessity of this
    protocol the hard way... false 53% retention reading on a
    thermally‑throttled run") is exactly the kind of admission peer
    reviewers want to see.
  - Eight conditions × 12 reps × 10,000‑sample percentile bootstrap is
    a defensible statistical design.
  - The Updated‑Evidence block (L2186) gives bootstrap CIs for k=64,
    128, 256, 512 and baseline; the cache‑fit reading survives the CI.

**Required changes.**
  1. **Abstract overclaim.** Currently the abstract says cache‑fit is
     the "primary" mechanism; the body (L9685) attributes the dominant
     share to *kernel fusion*, not L2 residency. Reorder to "kernel
     fusion is the dominant mechanism with cache‑fit as a secondary
     contributor." This is exactly the residual phrasing inconsistency
     N3 flags.
  2. **Cross‑vendor scope.** Restate `k* ≈ L2_MB × 48.0` as a single‑GPU
     prediction until at least one AMD/Apple/TPU measurement exists. If
     the AMD datapoint is not feasible before submission, mark it a
     "limitation, not a result" in the abstract.
  3. The k=2048 row at L767–769 ("warmup proxy" footnote) is an honest
     disclosure but it is buried. Add one sentence to the table caption
     so a reader of `tab:headline` alone is not misled by the
     `k=1536†` row.

**Optional but recommended.**
  - The bootstrap CI for `k=256` is `[33.3, 73.0]` (Updated Evidence).
    This is the L2‑working‑set transition. It is *much* wider than the
    other CIs and deserves a sentence: that point is genuinely noisy,
    not just a means‑and‑error point.

---

### Paper II — Geodesic Projection Pipeline (L2206)

**Verdict.** Promising; needs to commit to either MCR or shared‑rank as
the headline.

**Strengths.**
  - The MCR contamination disclosure at L2218–2240 (background workload
    eating ~3.5% throughput, caught by the variance gap) is excellent
    science.
  - The weight‑PCA flat‑fallback finding (L2241–2270) is exactly the
    kind of mechanistic confound that a PhD reviewer wants to see
    flagged.
  - The Updated‑Evidence per‑layer intrinsic‑dim grid (PCA95
    `334→60→1→1→273` across `{0,3,15,22,30}`) is a beautiful new
    signal.

**Required changes.**
  1. **MCR is "least confident" by your own admission (L2204), but the
     paper title and abstract still sell GP as MCR‑driven.** The
     measured truth is that at k=1024 MCR is statistically
     indistinguishable from shared‑rank. Either run the lower‑rank
     sweep `k∈{256, 512, 768}` you already prescribe in the body, or
     reframe the paper around the activation‑aware layer‑wise rank
     allocation (which the new intrinsic‑dim grid actually supports).
  2. **The intrinsic‑dim grid contradicts MCR's flat profile.** The
     real‑activation signal is `334→1` then back to `273`; the
     weight‑PCA MCR mode is flat (`~4×10⁻⁴`). This is the most
     interesting result in the paper and it currently sits in an
     Updated‑Evidence block at L3011. Move it into the body and use it
     as the motivation for activation‑aware MCR.

---

### Paper III — Geodesic Speculative Decoding (L3035)

**Verdict.** This is currently a partial result, and the partialness is
the contribution.

**Strengths.**
  - `tab:accept-collapse` (L3076–3083) is honest about the per‑prompt
    structure. The single surviving `k=256` prompt at α=56.2% is the
    interesting bit.
  - The mechanistic interpretation at L3098+ ("once compression
    destroys enough of the attention routing subspace, P_V concentrates
    on different tokens than P_D") is correct and well‑argued.

**Required changes.**
  1. **The k=512 row was abandoned to wallclock budget.** This is the
     most diagnostically useful point in the sweep (it is the
     transition between collapse and survival) and the paper cannot be
     a complete contribution without it. Either (a) execute the wproj
     cache that makes k=512 feasible at warm‑cache cost, or (b)
     reframe the paper as "compression‑induced acceptance collapse: a
     case study," demoting the spec‑decode×GRC composition to a
     conjecture supported by the k=256 single‑prompt survivor.
  2. **The α=56.2% > α=46.9% reading at k=256 (one prompt out of four)
     is intriguing but not statistically distinguishable from noise on
     n=1.** Either replicate it on additional prompts in the surviving
     class or remove it from the abstract.

---

### Paper IV — Organic Training Theory (L3531)

**Verdict.** Two papers in one. The negative (curvature warp) is solid;
the positive (learnable curvature warp v4) is asserted but not
quantified at the level the negative is.

**Strengths.**
  - The 0/32 sweep + 0/12 cross‑model replication (L3559+) is a clean,
    falsifiable, executed protocol. This is how to report a negative.
  - The covariant Riemannian‑exponential v2 is *worse* than v1 on
    Phi‑3.5‑mini (target error grows 9–24%); this is correctly read as
    *strengthening* the negative.

**Required changes.**
  1. **The "v4 CONFIRMED WORKING" claim at L3540 is supported only by
     a 12‑point ratio improvement (`0.868 → 0.237`).** This is far
     below the rigour bar set by the negative half of the paper. Either
     run v4 through the same (3 models × 2 dims × 2 protocols × 32
     configs × pass criterion) gauntlet, or demote v4 from "CONFIRMED
     WORKING" to "preliminary positive evidence on 12 points."
  2. The injectivity radius "FIXED" claim (L3534) needs a passage in
     the body (not just the bulleted summary) showing the calibration
     formula `C_LM = ρ_emp/ρ_vol` and the `<0.01%` agreement plot.

---

### Paper V — GRC Light Distillation (L4495)

**Verdict.** Tight, focused, methodologically careful. Few changes needed.

**Strengths.**
  - LoRA recovery protocol is conservative (β₂=0.95, warmstart 100
    steps before activating LoRA path). These are exactly the right
    knobs for a small‑corpus fit.

**Required changes.**
  1. **Reproducibility is currently a forward‑looking claim** (L4528+:
     "A complete empirical reproduction will be published alongside the
     populated results once the runner has executed"). This is a hard
     blocker for preprint release. Either complete the Phase 2 EC2 run
     before submission or downgrade Paper V to a protocol paper
     ("Phase 1 is run; Phase 2 protocol is preregistered").

---

### Paper VI — Task‑Level Impact (L5107)

**Verdict.** This paper is currently *infrastructure*, not *evidence*.

**Strengths.**
  - Clear preregistration of the four‑benchmark sweep (MMLU, GSM8K,
    HumanEval, IFEval) at six rank points.

**Required changes.**
  1. **Abstract reads: "Infrastructure complete. Measured results on
     SmolLM2‑135M..." (L5125).** That ellipsis is doing too much work.
     If MMLU/GSM8K/HumanEval/IFEval × six ranks have not been
     measured, this paper does not yet exist as a contribution; it is
     a protocol with a 135M smoke test. Remove from preprint or
     execute the sweep.

---

### Paper VII — FFN Cluster Compression (L5451)

**Verdict.** Good negative result well executed.

**Strengths.**
  - The activation‑weighted SVD failure (L5465+, weight‑norm proxy
    gives 1230 PPL = 45.2× baseline) is a genuine and useful
    null. The post‑mortem (massive‑activation phenomenon operates on
    runtime hidden states, not static weight columns; cite Sun et al.
    2024) is correct.
  - The "99.9% recovery" caveat at L5483–5494 is exactly how to
    contextualise a misleading headline. Keep it.

**Required changes.**
  1. The phrase "real‑activation‑weighted compression" at L5500 (1.99×
     baseline) is the actual deployment‑relevant number; surface it in
     the abstract.

---

### Paper VIII — GTC as RAG (L5987)

**Verdict.** The right comparison was hidden until you fixed the
caption.

**Strengths.**
  - The new caption ("0.17ms vs 450ms is in‑memory cache against
    on‑disk vector retrieval; the apples‑to‑apples baseline is
    in‑memory KV reuse, against which GTC measures ~15.5×") is the
    correct framing.

**Required changes.**
  1. **The 91.0/90.4/91.5% scale‑invariance result at L6014 is currently
     a 3‑point line.** Three models from two families is sub‑threshold
     for "scale‑invariant within ±0.5%." Either add a fourth scale
     (Qwen 7B, or SmolLM2 360M) or weaken the wording to "stable
     within ±0.5% across the three scales tested."
  2. **The B=10 → B=20 drop (97.9× → 60.0×) is unexplained** (L6028).
     Either explain it (cache pressure, MMU, what?) or call it
     anomalous.

---

### Paper IX — Cross‑GPU Transfer (L6141)

**Verdict.** This is a compact, well‑scoped paper. Few changes.

**Required changes.**
  1. **The portability claim hinges on a single second‑device check.**
     Add at least one more GPU before "cross‑GPU transfer" goes in the
     title. If only the L40S is available before submission, retitle
     to "Single‑Cross‑Vendor Portability of GRC."

---

### Paper X — CECI Model Grafting (L6270)

**Verdict.** The negative within‑model grafting result (N2: 0/120) is
the strongest part. The positive cross‑graft headline needs scope
language.

**Strengths.**
  - Per‑slot residual table (L6308+, V is consistently the hardest
    slot at 8–16% energy) is a clean diagnostic.
  - The depth‑band feasibility map (L6336+, viable ≤ ΔL=4, infeasible
    at ΔL≥8) is exactly the right unit of measurement.

**Required changes.**
  1. **The 0/120 within‑model graft null (N2 at L9683) is the load‑bearing
     scope statement for this paper.** It currently lives in the
     volume‑level negatives section. Surface it in *Paper X's own*
     abstract: "Cross‑architecture grafting is viable in the regime we
     tested; within‑architecture grafting is not."
  2. The Llama‑8B EC2 row is `TBD` for k ∈ {512, 1024, 1536, 2048} (L6232+).
     This is an active hole; queue or remove.

---

### Paper XI — Universal Geodesic Taxonomy (L7119)

**Verdict.** This paper has been the most rewritten and is now
honest. The remaining work is in framing.

**Strengths.**
  - `\Cref{sec:ugt-falsification}` (L7476) and `\Cref{sec:ugt-cross-model-universality}`
    (L7532) are model rebuttals to the original conjecture; the
    coordinate‑level rescue at `\Cref{sec:ugt-coordinate-structure}`
    (L7606) is properly hedged.
  - The Updated‑Evidence block at L7866 (Wielandt–Hoffman 7B
    *prediction* 0.999993, with 1.5B *measurement* 0.999971) is
    correctly labelled as a prediction.

**Required changes.**
  1. **Abstract still uses universal language.** Apply the suggested
     suffix from V1 above. Specifically: do not use "universal" in the
     title or abstract for a result whose subspace‑level form was not
     supported and whose layer‑wise form at 7B was Bonferroni‑rejected
     in the opposite direction. "Coordinate‑level structure of trained
     bases" would be more honest.
  2. **The 7B Wielandt–Hoffman prediction is a prediction.** The
     Updated‑Evidence block calls it that. Make sure the body of
     Paper XI does too — currently the bilateral UGT discussion at
     L7150+ runs the 1.5B and 7B numbers as if both were measured.

---

### Paper XII — Native Geodesic Training (L7895)

**Verdict.** Architecturally interesting; quantitatively incomplete.

**Strengths.**
  - NativeLinear formulation `W = B C Bᵀ` is clean and the parameter
    count `(k² + dk)` vs `d²` claim is straightforward.
  - RiemannianAdamW with QR retraction is a standard correct
    implementation.

**Required changes.**
  1. **The abstract claims "validated at 135M, 1.5B, and 7B"** (L7905).
     But the 7B validation in the volume is the single Wielandt–Hoffman
     prediction (Paper XI Updated Evidence) — *not* an end‑to‑end
     NativeLinear training run at 7B. Either run an actual 7B native
     training and report a loss curve, or restate as "validated at
     135M and 1.5B; predicted to scale to 7B via Wielandt–Hoffman
     bound."
  2. **`k* = L2_MB × 42.7` in the abstract is a different constant
     from Paper I's `k* ≈ L2_MB × 48.0`.** Reconcile: are these
     attention‑only vs full‑model? Different machines? Pick one and
     reference it.
  3. **KExpansion is asserted but not shown.** Add at least one loss
     curve figure showing automatic k growth.

---

### Paper XIII — Safe OGD (L8179)

**Verdict.** The structural identity is real. The numerical
verification (Updated Evidence at L8392) corroborates the algebra. Good
paper.

**Strengths.**
  - The new BP=NS check at L8392+ (160/160 trials at machine‑ε,
    σ_{k+1}‖x‖ bound holds at avg tightness 0.42) is the right unit
    test for the algebraic identity `Q_fᵀ P_safe = 0`.

**Required changes.**
  1. The `safe_loss_aczel_check.json` artefact (where all three
     aggregators allegedly satisfy all four Aczel axioms) is *not*
     used in the body — and that is correct, the result contradicts
     the Aczel theorem and is almost certainly a measurement bug.
     Keep skipping it. But add one sentence to Paper XIII's reproducibility
     section: "We do not rely on `safe_loss_aczel_check.json`; the
     reported axiom check appears anomalous and is excluded pending
     diagnosis."

---

### Paper XIV — Behavioral Geodesic Sniping (L8416)

**Verdict.** The exact 8‑category specificity table (Updated Evidence
at L8633) is the most important new artefact in the volume.

**Strengths.**
  - The honest split `{privacy 2.72, illegal_advice 2.65, phishing
    1.30, sycophancy 1.04}` clean vs `{jailbreak 0.54, misinformation
    0.37, self_harm 0.29, toxicity 0.19}` entangled is exactly the
    granularity a deployer needs.

**Required changes.**
  1. **The body of Paper XIV currently leads with the headline
     "specificity ≥ 1.3 across the board" reading.** The new artefact
     shows that holds for 3 of 8 categories, not 8 of 8. Replace the
     six‑category body table with the eight‑category Updated‑Evidence
     table; do not bury it.
  2. **The all‑Snipe (20 coords, 42.1% harm reduction at 1.8% benign
     loss) result needs the same per‑category breakdown.** Otherwise a
     reader cannot tell whether the 1.8% benign cost is concentrated
     in jailbreak/toxicity (where Snipe is known to entangle) or
     distributed.
  3. **The greedy ROI algorithm is described but its stopping rule is
     hand‑wavy.** Add the exact rule (cumulative damage threshold,
     and the threshold value).

---

### Paper XV — COG + TEH (L8671)

**Verdict.** Two papers in one suit. COG is exploratory; TEH is
quantitative.

**Strengths.**
  - The 4‑tier query recognition table (L8714) is a clean engineering
    spec.
  - TEH at 1.5B (Updated Evidence at L8968): 100% on 75 prompts at
    mean activation 20.3, 100% per‑category. This is a strong
    deployment‑scale result.

**Required changes.**
  1. **Behavioural‑residue invariance is reported uniformly in the
     body but the new artefact gives ratios `{2.92, 1.78, 1.81, 2.31,
     0.79}`** (Updated Evidence). Mean 1.92, min 0.79 *at the deepest
     probed layer*. Surface this in the body. The paper currently
     reads as if ratio ≥ 1 holds across all probed layers; the data
     say it holds for early/middle layers but not for the deepest.
  2. **TEH at 135M has mean_harmful 25.5 vs mean_benign 22.3** (Updated
     Evidence). That is a modest gap. The body should reflect "scale‑
     dependent: detection frontier resolves at 1.5B and is genuinely
     close to the benign distribution at 135M."
  3. **COG metric saturation at "~25 interactions" (L8702) is reported
     without an underlying figure.** Add one curve.
  4. **The "0.05ms to 4.1s" 4‑tier latency range** (L8674 caption) is
     an enormous span; the 4.1s figure is presumably the EXPLORE tier.
     Footnote: this is end‑to‑end including model forward, not just
     COG retrieval.

---

### Paper XVI — AGT Topology of Zeta Zeros (L9004)

**Verdict.** The most important paper in the volume to get right, and
the one most at risk of overclaiming. The Updated‑Evidence framing
(L9239+) is correct; the body needs to align with it.

**Strengths.**
  - The Updated Evidence at L9239+ is impressive: 74,949 critical‑line
    zeros (10,000 Odlyzko zeros1 + 10,000 Odlyzko zeros4 at height
    ~10²⁰ + 54,949 across 400 LMFDB L‑functions of degree 1+2),
    `‖D(s)‖ = 0` exactly, plus 5,000‑sample perturbation discrimination
    TPR=1.0/FPR=0.0 at four σ‑offsets.
  - The honest scope sentence at L9277+ ("the empirical content is
    that the framework *applies cleanly* at this scale… not… that all
    non‑trivial zeros lie on the critical line") is exactly the right
    claim.

**Required changes.**
  1. **The abstract still says "rank of this operator is exactly 1"
     and "geometric jury of 105 zeros votes with confidence
     J ≈ 1−10⁻³¹⁵"** (L9018+). Both are restatements of definitions:
     the rank‑1 observation follows from the feature design (all but
     one coordinate of D are identically zero on all of ℂ, see Paper
     XVIII Limitations at L9398), and the jury formula at this
     confidence violates the independence axiom because the 105 known
     zeros are consequences of the same theorem‑and‑computation
     pipeline (Paper XVIII Limitation 3, L9396). **Remove `J ≈ 1−10⁻³¹⁵`
     from the abstract entirely.** It is misleading to a reader who
     does not also read Paper XVIII.
  2. **The headline `‖D(s)‖ = 2|σ − 1/2|` is by construction.** State
     this in the abstract (currently it is in the Updated Evidence).
     The Updated Evidence sentence "the by‑construction caveat
     (`‖D(s)‖ = 2|σ−1/2|` is algebraic)" should be in *every* place
     the discriminator is reported.
  3. **The 74,949‑zero result is the new contribution; it deserves its
     own subsection in the body, not just a `\section*{Updated Evidence}`
     before the next paper.** Promote.

---

### Paper XVII — Analytic Continuation Manifold (L9286)

**Verdict.** This paper is currently the bridge that does not bridge.

**Strengths.**
  - The Z₂ separation 1.0 with critical_mean 1.4782 vs off_critical
    1.7616 (Updated Evidence at L9443+) is a genuine empirical
    statement.
  - The honest framing in the Updated Evidence ("the 0.009
    involution‑error of the learned embedder still makes this an
    empirical convergence statement, not a closed proof") is correct.

**Required changes.**
  1. **The paper's status field reads `PROVED` in the artefact** —
     don't put that in the manuscript without a sentence explaining
     it is a script status string, not a mathematical claim. The
     Updated‑Evidence block already does this; make sure no part of
     the body propagates the unhedged word "PROVED."
  2. **The "necessity argument" referenced from N5 (L9696)** —
     conditional on faithfulness of the learned `h` — is the actual
     scientific content of Paper XVII. Either state the conditional
     theorem precisely (Theorem: "If `h` is faithful in sense X, then
     `f∘ι ≡ f` follows.") with sense X spelled out, or reframe Paper
     XVII as "an empirical convergence study of the learned ACM
     embedder."

---

### Paper XVIII — The Bridge Protocol (L9469)

**Verdict.** The Limitations section (L9388–9420) is the best part of
the entire Riemann arc. It should be required reading for anyone who
reads Paper XVI's abstract.

**Strengths.**
  - The four numbered limitations (L9398–9420) are individually
    correct, mutually reinforcing, and devastating to any over‑reading
    of XVI–XVII.
  - The Updated Evidence at L9581+ (bulletproof 51/51 + e2e Stage1–5)
    is appropriately conservative on Stage 4 (PPL=404.5 collateral).

**Required changes.**
  1. **The honest Stage 4 PPL=404.5 number is the deployment story.**
     The pipeline does not compose for free. Surface this in the
     Paper XVIII abstract: "Stage 4 multi‑Snipe across 58 coordinates
     raises benign PPL to 404.5; the production pipeline uses the
     greedy budgeted variant of Paper XIV to avoid this." Currently
     the abstract is silent on it.
  2. **Stage 5 COG/TEH "blocks all 5 test queries with no expansion."**
     n=5 is too small for a deployment claim. Either expand to n≥30
     or label as "smoke test."

---

## Summary of must‑fix items before preprint release

In rough priority order:

  1. **Paper XVI abstract:** remove `J ≈ 1−10⁻³¹⁵`; state
     `‖D(s)‖ = 2|σ−1/2|` is algebraic. (V6, XVI.1, XVI.2.)
  2. **Papers XI–XV abstracts:** demote "universal/meaningful basis"
     to "coordinate‑level structure." (V1, XI.1, XII.1, XV.1, XV.2.)
  3. **Paper I abstract:** kernel fusion before cache‑fit; restate
     `48.0` as single‑vendor. (V5, I.1, I.2.)
  4. **Paper VI:** complete the four‑benchmark sweep or remove from
     preprint. (VI.1.)
  5. **Paper V Phase 2:** complete or downgrade to protocol paper.
     (V.1.)
  6. **Paper XII abstract:** "validated at 135M and 1.5B; 7B is a
     prediction." Reconcile `k* = L2_MB × 42.7` with Paper I's
     `48.0`. (XII.1, XII.2.)
  7. **Paper XIV body table:** replace 6‑category with 8‑category
     specificity table. (XIV.1.)
  8. **Paper III k=512 row:** execute the wproj cache or reframe.
     (III.1.)
  9. **Paper IV "v4 CONFIRMED WORKING":** demote unless run through
     the full 32‑config gauntlet. (IV.1.)
  10. **Updated‑Evidence sections:** promote from `\section*{}` to
      numbered subsections so they appear in the TOC. (V2.)
  11. **Replace residual "falsified" at L9686** with "not supported,
      with effect in the opposite direction." (V3.)

Items 1–7 are blocking. Items 8–11 are strongly recommended but a
preprint can be released without them if explicitly flagged in the
"open issues" section of the front matter.

---

## What this volume already does right

A blunt note to the authors. The May‑2026 evidence updates, the
explicit handling of negatives (N1–N5 at L9676+), the protocol
discipline in Paper I, the layer‑wise replication of the random‑basis
ablation in Paper XI, and the bullet‑listed limitations in Paper XVIII
are above the level of a typical arXiv preprint in this area. The
volume's core problem is *packaging*: an honest and impressive set of
results is occasionally surrounded by abstract‑level rhetoric that does
not match the body. Fixing the abstracts (items 1–3, 6) gets you 80%
of the way to a defensible preprint.

— End of review
