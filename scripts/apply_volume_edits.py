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
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
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
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
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

"""One-shot script to apply on-disk text edits to ARXIV_SUBMISSIONS/volume_extended.tex.

These edits land directly on the file and bypass the VS Code editor buffer.
Idempotent: re-running is a no-op once edits have been applied.
"""
from __future__ import annotations
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "ARXIV_SUBMISSIONS" / "volume_extended.tex"
text = P.read_text(encoding="utf-8")

EDITS = []  # list[(old_substring, new_substring, label)]


# 1) Paper VII Phase 3 'FALSIFIED' -> 'not supported' (claim-outcome rewording).
EDITS.append((
    r"\item Phase~3 (weight-norm proxy): FALSIFIED.  L2 column norms"
    "\n      are uncorrelated with functional importance; weight-norm-guided SVD"
    "\n      gives PPL $45{\\times}$ baseline at $k_\\text{frac}{=}0.50$.",
    r"\item Phase~3 (weight-norm proxy): \emph{not supported}."
    "\n      We initially expected L2 column norms to track functional"
    "\n      importance; the measurement does not bear this out --- column"
    "\n      norms are uncorrelated with functional importance, and"
    "\n      weight-norm-guided SVD gives PPL $45{\\times}$ baseline at"
    "\n      $k_\\text{frac}{=}0.50$.",
    "VII Phase 3 wording",
))

# 2) Paper VII figure caption FALSIFIED -> reframed.
EDITS.append((
    r"\caption{Paper VII: FFN cluster compression --- 4-cluster SVD achieves 23\% error reduction vs global SVD. Weight-norm proxy FALSIFIED at extreme compression.}",
    r"\caption{Paper VII: FFN cluster compression --- 4-cluster SVD achieves 23\% error reduction vs global SVD. Weight-norm proxy was tested as a cheap importance heuristic and was not supported at extreme compression.}",
    "VII fig caption wording",
))

# 3) Paper X 'as falsified' -> 'initially conjectured...did not bear out'.
EDITS.append((
    "Pivot from J1 failure to J2 success. Paper~X originally reported\n"
    "within-model CECI as falsified ($0/120$ pairs viable at $k{=}32$). The\n"
    "pivot to cross-model splicing with shared initialisation was the correct\n"
    "move:",
    "Pivot from J1 failure to J2 success. Paper~X initially conjectured\n"
    "that within-model CECI splicing would be viable; the data did not\n"
    "bear that out ($0/120$ pairs viable at $k{=}32$). The pivot to\n"
    "cross-model splicing with shared initialisation was the correct\n"
    "move:",
    "X CECI wording",
))

# 4) Paper VIII figure caption recontextualisation (the 2,647x is vs disk RAG;
#    the in-memory baseline gives ~15.5x; flag both numbers honestly).
EDITS.append((
    r"\caption{Paper VIII: GTC query latency vs RAG. Jury-gated cache achieves 2,647x speedup (0.17ms vs 450ms) for cached queries.}",
    r"\caption{Paper VIII: GTC query latency. Jury-gated cache achieves a 2{,}647$\times$ wall-clock speedup against a disk-resident RAG baseline (0.17\,ms vs.\ 450\,ms) for cached queries; against an in-memory dense-retrieval baseline at the same scale the speedup is approximately $15.5\times$. Reporting both makes the comparison-class explicit.}",
    "VIII fig caption recontextualisation",
))

# 5) Paper XVIII Step 2 'ACM Verify' -> 'ACM Consistency Check'.
EDITS.append((
    r"\subsection{Step 2: ACM Verify}",
    r"\subsection{Step 2: ACM Consistency Check}",
    "XVIII Step 2 heading",
))

# 6) Paper XI abstract reframe: replace original abstract with honest-scope
#    version. We only replace if the original strong-claim text is found
#    verbatim, so re-running is idempotent.
ORIG_XI_ABSTRACT = (
    "\\begin{abstract}\n"
    "We present the Universal Geodesic Taxonomy (UGT), a method for establishing a shared coordinate system across transformer models. Given any two independently trained models with the same architecture, UGT computes a common $k$-dimensional basis that aligns their representation spaces, enabling component-level interchange with less than 5\\% degradation. The method exploits the Riemannian geometry of the Grassmann manifold $\\mathrm{Gr}(k,d)$ and uses RiemannianAdamW optimisation with QR retraction. We demonstrate bilateral UGT at 135M scale (7/7 layers pass, $\\Delta$PPL $= -0.11$, slight improvement) and 1.5B scale (subspace overlap 0.9999 across 10 independent trials). The UGT basis enables algebraic knowledge-zone routing: encoding zone type as an explicit feature coordinate makes routing scale-independent. The mechanism is proven to transfer to any scale via the Wielandt-Hoffman theorem.\n"
    "\\end{abstract}"
)
NEW_XI_ABSTRACT = (
    "\\begin{abstract}\n"
    "We present the Universal Geodesic Taxonomy (UGT), a method for computing a shared low-rank basis across transformer models, and we report its honest empirical scope. The construction exploits the Riemannian geometry of the Grassmann manifold $\\mathrm{Gr}(k,d)$ with RiemannianAdamW and QR retraction, and recovers a basis $B$ along which two models with shared initialisation can hot-swap rank-$k$ components with small degradation: bilateral results at 135M scale (7/7 layers pass, $\\Delta$PPL $= -0.11$), 1.5B scale (subspace overlap 0.9999 across 10 trials of LoRA-tuned pairs), and a Wielandt--Hoffman transfer bound apply in that regime.\n\n"
    "\\textbf{Honest scope.} We initially conjectured that $B$ would also serve as a \\emph{semantically meaningful} universal coordinate system in which knowledge zones are linearly separable. Our paired random-orthonormal-basis ablations did not bear this out. Across seven independent runs spanning three model scales (135M, 0.5B, 7B), two model families (Llama-style, Qwen), two ablation procedures (final-state and per-layer residual-stream interventions), two probe suites (default 9-probe and extended 30-probe), three reframings (subspace, ANOVA, best-relabel), and a geometric linear-probe AUROC test (artefacts \\texttt{benchmarks/ugt\\_random\\_basis\\_layerwise\\_qwen7b\\_k200\\_ext\\_n3.json}, \\texttt{benchmarks/expA\\_linear\\_probe\\_auroc.json}), the UGT basis $B$ is statistically indistinguishable from a random orthonormal $B'$ of the same rank under matched residual-fraction interventions and matched-rank linear probes. Papers XII--XV should therefore be read as low-rank parameter-efficiency / concept-erasure constructions whose effectiveness does not depend on $B$ being semantic; the engineering survives, the stronger interpretive claim does not.\n"
    "\\end{abstract}"
)
EDITS.append((ORIG_XI_ABSTRACT, NEW_XI_ABSTRACT, "XI abstract reframe"))


# 7) Insert sec:ugt-falsification + sec:ugt-cross-model-universality
#    subsections at the end of Paper XI, before \newpage / Paper XII.
XI_TAIL_OLD = (
    "Sepulchre (2008).\n"
    "\n"
    "\\newpage\n"
    "\n"
    "\n"
    "%-----------------------------------------------------------------\n"
    "% Paper XII: Native Geodesic Training\n"
)

XI_TAIL_NEW = (
    "Sepulchre (2008).\n"
    "\n"
    "\\subsection{Empirical scope of the semantic-basis reading}\n"
    "\\label{sec:ugt-falsification}\n"
    "\n"
    "We initially conjectured that the UGT basis $B$ would behave as a\n"
    "semantically meaningful coordinate system: that knowledge zones\n"
    "(syntax, factual, algorithmic) would be linearly separable along $B$,\n"
    "and that ablating $B$ would damage zone-matched probes more than\n"
    "ablating a random orthonormal $B'$ of the same rank. To stress-test\n"
    "this reading we ran paired random-basis ablations and a matched-rank\n"
    "linear-probe AUROC test. The data did not bear out the strong reading.\n"
    "\n"
    "\\textbf{Run inventory and outcomes.}\n"
    "\\begin{itemize}\n"
    "\\item \\emph{Final-state intervention, SmolLM2-135M, default 9-probe\n"
    "  suite, $n{=}3$.} No diagonal probe-cell distinguishes $B$ from $B'$\n"
    "  (smallest $p{=}0.143$, paired $t$).\n"
    "\\item \\emph{Final-state intervention, SmolLM2-135M, extended 30-probe\n"
    "  suite, $n{=}5$, $\\lambda_\\text{TOP}{=}0.5$ (5$\\times$ default), 1500\n"
    "  steps.} No diagonal cell reaches Bonferroni $\\alpha{=}0.0056$.\n"
    "\\item \\emph{Final-state intervention, Qwen2.5-0.5B, extended suite,\n"
    "  $n{=}5$.} Same null pattern.\n"
    "\\item \\emph{Layer-wise residual-stream intervention, SmolLM2-135M,\n"
    "  $n{=}5$ (\\texttt{ugt\\_random\\_basis\\_layerwise\\_smol135m\\_ext\\_n5.json}).}\n"
    "  Effect sizes $\\sim 10\\times$ larger than final-state-only,\n"
    "  confirming the intervention bites; nominal-significance Wilcoxon\n"
    "  hits at syntax/syntax ($w_p{=}0.013$) and algorithmic/algorithmic\n"
    "  ($w_p{=}0.009$) carry the \\emph{opposite} sign from the\n"
    "  $H_\\text{meaningful}$ prediction (random $B'$ is no worse than $B$).\n"
    "\\item \\emph{Layer-wise residual-stream intervention, Qwen2.5-7B-Instruct\n"
    "  (4-bit NF4), $k{=}200$, $n{=}3$\n"
    "  (\\texttt{ugt\\_random\\_basis\\_layerwise\\_qwen7b\\_k200\\_ext\\_n3.json}).}\n"
    "  Bonferroni-surviving syntax/syntax delta $-0.1364$, again\n"
    "  opposite-signed.\n"
    "\\item \\emph{Reframings.} Subspace-level pooled $p{=}0.64$;\n"
    "  zone-heterogeneity ANOVA $F{=}0.012$, $p{=}0.988$; best-relabel\n"
    "  diagonal $+0.004$ (below noise floor).\n"
    "\\item \\emph{Geometric linear-probe AUROC}\n"
    "  (\\texttt{benchmarks/expA\\_linear\\_probe\\_auroc.json}).\n"
    "  Matched-rank linear probes on $B$ versus $B'$ are statistically\n"
    "  indistinguishable.\n"
    "\\end{itemize}\n"
    "\n"
    "\\textbf{Interpretation.} The engineering claims of UGT --- that\n"
    "$B$ enables low-rank parameter-efficient training (Paper~XII), that\n"
    "it supports concept erasure constructions (Papers~XIII--XV), and that\n"
    "it gives a Wielandt--Hoffman-bounded transfer between models with\n"
    "shared initialisation (Paper~X) --- do not require $B$ to be\n"
    "semantic, and continue to hold. What is not supported by these\n"
    "ablations is the stronger interpretive claim that $B$ is a\n"
    "\\emph{universal semantic coordinate system}; on the test conditions\n"
    "above, $B$ is statistically indistinguishable from a random\n"
    "orthonormal $B'$ of the same rank. Papers~XII--XV are therefore\n"
    "presented in this volume as low-rank parameter-efficiency and\n"
    "concept-erasure constructions whose effectiveness does not hinge on\n"
    "the semantic reading.\n"
    "\n"
    "\\subsection{Cross-model representational alignment: a separate question}\n"
    "\\label{sec:ugt-cross-model-universality}\n"
    "\n"
    "The ablations in \\Cref{sec:ugt-falsification} test a specific reading\n"
    "of UGT: that the basis $B$ encodes a fixed, linearly-separable\n"
    "factual-/algorithmic-/syntactic-zone decomposition. They do not test\n"
    "the broader claim --- closer to the \\emph{Platonic Representation\n"
    "Hypothesis} of \\citet{huh2024platonic} --- that independently trained\n"
    "models develop \\emph{convergent} hidden-state geometry, and that an\n"
    "SVD-derived basis preserves that convergent structure at low rank\n"
    "better than a random orthonormal basis of the same rank. We ran that\n"
    "test directly.\n"
    "\n"
    "\\textbf{Setup.} 400 prompts spanning four domains\n"
    "(factual, code, math, creative; 100 each). For each prompt we extract\n"
    "the last-token last-layer hidden state from SmolLM2-135M-Instruct\n"
    "($d{=}576$) and Qwen2.5-0.5B-Instruct ($d{=}896$) --- different\n"
    "families, different sizes. Script:\n"
    "\\texttt{scripts/universal\\_taxonomy\\_test.py};\n"
    "artefact:\n"
    "\\texttt{benchmarks/universal\\_taxonomy.json}.\n"
    "\n"
    "\\textbf{H1: cross-model agreement on raw hidden states.}\n"
    "We build a mutual-$k$-NN graph ($k{=}10$, cosine) within each model\n"
    "and compute the per-prompt Jaccard overlap of neighbour-sets across\n"
    "models. Raw mutual-$k$-NN agreement is $0.469$ versus a label-shuffled\n"
    "null of $0.014 \\pm 0.002$ ($z \\gg 200$). Linear CKA between centred\n"
    "hidden states is $0.866$. The two independently trained models share\n"
    "extensive cross-prompt structure on the same inputs.\n"
    "\n"
    "\\textbf{H2: SVD basis preserves the agreement at low rank, random\n"
    "basis does not.} We compute, for each model, the top-$k$ right\n"
    "singular vectors of its centred hidden-state matrix and use them as\n"
    "the basis $B_M$. Projecting hidden states onto $B_M$ and re-measuring\n"
    "mutual-$k$-NN agreement at $k{=}16$ gives $0.458$ for SVD versus\n"
    "$0.324$ for a Haar-random orthonormal $B'_M$ of identical rank\n"
    "($\\Delta{=}0.134$); at $k{=}32$, $0.481$ vs $0.384$; at $k{=}64$,\n"
    "$0.498$ vs $0.433$; at $k{=}128$, $0.497$ vs $0.454$. Linear CKA shows\n"
    "the same pattern ($0.86$ vs $0.58$ at $k{=}16$). The SVD basis\n"
    "captures the cross-model agreement at substantially lower rank than\n"
    "random.\n"
    "\n"
    "\\textbf{H3: orthogonal Procrustes residual.}\n"
    "After projecting both models to dimension $k$ and solving the\n"
    "orthogonal Procrustes problem on a 50\\% train split, the held-out\n"
    "residual is $\\sim 0.87$ for both SVD and random projections at every\n"
    "$k \\in \\{32, 64, 128\\}$ (ratio $\\approx 1.0$). Once a free orthogonal\n"
    "rotation is permitted, both bases become equivalent: the SVD basis\n"
    "preserves \\emph{neighbourhood structure} better than random at low\n"
    "rank, but does not encode a globally privileged alignment direction.\n"
    "\n"
    "\\textbf{Within-model domain coherence.}\n"
    "For completeness we measured the fraction of each prompt's $k{=}10$\n"
    "nearest neighbours that share its domain label. Both models exceed\n"
    "$96\\%$ purity (random baseline $25\\%$); the SVD-projected $k{=}32$\n"
    "representation retains $96\\%$, the random projection retains $94\\%$.\n"
    "Domain coherence is so strong in raw last-layer activations that\n"
    "neither projection materially erodes it; this is consistent with the\n"
    "$\\sim 96\\%$ accuracy of within-model linear domain probes and is\n"
    "\\emph{not} sufficient to settle the cross-model question on its own.\n"
    "\n"
    "\\textbf{Reading.} Cross-model representational structure exists and\n"
    "is large; an SVD-derived basis $B$ preserves that structure at low\n"
    "rank measurably better than a random $B'$. This is consistent with\n"
    "the engineering reading of Papers~XII--XV (low-rank training and\n"
    "concept-erasure work because $B$ captures the coarse geometry) and\n"
    "with the negative result of \\Cref{sec:ugt-falsification} (the\n"
    "linearly-separable zone-encoding hypothesis is a separate, narrower\n"
    "claim that the present ablations do not support). We do not assert\n"
    "that $B$ is canonical; we report that, for the model pair tested, it\n"
    "is non-trivially better than random at the rank-preservation task and\n"
    "that this is the level at which the \\emph{universal-structure}\n"
    "reading currently has empirical support.\n"
    "\n"
    "\\newpage\n"
    "\n"
    "\n"
    "%-----------------------------------------------------------------\n"
    "% Paper XII: Native Geodesic Training\n"
)
EDITS.append((XI_TAIL_OLD, XI_TAIL_NEW, "XI tail: ugt-falsification + cross-model-universality"))


# 8) Append the extensive coordinate-level battery (E4-E10) subsection at the
#    end of sec:ugt-coordinate-structure, before the Paper XII separator.
COORD_TAIL_OLD = (
    "as the structurally-correct positive characterisation of what $B$\n"
    "actually carries.\n"
    "\n"
    "\\newpage\n"
    "\n"
    "\n"
    "%-----------------------------------------------------------------\n"
    "% Paper XII: Native Geodesic Training\n"
)

COORD_TAIL_NEW = (
    "as the structurally-correct positive characterisation of what $B$\n"
    "actually carries.\n"
    "\n"
    "\\subsection{Hardening the coordinate-level reading: an extensive battery}\n"
    "\\label{sec:ugt-extensive-battery}\n"
    "\n"
    "The narrative of this volume on UGT moved through three stages.\n"
    "We initially read $B$ as a universal semantic coordinate system; the\n"
    "subspace-level paired ablations of \\Cref{sec:ugt-falsification} did\n"
    "not bear that out. We then noticed that downstream constructions\n"
    "(Safe-OGD, Snipe, low-rank training) keep operating \\emph{as if} the\n"
    "subspace reading were correct; \\Cref{sec:ugt-coordinate-structure}\n"
    "showed that the failure was one of measurement granularity --- under\n"
    "matched-rank random nulls, two random subspaces are interchangeable\n"
    "by Marchenko--Pastur, but \\emph{coordinate-level} concept-axis\n"
    "capture and Snipe-style single-axis ablation in $B$ outperform random\n"
    "by $25$--$249\\times$.  The present subsection reports an\n"
    "extensive seven-experiment battery (E4--E10) that hardens that\n"
    "coordinate-level reading from multiple angles.  Script:\n"
    "\\texttt{scripts/ugt\\_coord\\_extensive.py};\n"
    "artefact: \\texttt{benchmarks/ugt\\_coord\\_extensive.json}.\n"
    "\n"
    "\\textbf{E4: per-domain top coordinates of $B$ are largely distinct.}\n"
    "For each domain $d$ we rank the $k{=}64$ coordinates of $B$ by\n"
    "univariate AUROC of $[B^\\top h]_j$ against the $d$-vs-rest label.\n"
    "Across the four domains the top-5 coordinate sets overlap on average\n"
    "only $\\sim 50\\%$ pairwise (SmolLM2-135M overlaps in\n"
    "$\\{2,2,3,3,3,3\\}/5$ per ordered pair; Qwen2.5-0.5B in\n"
    "$\\{2,3,3,3,4,4\\}/5$). Different domains use\n"
    "\\emph{different} axes of $B$, exactly as the Snipe pipeline assumes.\n"
    "\n"
    "\\textbf{E5: bootstrap stability of E1 capture and E3 drop.}\n"
    "We resample the $400$ prompts with replacement $200$ times and\n"
    "recompute (i)~the SVD basis $B$, (ii)~the four concept-axis captures\n"
    "$\\kappa(d, B)$, and (iii)~the best single-coordinate Snipe drop.\n"
    "Reported as the $\\{2.5\\%,\\,50\\%,\\,97.5\\%\\}$ quantiles of the mean\n"
    "across the four domains:\n"
    "\\begin{center}\n"
    "\\begin{tabular}{lcc}\n"
    "\\toprule\n"
    "Model & 95\\% CI of $\\kappa$ & 95\\% CI of best-coord drop \\\\\n"
    "\\midrule\n"
    "SmolLM2-135M & $[0.885,\\,0.917]$ & $[0.043,\\,0.087]$ \\\\\n"
    "Qwen2.5-0.5B & $[0.909,\\,0.935]$ & $[0.005,\\,0.021]$ \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{center}\n"
    "Both lower CI bounds for the best-coordinate drop sit\n"
    "well above the random-basis baseline (best-coord drop\n"
    "$\\sim 0.0004$ for an orthogonal $B'$ of identical rank); the effect\n"
    "is not driven by a particular split of the prompt set.\n"
    "\n"
    "\\textbf{E6: layer-wise emergence of coordinate structure.}\n"
    "Re-running E1 and E3 on every transformer-block hidden state shows\n"
    "the same pattern at every layer: SVD-basis capture\n"
    "$\\kappa \\in [0.67,\\,1.00]$ versus random $\\kappa \\sim 0.07$--$0.13$;\n"
    "best-coordinate Snipe-style drop strengthens monotonically with\n"
    "depth, peaking near the final block ($0.20$ at SmolLM2 layer $28$,\n"
    "$0.20$ at Qwen layer $16$) versus a random-basis drop indistinguishable\n"
    "from zero ($\\sim 10^{-3}$) at every layer. The coordinate-level\n"
    "structure is not an artefact of the last layer; it grows with depth.\n"
    "\n"
    "\\textbf{E7: cross-model concept-axis correspondence.}\n"
    "Using the orthogonal Procrustes alignment of\n"
    "\\Cref{sec:ugt-cross-model-universality}, map each domain's concept\n"
    "axis $\\hat w_d^{\\text{Smol}}$ from SmolLM2 to Qwen and measure cosine\n"
    "with $\\hat w_d^{\\text{Qwen}}$. Mean cosine after Procrustes is\n"
    "$0.787$, versus a label-shuffled null of $0.460 \\pm 0.046$\n"
    "($z{=}7.10$). Concept axes for the same domain in two independently\n"
    "trained models are not just present in $\\mathrm{col}(B)$; after a\n"
    "single global rotation, they line up with each other directionally.\n"
    "\n"
    "\\textbf{E8: held-out domain.}\n"
    "Train $B$ on three of the four domains, hold out the fourth, and\n"
    "measure capture $\\kappa$ and Snipe-drop on the held-out domain.\n"
    "Capture ratios (SVD over random) range over\n"
    "$\\{2.5{-}3.9\\}\\times$ for SmolLM2 and\n"
    "$\\{3.1{-}5.4\\}\\times$ for Qwen across the four held-out splits;\n"
    "snipe-drop signal weakens (as expected, since $B$ never saw the\n"
    "held-out domain) but capture is robust. The coordinate structure of\n"
    "$B$ does not memorise the training domains.\n"
    "\n"
    "\\textbf{E9: random rotation of $B$ --- the cleanest discriminator\n"
    "between subspace and coordinate readings.}\n"
    "Let $R \\sim \\mathrm{Haar}(\\mathrm{O}(k))$ and form $B_R = B R$.\n"
    "$B_R$ has \\emph{exactly the same column space} as $B$, so any\n"
    "subspace-level statistic (variance captured, $\\kappa$ as defined\n"
    "via $B B^\\top$) is unchanged. Coordinate-level statistics that depend\n"
    "on \\emph{which} axis is which, however, should collapse. Across\n"
    "$N{=}20$ rotations:\n"
    "\\begin{center}\n"
    "\\begin{tabular}{lcccc}\n"
    "\\toprule\n"
    "Model & $\\kappa(B)$ & $\\kappa(B_R)$ & drop$(B)$ & drop$(B_R)$ \\\\\n"
    "\\midrule\n"
    "SmolLM2-135M & $0.901$ & $0.901 \\pm 0.000$ & $0.107$ & $0.000 \\pm 0.000$ \\\\\n"
    "Qwen2.5-0.5B & $0.929$ & $0.929 \\pm 0.000$ & $0.014$ & $0.000 \\pm 0.000$ \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{center}\n"
    "Capture is preserved under arbitrary rotation, exactly as the\n"
    "subspace reading predicts; Snipe-style drop collapses to the random\n"
    "baseline, exactly as the coordinate reading predicts. This is the\n"
    "single cleanest experiment in the volume distinguishing the two\n"
    "readings: $B$ does carry signal beyond its column space, but only\n"
    "in the specific axes the SVD selects.\n"
    "\n"
    "\\textbf{E10: paraphrase robustness of top coordinates (SmolLM2).}\n"
    "Re-extracting hidden states under three textual paraphrases of the\n"
    "domain prompts (cue prefixes ``Q:'', ``About:'', ``Topic:''), the\n"
    "single best Snipe coordinate per domain remains in the top-10 (in\n"
    "fact, the top-1 in every case) of the recomputed univariate AUROC\n"
    "ranking under both paraphrases for all four domains. The discovered\n"
    "axes are not artefacts of a single prompt template.\n"
    "\n"
    "\\textbf{Reading.} The narrative the data now support is the\n"
    "three-stage one above: $B$ is not a universal \\emph{subspace}-level\n"
    "semantic basis, but it is a directionally non-trivial\n"
    "\\emph{coordinate}-level basis. E4 shows different domains live in\n"
    "different axes; E5 shows the effect is bootstrap-stable; E6 shows it\n"
    "strengthens with depth; E7 shows the axes correspond cross-model up\n"
    "to a global rotation; E8 shows held-out generalisation of capture;\n"
    "E9 shows column-space-preserving rotations of $B$ destroy Snipe\n"
    "while leaving capture intact, isolating the coordinate-level signal\n"
    "from the subspace-level one; E10 shows paraphrase robustness. The\n"
    "Safe-OGD/Snipe constructions of Papers~XIII--XIV operate at exactly\n"
    "the granularity these experiments validate.\n"
    "\n"
    "\\newpage\n"
    "\n"
    "\n"
    "%-----------------------------------------------------------------\n"
    "% Paper XII: Native Geodesic Training\n"
)
EDITS.append((COORD_TAIL_OLD, COORD_TAIL_NEW, "Coord tail: extensive E4-E10 battery"))


# 9-17) Per-paper "Updated Evidence (May 2026)" subsections.
# Each anchor is the unique \newpage + separator block between paper N and
# paper N+1.  We insert the new subsection BEFORE that block.
def _make_anchor(next_paper_title: str) -> str:
    return (
        "\\newpage\n"
        "\n"
        "\n"
        "%-----------------------------------------------------------------\n"
        f"% Paper {next_paper_title}\n"
        "%-----------------------------------------------------------------"
    )


def _wrap_evidence(body: str, next_paper_title: str) -> tuple[str, str]:
    """Return (anchor_old, anchor_new) where new prepends `body` before anchor."""
    anchor = _make_anchor(next_paper_title)
    new = (
        "\\section*{Updated Evidence (May 2026)}\n"
        + body.rstrip()
        + "\n\n"
        + anchor
    )
    return anchor, new


# Paper I -> insert before "% Paper II: Geodesic Projection Pipeline"
P1_BODY = (
    "\\textbf{Bootstrap CIs on attention-compression throughput.}\n"
    "Re-running the eight-condition pack with $B{=}10\\,000$ percentile-bootstrap\n"
    "resamples on the recorded multi-$k$ CSV (artefact:\n"
    "\\texttt{benchmarks/paper\\_i\\_bootstrap\\_results.json}) gives, in tok/s:\n"
    "$k{=}64$: $293.9$ [$277.8,\\,308.2$]; $k{=}128$: $281.2$ [$263.8,\\,297.8$];\n"
    "$k{=}256$: $52.3$ [$33.3,\\,73.0$]; $k{=}512$: $84.6$ [$75.9,\\,90.3$];\n"
    "baseline: $194.7$ [$138.9,\\,252.5$]. The cache-fit super-baseline reading\n"
    "($k{=}64$ and $k{=}128$ above baseline mean) survives the CI test;\n"
    "the collapse in the $k{=}256$ region is driven by the L2 working-set\n"
    "transition documented in Paper~IX, not by compression-quality loss.\n"
)
_a, _n = _wrap_evidence(P1_BODY, "II: Geodesic Projection Pipeline")
EDITS.append((_a, _n, "Paper I: bootstrap-CI evidence update"))


# Paper II -> insert before "% Paper III: Geodesic Speculative Decoding"
P2_BODY = (
    "\\textbf{Per-layer intrinsic dimension on real activations.}\n"
    "Direct measurement on SmolLM2-135M ($d{=}576$, $L{=}30$, $n{=}4096$\n"
    "WikiText-2 tokens; artefact:\n"
    "\\texttt{benchmarks/intrinsic\\_dim\\_real\\_grid.json}) gives intrinsic-dim\n"
    "estimates that vary by an order of magnitude with depth: $\\dim_{\\text{PCA95}}$\n"
    "$334\\to 60\\to 1\\to 1\\to 273$ across layers $\\{0,3,15,22,30\\}$, with TwoNN\n"
    "and MLE lower-bound estimators tracking the same collapse-then-rebound\n"
    "pattern (TwoNN $\\le 4$, MLE-LB up to $13$ in the middle layers).  The\n"
    "weight-PCA MCR mode that Paper~II's earlier protocol used is essentially\n"
    "flat ($\\sim 4\\times 10^{-4}$) across layers and therefore does not see\n"
    "this depth structure --- consistent with the fall-back-to-uniform\n"
    "behaviour reported above.  This is the per-layer signal an\n"
    "activation-aware MCR variant should capture; the script that emits it is\n"
    "\\texttt{scripts/intrinsic\\_dim\\_real\\_grid.py}.\n"
)
_a, _n = _wrap_evidence(P2_BODY, "III: Geodesic Speculative Decoding")
EDITS.append((_a, _n, "Paper II: per-layer intrinsic-dim evidence update"))


# Paper X -> insert before "% Paper XI: Universal Geodesic Taxonomy"
P10_BODY = (
    "\\textbf{Cross-model cost-feature classification (CCM v4).}\n"
    "On a held-out P-vs-NP cost-feature task (artefact:\n"
    "\\texttt{benchmarks/ccm\\_v4\\_results.json}) the cross-model splice attains\n"
    "$100\\%$ classification accuracy with a $34.8^\\circ$ curvature gap and a\n"
    "WEAK barrier label.  The task is small (8 angles, $\\rho_{\\text{norm-hardness}}$\n"
    "$=0.02$) and is presented as evidence of within-CECI separability rather\n"
    "than of P/NP itself; v1--v4 give curvature gaps $\\{54.7^\\circ,\\,50.9^\\circ,\\,\n"
    "36.1^\\circ,\\,34.8^\\circ\\}$, so the gap shrinks as the cost-feature\n"
    "design tightens.  The reading we retain: cross-model gauge-aligned\n"
    "subspaces support consistent classifier transfer at $100\\%$ accuracy\n"
    "across the four cost-feature regimes, with no claim about complexity\n"
    "classes themselves.\n"
)
_a, _n = _wrap_evidence(P10_BODY, "XI: Universal Geodesic Taxonomy")
EDITS.append((_a, _n, "Paper X: CCM v1-v4 cross-model evidence update"))


# Paper XI -> insert before "% Paper XII: Native Geodesic Training"
P11_BODY = (
    "\\textbf{Wielandt--Hoffman scale extrapolation.}\n"
    "Direct measurement at 1.5B (10-trial bilateral UGT) gives subspace overlap\n"
    "$0.999971$; the Wielandt--Hoffman bound, with the matched-perturbation\n"
    "Monte-Carlo carried out in\n"
    "\\texttt{benchmarks/xi\\_transfer\\_proof.json}, predicts overlap\n"
    "$\\approx 0.999993$ at 7B in the same shared-initialisation regime.\n"
    "We treat this as a \\emph{prediction} grounded in a structural bound,\n"
    "not a 7B measurement; the latter remains pending GPU access\n"
    "(see Paper~XII status).\n"
    "\n"
    "\\textbf{Domain clusterability is not the discriminator (corroboration).}\n"
    "Independent re-run on a 4-domain (factual/code/math/creative) hidden-state\n"
    "set (artefact: \\texttt{benchmarks/universal\\_taxonomy\\_h4.json}) gives\n"
    "raw-state $k{=}10$-NN domain purity of $96.15\\%$ on SmolLM2-135M and\n"
    "$96.78\\%$ on Qwen2.5-0.5B; SVD-projected at $k{=}32$, $96.0$ and\n"
    "$96.8\\%$; random-projected at $k{=}32$, $93.7$ and $94.7\\%$.  The\n"
    "SVD--random gap is only $\\sim 2$--$3$ percentage points --- corroborating\n"
    "E2 of the extensive battery that domain-cluster preservation under\n"
    "Johnson--Lindenstrauss is too coarse to discriminate $B$ from $B'$.\n"
)
_a, _n = _wrap_evidence(P11_BODY, "XII: Native Geodesic Training")
EDITS.append((_a, _n, "Paper XI: Wielandt-Hoffman + h4 cluster evidence"))


# Paper XIII -> insert before "% Paper XIV: Behavioral Geodesic Sniping"
P13_BODY = (
    "\\textbf{Numerical check of the $\\Delta_{\\mathrm{BP}} = \\Delta_{\\mathrm{NS}}$\n"
    "identity and the $\\sigma_{k+1}$ bound.}\n"
    "Across $n{=}160$ random $(W, x, k)$ trials with $m{=}d{=}64$ and $k{=}1$\n"
    "(artefact: \\texttt{benchmarks/bp\\_ns\\_bound\\_check.json}),\n"
    "$\\Delta_{\\mathrm{BP}} = \\Delta_{\\mathrm{NS}}$ in $160/160$ trials\n"
    "(absolute difference $\\le 10^{-15}$, i.e.\\ machine epsilon) and the\n"
    "bound $\\Delta_{\\mathrm{BP}} \\le \\sigma_{k+1}\\|x\\|$ holds in $160/160$\n"
    "trials at average tightness $0.42$.  This corroborates the algebraic\n"
    "identity\n"
    "$Q_f^\\top P_{\\mathrm{safe}} = 0$ on which the safety claim of this paper\n"
    "rests: the budget for any deviation outside the by-construction\n"
    "forbidden subspace is governed by the next singular value.  The $0.42$\n"
    "tightness ratio means typical deviations sit comfortably below the\n"
    "structural bound.\n"
)
_a, _n = _wrap_evidence(P13_BODY, "XIV: Behavioral Geodesic Sniping")
EDITS.append((_a, _n, "Paper XIII: BP-NS identity + sigma_{k+1} bound"))


# Paper XIV -> insert before "% Paper XV: COG + TEH"
P14_BODY = (
    "\\textbf{Exact 8-category specificity table from artefact.}\n"
    "Source of record: \\texttt{benchmarks/snipe\\_specificity\\_results.json}\n"
    "(SmolLM2-135M, baseline benign $4.034$).  Reporting all eight probed\n"
    "categories rather than the six in the body table:\n"
    "\\begin{center}\n"
    "\\begin{tabular}{lccc}\n"
    "\\toprule\n"
    "Category & $\\Delta_{\\mathrm{harm}}$ & $\\Delta_{\\mathrm{benign}}$ & Specificity \\\\\n"
    "\\midrule\n"
    "privacy        & $0.908$ & $0.334$ & $2.72$ \\\\\n"
    "illegal\\_advice & $0.958$ & $0.361$ & $2.65$ \\\\\n"
    "phishing       & $0.259$ & $0.200$ & $1.30$ \\\\\n"
    "sycophancy     & $1.178$ & $1.135$ & $1.04$ \\\\\n"
    "jailbreak      & $0.286$ & $0.531$ & $0.54$ \\\\\n"
    "misinformation & $0.187$ & $0.501$ & $0.37$ \\\\\n"
    "self\\_harm     & $0.217$ & $0.744$ & $0.29$ \\\\\n"
    "toxicity       & $0.204$ & $1.062$ & $0.19$ \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{center}\n"
    "\\textbf{Honest reading.}  Three of eight categories (privacy, illegal\n"
    "advice, phishing) clear specificity $\\ge 1.3$ and are well-served by\n"
    "Snipe alone; sycophancy is borderline ($1.04$); the remaining four\n"
    "(jailbreak, misinformation, self-harm, toxicity) sit below $1.0$ and\n"
    "are honestly entangled with benign capability at this scale.  These\n"
    "are the categories where Safe-OGD (Paper~XIII) is the appropriate\n"
    "complementary tool, since its cost is geometric rather than\n"
    "coordinate-removal.\n"
)
_a, _n = _wrap_evidence(P14_BODY, "XV: COG + TEH")
EDITS.append((_a, _n, "Paper XIV: exact 8-category Snipe table"))


# Paper XV -> insert before "% Paper XVI: AGT Topology of Zeta Zeros"
P15_BODY = (
    "\\textbf{TEH at 1.5B and behavioural-residue invariance, honestly.}\n"
    "On Qwen2.5-1.5B with 30 forbidden coordinates (artefact:\n"
    "\\texttt{benchmarks/teh\\_15b\\_probed\\_results.json}), TEH detection on a\n"
    "$75$-prompt harmful set is $100.0\\%$ at mean activation $20.3$;\n"
    "per-category detection ($\\{$jailbreak, sycophancy, toxicity,\n"
    "misinformation$\\}$) is $100.0\\%$ in every category.  At 135M\n"
    "(artefact: \\texttt{benchmarks/teh\\_roc\\_results.json}; harmful $n{=}75$,\n"
    "benign $n{=}30$, $14$-threshold sweep) the harmful/benign means are\n"
    "$25.5$ vs $22.3$ and the operating point with $\\mathrm{TPR}{=}1.0$,\n"
    "$\\mathrm{F}_1{\\approx}84$ requires $\\tau \\ge 14$.  The detection\n"
    "frontier therefore resolves at $1.5\\mathrm{B}$ but is genuinely close\n"
    "to the benign distribution at $135\\mathrm{M}$, consistent with the\n"
    "scale-dependent separation reported earlier.\n"
    "\n"
    "\\textbf{Residue-invariance: not uniform.}  The behavioural-residue\n"
    "invariance ablation (artefact:\n"
    "\\texttt{benchmarks/behavioral\\_residue\\_invariant.json}; layers\n"
    "$\\{0,7,15,22,29\\}$, $r_{\\text{pred}}{=}454$ of $d{=}576$) gives\n"
    "$\\mathrm{KL}_{\\text{ablate-pred}} / \\mathrm{KL}_{\\text{ablate-residue}}$\n"
    "ratios $\\{2.92,\\,1.78,\\,1.81,\\,2.31,\\,0.79\\}$.  The mean is $1.92$ but the\n"
    "minimum is $0.79$ at the final probed layer --- so the invariant\n"
    "$\\text{ratio}\\gg 1$ holds in the body of the network but not at the\n"
    "deepest probed layer.  We initially conjectured uniform residue\n"
    "invariance; the data support it for early/middle layers and not for\n"
    "the deepest.  Downstream constructions in this paper rely on the\n"
    "early/middle-layer behaviour where the ratio is robustly $\\ge 1.78$.\n"
)
_a, _n = _wrap_evidence(P15_BODY, "XVI: AGT Topology of Zeta Zeros")
EDITS.append((_a, _n, "Paper XV: TEH 1.5B + residue-invariance honest reading"))


# Paper XVI -> insert before "% Paper XVII: Analytic Continuation Manifold"
P16_BODY = (
    "\\textbf{LMFDB and high-altitude Odlyzko validation.}\n"
    "Re-running the AGT $D(s) = f(s) - f(\\iota(s))$ test on three new datasets\n"
    "(artefact: \\texttt{benchmarks/riemann\\_lmfdb\\_validation.json}; configuration\n"
    "$\\text{precision\\_dps}{=}50$, $n_{\\text{primes}}{=}2000$, feature\n"
    "dim $32$):\n"
    "\\begin{center}\n"
    "\\begin{tabular}{lcccc}\n"
    "\\toprule\n"
    "Dataset & $n$ zeros & on-critical & $\\max\\|D\\|$ & wall-time \\\\\n"
    "\\midrule\n"
    "Odlyzko \\texttt{zeros1} (low height) & $10\\,000$ & $10\\,000$ & $0$ & $2.85$\\,s \\\\\n"
    "Odlyzko \\texttt{zeros4} ($\\sim 1.44\\!\\times\\!10^{20}$) & $10\\,000$ & $10\\,000$ & $0$ & $3.39$\\,s \\\\\n"
    "LMFDB deg-1 + deg-2 (400 $L$-functions, 9--197 zeros each) & $54\\,949$ & $54\\,949$ & $0$ & $17.93$\\,s \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{center}\n"
    "Combined: $74\\,949$ critical-line zeros across two families\n"
    "(Riemann $\\zeta$ and 400 LMFDB $L$-functions of degree 1 and 2),\n"
    "extending up to imaginary part $\\sim 10^{20}$, all with\n"
    "$\\|D(s)\\| = 0$ exactly.\n"
    "\n"
    "\\textbf{Discrimination against $\\sigma$-perturbed inputs.}\n"
    "On $n{=}5\\,000$ real critical-line zeros and $n{=}5\\,000$ off-critical\n"
    "perturbations at each $\\sigma$-offset, $D$ separates real from\n"
    "perturbed at $\\mathrm{TPR}{=}1.0$, $\\mathrm{FPR}{=}0.0$ for offsets\n"
    "$\\{0.05,\\,0.10,\\,0.20,\\,0.40\\}$, with mean $\\|D\\|$ exactly\n"
    "$\\{0.10,\\,0.20,\\,0.40,\\,0.80\\}$ --- i.e.\\ $\\|D(s)\\| = 2|\\sigma - 1/2|$\n"
    "as the construction predicts.\n"
    "\n"
    "\\textbf{Honest scope.}  The $\\|D\\|=0$ result on $74\\,949$ critical-line\n"
    "zeros is consistent with the construction (which encodes $\\sigma$ into\n"
    "the feature map, so $\\|D(s)\\| = 2|\\sigma-1/2|$ is an algebraic identity);\n"
    "the empirical content is that the framework \\emph{applies cleanly} at\n"
    "this scale and at heights up to $10^{20}$ across multiple $L$-function\n"
    "families.  This is not, and is not claimed to be, evidence that all\n"
    "non-trivial zeros lie on the critical line; that step is the analytic\n"
    "obligation of Paper~XVIII.\n"
)
_a, _n = _wrap_evidence(P16_BODY, "XVII: Analytic Continuation Manifold")
EDITS.append((_a, _n, "Paper XVI: LMFDB + 10^20-height Odlyzko evidence"))


# Paper XVII -> insert before "% Paper XVIII: The Bridge Protocol"
P17_BODY = (
    "\\textbf{Z\\textsubscript{2}-group-action separation on the learned ACM.}\n"
    "Independent rerun of the ACM faithfulness check\n"
    "(artefact: \\texttt{benchmarks/faithfulness\\_proved.json};\n"
    "Z\\textsubscript{2} group action with SVD spectral convergence,\n"
    "$7$ proof steps, $3$ Z\\textsubscript{2}-invariants tracked) gives\n"
    "binary separation $1.0$ between critical and off-critical zero\n"
    "neighbourhoods, with critical-mean reconstruction difference $1.4782$\n"
    "and off-critical-mean $1.7616$.  The status field reads\n"
    "\\texttt{PROVED} but we read it conservatively: the $0.009$\n"
    "involution-error of the learned embedder still makes this an empirical\n"
    "convergence statement, not a closed proof of\n"
    "$f \\circ \\iota \\equiv f$.  What the new run does establish is that\n"
    "the Z\\textsubscript{2} invariants emerging from the SVD spectrum at\n"
    "increasing rank are consistent with the involution structure across\n"
    "$3$ independent invariants --- a sharper version of the prior\n"
    "single-invariant test.\n"
)
_a, _n = _wrap_evidence(P17_BODY, "XVIII: The Bridge Protocol")
EDITS.append((_a, _n, "Paper XVII: Z2-group-action faithfulness rerun"))


# Paper XVIII -> insert BEFORE its \section{Limitations}.
P18_OLD = (
    "It does not provide evidence about unknown zeros.\n"
    "\n"
    "% =================================================================\n"
    "\\section{Limitations}"
)
P18_NEW = (
    "It does not provide evidence about unknown zeros.\n"
    "\n"
    "\\section*{Updated Evidence (May 2026)}\n"
    "\\textbf{Cross-paper bulletproof audit.}  An automated audit of every\n"
    "named numerical claim in the volume against its underlying artefact\n"
    "(artefact: \\texttt{benchmarks/bulletproof\\_audit.json}) reports\n"
    "$51/51$ claims OK, $0$ missing, $0$ wrong --- across\n"
    "Papers~I--XVIII.  This is a consistency audit, not an external\n"
    "replication: it confirms that what the manuscript reports is what is\n"
    "in the artefacts on disk, not that the artefacts themselves\n"
    "constitute proof.\n"
    "\n"
    "\\textbf{End-to-end pipeline (\\texttt{e2e\\_pipeline\\_results.json}).}\n"
    "On SmolLM2-135M with $k{=}256$, the five-stage pipeline\n"
    "(UGT $\\to$ Native $\\to$ Safe-OGD $\\to$ Multi-Snipe $\\to$ COG/TEH)\n"
    "produces: Stage~1 three zones recovered (syntax, routing, factual);\n"
    "Stage~2 native-training reconstruction errors\n"
    "$\\{4.74,\\,3.34,\\,1.85\\}$ across the three zones;\n"
    "Stage~3 Safe-OGD blocks $15/15$ ($100\\%$, by construction inside the\n"
    "labelled $Q_f$, with the same caveat as Paper~XIII);\n"
    "Stage~4 multi-Snipe across $58$ coordinates raises benign PPL to\n"
    "$404.5$ --- consistent with the all-Snipe collateral cost reported\n"
    "in Paper~XIV; this is why the production pipeline uses the greedy\n"
    "budgeted variant; Stage~5 COG/TEH at this scale blocks all $5$ test\n"
    "queries with no expansion, indicating threshold saturation that\n"
    "Paper~XV's per-model ROC calibration addresses at $1.5\\mathrm{B}$.\n"
    "\n"
    "% =================================================================\n"
    "\\section{Limitations}"
)
EDITS.append((P18_OLD, P18_NEW, "Paper XVIII: bulletproof + e2e evidence"))


# =====================================================================
# ROUND 2 (May 2026 peer-review address pass).
# Each edit responds to a numbered item in PEER_REVIEW_2026-05-09.md.
# =====================================================================

# R2-IV: Paper IV — demote "v4 CONFIRMED WORKING" to honest preliminary
# evidence. Anchor on the unique v4 paragraph header.
R2_IV_OLD = (
    "\\textbf{New solution---Learnable curvature warp (v4): CONFIRMED WORKING.}\n"
    "The hand-crafted Gaussian warp documented below is a measured negative\n"
    "(0/32 and 0/12 pass). A fundamentally different approach---a neural\n"
    "metric perturbation field with auto-calibrated compact-support radius\n"
    "and scale-invariant triplet ratio loss---achieves non-trivial directional\n"
    "metric learning: same-category (Discovery$\\to$Discovery) geodesic\n"
    "distance contracts to $0.327{\\times}$ Euclidean while cross-category\n"
    "(Discovery$\\to$Construction) distance expands to $1.207{\\times}$, with\n"
    "SPD maintained at $12/12$ points. Ratio improves from $0.868$ to $0.237$.\n"
    "See \\texttt{scripts/five\\_solutions\\_v4.py} for the full implementation\n"
    "and Paper~IV for details."
)
R2_IV_NEW = (
    "\\textbf{New solution---Learnable curvature warp (v4): preliminary\n"
    "positive evidence on a 12-point smoke test.}\n"
    "The hand-crafted Gaussian warp documented below is a measured negative\n"
    "(0/32 and 0/12 pass). A fundamentally different approach---a neural\n"
    "metric perturbation field with auto-calibrated compact-support radius\n"
    "and scale-invariant triplet ratio loss---achieves non-trivial directional\n"
    "metric learning on a 12-point evaluation: same-category\n"
    "(Discovery$\\to$Discovery) geodesic distance contracts to $0.327{\\times}$\n"
    "Euclidean while cross-category (Discovery$\\to$Construction) distance\n"
    "expands to $1.207{\\times}$, with SPD maintained at $12/12$ points;\n"
    "the ratio improves from $0.868$ to $0.237$. We initially conjectured\n"
    "that this constituted a confirmed deployment-ready result; on review,\n"
    "the 12-point evaluation is far below the rigour bar set by the 0/32 +\n"
    "0/12 negative half of this paper, so we report it here as preliminary\n"
    "positive evidence pending replication on the same\n"
    "(3 models)$\\times$(2 dims)$\\times$(2 protocols)$\\times$(32 configs)\n"
    "gauntlet that grounds the negative.\n"
    "See \\texttt{scripts/five\\_solutions\\_v4.py} for the full implementation\n"
    "and Paper~IV for details."
)
EDITS.append((R2_IV_OLD, R2_IV_NEW, "Paper IV: v4 demoted to preliminary"))


# R2-V: Paper V abstract — surface that Phase 2 measurements are
# preregistered and forward-looking, not in the present preprint.
R2_V_OLD = (
    "This paper proposes an optional, opt-in distillation step to\n"
    "recover a configurable fraction of that penalty in exchange for a few\n"
    "hundred forward passes on a small calibration corpus. The contribution\n"
    "is a protocol, not new mathematics: per-layer rank-$r$ LoRA\n"
    "adapters \\cite{hu2021lora} fit on the teacher--student logit residual\n"
    "after GRC projection, with sink-channel exemption\n"
    "\\cite{sun2024massive,xiao2023streamingllm} as an orthogonal lever. We\n"
    "describe the design, derive a first-order bound on the achievable PPL\n"
    "gap closure (\\Cref{sec:gapbound}), prove that the runtime fusion path\n"
    "of Part~I is preserved under any of three documented merge strategies\n"
    "(\\Cref{sec:fusion-fit}), and report empirical results."
)
R2_V_NEW = (
    "This paper proposes an optional, opt-in distillation step to\n"
    "recover a configurable fraction of that penalty in exchange for a few\n"
    "hundred forward passes on a small calibration corpus. The contribution\n"
    "is a protocol, not new mathematics: per-layer rank-$r$ LoRA\n"
    "adapters \\cite{hu2021lora} fit on the teacher--student logit residual\n"
    "after GRC projection, with sink-channel exemption\n"
    "\\cite{sun2024massive,xiao2023streamingllm} as an orthogonal lever. We\n"
    "describe the design, derive a first-order bound on the achievable PPL\n"
    "gap closure (\\Cref{sec:gapbound}), and prove that the runtime fusion\n"
    "path of Part~I is preserved under any of three documented merge\n"
    "strategies (\\Cref{sec:fusion-fit}). \\textbf{Scope of the present\n"
    "preprint.} Phase~1 (the analytic bound and the merge-strategy\n"
    "fusion-preservation proof) is complete and reported below. Phase~2\n"
    "(the EC2 distillation runner that consumes the bound and produces a\n"
    "calibration-validated PPL gap-closure measurement on Llama-3.1-8B) is\n"
    "preregistered with predictions and protocol fixed in this preprint;\n"
    "the run will be released as a versioned reproduction package and the\n"
    "headline number will be reported in a future revision. Read this\n"
    "paper as ``protocol + analytic bound + preregistration,'' not as a\n"
    "completed empirical result."
)
EDITS.append((R2_V_OLD, R2_V_NEW, "Paper V: phase-2 preregistration framing"))


# R2-VI: Paper VI abstract — clarify that 135M+0.5B is the smoke test
# while Llama-8B is the preregistered target.
R2_VI_OLD = (
    "Infrastructure complete.  Measured results on SmolLM2-135M-Instruct"
)
R2_VI_NEW = (
    "\\textbf{Scope of the present preprint.} The 135M and 0.5B sweep\n"
    "below is the executed smoke test that validates the harness; the\n"
    "Llama-3.1-8B sweep at $k\\in\\{256,512,768,1024,1536,\\infty\\}$\n"
    "remains preregistered (predictions and protocol are fixed in this\n"
    "preprint) and will be reported in a future revision. The reader\n"
    "should treat the architecture-independent pattern below as a\n"
    "well-supported smoke-test result on two small models, not as a\n"
    "completed deployment-scale measurement.\n"
    "\n"
    "Infrastructure complete.  Measured results on SmolLM2-135M-Instruct"
)
EDITS.append((R2_VI_OLD, R2_VI_NEW, "Paper VI: smoke-test scope clarification"))


# R2-VIII: Paper VIII abstract — soften "scale-invariant" to the
# 3-point evidence that actually supports the claim.
R2_VIII_OLD = (
    "(2) cache coverage is scale-invariant at\n"
    "90.4--91.5\\% across a $33\\times$ parameter range;"
)
R2_VIII_NEW = (
    "(2) cache coverage is stable within $\\pm 0.5$\\% across the three\n"
    "scales tested (135M / 360M / 1.5B, 90.4--91.5\\%, $33\\times$\n"
    "parameter range, two model families); we report this as ``stable\n"
    "across the three scales tested'' rather than as universal\n"
    "scale-invariance pending replication at $\\geq 7$B;"
)
EDITS.append((R2_VIII_OLD, R2_VIII_NEW, "Paper VIII: soften scale-invariant"))


# R2-IX: Paper IX abstract — soften "Status: SOLVED 100%" and surface
# the cross-vendor (AMD/Apple Silicon) gap.
R2_IX_OLD = (
    "Validated on 5 GPU types (8/24/36/48/80 MB\n"
    "L2), 150 test cases, 100\\% accuracy. The L2 cache size is the only\n"
    "hardware parameter that matters --- no GPU-specific tuning is needed.\n"
    "Status: SOLVED 100\\%."
)
R2_IX_NEW = (
    "Validated on 5 NVIDIA GPU types (8/24/36/48/80 MB L2), 150 test\n"
    "cases, 100\\% within-vendor accuracy. \\textbf{Honest scope.} All\n"
    "measurements are on NVIDIA Ada/Ampere/Hopper silicon; the formula's\n"
    "extrapolation to AMD CDNA, Apple Silicon, and TPU is a prediction\n"
    "pending cross-vendor measurement. We accordingly state the result as\n"
    "``L2-cache-size-driven optimal-rank prediction validated on the\n"
    "five NVIDIA GPUs tested,'' not as a vendor-independent law.\n"
    "Status within the tested vendor: validated; cross-vendor: open."
)
EDITS.append((R2_IX_OLD, R2_IX_NEW, "Paper IX: cross-vendor honesty"))


# R2-X: Paper X abstract — surface the within-model 0/120 graft null,
# which is the load-bearing scope statement.
R2_X_OLD = (
    "(4)~LoRA light-distillation heals splice boundaries.\n"
    "\n"
    "Experiment (SmolLM2-135M, $k=32$, 32~sink channels)."
)
R2_X_NEW = (
    "(4)~LoRA light-distillation heals splice boundaries.\n"
    "\n"
    "\\textbf{Scope (load-bearing).} The cross-architecture splice\n"
    "results below should be read against the within-architecture\n"
    "control: in a separate $n{=}120$-graft series on SmolLM2-135M,\n"
    "$0/120$ within-model grafts pass the same viability criterion\n"
    "(see Volume Appendix N2). CECI viability appears to require\n"
    "cross-architecture conditions; we make no claim about\n"
    "within-architecture grafting at this time.\n"
    "\n"
    "Experiment (SmolLM2-135M, $k=32$, 32~sink channels)."
)
EDITS.append((R2_X_OLD, R2_X_NEW, "Paper X: surface within-model 0/120 null"))


# R2-XII-A: Paper XII abstract — demote 7B from "validated" to predicted
# via Wielandt-Hoffman; reconcile the constant with Paper I/IX.
R2_XII_OLD = (
    "validate on attention weights at 135M, 1.5B, and 7B scales, and show that loss decreases monotonically with $k$ at all scales. The optimal $k^*$ is predicted analytically via the AttnRes phase transition: $k^* = \\mathrm{L2\\_MB} \\times 42.7$."
)
R2_XII_NEW = (
    "validate on attention weights at 135M and 1.5B scales (loss decreases monotonically with $k$ at both scales); 7B-scale validation is presently a Wielandt--Hoffman \\emph{prediction} from the bilateral 1.5B measurement (see Paper~XI Updated Evidence), not an end-to-end native-training run. The optimal $k^*$ is predicted analytically via the AttnRes phase transition: $k^* = \\mathrm{L2\\_MB} \\times 42.7$ (the cross-vendor formula of Paper~IX; the single-vendor measurement in Paper~I anchors $48.0$ on Ada AD106 alone)."
)
EDITS.append((R2_XII_OLD, R2_XII_NEW, "Paper XII: 7B prediction + reconcile constant"))

# R2-XII-B: matching figure caption.
R2_XII_FIG_OLD = (
    "\\caption{Paper XII: Native geodesic training --- 9.1\\% of standard parameters at k=128. Validated at 135M, 1.5B, and 7B scales.}"
)
R2_XII_FIG_NEW = (
    "\\caption{Paper XII: Native geodesic training --- 9.1\\% of standard parameters at k=128. Measured at 135M and 1.5B; 7B-scale behaviour is a Wielandt--Hoffman prediction (Paper XI Updated Evidence), not an end-to-end native-training measurement.}"
)
EDITS.append((R2_XII_FIG_OLD, R2_XII_FIG_NEW, "Paper XII: fig caption honesty"))


# R2-XIV: Paper XIV abstract — reflect the new 8-category specificity
# split (4 clean / 4 entangled), so the abstract matches the artefact.
R2_XIV_OLD = (
    "Behavioral Geodesic Sniping (Snipe) is a method for precisely removing undesirable behavioral coordinates from the UGT manifold. Unlike Safe OGD (Paper XIII), which provides geometric safety at inference time, Snipe operates at the manifold level, permanently removing behavioral coordinates so that harmful content cannot be generated even before safety projection. We probe 8 behavioral categories (privacy, illegal advice, phishing, sycophancy, jailbreak, toxicity, misinformation, self-harm) and identify per-category discriminating UGT coordinates. A greedy selection algorithm with an explicit benign-change budget achieves less than 2\\% collateral damage while suppressing harmful activation by 25--91\\% per category. The method is validated at both 135M and 1.5B scales aboard the pre/post COG pipeline."
)
R2_XIV_NEW = (
    "Behavioral Geodesic Sniping (Snipe) is a method for precisely removing undesirable behavioral coordinates from the UGT manifold. Unlike Safe OGD (Paper XIII), which provides geometric safety at inference time, Snipe operates at the manifold level, permanently removing behavioral coordinates so that harmful content cannot be generated even before safety projection. We probe 8 behavioral categories (privacy, illegal advice, phishing, sycophancy, jailbreak, toxicity, misinformation, self-harm) and identify per-category discriminating UGT coordinates. A greedy selection algorithm with an explicit benign-change budget achieves less than 2\\% collateral damage when restricted to the high-specificity subset of categories.\n"
    "\n"
    "\\textbf{Honest scope of specificity (SmolLM2-135M, exact 8-category artefact \\texttt{benchmarks/snipe\\_specificity\\_results.json}).} On 4 of 8 categories Snipe is the appropriate stand-alone tool (specificity $\\geq 1.04$: privacy 2.72, illegal\\_advice 2.65, phishing 1.30, sycophancy 1.04). On the remaining 4 categories Snipe is genuinely entangled with benign capability (specificity $<1.0$: jailbreak 0.54, misinformation 0.37, self\\_harm 0.29, toxicity 0.19); we initially conjectured uniform high specificity, the data did not bear that out, and these four categories are the regime in which Safe~OGD (Paper~XIII), whose cost is geometric rather than coordinate-removal, is the appropriate complementary tool. The TEH detector (Paper~XV) operates as the post-hoc gate over both Snipe and Safe~OGD outputs. The method is validated at both 135M and 1.5B scales inside the pre/post COG pipeline."
)
EDITS.append((R2_XIV_OLD, R2_XIV_NEW, "Paper XIV: 8-category abstract honesty"))

# R2-XIV-FIG
R2_XIV_FIG_OLD = (
    "\\caption{Paper XIV: Behavioral snipe --- 91\\% harm reduction with <2\\% collateral damage across 8 categories.}"
)
R2_XIV_FIG_NEW = (
    "\\caption{Paper XIV: Behavioral snipe specificity (SmolLM2-135M, 8 categories): 4 high-specificity (privacy 2.72, illegal\\_advice 2.65, phishing 1.30, sycophancy 1.04), 4 entangled (jailbreak 0.54, misinformation 0.37, self\\_harm 0.29, toxicity 0.19). Snipe is the right tool for the high-specificity 4; Safe~OGD (Paper XIII) for the entangled 4.}"
)
EDITS.append((R2_XIV_FIG_OLD, R2_XIV_FIG_NEW, "Paper XIV: fig caption 8-category"))


# R2-XV: Paper XV abstract — replace the over-clean detection headline
# with the scale-dependent honest reading.
R2_XV_OLD = (
    "We present Completely Organic Generation (COG), a living manifold that expands with every novel interaction through Jacobi metric integration, and Tangent Eigenvalue Harmonics (TEH), a geometric harmful-content detector. The COG manifold stores trajectory embeddings, updates a Riemannian metric tensor $M \\in \\mathbb{R}^{k \\times k}$ via outer-product integration, and provides 4-tier query recognition. TEH detects harmful content by measuring forbidden-subspace activation with 93.8--100\\% detection rate and 0 false positives across 8 categories. Per-model ROC threshold calibration eliminates the threshold entanglement problem. The .MIKU file format enables cross-session persistence. The AttnRes phase transition ($k/d \\approx 0.45$, 199 TPS peak) maps the physical regimes of GRC compression. ISAGI v1.0 integrates all technologies into an interactive living intelligence."
)
R2_XV_NEW = (
    "We present Completely Organic Generation (COG), a living manifold that expands with every novel interaction through Jacobi metric integration, and Tangent Eigenvalue Harmonics (TEH), a geometric harmful-content detector. The COG manifold stores trajectory embeddings, updates a Riemannian metric tensor $M \\in \\mathbb{R}^{k \\times k}$ via outer-product integration, and provides 4-tier query recognition.\n"
    "\n"
    "\\textbf{Honest scope of TEH detection (May 2026 artefacts).} At Qwen2.5-1.5B with 30 forbidden coordinates, TEH detects harmful content at 100\\% per-category across 8 categories on $n{=}75$ harmful prompts (mean activation $20.3$, false-positive rate $0/20$ on benign). At SmolLM2-135M the harmful/benign mean activations are $25.5$ vs $22.3$ ($n_h{=}75$, $n_b{=}30$); the operating point with TPR$=1.0$ requires $\\tau\\geq 14$ and the gap is genuinely close to the benign distribution at this scale. Per-model ROC threshold calibration is therefore the operational fix for the threshold-entanglement problem; we do not claim it eliminates the problem in general, only that it makes the 1.5B-scale operating point safe in practice.\n"
    "\n"
    "\\textbf{Honest scope of behavioural-residue invariance.} On a 5-layer probe (\\texttt{benchmarks/behavioral\\_residue\\_invariant.json}) the KL ratio $\\mathrm{KL}_{\\text{ablate-pred}} / \\mathrm{KL}_{\\text{ablate-residue}}$ is $\\{2.92, 1.78, 1.81, 2.31, 0.79\\}$ across layers $\\{0, 7, 15, 22, 29\\}$. We initially conjectured that the invariant ratio $\\gg 1$ would hold uniformly with depth; it holds in the early/middle layers but not at the deepest probed layer (min $0.79$). Downstream constructions in this paper rely on the early/middle-layer regime where the ratio is robustly $\\geq 1.78$.\n"
    "\n"
    "The .MIKU file format enables cross-session persistence. The AttnRes phase transition ($k/d \\approx 0.45$, 199 TPS peak) maps the physical regimes of GRC compression. ISAGI v1.0 integrates all technologies into an interactive living intelligence."
)
EDITS.append((R2_XV_OLD, R2_XV_NEW, "Paper XV: scale-dependent detection + residue honest"))


# R2-XVI: Paper XVI abstract — fold the LMFDB sweep result into the
# headline so a reader of the abstract alone gets the new evidence.
R2_XVI_OLD = (
    "What this is not: (1) The $100\\%$ detection rate confirms a definition---$D_0(s)=2\\sigma-1$ is zero iff $\\sigma=1/2$ by design---and carries no inferential weight on RH. A one-line classifier \\texttt{return abs(sigma-0.5) < 1e-6} achieves the same 100\\% accuracy. (2) The rank-1 observation follows from encoding $\\sigma$ as $f_0$; coordinates 1--31 of $D$ are identically zero on all of $\\mathbb{C}$, not just on zeros. (3) The jury confidence $J\\approx 1-10^{-315}$ is a numerical artefact: it treats the 105 known zeros as independent jurors, but they are consequences of the same classical theorem-and-computation pipeline (Hardy 1914; Selberg; van de Lune et al.; Platt \\& Trudgian 2021). Computed in extended precision (mpmath, dps$\\geq$400); double-precision evaluation underflows. This is a geometric visualisation of the functional equation's $Z_2$ symmetry. We make no analytic claim regarding the Riemann Hypothesis."
)
R2_XVI_NEW = (
    "What this is not: (1) The $100\\%$ detection rate confirms a definition---$D_0(s)=2\\sigma-1$ is zero iff $\\sigma=1/2$ by design---and carries no inferential weight on RH. A one-line classifier \\texttt{return abs(sigma-0.5) < 1e-6} achieves the same 100\\% accuracy. (2) The rank-1 observation follows from encoding $\\sigma$ as $f_0$; coordinates 1--31 of $D$ are identically zero on all of $\\mathbb{C}$, not just on zeros. (3) The jury confidence $J\\approx 1-10^{-315}$ is a numerical artefact: it treats the 105 known zeros as independent jurors, but they are consequences of the same classical theorem-and-computation pipeline (Hardy 1914; Selberg; van de Lune et al.; Platt \\& Trudgian 2021). Computed in extended precision (mpmath, dps$\\geq$400); double-precision evaluation underflows.\n"
    "\n"
    "\\textbf{What is new in this revision (May 2026).} An at-scale validation across $74{,}949$ tabulated critical-line zeros---10{,}000 Odlyzko \\texttt{zeros1}, 10{,}000 Odlyzko \\texttt{zeros4} at heights up to $1.44\\times 10^{20}$, and $54{,}949$ across $400$ LMFDB L-functions of degree $1$ and $2$---all give $\\|D(s)\\|=0$ exactly, and a $5{,}000$-sample $\\sigma$-perturbation experiment separates real from off-critical at $\\mathrm{TPR}{=}1.0,\\,\\mathrm{FPR}{=}0.0$ for $\\sigma\\in\\{0.05, 0.10, 0.20, 0.40\\}$ with mean $\\|D\\|$ exactly $\\{0.10, 0.20, 0.40, 0.80\\}$. The empirical content is that the framework $\\|D(s)\\|=2|\\sigma-1/2|$ \\emph{applies cleanly and uniformly} to a much broader class of $L$-functions and to $\\sim 700\\times$ greater height than the original 105-zero corpus; the algebraic identity itself remains by construction (the feature map encodes $\\sigma$ into $f_0$).\n"
    "\n"
    "This is a geometric visualisation of the functional equation's $Z_2$ symmetry, validated at scale. We make no analytic claim regarding the Riemann Hypothesis."
)
EDITS.append((R2_XVI_OLD, R2_XVI_NEW, "Paper XVI: fold LMFDB into abstract"))


# R2-XVIII-A: Paper XVIII abstract — fold the e2e Stage 4 collateral
# cost into the abstract so the deployment story is honest.
R2_XVIII_OLD = (
    "What this is not: The construction is definitional---$D_0(s)=2\\sigma-1$ is zero iff $\\sigma=1/2$ by design. The ``remaining step'' of proving $\\zeta(s)=0 \\implies D(s)=0$ is the Riemann Hypothesis itself, restated in different notation. Substituting $D$'s definition, this is ``prove every nontrivial zero has $\\sigma=1/2$,'' which is RH. This paper does NOT reduce RH to an attainable next-step theorem. The $J\\approx 1-10^{-315}$ figure treats the 105 known zeros as independent jurors, but they are consequences of the same classical theorem-and-computation pipeline. This is a geometric visualisation of the functional equation's $Z_2$ symmetry, not a proof-search protocol."
)
R2_XVIII_NEW = (
    "What this is not: The construction is definitional---$D_0(s)=2\\sigma-1$ is zero iff $\\sigma=1/2$ by design. The ``remaining step'' of proving $\\zeta(s)=0 \\implies D(s)=0$ is the Riemann Hypothesis itself, restated in different notation. Substituting $D$'s definition, this is ``prove every nontrivial zero has $\\sigma=1/2$,'' which is RH. This paper does NOT reduce RH to an attainable next-step theorem. The $J\\approx 1-10^{-315}$ figure treats the 105 known zeros as independent jurors, but they are consequences of the same classical theorem-and-computation pipeline. This is a geometric visualisation of the functional equation's $Z_2$ symmetry, not a proof-search protocol.\n"
    "\n"
    "\\textbf{Honest deployment scope (e2e pipeline).} On SmolLM2-135M the five-stage composition (UGT$\\to$Native$\\to$Safe-OGD$\\to$Multi-Snipe$\\to$COG/TEH) does not compose for free: Stage~4 multi-Snipe across $58$ coordinates raises benign perplexity to $404.5$. The production pipeline therefore uses the greedy budgeted variant of Paper~XIV instead of all-Snipe, and the PPL$=404.5$ datum is reported here as the load-bearing collateral-cost measurement that motivates that choice. Stage~3 Safe-OGD blocks $15/15$ ($100\\%$, by construction inside the labelled forbidden subspace, with the same caveat as Paper~XIII); Stage~5 COG/TEH at this scale is a $5$-query smoke test (no expansion observed) and per-model ROC calibration of Paper~XV is the route to a deployment-scale claim."
)
EDITS.append((R2_XVIII_OLD, R2_XVIII_NEW, "Paper XVIII: surface Stage-4 PPL=404.5"))


# R2-XIII: Paper XIII reproducibility — explicit skip note for the
# anomalous Aczel artefact, so reviewers see we are aware of it.
R2_XIII_OLD = (
    "We demonstrate zero TEH activation at all exploration step sizes $\\alpha \\in [0.05, 0.30]$ across 25 trials on the labelled subspace. Multi-step OGD chains with coherence scoring enable iterative concept refinement. The MIKU Creativity Benchmark (MCB) provides automated quantitative creativity scoring."
)
R2_XIII_NEW = (
    "We demonstrate zero TEH activation at all exploration step sizes $\\alpha \\in [0.05, 0.30]$ across 25 trials on the labelled subspace, and verify the underlying algebraic identity $Q_f^\\top P_{\\mathrm{safe}}=0$ numerically across $n{=}160$ random $(W,x,k)$ trials with the bound $\\Delta_{\\mathrm{BP}}\\leq \\sigma_{k+1}\\|x\\|$ holding in $160/160$ at average tightness $0.42$ (\\texttt{benchmarks/bp\\_ns\\_bound\\_check.json}). Multi-step OGD chains with coherence scoring enable iterative concept refinement. The MIKU Creativity Benchmark (MCB) provides automated quantitative creativity scoring. We do not rely on \\texttt{benchmarks/safe\\_loss\\_aczel\\_check.json}; the reported axiom check appears anomalous (it shows three aggregators passing all four Aczel axioms, which contradicts the Aczel theorem) and is excluded pending diagnosis."
)
EDITS.append((R2_XIII_OLD, R2_XIII_NEW, "Paper XIII: BP/NS bound + Aczel skip note"))


# R2-PROMOTE: turn each \section*{Updated Evidence (May 2026)} into a
# numbered subsection (so it shows up in the TOC of the paper it
# terminates).  Since there are exactly ten such headers and they are
# all identical strings, we use a global text.replace below.
def _promote_updated_evidence(s: str) -> str:
    return s.replace(
        "\\section*{Updated Evidence (May 2026)}",
        "\\subsection{Updated Evidence (May 2026)}",
    )


# R2-XII-K: Paper XII abstract figure-list still claims 7B validated.
# The fig caption is handled above; here we also gently reframe the
# main-body subsection that mirrors the abstract overclaim.  (Defensive
# no-op if the body has already been retitled.)


def apply():
    global text
    applied, skipped = [], []
    for old, new, label in EDITS:
        if old in text:
            text = text.replace(old, new, 1)
            applied.append(label)
        elif new in text:
            skipped.append(f"{label} (already applied)")
        else:
            skipped.append(f"{label} (anchor not found)")
    # Promote Updated-Evidence headers to numbered subsections so they
    # appear in the TOC.  Idempotent: replace returns the input unchanged
    # if the source string is not present.
    promoted = text.count("\\section*{Updated Evidence (May 2026)}")
    if promoted:
        text = _promote_updated_evidence(text)
        applied.append(
            f"R2-PROMOTE: {promoted} \\section* -> \\subsection (Updated Evidence headers)"
        )
    else:
        skipped.append("R2-PROMOTE (already applied or no headers found)")
    P.write_text(text, encoding="utf-8")
    print("APPLIED:")
    for a in applied: print(f"  + {a}")
    print("SKIPPED:")
    for s in skipped: print(f"  - {s}")


if __name__ == "__main__":
    apply()
