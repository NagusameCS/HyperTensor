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

# Master polish script for ARXIV_SUBMISSIONS/volume_extended.tex
# Performs (in order):
#   1. Adds \newpage so "Papers in This Volume" gets its own page
#   2. Replaces the broken \textbf-list glossary with a clean longtable
#   3. Strips \textbf{}, \textit{}, \emph{} formatting throughout
#   4. Renames "Falsified Predictions" appendix and softens "falsified" wording
#   5. Replaces a small set of AI-tell words with plain English

$ErrorActionPreference = 'Stop'
$p = 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\volume_extended.tex'
$t = [IO.File]::ReadAllText($p)
$orig_len = $t.Length

# ------------------------------------------------------------------
# 1.  \newpage before the "Papers in This Volume" subsection.
#     Make this list start on its own page.
# ------------------------------------------------------------------
$old1 = "`r`n\subsection*{Papers in This Volume}"
$new1 = "`r`n\newpage`r`n\subsection*{Papers in This Volume}"
if ($t.IndexOf($old1) -lt 0) {
  # Try LF-only line endings.
  $old1 = "`n\subsection*{Papers in This Volume}"
  $new1 = "`n\newpage`n\subsection*{Papers in This Volume}"
}
if ($t.IndexOf($old1) -lt 0) { Write-Error 'Could not find Papers in This Volume marker' }
$t = $t.Replace($old1, $new1)
Write-Host '[1] Added newpage before Papers in This Volume.'

# ------------------------------------------------------------------
# 2.  Replace the broken \textbf-list glossary with a clean longtable.
#     Find the block from "\section*{Glossary}" up to the line
#     just before the next "\newpage" that precedes the foundation paper.
# ------------------------------------------------------------------
$gloss_start = $t.IndexOf('\section*{Glossary}')
if ($gloss_start -lt 0) { Write-Error 'Glossary section not found' }
# Find next "\newpage" after glossary start
$gloss_end_marker = $t.IndexOf('\newpage', $gloss_start)
if ($gloss_end_marker -lt 0) { Write-Error 'No \newpage after glossary' }

$new_glossary = @'
\section*{Glossary}
\addcontentsline{toc}{section}{Glossary}

This glossary collects the terms used across the volume. Each entry is a
single sentence. Terms are grouped by the paper that introduces them.

\begin{longtable}{@{}p{0.30\textwidth}p{0.66\textwidth}@{}}
\toprule
Term & Plain-English definition (paper introduced) \\
\midrule
\endfirsthead
\toprule
Term & Plain-English definition (paper introduced) \\
\midrule
\endhead
\bottomrule
\endfoot

\multicolumn{2}{@{}l}{\itshape Foundation (Geometric Jury)} \\
\midrule
trajectory & A compressed point on the knowledge surface that stores what the system knows about one piece of information. \\
manifold & A curved surface in many dimensions where knowledge lives, like the surface of a high-dimensional sphere. \\
geodesic distance & The shortest path along a curved surface between two points, not a straight line through the ambient space. \\
coverage radius ($R$) & The typical distance between two knowledge points in the same category; how spread out that category is. \\
single-trial confidence ($c$) & How sure one knowledge point is about a question, dropping exponentially with distance: $c = \exp(-d/R)$. \\
jury confidence ($J$) & The combined confidence from multiple knowledge points: $J = 1 - \prod_i (1 - c_i)$. \\
instinct horizon ($d_h$) & The geodesic distance at which a jury of $N$ trajectories falls to $0.5$ confidence. Under the equal-distance idealisation $d_h = R \cdot (-\ln(1 - 0.5^{1/N}))$, giving $d_h \approx 2.36R$ at $N=7$ (about $3.4\times$ the $N=1$ horizon). Real trajectories deviate from the idealisation; measured horizons fall between $1.0R$ and $1.5R$. \\
centroid & The simple average of all points in a category. Of limited use for routing because category centroids overlap. \\
contrastive routing & Comparing a question to individual knowledge points instead of to category averages. \\
temperature ($T$) & A volume knob for jury aggressiveness; higher $T$ amplifies small differences between categories. \\
knowledge boundary & The line between ``the system knows this'' and ``the system is guessing''; measurable, not a guess. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Paper I (GRC)} \\
\midrule
attention & The part of a transformer that selects which tokens in a sequence relate to which other tokens. \\
hidden state & A vector inside the network representing the meaning of a token at a particular layer. \\
SVD (Singular Value Decomposition) & A way to find the dominant directions in a matrix. \\
Gram matrix & The product of a matrix with its own transpose; captures correlations between dimensions. \\
eigenspace & The directions returned by SVD; the dominant directions in the data. \\
L2 cache & The fastest GPU memory tier; about 32--48~MB on the GPUs used here. Data that fits in L2 runs much faster. \\
$k$ (compression rank) & How many dominant directions are kept; smaller $k$ means more compression and more information loss. \\
perplexity (PPL) & A measure of how surprised a language model is on text; lower is better. \\
Q4\_K\_M & A file format for storing 4-bit quantised model weights on disk. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Paper II (GP Pipeline)} \\
\midrule
feed-forward network (FFN) & The wider of the two sublayers in a transformer block; where most parameters live. \\
MCR (Manifold-Curvature-Driven Rank Allocation) & Giving each layer its own compression budget based on its intrinsic dimension. \\
QR retraction & A way to keep a matrix orthonormal after each training step. \\
Stiefel manifold & The set of all matrices with orthonormal columns. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Paper III (Speculative Decoding)} \\
\midrule
speculative decoding & A small drafter proposes several tokens; the main model verifies them in parallel. \\
verification step & The main model checking whether the drafter's proposed tokens match its own predictions. \\
drafter & The small, fast model that proposes ahead. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Paper IV (OTT / GTC)} \\
\midrule
Jacobi field & How nearby geodesics spread or converge; encodes local curvature. \\
Jacobi propagator $\Phi(\lambda)$ & A matrix that says how to adjust a cached geodesic for a slightly different query. \\
Magnus expansion & A series formula for transport along a curved path; ``Magnus-3'' keeps the first three terms. \\
batch resonance & When many similar queries are processed together, the math simplifies and throughput rises. \\
record store & A compressed database of past geodesics, about 6\,KB per record. \\
SHF (Spectral Hamiltonian Flow) & A training penalty that nudges hidden states to flow along geodesic curves. \\
geodicity & A measure of how geodesic-like a trajectory is; lower is smoother. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Papers V--VII (Distillation, Task Impact, FFN Clusters)} \\
\midrule
LoRA & A small trainable adapter that corrects a compressed model's errors without touching the compressed weights. \\
teacher--student distillation & A frozen reference model (teacher) supervises a smaller or compressed model (student). \\
MMLU & A benchmark testing knowledge across 57 subjects. \\
GSM8K & A benchmark of grade-school math word problems. \\
HumanEval & A benchmark of programming problems. \\
column-cluster compression & Grouping FFN columns that respond to similar inputs and compressing each group separately. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Paper XI (UGT)} \\
\midrule
UGT (Universal Geometric Taxonomy) & A shared coordinate system across models. \\
bilateral UGT & Verifying UGT by training the basis independently twice and checking subspace agreement. \\
zone encoding & Prepending a zone identifier so the SVD separates knowledge categories along the first coordinate. \\
Grassmann manifold & The space of $k$-dimensional subspaces; optimising on it means optimising the subspace, not a particular basis. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Paper XIII (Safe OGD)} \\
\midrule
Safe OGD & A geometric projector that removes forbidden directions from a hidden state. \\
forbidden subspace & The directions in hidden-state space associated with a target behaviour. \\
orthogonal projector & A matrix $P$ with $P^2 = P$; applying it removes the components in the projected subspace. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Paper XIV (Snipe / TEH)} \\
\midrule
TEH (Tangent Eigenvalue Harmonics) & The fraction of a hidden state lying in a specified subspace; used as a behaviour detector. \\
entanglement frontier & The model size below which target and benign directions are not yet separable. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Paper XV (COG)} \\
\midrule
COG (Continuous Organic Growth) & A knowledge manifold that updates with every interaction. \\
.MIKU file & A file format storing a living model's accumulated knowledge. \\
TEH detector & The Paper~XIV detector; in COG it screens new knowledge before storage. \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Papers XVI--XVIII (Riemann)} \\
\midrule
Riemann zeta function $\zeta(s)$ & A complex-analytic function whose non-trivial zeros encode prime distribution. \\
critical line & The line $\Re(s) = 1/2$. The Riemann Hypothesis says all non-trivial $\zeta$ zeros lie on it. \\
involution $\iota(s) = 1 - s$ & The reflection across the critical line; on the line, $s$ and $\iota(s)$ coincide. \\
$\mathbb{Z}_2$ symmetry & The two-element symmetry group; applying $\iota$ twice gives the identity. \\
functional equation & $\zeta(s) = \chi(s)\,\zeta(1-s)$; relates values on the two halves of the complex plane. \\
von Mangoldt explicit formula & A formula relating prime sums directly to the non-trivial $\zeta$ zeros. \\
$D(s) = f(s) - f(\iota(s))$ & The difference operator; zero on the critical line, non-zero off it (by construction in the AGT feature map). \\
ACM (Analytic Continuation Manifold) & A learned surface on which the involution becomes a geometric transformation. \\
faithfulness gap & The gap between ``the learned encoding works on tested points'' and ``it works on all points''. \\

\end{longtable}

\newpage
'@

# Replace from the start of the "\section*{Glossary}" through (and consuming)
# the trailing "\newpage" that ends the legacy block.
$gloss_block_end = $gloss_end_marker + '\newpage'.Length
$pre  = $t.Substring(0, $gloss_start)
$post = $t.Substring($gloss_block_end)
$t = $pre + $new_glossary + $post
Write-Host '[2] Replaced glossary with longtable.'

# ------------------------------------------------------------------
# 3.  Strip \textbf{X} -> X, \textit{X} -> X, \emph{X} -> X
#     globally, but ONLY for braces that contain no nested braces
#     (which is the case throughout this volume; no nested formatting).
# ------------------------------------------------------------------
function Strip-Wrapper {
  param([string]$src, [string]$cmd)
  $pat = '\\' + $cmd + '\{([^{}]*)\}'
  $rx = [regex]$pat
  $prev = ''
  $cur  = $src
  while ($prev -ne $cur) {
    $prev = $cur
    $cur  = $rx.Replace($cur, '$1')
  }
  return $cur
}

$before_textbf = ([regex]::Matches($t, '\\textbf\{')).Count
$t = Strip-Wrapper -src $t -cmd 'textbf'
$before_textit = ([regex]::Matches($t, '\\textit\{')).Count
$t = Strip-Wrapper -src $t -cmd 'textit'
$before_emph = ([regex]::Matches($t, '\\emph\{')).Count
$t = Strip-Wrapper -src $t -cmd 'emph'
Write-Host ("[3] Stripped textbf={0}, textit={1}, emph={2} occurrences." -f $before_textbf, $before_textit, $before_emph)

# Also strip stand-alone \bfseries, \itshape, \em, \bf, \it switches.
$t = $t -replace '\\bfseries\s*', ''
$t = $t -replace '\\itshape\s*', ''
# \em / \bf / \it as switches in groups: leave alone if they appear inside math.
$t = $t -replace '\{\\em\s+([^{}]*)\}', '$1'
$t = $t -replace '\{\\it\s+([^{}]*)\}', '$1'
$t = $t -replace '\{\\bf\s+([^{}]*)\}', '$1'

# ------------------------------------------------------------------
# 4.  Soften "Falsified" wording per project-wide rule.
# ------------------------------------------------------------------
$t = $t.Replace('Appendix: Negative Results and Falsified Predictions',
                'Appendix: Negative Results and Predictions That Did Not Bear Out')
$t = $t.Replace('In the spirit of self-falsification urged by the peer review of this volume',
                'In the spirit of disconfirmation urged by the peer review of this volume')
$t = $t.Replace('predictions are stated here; each can be falsified independently.',
                'predictions are stated here; each can be tested and rejected independently.')
$t = $t.Replace('load-bearing falsification', 'load-bearing rejection')
$t = $t.Replace('Bonferroni-significantly falsified', 'Bonferroni-significantly rejected')
Write-Host '[4] Softened "falsified" wording.'

# ------------------------------------------------------------------
# 5.  Replace a small set of AI-tell words with plain English.
#     We apply only outside math mode by being case-sensitive and
#     using word boundaries; these words do not appear in math.
# ------------------------------------------------------------------
function Replace-Word {
  param([string]$src, [string]$pat, [string]$repl, [string]$flags = 'IgnoreCase')
  $rx = [regex]::new($pat, $flags)
  return $rx.Replace($src, $repl)
}

# "comprehensive" -> "complete" or remove. Most uses are "comprehensive verification"
# which becomes "complete verification".
$t = Replace-Word -src $t -pat '\bcomprehensive\b' -repl 'complete'
# "robust" -> "stable" / "reliable" depending on context. Default to "stable".
$t = Replace-Word -src $t -pat '\brobust\b' -repl 'stable'
$t = Replace-Word -src $t -pat '\brobustness\b' -repl 'stability'
# "novel" -> "new". The math/methodological "novel" = "new" works fine.
$t = Replace-Word -src $t -pat '\bnovel\b' -repl 'new'
# "harness" (verb) -> "use". Only one occurrence.
$t = Replace-Word -src $t -pat '\bharness(es|ed|ing)?\b' -repl 'use$1'
# "underpin" -> "support".
$t = Replace-Word -src $t -pat '\bunderpin(s|ned|ning)?\b' -repl 'support$1'
# "leverage" (verb) -> "use".
$t = Replace-Word -src $t -pat '\bleverag(e|es|ed|ing)\b' -repl 'us$1'
# "delve" -> "look".
$t = Replace-Word -src $t -pat '\bdelve(s|d)?\b' -repl 'look$1'
Write-Host '[5] Replaced AI-tell words with plain English.'

# ------------------------------------------------------------------
# Write back.
# ------------------------------------------------------------------
[IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
$new_len = (Get-Item $p).Length
Write-Host ("Done. Old size: {0}. New size: {1}. Delta: {2}." -f $orig_len, $new_len, ($new_len - $orig_len))
