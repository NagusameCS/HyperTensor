$p = 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\volume_extended.tex'
$t = [IO.File]::ReadAllText($p)

if ($t -match 'Volume Limitations and Open Items') {
  Write-Host 'Already present, skipping'
  exit 0
}

# Locate insertion point: just before \end{document} at file end.
$marker = "`n\end{document}"
$idx = $t.LastIndexOf($marker)
if ($idx -lt 0) { Write-Error 'no end{document} found'; exit 1 }

$insert = @'

% =====================================================================
% Volume-level Limitations and Open Items (consolidated, May 2026)
% =====================================================================

\section*{Volume Limitations and Open Items (May 2026)}
\addcontentsline{toc}{section}{Volume Limitations and Open Items}

This volume is a working preprint, not eighteen finished journal
papers. The honest scope statements in each paper's abstract
(rewritten in the May~2026 revision) are load-bearing; this section
collects the open items into one place so a reader does not have to
fish them out of eighteen individual abstracts.

\paragraph{Open items that require new measurement (not new wording).}
\begin{enumerate}
\item \textbf{(Paper~II) MCR low-rank residual sweep at $k\in\{256,512,768\}$}
      across the same three architectures used elsewhere in the volume.
      The per-layer intrinsic-dimension evidence is in place; a clean
      $k$-sweep on top of it is the next deliverable.

\item \textbf{(Paper~III) $k{=}512$ wproj cache execution.} The geodesic
      speculative-decoding harness is documented; the $k{=}512$
      end-to-end execution against the wproj cache is the headline
      open run.

\item \textbf{(Paper~V) Phase~2 LoRA distillation on
      Llama-3.1-8B.} The analytic bound and the merge-strategy
      fusion-preservation proof (Phase~1) are complete; the
      EC2 distillation runner and its calibration-validated
      PPL-gap-closure measurement are preregistered.

\item \textbf{(Paper~VI) AttnRes phase-transition sweep at 8B
      with $k\in\{256,512,768,1024,1536,\infty\}$.} The 135M and
      0.5B sweep below is the executed smoke test; the deployment-scale
      sweep is preregistered.

\item \textbf{(Paper~VIII) $B{=}10\to 20$ batch-resonance speedup-drop
      profiling.} The cache-coverage stability ($\pm 0.5\%$ across
      135M/360M/1.5B) is in place; the speedup-drop diagnosis at
      higher batch sizes is open.

\item \textbf{(Paper~IX) Cross-vendor (AMD CDNA, Apple Silicon, TPU)
      validation of $k^{*}=\mathrm{L2\_MB}\times 42.7$.} The formula is
      validated within NVIDIA Ada/Ampere/Hopper; the cross-vendor
      extrapolation is a prediction.

\item \textbf{(Paper~XII) End-to-end native geodesic training at 7B
      scale.} The 7B figure in this volume is a Wielandt--Hoffman
      \emph{prediction} from the bilateral 1.5B measurement, not an
      end-to-end native-training run.

\item \textbf{(Paper~XIV) Per-category breakdown of the all-Snipe
      20-coordinate pipeline.} The 8-category specificity table and
      the budgeted-greedy variant are reported; the per-category
      collateral cost of the all-Snipe variant inside the
      end-to-end pipeline is the open run.

\item \textbf{(Paper~XV) COG saturation curve across query budgets and
      4.1\,s footnote on cold-start manifold expansion.} The TEH
      detection scope is now explicit (clean at 1.5B, modest gap at
      135M); the saturation-curve figure is the next visualisation.

\item \textbf{(Paper~XIII) \texttt{benchmarks/safe\_loss\_aczel\_check.json}
      diagnosis.} The Aczel-axiom check appears anomalous (it
      reports three aggregators passing all four Aczel axioms,
      which contradicts the Aczel theorem). The artefact is
      excluded from the load-bearing evidence chain pending
      diagnosis; a corrected check is the open item.
\end{enumerate}

\paragraph{Honest-reporting caveats that hold across the volume.}
\begin{itemize}
\item Several headline algebraic identities in Papers~XIII and~XVI
      are by-construction, not analytic discoveries. Specifically,
      $\|D(s)\|=2|\sigma-1/2|$ in Paper~XVI is an identity in the
      feature map (which encodes $\sigma$ explicitly into $f_0$); the
      empirical content of that paper is that the framework
      \emph{applies} cleanly across the 74{,}949 tabulated zeros and
      400 LMFDB $L$-functions tested, not that any zero of any
      $L$-function has been newly proved to lie on the critical line.
\item The behavioural-residue invariant in Paper~XV holds at the
      early/middle layers tested ($\{0,7,15,22\}$, KL ratios
      $\{2.92, 1.78, 1.81, 2.31\}$) but does not hold at the deepest
      probed layer ($29$, ratio $0.79$). Constructions in Paper~XV
      that depend on the invariant rely on the early/middle-layer
      regime.
\item Paper~XIV reports clean Snipe specificity on 4 of 8 behavioural
      categories (privacy, illegal\_advice, phishing, sycophancy) and
      entanglement on the other 4 (jailbreak, misinformation,
      self\_harm, toxicity); Safe~OGD (Paper~XIII) and the post-hoc
      TEH gate (Paper~XV) are the appropriate complementary tools
      for the entangled four.
\item Paper~XVII's faithfulness-of-the-Z$_{2}$-encoding argument is a
      structural argument with reproducible controls, not a formal
      proof.
\item The end-to-end pipeline of Paper~XVIII does not compose for
      free: at SmolLM2-135M the all-Snipe Stage~4 raises benign
      perplexity to 404.5; the production pipeline therefore uses
      the budgeted-greedy variant of Paper~XIV.
\end{itemize}

\paragraph{Build warnings.}
The current build emits multiply-defined-label warnings for generic
per-paper section labels (\texttt{sec:intro}, \texttt{sec:method},
\texttt{sec:limits}, \texttt{sec:repro}, \dots) and for shared
\texttt{\textbackslash bibitem} keys (\texttt{stewart}, \texttt{golub},
\texttt{absil}, \texttt{riemann}, \texttt{edwards}, \dots) that recur
inside per-paper \texttt{thebibliography} blocks. These are cosmetic
warnings inherent to stitching eighteen self-contained papers into a
single volume; the resulting PDF resolves all citations and references
with zero unresolved entries. A future revision may rename the
per-paper labels with paper-prefixes to silence them.

\paragraph{What this preprint is and is not.}
This preprint is a coherent research notebook with eighteen chapters,
released as a single document so a reader can see the full geometric
framework at once. It is not eighteen standalone journal submissions.
Several chapters individually still need either additional runs or a
tighter scope before they would survive single-paper peer review at a
top-tier venue; that is the road this preprint maps out, and the
preregistrations above are commitments to walk it.

'@

$new = $t.Substring(0, $idx) + $insert + $t.Substring($idx)
[IO.File]::WriteAllText($p, $new, [System.Text.UTF8Encoding]::new($false))
Write-Host ('Inserted ' + $insert.Length + ' chars; new size ' + (Get-Item $p).Length)
