$p = (Resolve-Path 'ARXIV_SUBMISSIONS/volume_extended.tex').Path
$t = [IO.File]::ReadAllText($p)
$orig = $t.Length

$nl = "`r`n"

$t = $t.Replace("The throughput model of$nl\Cref{eq:spec} predicts", "The closed-form throughput model predicts")
$t = $t.Replace("The throughput model of`n\Cref{eq:spec} predicts", "The closed-form throughput model predicts")
$t = $t.Replace("\Cref{eq:spec} is the peak", "The closed-form throughput model peaks")
$t = $t.Replace("\Cref{sec:why})", "discussed below)")
$t = $t.Replace("Under the rejection rule of \Cref{eq:accept}", "Under the standard speculative-decoding rejection rule")
$t = $t.Replace('$T_V(k)$ in \Cref{eq:spec}.}', '$T_V(k)$ in the closed-form throughput model.}')
$t = $t.Replace("templates (Sec.~\ref{sec:formal}),", "templates (the formal addendum below),")
$t = $t.Replace("Eq.~\eqref{eq:gtc-correction} is then a first-order Taylor expansion", "The cached-propagator correction equation is a first-order Taylor expansion")
$t = $t.Replace("the empirical finding of$nl\Cref{sec:formal} (Empirical feasibility stub):", "the formal addendum's empirical feasibility stub:")
$t = $t.Replace("the empirical finding of`n\Cref{sec:formal} (Empirical feasibility stub):", "the formal addendum's empirical feasibility stub:")
$t = $t.Replace("the joint-training premise of$nl\Cref{sec:formal},", "the joint-training premise of the formal addendum,")
$t = $t.Replace("the joint-training premise of`n\Cref{sec:formal},", "the joint-training premise of the formal addendum,")
$t = $t.Replace("(\Cref{tab:ugt-concept-capture};", "(see UGT zone-recovery artefact below;")

[IO.File]::WriteAllText($p, $t)
"Bytes: $orig -> $($t.Length)"
$remaining = (Select-String -Path $p -Pattern '\\(Cref|cref|ref|eqref)\{(eq:accept|eq:gtc-correction|eq:spec|sec:formal|sec:why|tab:ugt-concept-capture)\}' | Measure-Object).Count
"Remaining broken refs: $remaining"
