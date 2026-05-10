$p = 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\volume_extended.tex'
$t = [IO.File]::ReadAllText($p)
$old = "`r`n\section*{Glossary}"
$new = "`r`n\newpage`r`n\section*{Glossary}"
if ($t.IndexOf($old) -lt 0) { $old = "`n\section*{Glossary}"; $new = "`n\newpage`n\section*{Glossary}" }
if ($t.IndexOf($old) -lt 0) { Write-Error 'Glossary marker not found' }
# Avoid double-newpage if one already precedes.
$idx = $t.IndexOf($old)
$prev200 = $t.Substring([Math]::Max(0,$idx-200), [Math]::Min(200,$idx))
if ($prev200 -match '\\newpage\s*$') { Write-Host 'Already has newpage before Glossary; skipping.' }
else {
  $t = $t.Substring(0,$idx) + $new + $t.Substring($idx + $old.Length)
  [IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
  Write-Host 'Added newpage before Glossary.'
}
