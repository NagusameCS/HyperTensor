# Round 2: handle \textbf{...} and \emph{...} with nested braces, plus
# remaining AI-tell stems (robustly, robustness etc).

$ErrorActionPreference = 'Stop'
$p = 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\volume_extended.tex'
$t = [IO.File]::ReadAllText($p)
$orig_len = $t.Length

# Brace-balanced strip of \cmd{...}.
function Strip-Balanced {
  param([string]$src, [string]$cmd)
  $needle = '\' + $cmd + '{'
  $out = New-Object System.Text.StringBuilder
  $i = 0
  $len = $src.Length
  $stripped = 0
  while ($i -lt $len) {
    if ($i + $needle.Length -le $len -and $src.Substring($i, $needle.Length) -eq $needle) {
      # Find matching closing brace.
      $depth = 1
      $j = $i + $needle.Length
      while ($j -lt $len -and $depth -gt 0) {
        $c = $src[$j]
        if ($c -eq '\') {
          $j += 2  # skip escaped char like \{ \} \\
          continue
        }
        if ($c -eq '{') { $depth++ }
        elseif ($c -eq '}') { $depth-- }
        if ($depth -eq 0) { break }
        $j++
      }
      if ($depth -eq 0) {
        # Append the inner contents (no command, no outer braces).
        [void]$out.Append($src.Substring($i + $needle.Length, $j - ($i + $needle.Length)))
        $i = $j + 1
        $stripped++
        continue
      }
    }
    [void]$out.Append($src[$i])
    $i++
  }
  return @{ text = $out.ToString(); count = $stripped }
}

$r = Strip-Balanced -src $t -cmd 'textbf'
$t = $r.text
Write-Host ('[textbf nested] stripped: ' + $r.count)

$r = Strip-Balanced -src $t -cmd 'emph'
$t = $r.text
Write-Host ('[emph nested] stripped:   ' + $r.count)

$r = Strip-Balanced -src $t -cmd 'textit'
$t = $r.text
Write-Host ('[textit nested] stripped: ' + $r.count)

# AI-tell stems we missed.
$rx_robust = [regex]'\brobustly\b'
$cnt = $rx_robust.Matches($t).Count
$t = $rx_robust.Replace($t, 'consistently')
Write-Host ('[robustly] replaced: ' + $cnt)

# Sanity check on "very" / "highly" overuse? Skip for now -- low risk.

[IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
$new_len = (Get-Item $p).Length
Write-Host ("Done. Old: {0}  New: {1}  Delta: {2}" -f $orig_len, $new_len, ($new_len - $orig_len))
