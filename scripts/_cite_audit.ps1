$tex = [IO.File]::ReadAllText('c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\volume_extended.tex')
$cite = [regex]::Matches($tex, '\\cite\{').Count
$citep = [regex]::Matches($tex, '\\citep\{').Count
$autocite = [regex]::Matches($tex, '\\autocite\{').Count
$pb = [regex]::Matches($tex, '\\printbibliography').Count
$add = [regex]::Matches($tex, '\\addbibresource').Count
Write-Host "cite=$cite citep=$citep autocite=$autocite printbib=$pb addbib=$add"
# Distinct keys
$keys = [regex]::Matches($tex, '\\cite[a-z]*\*?(?:\[[^\]]*\])?\{([^}]+)\}') | ForEach-Object { $_.Groups[1].Value -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Sort-Object -Unique
Write-Host ('distinct cite keys: ' + $keys.Count)
$bibkeys = (Select-String -Path 'c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\refs.bib' -Pattern '^@\w+\{([^,]+),').Matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
Write-Host ('refs.bib entries:   ' + $bibkeys.Count)
$missing = $keys | Where-Object { $bibkeys -notcontains $_ }
Write-Host ('cited but NOT in refs.bib: ' + ($missing | Measure-Object).Count)
if ($missing) { $missing | ForEach-Object { Write-Host "  MISSING: $_" } }
$unused = $bibkeys | Where-Object { $keys -notcontains $_ }
Write-Host ('refs.bib entries NOT cited: ' + ($unused | Measure-Object).Count)
if ($unused.Count -gt 0 -and $unused.Count -le 30) { $unused | ForEach-Object { Write-Host "  unused: $_" } }
