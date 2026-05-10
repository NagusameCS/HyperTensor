$b = [IO.File]::ReadAllText('c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\refs.bib')
$arxiv = [regex]::Matches($b,'(?im)eprint\s*=\s*[\{"]([^\}",]+)[\}"]') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
$doi   = [regex]::Matches($b,'(?im)doi\s*=\s*[\{"]([^\}",]+)[\}"]')    | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
$url   = [regex]::Matches($b,'(?im)url\s*=\s*[\{"]([^\}"]+)[\}"]')     | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

# Also pull bare arXiv:NNNN.NNNNN from note/howpublished fields and inline bibitems
$tex = [IO.File]::ReadAllText('c:\Users\legom\HyperTensor\ARXIV_SUBMISSIONS\volume_extended.tex')
$arxivBare = [regex]::Matches($b + "`n" + $tex, '(?i)arXiv[:\s]+(\d{4}\.\d{4,5})') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

$allArxiv = ($arxiv + $arxivBare) | Sort-Object -Unique

Write-Host ("=== arXiv IDs (" + $allArxiv.Count + ") ===")
$allArxiv | ForEach-Object { Write-Host ("arXiv:" + $_) }
Write-Host ""
Write-Host ("=== DOIs (" + $doi.Count + ") ===")
$doi | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host ("=== URLs (" + $url.Count + ") ===")
$url | ForEach-Object { Write-Host $_ }
