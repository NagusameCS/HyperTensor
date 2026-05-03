param([string[]]$Papers = @("A","B","C","D","E"))

$root = $PSScriptRoot
$map = @{
    A = "grc-attention-compression"
    B = "geodesic-projection-pipeline"
    C = "geodesic-speculative-decoding"
    D = "ott-gtc-manifold-runtime"
    E = "grc-light-distillation"
}

foreach ($p in $Papers) {
    $name = $map[$p]
    $dir  = Join-Path $root "ARXIV_SUBMISSIONS\paper-$p"
    Write-Host "`n=== Paper $p : $name ===" -ForegroundColor Cyan
    Set-Location $dir

    # Pass 1
    $r1 = pdflatex -interaction=nonstopmode "$name.tex" 2>&1
    $fatal = $r1 | Select-String "Fatal error"
    if ($fatal) { Write-Host "  PASS1 FATAL: $fatal" -ForegroundColor Red }
    else { Write-Host "  pass1 ok" }

    # biber
    $rb = biber $name 2>&1
    $berr = $rb | Select-String "ERROR -"
    if ($berr) { Write-Host "  BIBER ERR: $berr" -ForegroundColor Yellow }
    else { Write-Host "  biber ok" }

    # Pass 2
    $r2 = pdflatex -interaction=nonstopmode "$name.tex" 2>&1
    # Pass 3
    $r3 = pdflatex -interaction=nonstopmode "$name.tex" 2>&1
    $out = $r3 | Select-String "Output written"
    if ($out) { Write-Host "  $out" -ForegroundColor Green }
    else { Write-Host "  PASS3 no output - checking..." -ForegroundColor Red; $r3 | Select-String "Fatal|Error" | Select-Object -First 5 }

    Set-Location $root
}

Write-Host "`nDone. PDFs:"
Get-ChildItem ARXIV_SUBMISSIONS -Recurse -Filter "*.pdf" | Select-Object @{N="Paper";E={$_.Directory.Name}}, Name, @{N="KB";E={[math]::Round($_.Length/1KB)}}
