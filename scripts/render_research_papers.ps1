# PowerShell port of scripts/render_research_papers.sh.
# Requires pandoc 3+ and a compiled .bbl in each paper directory.

$ErrorActionPreference = 'Continue'
$root   = (Resolve-Path "$PSScriptRoot\..").Path
$submit = Join-Path $root 'ARXIV_SUBMISSIONS'
$out    = Join-Path $root 'docs\research'
$tpl    = Join-Path $root 'docs\assets\research-paper.template.html'

# Ensure pandoc is reachable.
$pandoc = (Get-Command pandoc -ErrorAction SilentlyContinue).Source
if (-not $pandoc) {
    $candidate = "$env:USERPROFILE\AppData\Local\Pandoc\pandoc.exe"
    if (Test-Path $candidate) { $pandoc = $candidate }
}
if (-not $pandoc) { throw 'pandoc not found' }

New-Item -ItemType Directory -Force -Path $out, (Join-Path $out 'figures') | Out-Null

function Render-One {
    param(
        [string]$letter,
        [string]$stem,
        [string]$tag,
        [string]$title
    )
    $src     = Join-Path $submit "paper-$letter\$stem.tex"
    $bib     = Join-Path $submit "paper-$letter\refs.bib"
    $figdir  = Join-Path $submit "paper-$letter\figures"
    $lower   = $letter.ToLower()
    $outfile = Join-Path $out "paper-$lower-$stem.html"

    Write-Host "=== Paper $letter ==="

    if (Test-Path $figdir) {
        $figdest = Join-Path $out "figures\paper-$letter"
        New-Item -ItemType Directory -Force -Path $figdest | Out-Null
        Get-ChildItem -Path $figdir -Filter *.pdf -ErrorAction SilentlyContinue |
            Copy-Item -Destination $figdest -Force
    }

    $resourcePath = ".;$(Join-Path $submit "paper-$letter");$(Join-Path $submit "paper-$letter\figures");$(Join-Path $root 'docs\data');$root"

    # Pre-process: inline \input{...} so pandoc sees a self-contained .tex.
    # Pandoc resolves \input relative to a path it does not search; we expand
    # it by reading the included file and substituting its contents in place.
    $rawTex = Get-Content $src -Raw
    $expandedTex = $rawTex
    while ($true) {
        $m = [regex]::Match($expandedTex, '\\input\{([^}]+)\}')
        if (-not $m.Success) { break }
        $inc = $m.Groups[1].Value
        $candidates = @(
            (Join-Path (Split-Path $src) $inc),
            (Join-Path $root $inc),
            (Join-Path $root 'docs\data' (Split-Path -Leaf $inc))
        )
        $resolved = $null
        foreach ($cand in $candidates) {
            if (Test-Path $cand)        { $resolved = (Get-Content $cand        -Raw); break }
            if (Test-Path "$cand.tex")  { $resolved = (Get-Content "$cand.tex"  -Raw); break }
        }
        if ($null -eq $resolved) { $resolved = "% [unresolved input: $inc]" }
        $expandedTex = $expandedTex.Substring(0, $m.Index) + $resolved + $expandedTex.Substring($m.Index + $m.Length)
    }
    $tmpTex = Join-Path $env:TEMP "render_$letter`_$stem.tex"
    Set-Content -Path $tmpTex -Value $expandedTex -NoNewline -Encoding UTF8

    & $pandoc $tmpTex `
        --from=latex `
        --to=html5 `
        --mathjax `
        --standalone `
        --template=$tpl `
        --metadata=title:$title `
        --metadata=tag:$tag `
        --metadata="author:HyperTensor Project (William Ken Ohara Stewart)" `
        --metadata=date:"April 2026" `
        --metadata=arxiv-pdf:"/pdfs/$stem.pdf" `
        --metadata=tex-source:"https://github.com/NagusameCS/HyperTensor/blob/main/ARXIV_SUBMISSIONS/paper-$letter/$stem.tex" `
        --metadata=lang:en `
        --citeproc --bibliography=$bib `
        --resource-path=$resourcePath `
        --toc --toc-depth=2 `
        --section-divs `
        --shift-heading-level-by=1 `
        --wrap=preserve `
        -o $outfile 2>&1 | Out-Host

    if (Test-Path $outfile) {
        $c = Get-Content $outfile -Raw
        $c = [regex]::Replace($c, "figures/([a-zA-Z0-9_-]+)\.pdf", "figures/paper-$letter/`$1.png")
        Set-Content -Path $outfile -Value $c -NoNewline
        Write-Host "  -> $outfile"
    }
}

Render-One A grc-attention-compression          'Paper A - GRC'        'Geodesic Runtime Compression: a calibration-free, super-baseline attention compression'
Render-One B geodesic-projection-pipeline       'Paper B - GP'         'Geodesic Projection: per-layer rank, MCR allocation, and the depth-sink shortcut'
Render-One C geodesic-speculative-decoding      'Paper C - OTT-Decode' 'Geodesic Speculative Decoding: OTT-aware verifier with EOS-aware acceptance'
Render-One D ott-gtc-manifold-runtime           'Paper D - OTT/GTC'    'Organic Training Theory and the GTC Manifold Runtime'
Render-One E grc-light-distillation             'Paper E - Distill'    'Light Distillation for Calibration-Permitted Low-Rank Attention Compression'

Write-Host ''
Write-Host "All research papers rendered to $out"
Get-ChildItem -Path $out -Filter *.html | Select-Object Name
