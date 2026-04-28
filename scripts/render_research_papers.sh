#!/usr/bin/env bash
# scripts/render_research_papers.sh
# Render the four LaTeX research papers in ARXIV_SUBMISSIONS/ to standalone
# HTML pages under docs/research/, using the same visual language as the
# engineering papers in docs/papers/.
#
# Requires: pandoc (>= 3), the ARXIV_SUBMISSIONS/ tree compiled enough that
# .bbl files exist (so citations can be resolved).  Math is delegated to
# MathJax in the rendered page; we ask pandoc for raw HTML+TeX maths.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT="$ROOT/ARXIV_SUBMISSIONS"
OUT="$ROOT/docs/research"
TPL="$ROOT/docs/assets/research-paper.template.html"

mkdir -p "$OUT" "$OUT/figures"

# Tag, file stem, friendly title for each paper.
render_one () {
  local letter="$1" stem="$2" tag="$3" title="$4"
  local src="$SUBMIT/paper-$letter/$stem.tex"
  local bib="$SUBMIT/paper-$letter/refs.bib"
  local figdir="$SUBMIT/paper-$letter/figures"
  local outfile="$OUT/paper-$(echo "$letter" | tr '[:upper:]' '[:lower:]')-${stem}.html"

  echo "=== Paper $letter ==="
  # Mirror figures (PDFs become unusable in browsers; convert if needed).
  if [ -d "$figdir" ]; then
    mkdir -p "$OUT/figures/paper-$letter"
    cp "$figdir"/*.pdf "$OUT/figures/paper-$letter/" 2>/dev/null || true
    # Convert PDFs to PNG for browser display when available.
    if command -v sips >/dev/null 2>&1; then
      for pdf in "$figdir"/*.pdf; do
        [ -e "$pdf" ] || continue
        local base
        base="$(basename "$pdf" .pdf)"
        sips -s format png "$pdf" --out "$OUT/figures/paper-$letter/$base.png" >/dev/null 2>&1 || true
      done
    fi
  fi

  pandoc "$src" \
    --from=latex \
    --to=html5 \
    --mathjax \
    --standalone \
    --template="$TPL" \
    --metadata=title:"$title" \
    --metadata=tag:"$tag" \
    --metadata=author:"HyperTensor Project (William Ken Ohara Stewart)" \
    --metadata=date:"April 2026" \
    --metadata=arxiv-pdf:"../../ARXIV_SUBMISSIONS/paper-$letter/$stem.pdf" \
    --metadata=tex-source:"../../ARXIV_SUBMISSIONS/paper-$letter/$stem.tex" \
    --metadata=lang:en \
    --citeproc --bibliography="$bib" \
    --resource-path=".:$SUBMIT/paper-$letter:$SUBMIT/paper-$letter/figures" \
    --toc --toc-depth=2 \
    --section-divs \
    --shift-heading-level-by=1 \
    --wrap=preserve \
    -o "$outfile" \
    2> "$OUT/.paper-$letter.log" || {
      echo "  pandoc errors (see $OUT/.paper-$letter.log):"
      tail -20 "$OUT/.paper-$letter.log" | sed 's/^/    /'
    }

  # Rewrite figure references so PDFs load as PNGs and from the local dir.
  if [ -f "$outfile" ]; then
    sed -i '' -E "s|figures/([a-zA-Z0-9_-]+)\\.pdf|figures/paper-$letter/\\1.png|g" "$outfile"
    echo "  -> $outfile"
  fi
}

render_one A grc-attention-compression          "Paper A · GRC"      "Geodesic Runtime Compression: a calibration-free, super-baseline attention compression"
render_one B geodesic-projection-pipeline       "Paper B · GP"       "Geodesic Projection: per-layer rank, MCR allocation, and the depth-sink shortcut"
render_one C geodesic-speculative-decoding      "Paper C · OTT-Decode" "Geodesic Speculative Decoding: OTT-aware verifier with EOS-aware acceptance"
render_one D ott-gtc-manifold-runtime           "Paper D · OTT/GTC"  "Organic Training Theory and the GTC Manifold Runtime"

echo
echo "All research papers rendered to $OUT"
ls -1 "$OUT"/*.html
