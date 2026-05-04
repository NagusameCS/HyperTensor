#!/usr/bin/env python3
"""Build unified whitepaper by concatenating all 12 engineering paper HTML bodies."""
import os, re

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'papers')
papers = [
    ('00-introduction.html', 'Paper 0: Introduction --- Tensors, Attention, and Transformer Geometry'),
    ('01-attention-compression.html', 'Paper 1: GRC --- Geodesic Runtime Compression (106.27% throughput)'),
    ('02-geodesic-projection.html', 'Paper 2: GP --- Geodesic Projection Pipeline (Full Multi-Slot)'),
    ('03-speculative-decoding.html', 'Paper 3: GSD --- Geodesic Speculative Decoding (38.5% acceptance)'),
    ('04-organic-training-theory.html', 'Paper 4: OTT --- Organic Training Theory (Riemannian Framework)'),
    ('05-gtc-ott-runtime.html', 'Paper 5: GTC + OTT Runtime Anchor (97x batched-Jacobi gain)'),
    ('06-adaptive-compression.html', 'Paper 6: Adaptive Layer --- Phase-Aware, Thermal-Coupled, Online'),
    ('11-ugt-taxonomy.html', 'Paper XI: UGT --- Universal Geodesic Taxonomy (Bilateral 0.9999 overlap)'),
    ('12-native-training.html', 'Paper XII: Native Geodesic Training (NativeLinear, KExpansion)'),
    ('13-safe-ogd.html', 'Paper XIII: Safe OGD --- Orthogonal Geodesic Deviation (0% TEH)'),
    ('14-behavioral-snipe.html', 'Paper XIV: Snipe --- Behavioral Geodesic Sniping (<2% collateral)'),
    ('15-cog-teh.html', 'Paper XV: COG+TEH --- Completely Organic Generation + TEH Detection'),
]

# Extract CSS from 01 paper
with open(os.path.join(root, '01-attention-compression.html'), 'r', encoding='utf-8') as f:
    c01 = f.read()

css_start = c01.find('<style>')
css_end = c01.find('</style>') + len('</style>')
css = c01[css_start:css_end]

# Extract paper body from each
bodies = []
for fname, title in papers:
    fpath = os.path.join(root, fname)
    if not os.path.exists(fpath):
        print(f'MISSING: {fname}')
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()
    
    body = None
    
    # Strategy 1: <article class="paper-body"> (Papers XI-XV)
    m = re.search(r'<article class="paper-body">(.*?)</article>', c, re.DOTALL)
    if m:
        body = m.group(1)
    
    # Strategy 2: <header id="hero"> to </body> (Papers 1-6)
    if not body:
        hero_start = c.find('<header id="hero">')
        if hero_start < 0:
            hero_start = c.find('<div id="hero">')
        body_end = c.rfind('</body>')
        if body_end < 0:
            body_end = c.rfind('</html>')
        if hero_start > 0 and body_end > hero_start:
            # Extract from hero to end, strip the closing body/html
            body = c[hero_start:body_end]
            # Remove closing tags we'll add back
            body = re.sub(r'</body>\s*</html>\s*$', '', body)
    
    # Strategy 3: <main> or first <section> to </body> (Paper 0)
    if not body:
        for tag in ['<main>', '<section']:
            tag_start = c.find(tag)
            if tag_start > 0:
                body_end = c.rfind('</body>')
                if body_end < 0:
                    body_end = len(c)
                body = c[tag_start:body_end]
                break
    
    if not body:
        print(f'NO BODY: {fname}')
        continue
    
    bodies.append((title, body))
    print(f'OK: {fname} ({len(body):,} chars)')

# Build TOC
toc_items = '\n'.join(f'      <li><a href="#paper{i}">{title}</a></li>' for i, (title, _) in enumerate(bodies))

# Build paper sections
sections = ''
for i, (title, body) in enumerate(bodies):
    sections += f'    <hr class="paper-divider" data-title="{title}" id="paper{i}">\n'
    sections += f'    <article class="paper-body">\n{body}\n    </article>\n\n'

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HyperTensor Unified Engineering Whitepaper --- All 12 Papers</title>
<link rel="canonical" href="https://nagusamecs.github.io/HyperTensor/papers/whitepaper.html">
<meta name="robots" content="index, follow, max-image-preview:large">
<meta name="GPTBot" content="noindex, nofollow">
<meta name="ClaudeBot" content="noindex, nofollow">
<meta name="CCBot" content="noindex, nofollow">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&amp;family=IBM+Plex+Mono:wght@400;500&amp;display=swap" rel="stylesheet">
<script>MathJax = {{ tex: {{ inlineMath: [['$','$'],['\\(','\\)']], displayMath: [['$$','$$']], tags: 'ams' }}, svg: {{ fontCache: 'global' }} }};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<link rel="stylesheet" href="../assets/reader.css">
<link rel="stylesheet" href="../assets/paper.css">
<script src="../assets/reader.js"></script>
<link rel="stylesheet" href="../assets/paper-style.css">
{css}
<style>
  body {{ max-width: 820px; margin: 0 auto; }}
  .paper-divider {{ border: none; border-top: 3px solid #c0392b; margin: 3rem 0 2rem; position: relative; }}
  .paper-divider::after {{ content: attr(data-title); position: absolute; top: -0.85rem; left: 50%; transform: translateX(-50%); background: #fff; padding: 0 1.5rem; font-weight: 700; font-size: 1.1rem; color: #c0392b; white-space: nowrap; max-width: 90vw; overflow: hidden; text-overflow: ellipsis; }}
  nav {{ position: sticky; top: 0; z-index: 100; background: rgba(255,255,255,0.95); backdrop-filter: blur(12px); border-bottom: 1px solid #e5e5e5; padding: 0.6rem 1.5rem; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }}
  nav a {{ font-size: 0.8rem; font-weight: 500; color: #5c5c5c; text-decoration: none; }}
  nav a:hover {{ color: #1a1a1a; }}
  nav .brand {{ font-weight: 700; color: #1a1a1a; }}
  #toc {{ max-width: 820px; margin: 2rem auto 0; padding: 0 1.5rem; }}
  #toc details {{ margin-bottom: 1rem; }}
  #toc summary {{ font-weight: 600; cursor: pointer; font-size: 1.1rem; }}
  #toc ol {{ columns: 2; column-gap: 2rem; }}
  #toc li {{ margin: 0.2rem 0; font-size: 0.9rem; }}
</style>
</head>
<body>

<nav>
  <a class="brand" href="../index.html">NagusameCS</a>
  <a href="../index.html">Home</a>
  <a href="../index.html#research">Research</a>
  <a href="https://github.com/NagusameCS/HyperTensor">GitHub</a>
</nav>

<div id="toc">
<details open>
<summary>Table of Contents --- 12 Engineering Papers (Papers 0--6 + XI--XV)</summary>
<ol>
{toc_items}
</ol>
</details>
</div>

{sections}
</body>
</html>'''

out_path = os.path.join(root, 'whitepaper.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'\nWritten: {out_path}')
print(f'Total: {len(html):,} chars, {len(bodies)} papers spliced')
