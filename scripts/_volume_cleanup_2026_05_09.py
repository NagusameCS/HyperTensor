"""Analyze and clean Updated Evidence duplicates + small cosmetic fixes in volume_extended.tex.

What it does:
  1. Detects every "\subsection{Updated Evidence (May 2026)} ... " block and dedupes
     adjacent (or nearby) blocks whose body text is identical.
  2. Replaces "Here is the core idea in one paragraph. " with "The core idea is that ".
  3. Ensures every "\printbibliography" and every per-paper "\begin{thebibliography}"
     starts on a fresh page (inserts a "\clearpage" right before, if missing).
  4. Reports stats so we can confirm before writing.

Writes back via UTF-8 (no BOM) to match the existing file. Use --apply to write.
"""
from __future__ import annotations
import re, sys, pathlib

PATH = pathlib.Path('ARXIV_SUBMISSIONS/volume_extended.tex')
src = PATH.read_text(encoding='utf-8')

# ------------------------------------------------------------ 1. UE dedupe
ue_re = re.compile(
    r'(\\subsection\{Updated Evidence \(May 2026\)\}\s*\n.*?)'
    r'(?=\\subsection|\\section|\\chapter|\\clearpage|\\printbibliography|\\end\{document\})',
    re.DOTALL,
)
matches = list(ue_re.finditer(src))
print(f'UE blocks matched: {len(matches)}')

# Build list (start, end, body_norm)
def norm(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip()
    return s

groups = []
for m in matches:
    groups.append((m.start(), m.end(), norm(m.group(1))))

# Identify runs of UE blocks whose first ~200 chars of body match (same evidence
# paragraph repeated; the last block in each run usually carries an extra
# closing comment / "\newpage", so we keep the longest and drop the rest).
def head_key(body: str) -> str:
    # Strip heading and take first ~200 normalised chars of the paragraph.
    body = re.sub(r'^\\subsection\{Updated Evidence \(May 2026\)\}\s*', '', body)
    return norm(body)[:200]

to_drop = []
i = 0
while i < len(groups):
    j = i + 1
    key_i = head_key(groups[i][2])
    while j < len(groups) and head_key(groups[j][2]) == key_i:
        j += 1
    # cluster is groups[i:j]
    if j - i > 1:
        cluster = list(range(i, j))
        keep = max(cluster, key=lambda k: len(groups[k][2]))
        to_drop.extend(k for k in cluster if k != keep)
    i = j
print(f'cluster duplicates to drop: {len(to_drop)}')

# Apply drops (back to front to preserve offsets).
new_src = src
for idx in sorted(to_drop, reverse=True):
    s, e, _ = groups[idx]
    new_src = new_src[:s] + new_src[e:]

# ------------------------------------------------------------ 2. core idea fix
old_core = 'Here is the core idea in one paragraph. Every time the model processes a piece'
new_core = 'The core idea is that every time the model processes a piece'
core_count = new_src.count(old_core)
new_src = new_src.replace(old_core, new_core)
print(f'core-idea replacements: {core_count}')

# Catch any other "Here is the core idea in one paragraph." prefaces.
generic = 'Here is the core idea in one paragraph. '
remaining = new_src.count(generic)
new_src = new_src.replace(generic, 'The core idea is that ')
if remaining:
    print(f'generic core-idea replacements: {remaining}')

# ------------------------------------------------------------ 3. references on own page
# Each \printbibliography → preceded by \clearpage (idempotent).
def ensure_clearpage_before(text: str, marker: str) -> tuple[str, int]:
    out = []
    cursor = 0
    inserted = 0
    for m in re.finditer(re.escape(marker), text):
        out.append(text[cursor:m.start()])
        # Look back at up to 80 chars to see if a \clearpage already precedes.
        prefix = text[max(0, m.start() - 80):m.start()]
        if '\\clearpage' not in prefix and '\\newpage' not in prefix:
            out.append('\\clearpage\n')
            inserted += 1
        out.append(text[m.start():m.end()])
        cursor = m.end()
    out.append(text[cursor:])
    return ''.join(out), inserted

new_src, n1 = ensure_clearpage_before(new_src, '\\printbibliography')
print(f'\\clearpage inserted before \\printbibliography: {n1}')
new_src, n2 = ensure_clearpage_before(new_src, '\\begin{thebibliography}')
print(f'\\clearpage inserted before \\begin{{thebibliography}}: {n2}')

# ------------------------------------------------------------ write
if '--apply' in sys.argv:
    PATH.write_text(new_src, encoding='utf-8', newline='\n')
    print('WROTE', PATH)
else:
    diff_lines = src.count('\n') - new_src.count('\n')
    print(f'(dry run) net line delta: -{diff_lines}')
