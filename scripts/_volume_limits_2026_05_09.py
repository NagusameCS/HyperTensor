"""Cut Verification Status section, rename 'Volume Limitations and Open Items'
to just 'Limitations', and remove the build-warnings paragraph.
"""
import pathlib

P = pathlib.Path('ARXIV_SUBMISSIONS/volume_extended.tex')
src = P.read_text(encoding='utf-8')
orig = src

# 1. Cut the Verification Status section entirely (header + single body paragraph).
old_vs = (
    '\\section*{Verification Status}\n'
    'Quantitative claims in this volume are catalogued in the repository file '
    '\\texttt{VERIFICATION\\_STATUS.md} with one of four tags: REAL (measured on hardware), '
    'SIM (synthetic or Monte Carlo), MIXED (combination), or UNVERIFIED (compute-bound or '
    'outstanding). Claims tagged SIM or UNVERIFIED in that file are noted as such in the '
    'body where space permits. The verification infrastructure is in '
    '\\texttt{scripts/complete\\_verify.py}. See also \\texttt{BENCHMARK\\_PROTOCOL.md} '
    'and \\texttt{complete\\_STATE.md} in the repository root.\n'
    '\n'
    '\n'
)
assert old_vs in src, "Verification Status block not found verbatim"
src = src.replace(old_vs, '')

# 2. Rename the section heading.
src = src.replace(
    '\\section*{Volume Limitations and Open Items (May 2026)}\n'
    '\\addcontentsline{toc}{section}{Volume Limitations and Open Items}\n',
    '\\section*{Limitations}\n'
    '\\addcontentsline{toc}{section}{Limitations}\n',
)

# 3. Remove the Build-warnings paragraph (full block including trailing blank line).
old_bw = (
    '\\paragraph{Build warnings.}\n'
    'The current build emits multiply-defined-label warnings for generic\n'
    'per-paper section labels (\\texttt{sec:intro}, \\texttt{sec:method},\n'
    '\\texttt{sec:limits}, \\texttt{sec:repro}, \\dots) and for shared\n'
    '\\texttt{\\textbackslash bibitem} keys (\\texttt{stewart}, \\texttt{golub},\n'
    '\\texttt{absil}, \\texttt{riemann}, \\texttt{edwards}, \\dots) that recur\n'
    'inside per-paper \\texttt{thebibliography} blocks. These are cosmetic\n'
    'warnings inherent to stitching eighteen self-contained papers into a\n'
    'single volume; the resulting PDF resolves all citations and references\n'
    'with zero unresolved entries. A future revision may rename the\n'
    'per-paper labels with paper-prefixes to silence them.\n'
    '\n'
)
assert old_bw in src, "Build warnings block not found verbatim"
src = src.replace(old_bw, '')

# 4. Soften the lead paragraph: drop the "load-bearing" / "fish them out" framing.
old_lead = (
    'This volume is a working preprint, not eighteen finished journal\n'
    "papers. The scope statements in each paper's abstract\n"
    '(rewritten in the May~2026 revision) are load-bearing; this section\n'
    'collects the open items into one place so a reader does not have to\n'
    'fish them out of eighteen individual abstracts.\n'
)
new_lead = (
    'The scope statement in each paper bounds what that paper claims.\n'
    'This section collects the cross-volume limitations and the open\n'
    'measurements that would extend them.\n'
)
src = src.replace(old_lead, new_lead)

assert src != orig, "no changes applied"
P.write_text(src, encoding='utf-8', newline='\n')
print('size before:', len(orig.encode('utf-8')))
print('size after :', len(src.encode('utf-8')))
print('delta:', len(src.encode('utf-8')) - len(orig.encode('utf-8')))
print('Verification Status remaining:', src.count('Verification Status'))
print('Build warnings remaining:', src.count('Build warnings'))
print('Volume Limitations remaining:', src.count('Volume Limitations'))
print("'Limitations' header present:", '\\section*{Limitations}' in src)
