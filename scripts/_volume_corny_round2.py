"""Round-2 corny softening: two purple-prose lines."""
import pathlib

P = pathlib.Path('ARXIV_SUBMISSIONS/volume_extended.tex')
src = P.read_text(encoding='utf-8')
orig = src

edits = [
    (
        'The surprise came at a specific compression level: 1,024 dimensions kept out',
        'The non-trivial result was at a specific compression level: 1,024 dimensions kept out',
    ),
    (
        "metric's dominant eigendirections shift. Over thousands of interactions, the\n"
        "manifold naturally forgets noise and crystallizes profound concepts.",
        "metric's dominant eigendirections shift. Over thousands of interactions, the\n"
        "manifold attenuates noise directions and reinforces frequently revisited ones.",
    ),
]

for old, new in edits:
    assert old in src, f'NOT FOUND: {old[:80]!r}'
    src = src.replace(old, new)

assert src != orig
P.write_text(src, encoding='utf-8', newline='\n')
print('size before:', len(orig.encode('utf-8')))
print('size after :', len(src.encode('utf-8')))
print('crystallizes remaining:', src.count('crystallizes'))
print('The surprise came remaining:', src.count('The surprise came'))
