import os

banner = '\n<div style="background:#eef4ff;border:1px solid #2d6be4;border-radius:8px;padding:12px 20px;margin:12px 2rem;max-width:780px;font-size:0.85rem;color:#1a1a1a;">\nHyperTensor Framework Update (May 3, 2026): This paper is part of a 30-paper research program. The complete framework now includes the k-manifold living-model stack (Papers XI&ndash;XV, 96% complete), a geometric approach to the Riemann Hypothesis via Z<sub>2</sub> symmetry (Papers XVI&ndash;XVIII), and the ISAGI adaptive living model. See the <a href="../whitepaper.html" style="color:#2d6be4;">unified whitepaper</a> and <a href="https://github.com/NagusameCS/HyperTensor" style="color:#2d6be4;">GitHub repository</a> for the full research program.\n</div>\n'

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'papers')
for fname in sorted(os.listdir(root)):
    if fname.endswith('.html') and fname[0].isdigit():
        fpath = os.path.join(root, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            c = f.read()
        if 'HyperTensor Framework Update' not in c:
            c = c.replace('</nav>', '</nav>\n' + banner)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(c)
            print(f'  Updated: {fname}')

print('Done')
