import re
t=open('ARXIV_SUBMISSIONS/volume_extended.tex',encoding='utf-8').read()
# Find each thebibliography block and its bibitem count.
for m in re.finditer(r'\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}', t, re.DOTALL):
    line=t[:m.start()].count('\n')+1
    body=m.group(1)
    items=re.findall(r'\\bibitem', body)
    # Find nearest preceding \section{Paper N: ...}
    prev=t[:m.start()]
    pm = list(re.finditer(r'\\section\{Paper\s+([IVX]+):[^}]*\}', prev))
    paper = pm[-1].group(1) if pm else '?'
    print(f'Paper {paper:>5s}  L{line:5d}  bibitems={len(items)}')
