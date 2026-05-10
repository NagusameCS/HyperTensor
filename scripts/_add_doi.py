import pathlib
p = pathlib.Path('ARXIV_SUBMISSIONS/volume_extended.tex')
t = p.read_text(encoding='utf-8')

needle = '\\maketitle\n\n\\section*{Foreword}'
print('matches:', t.count(needle))

doi_block = '\\begin{center}\\small DOI: \\href{https://doi.org/10.5281/zenodo.20077378}{10.5281/zenodo.20077378}\\end{center}\n\n'
replacement = '\\maketitle\n\n' + doi_block + '\\section*{Foreword}'

if t.count(needle) == 1:
    t = t.replace(needle, replacement, 1)
    p.write_text(t, encoding='utf-8', newline='\n')
    print('DOI inserted, count in file:', t.count('20077378'))
else:
    print('FAIL')
