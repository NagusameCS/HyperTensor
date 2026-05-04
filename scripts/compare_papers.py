import re, os
root = r'c:\Users\legom\HyperTensor\docs\papers'
for fn in ['01-attention-compression.html','11-ugt-taxonomy.html','12-native-training.html']:
    with open(os.path.join(root,fn),'r',encoding='utf-8') as f:
        c = f.read()
    has_inline = '<style>' in c[:3000]
    has_linked = 'paper.css' in c[:3000] or 'paper-style.css' in c[:3000]
    has_plotly = 'plotly' in c[:2000]
    has_hero_div = 'id="hero"' in c[:5000]
    has_hero_tag = 'hero-tag' in c[:5000]
    has_hero_title = 'hero-title' in c[:5000]
    has_article = '<article' in c[:10000]
    print(f'{fn}: inline_css={has_inline} linked_css={has_linked} plotly={has_plotly} hero={has_hero_div} article={has_article} size={len(c):,}')

# Check Paper 1 nav links
with open(os.path.join(root,'01-attention-compression.html'),'r',encoding='utf-8') as f:
    c1 = f.read()
nav1 = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', c1[c1.find('<nav>'):c1.find('</nav>')])
print(f'\nPaper 1 nav: {nav1}')

with open(os.path.join(root,'11-ugt-taxonomy.html'),'r',encoding='utf-8') as f:
    c11 = f.read()
nav11 = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', c11[c11.find('<nav>'):c11.find('</nav>')])
print(f'Paper XI nav: {nav11}')
