import re, os
root = r'c:\Users\legom\HyperTensor\docs\papers'
for fname in ['00-introduction.html','01-attention-compression.html','02-geodesic-projection.html']:
    with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
        c = f.read()
    # Find all article-like containers
    for tag in ['article', 'section', 'main', 'div']:
        matches = list(re.finditer(rf'<{tag}\b[^>]*>', c))
        if matches:
            for m in matches[:3]:
                print(f'{fname}: <{tag}{m.group()[len(tag)+1:]}')
    # Check for </nav> position vs content start
    nav_end = c.find('</nav>')
    article_start = c.find('<article')
    section_start = c.find('<section')
    hero_id = re.search(r'id="hero"', c)
    hero_start = hero_id.start() if hero_id else -1
    print(f'  nav_end={nav_end}, article_start={article_start}, section_start={section_start}, hero={hero_start}')
    # Show chars around nav end
    print(f'  after nav: {c[nav_end:nav_end+200][:100]}')
    print()
