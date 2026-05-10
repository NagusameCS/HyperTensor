import urllib.request, re
hdr = {'User-Agent':'Mozilla/5.0 (verifier)'}
for aid in ['2301.08243','2404.08471']:
    req = urllib.request.Request(f'https://arxiv.org/abs/{aid}', headers=hdr)
    html = urllib.request.urlopen(req, timeout=30).read().decode('utf-8','ignore')
    t = re.search(r'<meta name="citation_title" content="([^"]+)"', html).group(1)
    a = re.findall(r'<meta name="citation_author" content="([^"]+)"', html)
    d = re.search(r'<meta name="citation_date" content="([^"]+)"', html)
    print(f'{aid}')
    print(f'  TITLE   : {t}')
    print(f'  AUTHORS : {a[:8]}')
    print(f'  DATE    : {d.group(1) if d else "?"}')
