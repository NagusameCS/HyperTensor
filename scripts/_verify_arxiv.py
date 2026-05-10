import urllib.request, re, time, sys
ids = ['2203.02155','2212.08073','2307.02483','2404.08471','2212.04089']
hdr = {'User-Agent':'Mozilla/5.0 (HyperTensor citation verifier)'}
for aid in ids:
    url = f'https://arxiv.org/abs/{aid}'
    try:
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=30) as r:
            html = r.read().decode('utf-8','ignore')
        m = re.search(r'<meta name="citation_title" content="([^"]+)"', html)
        title = m.group(1) if m else '<NO TITLE>'
        print(f'OK  {aid}  {title[:100]}')
    except Exception as e:
        print(f'ERR {aid}  {e}')
    time.sleep(6)
