import re
t=open('ARXIV_SUBMISSIONS/volume_extended.tex',encoding='utf-8').read()
for pat in ['Honest scope','honest scope','Honest Scope','bulletproof','Bulletproof','BULLETPROOF']:
    for m in re.finditer(re.escape(pat), t):
        s=max(0,m.start()-60); e=min(len(t),m.end()+160)
        line=t[:m.start()].count('\n')+1
        ctx=t[s:e].replace('\n',' ')
        print(f'L{line:5d} [{pat}] ...{ctx}...')
