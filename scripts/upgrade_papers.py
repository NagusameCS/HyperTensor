#!/usr/bin/env python3
"""Upgrade Papers XI-XV to match Paper 1's visual standard with section-anchor nav."""
import re, os

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'papers')

# Paper section ID maps (h2 text -> section id)
section_maps = {
    '11-ugt-taxonomy.html': {
        'Abstract': 'abstract',
        '1. The UGT Construction': 'construction',
        '2. Bilateral UGT: Cross-Model Component Interchange': 'bilateral',
        '3. Zone Specialisation': 'zones',
        '4. CECI Validation': 'ceci',
        '5. Implementation': 'implementation',
        '6. Status and Remaining Work': 'status',
    },
    '12-native-training.html': {
        'Abstract': 'abstract',
        '1. NativeLinear Architecture': 'architecture',
        '2. KExpansion Scheduler': 'kexpansion',
        '3. Measured Results': 'results',
        '4. Analytic k* via AttnRes Phase Transition': 'analytic-k',
        '5. Implementation': 'implementation',
        '6. Status': 'status',
    },
    '13-safe-ogd.html': {
        'Abstract': 'abstract',
        '1. The Safety Problem': 'problem',
        '2. The Safe Subspace Projector': 'projector',
        '3. Multi-Step OGD Chains': 'chains',
        '4. MIKU Creativity Benchmark (MCB)': 'mcb',
        '5. Measured Results': 'results',
        '6. Implementation': 'implementation',
        '7. Status': 'status',
    },
    '14-behavioral-snipe.html': {
        'Abstract': 'abstract',
        '1. The Behavioral Coordinate Hypothesis': 'hypothesis',
        '2. Category Probing': 'probing',
        '3. Greedy Selection with Benign Budget': 'greedy',
        '4. Measured Results': 'results',
        '5. Pre/Post COG Pipeline': 'pipeline',
        '6. Implementation': 'implementation',
        '7. Status': 'status',
    },
    '15-cog-teh.html': {
        'Abstract': 'abstract',
        '1. COG: The Living Manifold': 'cog',
        '2. TEH: Tangent Eigenvalue Harmonics': 'teh',
        '3. AttnRes Phase Transition': 'attnres',
        '4. ISAGI: The Complete Living Model': 'isagi',
        '5. Implementation': 'implementation',
        '6. Status': 'status',
    },
}

for fname, smap in section_maps.items():
    fpath = os.path.join(root, fname)
    if not os.path.exists(fpath):
        print(f'MISSING: {fname}')
        continue
    
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()
    
    orig = c
    
    # 1. Wrap h2 sections in <section id="..."> tags
    # Pattern: <h2>Section Title</h2>\n\n...content...\n\n<h2> or </article>
    for section_title, section_id in smap.items():
        # Find this h2
        pattern = re.compile(
            rf'(<h2>\s*{re.escape(section_title)}\s*</h2>)',
            re.IGNORECASE
        )
        m = pattern.search(c)
        if not m:
            # Try without number prefix
            base_title = re.sub(r'^\d+\.\s*', '', section_title)
            pattern = re.compile(
                rf'(<h2>\s*{re.escape(base_title)}\s*</h2>)',
                re.IGNORECASE
            )
            m = pattern.search(c)
        if m:
            h2_tag = m.group(0)
            section_open = f'<section id="{section_id}">\n{h2_tag}'
            c = c.replace(h2_tag, section_open, 1)
            print(f'  {fname}: wrapped "{section_title[:40]}" as #{section_id}')
        else:
            print(f'  {fname}: NOT FOUND "{section_title[:40]}"')
    
    # Close unmatched <section> tags before next <section> or </article>
    # Find all <section id="..."> positions and insert </section> before the next one or </article>
    section_starts = list(re.finditer(r'<section id="([^"]+)">', c))
    for i, m in enumerate(section_starts):
        sid = m.group(1)
        start_pos = m.start()
        if i + 1 < len(section_starts):
            end_pos = section_starts[i+1].start()
        else:
            end_pos = c.find('</article>', start_pos)
            if end_pos < 0:
                end_pos = len(c)
        # Insert </section> before the end marker
        c = c[:end_pos] + '\n</section>\n' + c[end_pos:]
        # Recalculate positions after insertion
        section_starts = list(re.finditer(r'<section id="([^"]+)">', c))
    
    # 2. Update nav to include section links
    nav_start = c.find('<nav>')
    nav_end = c.find('</nav>')
    nav_content = c[nav_start:nav_end]
    
    # Build section-anchor links
    section_links = '\n'.join(
        f'    <li><a href="#{sid}">{title[:40].strip()}</a></li>'
        for title, sid in smap.items() if sid != 'abstract'
    )
    
    # Insert section links into nav (after "All papers" and "Intro")
    # Find the "Paper XV" (last paper link) or "GitHub" link position
    github_pos = nav_content.rfind('<li><a href="https://github.com')
    if github_pos > 0:
        new_nav = nav_content[:github_pos] + section_links + '\n    ' + nav_content[github_pos:]
    else:
        new_nav = nav_content + '\n' + section_links
    
    c = c[:nav_start] + new_nav + c[nav_end:]
    
    if c != orig:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f'  UPDATED: {fname}')
    else:
        print(f'  UNCHANGED: {fname}')

print('\nDone')
