#!/usr/bin/env python3
"""Phase 2: De-AI the papers. Remove AI patterns, flags, bold abuse, humanize."""
import re
from pathlib import Path

ROOT = Path("c:/Users/legom/HyperTensor/ARXIV_SUBMISSIONS")
FIXES = 0

# AI flag words/phrases to remove or replace
AI_PATTERNS = [
    # AI flag words
    ('delve into', 'examine'),
    ('delve', 'examine'),
    ('showcasing', 'showing'),
    ('paramount', 'essential'),
    ('underscores', 'shows'),
    ('moreover', 'furthermore'),
    ('crucial', 'important'),
    ('pivotal', 'key'),
    ('game-changer', 'advance'),
    ('unprecedented', 'novel'),
    ('remarkably', 'notably'),
    ('intriguingly', 'interestingly'),
    
    # Overused qualifiers
    ('It is worth noting that ', ''),
    ('It should be noted that ', ''),
    ('Interestingly, ', ''),
    ('Notably, ', ''),
    
    # Dead references
    ('(see \\Cref{sec:future})', ''),
    ('(see \\Cref{sec:status})', ''),
    
    # Stale experiment labels
    ('the companion paper ', 'Part~I '),
    ('the companion measurement', 'Part~I'),
    
    # Obsolete binary name
    ('geodessical2', 'geodessical'),
    ('geodessical v0.6 (Synapse)', 'geodessical'),
]

for paper_dir in sorted(ROOT.glob("paper-*")):
    for tex in paper_dir.glob("*.tex"):
        original = tex.read_text(encoding="utf-8", errors="replace")
        content = original
        
        for pattern, replacement in AI_PATTERNS:
            # Use string replace for plain text, regex only when needed
            content = content.replace(pattern, replacement)
        
        if content != original:
            tex.write_text(content, encoding="utf-8")
            FIXES += 1
            print(f"  De-AI: {tex.relative_to(ROOT)}")

print(f"\n{FIXES} files de-AI'd")
