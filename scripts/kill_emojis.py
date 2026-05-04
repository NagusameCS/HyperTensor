# -*- coding: utf-8 -*-
"""Kill all emojis and status markers from specified files. Replace with ASCII."""
import os, sys, glob

REPLACEMENTS = {
    # Status markers
    '[OK]': '[OK]',
    '[XX]': '[XX]', 
    '[!!]': '[!!]',
    # Decorative
    '': '',
    '*': '*',
    '*': '*',
    'Note:': 'Note:',
    '': '',
    '->': '->',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    # Dashboard
    '[CHART]': '[CHART]',
    '[UP]': '[UP]',
    '': '',
    '': '',
    '': '',
    '<->': '<->',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
}

TARGET_DIRS = [
    'scripts',
]

def kill_emojis_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for emoji, replacement in REPLACEMENTS.items():
        content = content.replace(emoji, replacement)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(root)  # go up from scripts/ to repo root
    
    changed = 0
    for d in TARGET_DIRS:
        for root_dir, dirs, files in os.walk(os.path.join(root, d)):
            # Skip session_dossier
            if 'session_dossier' in root_dir:
                continue
            for fname in files:
                if fname.endswith('.py') or fname.endswith('.md') or fname.endswith('.html'):
                    fpath = os.path.join(root_dir, fname)
                    if kill_emojis_in_file(fpath):
                        changed += 1
                        rel = os.path.relpath(fpath, root)
                        print(f'  Cleaned: {rel}')
    
    print(f'\nFiles cleaned: {changed}')

if __name__ == '__main__':
    main()
