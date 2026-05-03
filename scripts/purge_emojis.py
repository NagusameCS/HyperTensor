#!/usr/bin/env python3
"""
Emoji purger — removes all emoji characters from text files recursively.
"""
import re, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # misc symbols
    "\U0001F680-\U0001F6FF"  # transport
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed
    "\U0001F900-\U0001F9FF"  # supplemental
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # more
    "\U00002600-\U000026FF"  # misc
    "\U00002700-\U000027BF"  # more dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"              # ZWJ
    "\U0000FE0F"              # variation selector-16
    "\U000020E3"              # combining enclosing keycap
    "" "" "" "" "" "" ""  # common used in our repo
    "]+",
    flags=re.UNICODE
)

TARGET_EXTS = {'.tex','.md','.py','.json','.ps1','.sh','.c','.h','.txt','.csv','.modelfile'}
SKIP_DIRS = {'.git','.venv','venv','__pycache__','node_modules','build_host','build_rpi','.cache'}
SKIP_NAMES = {'.gitignore','.gitattributes'}

count = 0
for fp in ROOT.rglob('*'):
    if fp.is_dir(): continue
    if fp.suffix.lower() not in TARGET_EXTS and fp.suffix: continue
    if any(p in SKIP_DIRS for p in fp.parts): continue
    if fp.name in SKIP_NAMES: continue
    
    try:
        text = fp.read_text(encoding='utf-8', errors='replace')
        cleaned = EMOJI_RE.sub('', text)
        if cleaned != text:
            fp.write_text(cleaned, encoding='utf-8')
            count += 1
            if count <= 30:
                match_chars = set(text) - set(cleaned)
                emojis = ''.join(sorted(c for c in match_chars if ord(c) > 127)[:10])
                print(f"  Cleaned: {fp.relative_to(ROOT)} [{emojis}]")
    except Exception:
        pass

print(f"\nEmoji purge complete: {count} files cleaned")
