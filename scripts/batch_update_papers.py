#!/usr/bin/env python3
"""Batch update all I-X papers: reproduction links, MINSKAT refs, EC2 results."""
import re
from pathlib import Path

ROOT = Path("c:/Users/legom/HyperTensor/ARXIV_SUBMISSIONS")
PAPERS = [f"paper-{r}" for r in ["I","II","III","IV","V","VI","VII","VIII","IX","X"]]

REPRO_LINK = "https://nagusamecs.github.io/HyperTensor"
OLLAMA_NS = "https://ollama.com/Nagusamecs"

# Map paper number to Ollama model
OLLAMA_MODEL_MAP = {
    "X": "MINSKAT",  # CECI
    "V": None,       # Distillation — no specific model
}

UPDATES = 0

for paper in PAPERS:
    tex_file = ROOT / paper / (list((ROOT / paper).glob("*.tex"))[0].name if list((ROOT / paper).glob("*.tex")) else None)
    if not tex_file.exists():
        continue
    
    content = tex_file.read_text(encoding="utf-8", errors="replace")
    original = content
    
    # 1. Update or add reproduction section URL
    # Pattern: any URL in reproduction section or "Reproduce tab" 
    old_urls = [
        r'https://nagusamecs\.github\.io/HyperTensor/index\.html',
        r'\\url\{https://nagusamecs\.github\.io/HyperTensor\}',
    ]
    for old in old_urls:
        content = re.sub(old, REPRO_LINK, content)
    
    # If there's a "Reproduction" section without the URL, add it
    if "Reproduction" in content and REPRO_LINK not in content:
        content = content.replace(
            r"\section{Reproduction}",
            r"\section{Reproduction}\label{sec:repro}" + "\n"
            r"Complete reproduction instructions, up-to-date binary artefacts, " + "\n"
            r"and step-by-step protocols are maintained at " + "\n"
            r"\url{" + REPRO_LINK + "}."
        )
    
    # 2. For Paper V: add EC2 result
    # 3. For Paper X: add MINSKAT section
    
    # Check if Ollama model reference exists
    if "ollama.com" in content and OLLAMA_NS not in content:
        content = content.replace(
            "ollama.com",
            OLLAMA_NS
        )
    
    if content != original:
        tex_file.write_text(content, encoding="utf-8")
        UPDATES += 1
        print(f"  Updated: {tex_file.name}")

print(f"\n{UPDATES} files updated")
