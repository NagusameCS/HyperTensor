#!/usr/bin/env python3
"""EC2 setup: prepare calibration corpus for distill runner."""
import os
os.makedirs("data", exist_ok=True)

from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join([t for t in ds["text"][:50000] if t])
with open("data/wikitext2_train_5k.txt", "w") as f:
    f.write(text)
print(f"Corpus: {len(text)} chars, ~{len(text.split())} words")
