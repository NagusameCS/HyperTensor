#!/usr/bin/env python3
"""Prepare language-only training data for dedicated model training."""
import json, re, os
from datasets import load_dataset, concatenate_datasets

OUTPUT_DIR = "configs\language_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

exclude = ["\\b\\d{4,}\\b", "[+\\-*/=\\^]{2,}", "\\\\begin\\{", "\\\\frac\\{", "\\$.*\\$"]
include = ["[a-zA-Z]{3,}"]

def filter_language(text: str) -> bool:
    """Return True if text is pure language content."""
    if not text or len(text.strip()) < 20:
        return False
    # Must match at least one include pattern
    if not any(re.search(p, text) for p in include):
        return False
    # Must not match any exclude pattern
    if any(re.search(p, text, re.IGNORECASE) for p in exclude):
        return False
    return True

all_data = []
for ds_info in [{"name": "wikitext", "path": "wikitext", "config": "wikitext-103-raw-v1", "split": "train", "filter": "language", "max_samples": 50000}, {"name": "bookcorpus", "path": "bookcorpus", "split": "train", "filter": "language", "max_samples": 20000}]:
    print(f"Loading {ds_info['name']}...")
    try:
        ds = load_dataset(ds_info["path"], split=ds_info["split"], trust_remote_code=True)
        # Format if needed
        fmt = ds_info.get("format", "{text}")
        filtered = []
        for row in ds:
            if len(filtered) >= ds_info.get("max_samples", 50000):
                break
            text = fmt.format(**row) if fmt else row.get("text", "")
            if filter_language(text):
                filtered.append({"text": text})
        print(f"  {ds_info['name']}: {len(filtered)} filtered samples")
        all_data.extend(filtered)
    except Exception as e:
        print(f"  {ds_info['name']}: SKIP ({e})")

# Save
output_path = os.path.join(OUTPUT_DIR, "language_only_train.jsonl")
with open(output_path, "w") as f:
    for item in all_data:
        f.write(json.dumps(item) + "\n")

print(f"\nTotal: {len(all_data)} samples -> {output_path}")
print(f"Estimated tokens: ~{len(all_data) * 200} (rough)")
