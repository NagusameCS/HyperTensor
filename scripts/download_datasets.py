#!/usr/bin/env python3
"""Download benchmark datasets for Paper F."""
import os, json, sys

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

os.makedirs("data/mmlu/test", exist_ok=True)
os.makedirs("data/mmlu/dev", exist_ok=True)

# MMLU
print("Downloading MMLU...")
try:
    ds = load_dataset('cais/mmlu', 'all', split='test', trust_remote_code=True)
    subjects = sorted(set(ds['subject']))
    for subj in subjects:
        subj_data = [row for row in ds if row['subject'] == subj]
        csv_path = f'data/mmlu/test/{subj}_test.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('question,A,B,C,D,answer\n')
            for row in subj_data[:200]:
                choices = row['choices']
                q = row['question'].replace('"', "'").replace('\n', ' ')
                f.write(f'"{q}","{choices[0]}","{choices[1]}","{choices[2]}","{choices[3]}",{row["answer"]}\n')
        print(f'  {subj}: {len(subj_data)} questions')
    print(f'MMLU: {len(subjects)} subjects done')
except Exception as e:
    print(f'MMLU failed: {e}')

# GSM8K
print("Downloading GSM8K...")
try:
    gsm8k = load_dataset('gsm8k', 'main', split='test')
    with open('data/gsm8k_test.jsonl', 'w', encoding='utf-8') as f:
        for row in gsm8k:
            f.write(json.dumps({'question': row['question'], 'answer': row['answer']}) + '\n')
    print(f'GSM8K: {len(gsm8k)} questions')
except Exception as e:
    print(f'GSM8K failed: {e}')

print("Done.")
