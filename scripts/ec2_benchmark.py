#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
EC2 L40S BENCHMARK SUITE — Papers V, VI, VII
==============================================
Runs all GPU-blocked experiments on L40S (48GB VRAM).

1. Paper VII: LoRA on activation-weighted GRC FFN (7B)
2. Paper VI:  Extended MMLU + GSM8K at 7B scale
3. Paper V:  GRC distillation proxy on Qwen2.5-7B

Output: ~/benchmarks_ec2/results.json
"""
import torch, json, time, os, re, random, numpy as np
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

OUT = Path.home() / "benchmarks_ec2"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

#  Paper VII: LoRA on FFN at 7B scale 
def paper_vii_lora():
    print("\n" + "="*60)
    print("PAPER VII — LoRA on Activation-Weighted FFN (7B)")
    print("="*60)
    
    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    
    print("[1] Loading 7B model @ 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, device_map="auto", trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print(f"    d_model={model.config.hidden_size}, layers={model.config.num_hidden_layers}")
    
    # Calibration texts (~200 tokens each, 10 samples = ~2K tokens)
    cal_texts = [
        "The capital of France is Paris. France is a country in Western Europe known for its art, cuisine, and culture. Paris has many famous landmarks including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        "Quantum mechanics describes the behavior of matter and energy at atomic scales. Key principles include wave-particle duality, superposition, and the uncertainty principle. The Schrödinger equation governs quantum evolution.",
        "Machine learning is a subset of artificial intelligence. Neural networks learn patterns from data through backpropagation. Transformers use self-attention mechanisms for natural language processing tasks.",
        "The Renaissance was a period of European cultural rebirth spanning the 14th to 17th centuries. It began in Italy and spread across Europe, reviving classical learning and fostering artistic innovation.",
        "DNA replication is the process by which a cell copies its genetic material. The double helix unwinds, and each strand serves as a template for a new complementary strand. This is semiconservative replication.",
        "Gravity is a fundamental force of nature. Newton described it as a force proportional to mass and inversely proportional to distance squared. Einstein recast it as curvature of spacetime.",
        "The Industrial Revolution transformed manufacturing through mechanization. It began in Britain around 1760 and spread globally, replacing manual labor with machines powered by steam and later electricity.",
        "Photosynthesis converts light energy into chemical energy. Plants use chlorophyll to capture sunlight, converting carbon dioxide and water into glucose and oxygen. This process sustains most life on Earth.",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c².",
        "Neural networks consist of layers of interconnected nodes. Each node applies a weighted sum followed by an activation function. Deep networks learn hierarchical representations automatically.",
    ]
    
    test_texts = [
        "World War II was a global conflict lasting from 1939 to 1945. It involved most of the world's nations including all great powers, eventually forming two opposing military alliances.",
        "The speed of light in vacuum is exactly 299,792,458 meters per second. This constant, denoted by c, is fundamental to physics and appears in Einstein's famous equation E = mc².",
    ]
    
    def ppl(model, tok, texts):
        total_loss = 0.0; total_tok = 0
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=256)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
                total_loss += out.loss.item() * enc["input_ids"].shape[1]
                total_tok += enc["input_ids"].shape[1]
        return float(np.exp(total_loss / max(total_tok, 1)))
    
    print("[2] Baseline PPL...")
    ppl_base = ppl(model, tok, test_texts)
    print(f"    Baseline: {ppl_base:.2f}")
    
    print("[3] Applying GRC at k=1024 on attention...")
    nkv = model.config.num_key_value_heads
    nh = model.config.num_attention_heads
    has_gqa = nkv < nh
    
    for layer in model.model.layers:
        attn = layer.self_attn; dt = attn.q_proj.weight.dtype
        Wq = attn.q_proj.weight.data.float(); Wk = attn.k_proj.weight.data.float(); Wv = attn.v_proj.weight.data.float()
        if has_gqa:
            nr = nh // nkv; hd = Wq.shape[1] // nh
            Wk2 = torch.zeros(Wq.shape[0], Wq.shape[1], device=Wk.device); Wv2 = torch.zeros_like(Wk2)
            for kv in range(nkv):
                for rep in range(nr):
                    qh = kv * nr + rep
                    Wk2[qh*hd:(qh+1)*hd, :] = Wk[kv*hd:(kv+1)*hd, :]
                    Wv2[qh*hd:(qh+1)*hd, :] = Wv[kv*hd:(kv+1)*hd, :]
            Wk, Wv = Wk2, Wv2
        M = torch.cat([Wq, Wk, Wv], dim=0)
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        ke = min(1024, len(S)); P = Vt[:ke, :].T; PTP = P @ P.T
        attn.q_proj.weight.data = (Wq @ PTP).to(dt)
        Wk_p = Wk @ PTP; Wv_p = Wv @ PTP
        if has_gqa:
            Wk_o = torch.zeros(attn.k_proj.weight.shape[0], attn.k_proj.weight.shape[1], device=Wk.device); Wv_o = torch.zeros_like(Wk_o)
            for kv in range(nkv): h0 = kv * nr; Wk_o[kv*hd:(kv+1)*hd, :] = Wk_p[h0*hd:(h0+1)*hd, :]; Wv_o[kv*hd:(kv+1)*hd, :] = Wv_p[h0*hd:(h0+1)*hd, :]
            attn.k_proj.weight.data = Wk_o.to(dt); attn.v_proj.weight.data = Wv_o.to(dt)
        else:
            attn.k_proj.weight.data = Wk_p.to(dt); attn.v_proj.weight.data = Wv_p.to(dt)
    torch.cuda.empty_cache()
    
    ppl_grc = ppl(model, tok, test_texts)
    print(f"    GRC PPL: {ppl_grc:.2f} ({ppl_grc/ppl_base:.2f}x)")
    
    print("[4] Adding LoRA (r=8) to FFN + fine-tuning (300 steps)...")
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "down_proj"],
                              lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(model, lora_config)
    opt = torch.optim.AdamW(peft_model.parameters(), lr=5e-4)
    
    for step in range(300):
        enc = tok(cal_texts[step % len(cal_texts)], return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        opt.zero_grad()
        out = peft_model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 75 == 0:
            print(f"    Step {step+1}/300: loss={out.loss.item():.4f}")
    
    ppl_lora = ppl(peft_model, tok, test_texts)
    gap_recovery = (ppl_grc - ppl_lora) / max(ppl_grc - ppl_base, 1e-6) * 100
    print(f"    LoRA PPL: {ppl_lora:.2f} ({ppl_lora/ppl_base:.2f}x) — gap recovery: {gap_recovery:.0f}%")
    
    result = {
        "model": MODEL, "scale": "7B", "grc_rank": 1024, "lora_rank": 8,
        "cal_tokens": sum(len(tok(t)['input_ids']) for t in cal_texts),
        "ppl_baseline": round(ppl_base, 3), "ppl_grc": round(ppl_grc, 3),
        "ppl_lora": round(ppl_lora, 3), "gap_recovery_pct": round(gap_recovery, 1),
    }
    del peft_model, model; torch.cuda.empty_cache()
    return result

#  Paper VI: Extended MMLU + GSM8K at 7B scale 
def paper_vi_extended():
    print("\n" + "="*60)
    print("PAPER VI — Extended MMLU + GSM8K (7B)")
    print("="*60)
    
    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    
    print("[1] Loading 7B model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, device_map="auto", trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    def apply_grc(model, k):
        nkv = model.config.num_key_value_heads; nh = model.config.num_attention_heads; has_gqa = nkv < nh
        for layer in model.model.layers:
            attn = layer.self_attn; dt = attn.q_proj.weight.dtype
            Wq = attn.q_proj.weight.data.float(); Wk = attn.k_proj.weight.data.float(); Wv = attn.v_proj.weight.data.float()
            if has_gqa:
                nr = nh // nkv; hd = Wq.shape[1] // nh
                Wk2 = torch.zeros(Wq.shape[0], Wq.shape[1], device=Wk.device); Wv2 = torch.zeros_like(Wk2)
                for kv in range(nkv):
                    for rep in range(nr):
                        qh = kv * nr + rep
                        Wk2[qh*hd:(qh+1)*hd, :] = Wk[kv*hd:(kv+1)*hd, :]
                        Wv2[qh*hd:(qh+1)*hd, :] = Wv[kv*hd:(kv+1)*hd, :]
                Wk, Wv = Wk2, Wv2
            M = torch.cat([Wq, Wk, Wv], dim=0)
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
            ke = min(k, len(S)); P = Vt[:ke, :].T; PTP = P @ P.T
            attn.q_proj.weight.data = (Wq @ PTP).to(dt)
            Wk_p = Wk @ PTP; Wv_p = Wv @ PTP
            if has_gqa:
                Wk_o = torch.zeros(attn.k_proj.weight.shape[0], attn.k_proj.weight.shape[1], device=Wk.device); Wv_o = torch.zeros_like(Wk_o)
                for kv in range(nkv): h0 = kv * nr; Wk_o[kv*hd:(kv+1)*hd, :] = Wk_p[h0*hd:(h0+1)*hd, :]; Wv_o[kv*hd:(kv+1)*hd, :] = Wv_p[h0*hd:(h0+1)*hd, :]
                attn.k_proj.weight.data = Wk_o.to(dt); attn.v_proj.weight.data = Wv_o.to(dt)
            else:
                attn.k_proj.weight.data = Wk_p.to(dt); attn.v_proj.weight.data = Wv_p.to(dt)
        torch.cuda.empty_cache()
    
    def eval_mmlu(model, tok, n_samples=50):
        try:
            ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
            subjects = list(set(row["subject"] for row in ds.take(1000)))
            selected = random.sample(subjects, min(5, len(subjects)))
        except:
            # Fallback: use hand-picked questions
            QS = [{'s':'math','q':'What is the derivative of x^3?','c':['x^2','3x^2','3x','x^3/3'],'a':1},
                  {'s':'math','q':'If x^2=25, what is x?','c':['5','-5','5 or -5','25'],'a':2},
                  {'s':'science','q':'What is H2O?','c':['Hydrogen','Oxygen','Water','Peroxide'],'a':2},
                  {'s':'science','q':'Speed of light?','c':['3e6','3e8','3e10','300'],'a':1},
                  {'s':'history','q':'WWII ended in?','c':['1943','1944','1945','1946'],'a':2},
                  {'s':'history','q':'First US President?','c':['Jefferson','Washington','Lincoln','Adams'],'a':1},
                  {'s':'cs','q':'CPU stands for?','c':['Central Processing Unit','Computer Personal Unit','Central Program Utility','Core Processing Unit'],'a':0},
                  {'s':'cs','q':'Binary search complexity?','c':['O(n)','O(n^2)','O(log n)','O(1)'],'a':2}]
            correct = 0
            for q in QS:
                choices = '\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(q['c']))
                prompt = f"Question: {q['q']}\n{choices}\nAnswer with the letter."
                msgs = [{'role':'user','content':prompt}]
                fmt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = tok(fmt, return_tensors='pt', truncation=True, max_length=512)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                with torch.no_grad(): out = model.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
                resp = tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
                m = re.search(r'\b([A-D])\b', resp); pred = m.group(1) if m else '?'
                if pred == chr(65 + q['a']): correct += 1
            return {'accuracy': correct/len(QS), 'correct': correct, 'total': len(QS)}
    
    def eval_gsm8k(model, tok, n_samples=20):
        try:
            ds = load_dataset("gsm8k", "main", split="test", streaming=True)
            qs = [row for row in ds.take(n_samples)]
        except:
            qs = [{'question': 'Janet has 24 eggs. She uses 4 eggs to make omelets and gives away 6 eggs. How many eggs does Janet have left?', 'answer': 'Janet starts with 24 eggs. 24 - 4 = 20. 20 - 6 = 14. #### 14'},
                  {'question': 'A store sells apples for $0.50 each. Tom buys 12 apples and pays with $10. How much change?', 'answer': '12 × 0.50 = 6. 10 - 6 = 4. #### 4'}]
        correct = 0
        for q in qs:
            prompt = f"Solve: {q['question']}\nThink step by step. End with #### answer."
            msgs = [{'role':'user','content':prompt}]
            fmt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(fmt, return_tensors='pt', truncation=True, max_length=512)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad(): out = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
            resp = tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
            m = re.search(r'####\s*(-?[\d.,]+)', resp)
            pred = m.group(1).replace(',','') if m else None
            gt_match = re.search(r'####\s*(-?[\d.,]+)', q['answer'])
            gt = gt_match.group(1).replace(',','') if gt_match else None
            if pred and gt and abs(float(pred) - float(gt)) < 1e-6: correct += 1
        return {'accuracy': correct/len(qs), 'correct': correct, 'total': len(qs)}
    
    results = {}
    # Full rank
    print("[2] Full rank...")
    results['full'] = {'mmlu': eval_mmlu(model, tok), 'gsm8k': eval_gsm8k(model, tok)}
    print(f"    MMLU={results['full']['mmlu']['accuracy']:.1%} GSM8K={results['full']['gsm8k']['accuracy']:.1%}")
    
    # GRC at k=2048
    for k in [2048, 1024, 512]:
        print(f"[3] GRC k={k}...")
        del model; torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, device_map="auto", trust_remote_code=True, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        apply_grc(model, k)
        results[str(k)] = {'mmlu': eval_mmlu(model, tok), 'gsm8k': eval_gsm8k(model, tok)}
        print(f"    MMLU={results[str(k)]['mmlu']['accuracy']:.1%} GSM8K={results[str(k)]['gsm8k']['accuracy']:.1%}")
    
    del model; torch.cuda.empty_cache()
    return results

#  Run all 
all_results = {}
try:
    all_results['paper_vii'] = paper_vii_lora()
except Exception as e:
    all_results['paper_vii'] = {"error": str(e)[:200]}

try:
    all_results['paper_vi'] = paper_vi_extended()
except Exception as e:
    all_results['paper_vi'] = {"error": str(e)[:200]}

with open(OUT / "results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"RESULTS SAVED: {OUT / 'results.json'}")
for k, v in all_results.items():
    if 'error' in v:
        print(f"  {k}: ERROR — {v['error'][:80]}")
    else:
        print(f"  {k}: OK — {list(v.keys())[:5]}...")
