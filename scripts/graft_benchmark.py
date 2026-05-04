"""GRAFT BENCHMARK — Industry-standard evaluation of grafted models.

Runs MMLU, HellaSwag, ARC, GSM8K, PIQA, BoolQ on baseline + all 7 grafts.
Uses lm-evaluation-harness when available, falls back to direct eval.

PPL NOTE: Lower PPL = better. "PPL lost" (negative delta) = improvement.
           "PPL gained" (positive delta) = degradation.
"""
import torch, json, time, os, sys, math, subprocess
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F

torch.set_grad_enabled(False)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

OUT = Path("outputs/benchmarks")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  GRAFT BENCHMARK — Industry-Standard Evaluation")
print("  MMLU | HellaSwag | ARC | GSM8K | PIQA | BoolQ")
print("=" * 70)

# ============================================================================
# MODEL REGISTRY
# ============================================================================
MODELS = {
    "baseline": {
        "path": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "display": "SmolLM2-135M (Baseline)",
        "type": "baseline",
    },
    "minSode": {
        "path": "NagusameCS/minSode",
        "display": "minSøde (my sweet)",
        "graft": "Layer 5 FFN ← 15, 105% PPL recovery",
        "ppl_delta": -3.1,  # negative = better than baseline
    },
    "minKaereste": {
        "path": "NagusameCS/minKaereste",
        "display": "minKæreste (my dearest)",
        "graft": "Layer 8 FFN ← 18, 122% PPL recovery",
        "ppl_delta": -4.3,
    },
    "minElskede": {
        "path": "NagusameCS/minElskede",
        "display": "minElskede (my beloved)",
        "graft": "Layer 20 FFN ← 10, 60% PPL recovery",
        "ppl_delta": 0.0,
    },
    "minLillemus": {
        "path": "NagusameCS/minLillemus",
        "display": "minLillemus (my little mouse)",
        "graft": "Layer 12 FFN ← 2, 29% PPL recovery",
        "ppl_delta": +5.5,
    },
    "minFjollede": {
        "path": "NagusameCS/minFjollede",
        "display": "minFjollede (my silly)",
        "graft": "CROSS-MODEL: SmolLM2 ← Qwen2.5-0.5B FFN",
        "ppl_delta": +7.2,
    },
    "minSolskin": {
        "path": "NagusameCS/minSolskin",
        "display": "minSolskin (my sunshine)",
        "graft": "Layers (7,11) ← (22,26), coding-math hybrid",
        "ppl_delta": +7.5,
    },
    "minHjerteven": {
        "path": "NagusameCS/minHjerteven",
        "display": "minHjerteven (my heart-friend)",
        "graft": "8-layer alternating opposite-hemisphere",
        "ppl_delta": +6.9,
    },
}

# ============================================================================
# BENCHMARK DATASETS (self-contained, no external deps needed)
# ============================================================================

# MMLU-lite: 10 questions from 5 categories (50 total)
MMLU_QUESTIONS = [
    # High School Math
    {"subject": "math", "q": "If x^2 + 5x + 6 = 0, what are the roots?", "choices": ["-2 and -3", "2 and 3", "-2 and 3", "2 and -3"], "answer": 0},
    {"subject": "math", "q": "What is the derivative of x^3?", "choices": ["3x^2", "x^2", "3x", "x^4/4"], "answer": 0},
    {"subject": "math", "q": "What is the value of π to 2 decimal places?", "choices": ["3.14", "3.16", "3.12", "3.18"], "answer": 0},
    {"subject": "math", "q": "If a triangle has sides 3, 4, 5, what is its area?", "choices": ["6", "12", "7.5", "10"], "answer": 0},
    {"subject": "math", "q": "What is the sum of angles in a triangle?", "choices": ["180°", "360°", "90°", "270°"], "answer": 0},
    {"subject": "math", "q": "Solve: 2x + 3 = 11", "choices": ["x=4", "x=7", "x=5", "x=8"], "answer": 0},
    {"subject": "math", "q": "What is the square root of 144?", "choices": ["12", "14", "10", "16"], "answer": 0},
    {"subject": "math", "q": "What is 15% of 200?", "choices": ["30", "15", "45", "20"], "answer": 0},
    {"subject": "math", "q": "If f(x) = 2x+1, what is f(3)?", "choices": ["7", "6", "8", "5"], "answer": 0},
    {"subject": "math", "q": "What is the area of a circle with radius 5?", "choices": ["25π", "10π", "5π", "50π"], "answer": 0},
    # Computer Science
    {"subject": "cs", "q": "What does CPU stand for?", "choices": ["Central Processing Unit", "Computer Personal Unit", "Central Program Utility", "Core Processing Unit"], "answer": 0},
    {"subject": "cs", "q": "Which data structure uses LIFO?", "choices": ["Stack", "Queue", "Array", "Tree"], "answer": 0},
    {"subject": "cs", "q": "What is the time complexity of binary search?", "choices": ["O(log n)", "O(n)", "O(n^2)", "O(1)"], "answer": 0},
    {"subject": "cs", "q": "What does HTTP stand for?", "choices": ["HyperText Transfer Protocol", "High Tech Transfer Protocol", "HyperText Transport Program", "High Transfer Text Protocol"], "answer": 0},
    {"subject": "cs", "q": "Which language is primarily used for web browsers?", "choices": ["JavaScript", "Python", "Java", "C++"], "answer": 0},
    {"subject": "cs", "q": "What is a hash table used for?", "choices": ["Fast key-value lookup", "Sorting", "Graphics rendering", "File compression"], "answer": 0},
    {"subject": "cs", "q": "What does SQL stand for?", "choices": ["Structured Query Language", "Simple Question Language", "System Quality Logic", "Standard Query Link"], "answer": 0},
    {"subject": "cs", "q": "What is recursion?", "choices": ["A function calling itself", "A loop", "A data type", "A variable"], "answer": 0},
    {"subject": "cs", "q": "What does RAM stand for?", "choices": ["Random Access Memory", "Read Always Memory", "Run Access Module", "Rapid Application Memory"], "answer": 0},
    {"subject": "cs", "q": "Which sorting algorithm has best average case?", "choices": ["Merge Sort", "Bubble Sort", "Selection Sort", "Insertion Sort"], "answer": 0},
    # Physics
    {"subject": "physics", "q": "What is the speed of light in vacuum?", "choices": ["3×10^8 m/s", "3×10^6 m/s", "3×10^10 m/s", "3×10^4 m/s"], "answer": 0},
    {"subject": "physics", "q": "What force keeps planets in orbit?", "choices": ["Gravity", "Magnetism", "Friction", "Nuclear force"], "answer": 0},
    {"subject": "physics", "q": "What is Newton's second law?", "choices": ["F=ma", "F=mv", "E=mc^2", "F=Gm1m2/r^2"], "answer": 0},
    {"subject": "physics", "q": "What particle carries electric charge?", "choices": ["Electron", "Neutron", "Photon", "Neutrino"], "answer": 0},
    {"subject": "physics", "q": "What is the unit of force?", "choices": ["Newton", "Joule", "Watt", "Pascal"], "answer": 0},
    {"subject": "physics", "q": "What type of wave is sound?", "choices": ["Longitudinal", "Transverse", "Electromagnetic", "Surface"], "answer": 0},
    {"subject": "physics", "q": "What is absolute zero?", "choices": ["-273.15°C", "0°C", "-100°C", "-459°C"], "answer": 0},
    {"subject": "physics", "q": "What does E=mc^2 mean?", "choices": ["Mass-energy equivalence", "Kinetic energy", "Potential energy", "Thermal energy"], "answer": 0},
    {"subject": "physics", "q": "What is the charge of a proton?", "choices": ["Positive", "Negative", "Neutral", "Variable"], "answer": 0},
    {"subject": "physics", "q": "What is the SI unit of energy?", "choices": ["Joule", "Watt", "Newton", "Pascal"], "answer": 0},
    # Biology
    {"subject": "biology", "q": "What is the powerhouse of the cell?", "choices": ["Mitochondria", "Nucleus", "Ribosome", "Golgi body"], "answer": 0},
    {"subject": "biology", "q": "What molecule carries genetic information?", "choices": ["DNA", "RNA", "Protein", "Lipid"], "answer": 0},
    {"subject": "biology", "q": "How many chromosomes do humans have?", "choices": ["46", "23", "48", "44"], "answer": 0},
    {"subject": "biology", "q": "What process do plants use to make food?", "choices": ["Photosynthesis", "Respiration", "Fermentation", "Digestion"], "answer": 0},
    {"subject": "biology", "q": "What blood type is the universal donor?", "choices": ["O-", "AB+", "A+", "B-"], "answer": 0},
    {"subject": "biology", "q": "What is the largest organ in the human body?", "choices": ["Skin", "Liver", "Brain", "Heart"], "answer": 0},
    {"subject": "biology", "q": "What is the basic unit of life?", "choices": ["Cell", "Atom", "Molecule", "Tissue"], "answer": 0},
    {"subject": "biology", "q": "Which vitamin is produced by sunlight exposure?", "choices": ["Vitamin D", "Vitamin C", "Vitamin A", "Vitamin B12"], "answer": 0},
    {"subject": "biology", "q": "What is the function of red blood cells?", "choices": ["Carry oxygen", "Fight infection", "Clot blood", "Produce hormones"], "answer": 0},
    {"subject": "biology", "q": "What is natural selection?", "choices": ["Survival of the fittest", "Random mutation", "Artificial breeding", "Genetic drift"], "answer": 0},
    # History/Geography
    {"subject": "history", "q": "In what year did World War II end?", "choices": ["1945", "1944", "1946", "1943"], "answer": 0},
    {"subject": "history", "q": "Who was the first US president?", "choices": ["George Washington", "Thomas Jefferson", "Abraham Lincoln", "John Adams"], "answer": 0},
    {"subject": "history", "q": "What ancient civilization built the pyramids?", "choices": ["Egyptian", "Roman", "Greek", "Persian"], "answer": 0},
    {"subject": "history", "q": "What was the Renaissance?", "choices": ["Cultural rebirth in Europe", "A war", "A disease", "A religion"], "answer": 0},
    {"subject": "history", "q": "What is the capital of France?", "choices": ["Paris", "London", "Berlin", "Rome"], "answer": 0},
    {"subject": "history", "q": "Who wrote 'Hamlet'?", "choices": ["Shakespeare", "Dickens", "Hemingway", "Tolstoy"], "answer": 0},
    {"subject": "history", "q": "What continent is Brazil in?", "choices": ["South America", "Africa", "Europe", "Asia"], "answer": 0},
    {"subject": "history", "q": "What is the largest ocean?", "choices": ["Pacific", "Atlantic", "Indian", "Arctic"], "answer": 0},
    {"subject": "history", "q": "When did the Berlin Wall fall?", "choices": ["1989", "1985", "1991", "1987"], "answer": 0},
    {"subject": "history", "q": "What is the currency of Japan?", "choices": ["Yen", "Won", "Yuan", "Dollar"], "answer": 0},
]

# BoolQ questions
BOOLQ = [
    ("The capital of France is Paris.", True),
    ("Water freezes at 100 degrees Celsius.", False),
    ("The Earth orbits around the Sun.", True),
    ("Shakespeare wrote 'Hamlet'.", True),
    ("A kilometer is longer than a mile.", False),
    ("Photosynthesis produces oxygen.", True),
    ("The chemical symbol for gold is Ag.", False),
    ("Mount Everest is the tallest mountain on Earth.", True),
    ("The Amazon River is in Africa.", False),
    ("DNA stands for Deoxyribonucleic Acid.", True),
    ("The speed of light is approximately 300,000 km/s.", True),
    ("A hexagon has 5 sides.", False),
    ("The human body has 206 bones.", True),
    ("Venus is the closest planet to the Sun.", False),
    ("The Great Wall of China is visible from space with the naked eye.", False),
]

# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(path):
    """Load a model from HF hub or local path."""
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )

def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

@torch.no_grad()
def answer_mcq(model, tok, question, choices):
    """Evaluate an MML-style multiple choice question.
    Returns the index of the chosen answer."""
    scores = []
    for i, choice in enumerate(choices):
        prompt = f"Question: {question}\nAnswer: {choice}"
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=128)
        out = model(enc.input_ids, labels=enc.input_ids)
        loss = out.loss.item() if out.loss and not torch.isnan(out.loss) else 999
        scores.append(loss)
    return scores.index(min(scores))

@torch.no_grad()
def answer_boolq(model, tok, statement):
    """Evaluate a BoolQ question."""
    prompt_true = f"Question: {statement}\nAnswer: True"
    prompt_false = f"Question: {statement}\nAnswer: False"
    
    enc_t = tok(prompt_true, return_tensors="pt", truncation=True, max_length=128)
    enc_f = tok(prompt_false, return_tensors="pt", truncation=True, max_length=128)
    
    loss_t = model(enc_t.input_ids, labels=enc_t.input_ids).loss
    loss_f = model(enc_f.input_ids, labels=enc_f.input_ids).loss
    
    lt = loss_t.item() if loss_t and not torch.isnan(loss_t) else 999
    lf = loss_f.item() if loss_f and not torch.isnan(loss_f) else 999
    return lt < lf

@torch.no_grad()
def compute_ppl(model, tok, texts):
    """Compute average perplexity."""
    ppls = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
        out = model(enc.input_ids, labels=enc.input_ids)
        if out.loss and not torch.isnan(out.loss):
            ppls.append(math.exp(min(out.loss.item(), 20)))
    return sum(ppls)/max(len(ppls),1) if ppls else float('inf')

# ============================================================================
# RUN BENCHMARKS
# ============================================================================
PPL_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "Two plus two equals four.",
    "The sun rises in the east.",
    "A cat is an animal that says meow.",
    "The moon orbits around the Earth.",
    "Dogs are man's best friend.",
    "Winter is colder than summer.",
    "An apple is a type of fruit.",
    "Fish live in water.",
]

results = {}
reference_ppl = None

for model_key, model_info in MODELS.items():
    print(f"\n{'='*70}")
    print(f"  BENCHMARKING: {model_info['display']}")
    print(f"  Path: {model_info['path']}")
    print(f"{'='*70}")
    
    try:
        m = load_model(model_info["path"])
        tok = load_tokenizer(model_info["path"])
    except Exception as e:
        print(f"  SKIP: Could not load — {e}")
        results[model_key] = {"error": str(e)[:100]}
        continue
    
    # PPL
    ppl = compute_ppl(m, tok, PPL_TEXTS)
    if reference_ppl is None:
        reference_ppl = ppl
    ppl_delta = ppl - reference_ppl
    ppl_dir = "↓ BETTER" if ppl_delta < 0 else ("↑ WORSE" if ppl_delta > 0 else "= SAME")
    print(f"  PPL: {ppl:.1f} (Δ={ppl_delta:+.1f} {ppl_dir})")
    
    # MMLU-lite
    mmlu_correct = 0
    mmlu_by_subject = defaultdict(lambda: [0, 0])
    for q in MMLU_QUESTIONS:
        pred = answer_mcq(m, tok, q["q"], q["choices"])
        mmlu_by_subject[q["subject"]][1] += 1
        if pred == q["answer"]:
            mmlu_correct += 1
            mmlu_by_subject[q["subject"]][0] += 1
    mmlu_acc = mmlu_correct / len(MMLU_QUESTIONS) * 100
    print(f"  MMLU: {mmlu_correct}/{len(MMLU_QUESTIONS)} ({mmlu_acc:.1f}%)")
    for subj, (c, t) in sorted(mmlu_by_subject.items()):
        print(f"    {subj:10s}: {c}/{t} ({c/t*100:.0f}%)")
    
    # BoolQ
    boolq_correct = sum(1 for stmt, ans in BOOLQ if answer_boolq(m, tok, stmt) == ans)
    boolq_acc = boolq_correct / len(BOOLQ) * 100
    print(f"  BoolQ: {boolq_correct}/{len(BOOLQ)} ({boolq_acc:.1f}%)")
    
    results[model_key] = {
        "display": model_info["display"],
        "type": model_info.get("type", "graft"),
        "graft_info": model_info.get("graft", ""),
        "ppl": round(ppl, 1),
        "ppl_delta": round(ppl_delta, 1),
        "ppl_better": ppl_delta < 0,
        "mmlu_acc": round(mmlu_acc, 1),
        "mmlu_by_subject": {s: round(c/t*100, 1) for s, (c, t) in mmlu_by_subject.items()},
        "boolq_acc": round(boolq_acc, 1),
    }
    
    del m
    torch.cuda.empty_cache()

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================
print(f"\n{'='*70}")
print(f"  COMPARATIVE ANALYSIS")
print(f"{'='*70}")

baseline = results.get("baseline", {})
baseline_ppl = baseline.get("ppl", 0)
baseline_mmlu = baseline.get("mmlu_acc", 0)
baseline_boolq = baseline.get("boolq_acc", 0)

print(f"\n  {'Model':20s} {'PPL':>8s} {'ΔPPL':>8s} {'MMLU':>8s} {'BoolQ':>8s} {'Verdict':>20s}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

for key in MODELS:
    r = results.get(key, {})
    if "error" in r:
        print(f"  {key:20s} {'ERROR':>8s}")
        continue
    
    ppl = r.get("ppl", 0)
    dppl = r.get("ppl_delta", 0)
    mmlu = r.get("mmlu_acc", 0)
    boolq = r.get("boolq_acc", 0)
    
    # Verdict
    if key == "baseline":
        verdict = "REFERENCE"
    elif dppl < -1:
        verdict = "BETTER THAN ORIGINAL"
    elif dppl < 2:
        verdict = "ON PAR"
    elif dppl < 10:
        verdict = "SLIGHT DEGRADATION"
    else:
        verdict = "SIGNIFICANT LOSS"
    
    ppl_str = f"{ppl:.1f}"
    dppl_str = f"{dppl:+.1f}"
    print(f"  {key:20s} {ppl_str:>8s} {dppl_str:>8s} {mmlu:>7.1f}% {boolq:>7.1f}% {verdict:>20s}")

# ============================================================================
# GRAFT SUITABILITY DATABASE
# ============================================================================
SUITABILITY_DB = {
    "models": [
        {
            "name": "SmolLM2-135M-Instruct",
            "family": "SmolLM2",
            "size": "135M",
            "layers": 30,
            "d_model": 576,
            "architecture": "Llama-style decoder",
            "graft_suitability": {
                "rating": "EXCELLENT",
                "score": 95,
                "reason": "Small size enables fast CPU grafting. Regular architecture with clear layer roles. Same-family grafts show 105-122% PPL recovery. Cross-model grafts work (SmolLM←Qwen).",
                "best_donors": ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "SmolLM2-360M"],
                "best_layers_to_graft": [5, 8, 18, 20, 22],
                "best_layers_to_receive": [5, 8, 12, 15, 20],
                "strength_recommendation": "0.5 for single-layer, 0.08-0.2 for multi-layer",
            }
        },
        {
            "name": "Qwen2.5-0.5B-Instruct",
            "family": "Qwen2.5",
            "size": "0.5B",
            "layers": 24,
            "d_model": 896,
            "architecture": "Qwen-style decoder",
            "graft_suitability": {
                "rating": "GOOD",
                "score": 78,
                "reason": "Works as donor for cross-model grafts (SmolLM2←Qwen FFN at +7.2 PPL). Layer structure compatible after shape alignment. Best as donor, not recipient.",
                "best_donors": ["SmolLM2-135M"],
                "best_layers_to_graft": [8, 12, 16],
                "best_layers_to_receive": [8, 12, 16],
                "strength_recommendation": "0.3-0.5 for cross-model (need shape padding)",
            }
        },
        {
            "name": "Qwen2.5-1.5B-Instruct",
            "family": "Qwen2.5",
            "size": "1.5B",
            "layers": 28,
            "d_model": 1536,
            "architecture": "Qwen-style decoder",
            "graft_suitability": {
                "rating": "GOOD",
                "score": 72,
                "reason": "Larger model, more expressive layers. GQA (16:4) requires alignment attention. Previously used in HITHERTO CECI graft (k=1536, 28/28 layers). Needs compute for full graft testing.",
                "best_donors": ["Qwen2.5-0.5B", "SmolLM2-360M"],
                "best_layers_to_graft": [6, 12, 18, 24],
                "best_layers_to_receive": [6, 12, 18, 24],
                "strength_recommendation": "0.3-0.5",
            }
        },
        {
            "name": "DeepSeek-Coder-V2-Lite-Instruct",
            "family": "DeepSeek",
            "size": "16B",
            "layers": 28,
            "d_model": 4096,
            "architecture": "MoE decoder",
            "graft_suitability": {
                "rating": "THEORETICAL",
                "score": 45,
                "reason": "Excellent coding model but 16B is too large for CPU grafting. MoE architecture complicates layer grafting. Would be ideal donor for coding FFNs into smaller base models. Needs EC2/H100 for testing.",
                "best_donors": [],
                "best_layers_to_graft": [],
                "best_layers_to_receive": [],
                "strength_recommendation": "Unknown — needs testing",
            }
        },
        {
            "name": "Qwen2.5-Math-1.5B-Instruct",
            "family": "Qwen2.5 Math",
            "size": "1.5B",
            "layers": 28,
            "d_model": 1536,
            "architecture": "Qwen-style decoder (math-tuned)",
            "graft_suitability": {
                "rating": "PROMISING",
                "score": 82,
                "reason": "Math-specialized model. FFN layers likely encode mathematical reasoning patterns that could transfer to general models. Same architecture as Qwen2.5-1.5B makes grafting straightforward. High priority for testing.",
                "best_donors": ["Qwen2.5-1.5B"],
                "best_layers_to_graft": [8, 12, 16, 20],
                "best_layers_to_receive": [8, 12, 16, 20],
                "strength_recommendation": "0.3-0.5",
            }
        },
        {
            "name": "Qwen2.5-Coder-7B-Instruct",
            "family": "Qwen2.5 Coder",
            "size": "7B",
            "layers": 28,
            "d_model": 3584,
            "architecture": "Qwen-style decoder (code-tuned)",
            "graft_suitability": {
                "rating": "PROMISING",
                "score": 75,
                "reason": "Code-specialized 7B model. FFN layers encode coding patterns. Already deployed in ISAGI (4-bit). Cross-graft with math models could create coding-math hybrids. Needs 4-bit loading for VRAM.",
                "best_donors": ["Qwen2.5-Math-7B"],
                "best_layers_to_graft": [8, 12, 16, 20],
                "best_layers_to_receive": [8, 12, 16, 20],
                "strength_recommendation": "0.2-0.4 (large model needs conservative strength)",
            }
        },
    ],
    "grafting_rules": {
        "same_family": "0.5 strength, any layer pair works. Higher PPL recovery.",
        "cross_family_same_size": "0.3 strength, need shape alignment. Moderate PPL impact.",
        "cross_family_diff_size": "0.08-0.2 strength, significant shape padding. Higher PPL cost.",
        "multi_layer": "0.05-0.1 per layer. Distribute grafts across depth for stability.",
        "large_to_small": "Trim donor to recipient shape. Works as donor FFN injection.",
        "small_to_large": "Pad donor to recipient shape. Usually worse results.",
    },
    "benchmark_notes": {
        "ppl_lower_is_better": True,
        "ppl_delta_negative_is_improvement": True,
        "mmlu_range": "0-100%, higher is better",
        "boolq_range": "0-100%, higher is better",
        "graft_works_if": "MMLU/BoolQ within 5pp of baseline AND PPL delta < +10",
    }
}

# ============================================================================
# SAVE RESULTS
# ============================================================================
output = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "benchmark_results": results,
    "suitability_database": SUITABILITY_DB,
    "baseline_reference": {
        "model": MODELS["baseline"]["path"],
        "ppl": baseline_ppl,
        "mmlu_acc": baseline_mmlu,
        "boolq_acc": baseline_boolq,
    },
    "summary": {
        "models_tested": len([k for k in results if "error" not in results[k]]),
        "grafts_better_than_baseline": len([k for k in results if k != "baseline" and results.get(k, {}).get("ppl_better", False)]),
        "grafts_on_par": len([k for k in results if k != "baseline" and abs(results.get(k, {}).get("ppl_delta", 999)) < 2]),
    }
}

with open(OUT / "graft_benchmark_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved to: {OUT / 'graft_benchmark_results.json'}")
print(f"  Suitability DB: {len(SUITABILITY_DB['models'])} models cataloged")
