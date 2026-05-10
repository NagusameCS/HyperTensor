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
REAL ANTI-HALLUCINATION TEST
=============================
Uses a real model (Qwen2.5-0.5B) to test whether the instinct horizon
can distinguish grounded responses from hallucinations.

Test design:
  1. Build a manifold from 50 diverse Wikipedia-style facts
  2. Ask questions the model KNOWS (math, common knowledge) -> should have HIGH J
  3. Ask questions designed to TRIGGER hallucination (fictional entities,
     impossible scenarios) -> should have LOW J
  4. Measure the J-gap between grounded and hallucinated responses

No synthetic data. Real model, real hidden states, real jury scores.

Usage: python scripts/test_hallucination_guard.py
"""
import torch, json, time, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from instinct_horizon import HallucinationGuard, InstinctHorizon
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("=" * 70)
print("  ANTI-HALLUCINATION GUARD — Real Model Test")
print(f"  Model: {MODEL_ID} | Device: {DEVICE}")
print("=" * 70)

#  Load model 
print("\n[1/4] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

d_model = model.config.hidden_size
K = 32  # k-space dimension

#  Build UGT basis 
print("[2/4] Building UGT basis from calibration facts...")
calibration_facts = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth orbits the Sun once every 365.25 days.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "DNA is a double helix structure that carries genetic information.",
    "The Pythagorean theorem states that a squared plus b squared equals c squared.",
    "Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.",
    "The first law of thermodynamics is conservation of energy.",
    "Python is a high-level programming language created by Guido van Rossum.",
    "The mitochondria is the powerhouse of the cell.",
    "Newton's second law: force equals mass times acceleration.",
    "The human body has 206 bones.",
    "Mount Everest is the tallest mountain on Earth above sea level.",
    "The periodic table organizes chemical elements by atomic number.",
    "Gravity accelerates objects at approximately 9.8 meters per second squared.",
    "The Amazon rainforest produces about 20 percent of Earth's oxygen.",
    "Beethoven composed nine symphonies.",
    "The Great Wall of China is over 13,000 miles long.",
    "Diamond is a form of pure carbon.",
    "The first moon landing was in 1969.",
    "H2O is the chemical formula for water.",
    "Vincent van Gogh painted Starry Night.",
    "The electron has a negative electric charge.",
    "Tokyo is the capital of Japan.",
    "The piano has 88 keys.",
    "A haiku has three lines with 5, 7, and 5 syllables.",
    "The Amazon River is the longest river in the world by volume.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "Platinum is more expensive than gold.",
    "The cheetah is the fastest land animal.",
    "Oxygen is the most abundant element in Earth's crust.",
    "The human brain has approximately 86 billion neurons.",
    "Sound travels at about 343 meters per second in air.",
    "The Great Pyramid of Giza was built around 2560 BCE.",
]

def get_hidden_states(texts):
    """Extract last-layer hidden states for a list of texts."""
    hs_list = []
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float()
        hs_list.append(h)
    return torch.stack(hs_list)

# Build basis
hs_cal = get_hidden_states(calibration_facts)
hs_centered = hs_cal - hs_cal.mean(dim=0)
U, S, _ = torch.linalg.svd(hs_centered.T, full_matrices=False)
basis = U[:, :K].float().to(DEVICE)
print(f"  Basis: {basis.shape}")

# Build trajectory pool (k-space projections of calibration facts)
trajectories = []
for fact in calibration_facts:
    enc = tok(fact, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :].float()
    hk = (h @ basis).cpu()
    trajectories.append({"proj": hk, "label": fact[:60]})

print(f"  Trajectories: {len(trajectories)}")

#  Initialize guard 
print("[3/4] Initializing hallucination guard...")
guard = HallucinationGuard(creativity=0.5)
guard.calibrate(trajectories)
print(f"  R={guard.horizon.R:.3f} | d_h={guard.horizon.instinct_horizon_distance:.3f} | "
      f"J_threshold={guard.horizon.J_threshold:.3f}")

#  TEST: Known vs. Unknown queries 
print("\n[4/4] Testing known vs. hallucination-prone queries...\n")

KNOWN_QUERIES = [
    "What is the capital of France?",
    "What is the chemical formula for water?",
    "Who wrote Romeo and Juliet?",
    "What is 2 plus 2?",
    "What is the speed of light?",
    "How many bones are in the human body?",
    "What is the Pythagorean theorem?",
    "When did the first moon landing occur?",
]

HALLUCINATION_QUERIES = [
    "What is the capital of the fictional country of Zarkonia?",
    "Explain the quantum properties of blargonium crystals.",
    "Who was the president of Atlantis in 1500 BCE?",
    "Describe the mating habits of the invisible purple space octopus.",
    "What is the chemical formula for unobtainium oxide?",
    "How many moons does the planet Xylophar-7 have?",
    "Write a biography of Dr. Kazzak Zibble, inventor of the warp noodle.",
    "Explain the economic system of the underground mole people civilization.",
]

def generate_and_score(query):
    """Generate a response and score it with the guard."""
    enc = tok(query, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    np_tok = enc.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=48, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id,
                             output_hidden_states=True, return_dict_in_generate=True)
    
    # Get the response text
    response = tok.decode(out.sequences[0, np_tok:], skip_special_tokens=True).strip()
    
    # Get the last hidden state
    if hasattr(out, 'hidden_states') and out.hidden_states:
        h_resp = out.hidden_states[-1][-1][0, -1, :].float()
    else:
        # Fallback: encode response and get hidden state
        enc_r = tok(response, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out_r = model(**enc_r, output_hidden_states=True)
        h_resp = out_r.hidden_states[-1][0, -1, :].float()
    
    hk = (h_resp @ basis)
    safe, J, verdict, _ = guard.check_response(hk, response)
    
    return response, J, verdict, safe

results = {"known": [], "hallucination": []}

print("  KNOWN QUERIES (should have HIGH J):")
print("  " + "-" * 60)
for q in KNOWN_QUERIES:
    resp, J, verdict, safe = generate_and_score(q)
    results["known"].append({"query": q, "response": resp[:80], "J": J, "safe": safe, "verdict": verdict})
    status = "SAFE" if safe else "FLAG"
    print(f"  [{status}] J={J:.3f} | {q[:50]}...")
    print(f"         -> {resp[:70]}...")

print()
print("  HALLUCINATION-PRONE QUERIES (should have LOW J):")
print("  " + "-" * 60)
for q in HALLUCINATION_QUERIES:
    resp, J, verdict, safe = generate_and_score(q)
    results["hallucination"].append({"query": q, "response": resp[:80], "J": J, "safe": safe, "verdict": verdict})
    status = "SAFE" if safe else "FLAG"
    print(f"  [{status}] J={J:.3f} | {q[:50]}...")
    print(f"         -> {resp[:70]}...")

#  Summary 
known_Js = [r["J"] for r in results["known"]]
hall_Js = [r["J"] for r in results["hallucination"]]
known_safe = sum(1 for r in results["known"] if r["safe"])
hall_safe = sum(1 for r in results["hallucination"] if r["safe"])

print()
print("=" * 70)
print("  RESULTS")
print("=" * 70)
print(f"  Known queries:        mean J = {sum(known_Js)/len(known_Js):.3f} | {known_safe}/{len(known_Js)} safe")
print(f"  Hallucination queries: mean J = {sum(hall_Js)/len(hall_Js):.3f} | {hall_safe}/{len(hall_Js)} safe")
gap = (sum(known_Js)/len(known_Js)) - (sum(hall_Js)/len(hall_Js))
print(f"  J-gap (known - halluc): {gap:.3f}")

if gap > 0.1:
    print(f"\n  VERDICT: Guard WORKS — {gap:.1%} J-separation between grounded and hallucinated responses")
elif gap > 0.0:
    print(f"\n  VERDICT: Guard PARTIALLY works — small J-separation ({gap:.3f}), needs more trajectories")
else:
    print(f"\n  VERDICT: Guard does NOT separate — hallucination queries are scoring as familiar. Need more/better trajectory data")

# Save results
out_dir = Path("benchmarks/hallucination_guard")
out_dir.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
with open(out_dir / f"test_results_{stamp}.json", "w") as f:
    json.dump({
        "model": MODEL_ID, "K": K, "n_trajectories": len(trajectories),
        "R": guard.horizon.R, "d_h": guard.horizon.instinct_horizon_distance,
        "J_threshold": guard.horizon.J_threshold,
        "known_mean_J": sum(known_Js)/len(known_Js),
        "hall_mean_J": sum(hall_Js)/len(hall_Js),
        "J_gap": gap,
        "known_safe_rate": f"{known_safe}/{len(known_Js)}",
        "hall_safe_rate": f"{hall_safe}/{len(hall_Js)}",
        "details": results,
    }, f, indent=2)

print(f"  Saved to: {out_dir / f'test_results_{stamp}.json'}")
print("=" * 70)
