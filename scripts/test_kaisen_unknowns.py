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

"""Test Kaisen on genuinely unanswerable questions. Real model, real output."""
import torch, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from instinct_horizon import InstinctHorizon

torch.manual_seed(42); torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
M = "Qwen/Qwen2.5-0.5B-Instruct"; K = 16

print("="*55)
print("  KAISEN: UNKNOWNS TARGETING TEST")
print(f"  Model: {M}")
print("="*55)

# Load
print("\n[load]"); model = AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(M,trust_remote_code=True); tok.pad_token=tok.eos_token
d=model.config.hidden_size

def h_state(text):
    enc=tok(text,return_tensors="pt",truncation=True,max_length=64).to(DEV)
    with torch.no_grad(): out=model(**enc,output_hidden_states=True)
    return out.hidden_states[-1][0,-1,:].float()

# Build manifold: 40 known facts
facts=[
    "Paris is the capital of France.","Water boils at 100 Celsius.",
    "The Earth orbits the Sun.","Photosynthesis produces oxygen.",
    "Light speed is 299792458 m/s.","DNA is a double helix.",
    "F=ma is Newton's second law.","206 bones in human body.",
    "Gravity accelerates at 9.8 m/s^2.","Amazon is the largest rainforest.",
    "Beethoven wrote 9 symphonies.","Diamond is pure carbon.",
    "Moon landing was 1969.","H2O is water.",
    "Electrons have negative charge.","Tokyo is Japan's capital.",
    "Cheetah is fastest land animal.","Oxygen most abundant in crust.",
    "Great Wall is 13000+ miles.","Piano has 88 keys.",
    "Haiku is 5-7-5 syllables.","Mona Lisa by da Vinci.",
    "Mitochondria is cell powerhouse.","Integral of x is x^2/2.",
    "Python created by Guido van Rossum.","Mount Everest tallest above sea.",
    "First law of thermodynamics: conservation.","Germanium is element 32.",
    "Shakespeare wrote 37 plays.","Venus is the hottest planet.",
    "Diamond is hardest natural substance.","Dolphins use echolocation.",
    "First computer bug was a moth.","Octopuses have 3 hearts.",
    "Bananas are technically berries.","Honey never spoils.",
    "Lightning is 5x hotter than sun surface.","A day on Venus > year on Venus.",
]
hs=torch.stack([h_state(f) for f in facts])
hs_c=hs-hs.mean(0); U,S,_=torch.linalg.svd(hs_c.T,full_matrices=False)
basis=U[:,:K].float().to(DEV)
def to_k(h): return (h@basis).cpu()

trajs=[{"proj":to_k(h_state(f)),"label":f} for f in facts]
jury=InstinctHorizon(creativity=0.3); jury.calibrate(trajs)
print(f"  Manifold: {len(facts)} facts | k={K} | R={jury.R:.2f} | d_h={jury.instinct_horizon_distance:.2f}")

#  Test questions 
tests=[
    ("KNOWN","What is the capital of France?"),
    ("KNOWN","Explain Newton's second law."),
    ("KNOWN","What is the speed of light?"),
    ("UNKNOWN","What is the capital of the fictional country of Zarkonia?"),
    ("UNKNOWN","How many moons does the planet Xylophar-7 have?"),
    ("UNKNOWN","Describe the biology of invisible purple space octopuses."),
    ("UNKNOWN","What is the chemical formula for blargonium oxide?"),
    ("UNKNOWN","Who was president of Atlantis in 1500 BCE?"),
]

def generate(prompt, mt=48):
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=256).to(DEV)
    np=enc.input_ids.shape[1]
    with torch.no_grad():
        out=model.generate(**enc,max_new_tokens=mt,do_sample=True,temperature=0.7,top_p=0.9,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,np:],skip_special_tokens=True).strip()

print(f"\n{'Type':<9s} {'J_before':>8s} {'J_after':>8s} {'Verdict':>22s} {'Exp?':>4s} Query")
print("-"*75)

stats={"KNOWN":{"J_before":[],"J_after":[],"expanded":[]},
       "UNKNOWN":{"J_before":[],"J_after":[],"expanded":[]}}

for typ,q in tests:
    # Before: measure J for the query
    J_b,_=jury.jury_confidence(to_k(h_state(q)))
    
    # Generate response
    resp=generate(q)
    
    # After: measure J for the response
    J_a,_=jury.jury_confidence(to_k(h_state(resp[:200])))
    
    # Should Kaisen expand? (response outside horizon = novel ground)
    novel=J_a<jury.J_threshold
    if novel: trajs.append({"proj":to_k(h_state(resp[:200])),"label":f"scout:{q[:40]}"})
    
    verdict="FAMILIAR" if J_a>=0.85 else("HORIZON" if J_a>=0.5 else("SCOUT ZONE" if J_a>=0.25 else("EXPLORE" if J_a>=0.05 else"DEEP VOID")))
    
    print(f"{typ:<9s} {J_b:8.4f} {J_a:8.4f} {verdict:>22s} {'+' if novel else '':>4s} {q[:52]}")
    
    stats[typ]["J_before"].append(J_b)
    stats[typ]["J_after"].append(J_a)
    stats[typ]["expanded"].append(novel)

# Recalibrate
jury2=InstinctHorizon(creativity=0.3); jury2.calibrate(trajs)

print(f"\n{'='*55}")
print(f"  RESULTS")
print(f"{'='*55}")
for typ in["KNOWN","UNKNOWN"]:
    s=stats[typ]; n=len(s["J_before"])
    if n:
        print(f"  {typ}: J_before={sum(s['J_before'])/n:.3f} -> J_after={sum(s['J_after'])/n:.3f} | expanded={sum(s['expanded'])}/{n}")
print(f"  Trajectories: {len(trajs)} (was {len(facts)})")
print(f"  R: {jury.R:.3f} -> {jury2.R:.3f}")
print(f"  d_h: {jury.instinct_horizon_distance:.3f} -> {jury2.instinct_horizon_distance:.3f}")
print(f"  KAISEN {'EXPANDED THE FRONTIER' if len(trajs)>len(facts) else 'DID NOT EXPAND'}")
print(f"{'='*55}")
