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

"""Kaisen vs. REAL problems — things we genuinely need to figure out."""
import torch, sys, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from instinct_horizon import InstinctHorizon

torch.manual_seed(42); torch.set_grad_enabled(False)
M="Qwen/Qwen2.5-1.5B-Instruct"; DEV="cuda"; K=32

print("="*60)
print("  KAISEN vs. REAL PROBLEMS")
print(f"  Model: {M}")
print("="*60)

# Load
print("[load]")
model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True)
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True); tok.pad_token=tok.eos_token
d=model.config.hidden_size

def hs(t):
    enc=tok(t,return_tensors="pt",truncation=True,max_length=128).to(DEV)
    with torch.no_grad(): o=model(**enc,output_hidden_states=True)
    return o.hidden_states[-1][0,-1,:].float()

def gen(p,mt=128):
    enc=tok(p,return_tensors="pt",truncation=True,max_length=512).to(DEV)
    np=enc.input_ids.shape[1]
    with torch.no_grad(): o=model.generate(**enc,max_new_tokens=mt,do_sample=True,temperature=0.7,top_p=0.9,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0,np:],skip_special_tokens=True).strip()

# Build a broad manifold: 50 diverse facts
facts=[f"{i}. The capital of France is Paris." for i in range(5)]
facts+=[f"{i}. Water freezes at 0 Celsius and boils at 100." for i in range(5)]
facts+=[f"{i}. The speed of light is approximately 300 million meters per second." for i in range(5)]
facts+=[f"{i}. The Pythagorean theorem states a^2+b^2=c^2." for i in range(5)]
facts+=[f"{i}. DNA carries genetic information in a double helix." for i in range(5)]
facts+=[f"{i}. Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet." for i in range(5)]
facts+=[f"{i}. Newton's second law: force equals mass times acceleration." for i in range(5)]
facts+=[f"{i}. The derivative of x^n is n*x^(n-1)." for i in range(5)]
facts+=[f"{i}. Photosynthesis: CO2 + H2O + light -> glucose + O2." for i in range(5)]
facts+=[f"{i}. Python was created by Guido van Rossum in 1991." for i in range(5)]

hs_stack=torch.stack([hs(f) for f in facts[:50]])
hs_c=hs_stack-hs_stack.mean(0); U,S,_=torch.linalg.svd(hs_c.T,full_matrices=False)
basis=U[:,:K].float().to(DEV)
def to_k(h): return (h@basis).cpu()

trajs=[{"proj":to_k(hs(f)),"label":f} for f in facts[:50]]
jury=InstinctHorizon(creativity=0.3); jury.calibrate(trajs)
print(f"  Manifold: {len(trajs)} facts | k={K} | R={jury.R:.2f} | d_h={jury.instinct_horizon_distance:.2f}")

# REAL PROBLEMS — things the model must genuinely figure out
# Each has a verifiable answer we can check
PROBLEMS=[
    ("MATH","Find all integer solutions to x^2 + y^2 = 25 where x and y are positive integers.",
     "Answer: (3,4) and (4,3). Verify by checking both pairs."),
    
    ("LOGIC","There are 5 houses in a row, each of a different color. The green house is immediately to the left of the white house. The red house is in the middle. Where is the blue house?",
     "Verify by checking all constraints."),
    
    ("PATTERN","Continue this sequence: 1, 1, 2, 3, 5, 8, 13, ? What is the next number and what is the pattern?",
     "Answer: 21 (Fibonacci). Verify by checking rule: each term is sum of previous two."),
    
    ("ESTIMATION","If I fold a piece of paper 0.1mm thick in half 42 times, approximately how thick would it be? Show your work.",
     "Answer: 0.1mm * 2^42 ≈ 0.1 * 4.398e12 ≈ 439,800 km. Verify exponentiation."),
]

for title, problem, check in PROBLEMS:
    print(f"\n{'='*60}")
    print(f"  PROBLEM: {title}")
    print(f"  {'='*60}")
    print(f"  Q: {problem[:100]}...")
    
    # Measure pre-scout J
    J_pre,_=jury.jury_confidence(to_k(hs(problem)))
    in_out="INSIDE" if J_pre>=jury.J_threshold else "OUTSIDE"
    print(f"  J_before: {J_pre:.3f} ({in_out} manifold)")
    
    # Kaisen scout prompt — push beyond known territory
    scout_prompt=f"""You are an expert problem solver. Solve this step by step. Show ALL your work.

{problem}

Think carefully. If you're not sure, reason through it logically.
End with: ANSWER: [your final answer]"""
    
    response=gen(scout_prompt,mt=200)
    
    # Measure post-scout J
    J_post,_=jury.jury_confidence(to_k(hs(response[:300])))
    delta=J_post-J_pre
    arrow="DOWN (novel territory)" if delta<-0.05 else ("UP (familiar)" if delta>0.05 else "SAME")
    
    print(f"\n  RESPONSE:")
    for line in response.split('\n')[:12]:
        print(f"    {line[:100]}")
    if len(response.split('\n'))>12:
        print(f"    ... ({len(response.split('\n'))-12} more lines)")
    
    print(f"\n  J_after: {J_post:.3f} (Δ={delta:+.3f} {arrow})")
    
    # Verification: ask the model to check its own answer
    verify_prompt=f"Here is a solution to a problem:\n\n{response[:400]}\n\nCheck this solution. The verification criteria: {check}\n\nIs the solution correct? Answer YES or NO, and explain why."
    verification=gen(verify_prompt,mt=80)
    is_correct="YES" in verification.upper().split('\n')[0] if verification else False
    
    print(f"  Self-verify: {'CORRECT' if is_correct else 'NEEDS WORK'} — {verification[:120]}...")
    
    # Expand if novel
    if J_post<jury.J_threshold:
        trajs.append({"proj":to_k(hs(response[:300])),"label":f"kaisen:{title}"})
        print(f"  [EXPAND] Frontier advanced — {len(trajs)} trajectories")

# Recalibrate
jury2=InstinctHorizon(creativity=0.3); jury2.calibrate(trajs)
print(f"\n{'='*60}")
print(f"  FINAL: R={jury.R:.2f} -> {jury2.R:.2f} | d_h={jury.instinct_horizon_distance:.2f} -> {jury2.instinct_horizon_distance:.2f}")
print(f"  Trajectories: {len(trajs)} (was 50)")
print(f"{'='*60}")
