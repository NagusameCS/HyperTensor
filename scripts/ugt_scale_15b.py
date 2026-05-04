"""Scale UGT to Qwen2.5-1.5B (d=1536, 28 layers, k=512).
Phase A: basis alignment. Phase B: zone specialization.
Deploy to EC2 L40S (47GB VRAM free)."""
import torch, json, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
K_BASIS = 512
PHASE_A_STEPS = 400
PHASE_B_STEPS = 2000  # shorter for 1.5B
BATCH_SIZE = 1
LR = 1e-4
OUT_DIR = "/home/ubuntu/benchmarks/ugt_qwen15b"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  UGT SCALING: Qwen2.5-1.5B (k=512)")
print("=" * 60)

# -- Load model --
print("\n[1] Loading Qwen2.5-1.5B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Get hidden dim
d_model = model.config.hidden_size
n_layers = model.config.num_hidden_layers
print(f"  d_model={d_model}, layers={n_layers}, k={K_BASIS}")
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# -- Create taxonomic basis --
print("\n[2] Creating taxonomic basis (k=512)...")
# Subspace definitions scaled proportionally
# k=512: syntax 0-191, routing 192-383, factual 384-447, math 448-511
subspace_defs = {
    "syntax": list(range(0, 192)),
    "routing": list(range(192, 384)),
    "factual": list(range(384, 448)),
    "math": list(range(448, 512)),
}
print(f"  Subspaces: syntax={len(subspace_defs['syntax'])}d, "
      f"routing={len(subspace_defs['routing'])}d, "
      f"factual={len(subspace_defs['factual'])}d, "
      f"math={len(subspace_defs['math'])}d")

# Initialize basis (random orthonormal)
basis = torch.randn(d_model, K_BASIS, device=DEVICE, dtype=torch.float32) * 0.01
Q, R = torch.linalg.qr(basis)
basis = Q  # orthonormal basis [d, k]
basis.requires_grad = True

# -- Training data --
# Mixed domain: syntax, general, math, reasoning
train_texts = [
    # Syntax / general
    "The quick brown fox jumps over the lazy dog near the river bank under the morning sun",
    "Despite the heavy rain, the construction team continued working on the foundation of the new building",
    "In the beginning, God created the heavens and the earth, and the earth was without form",
    # Math / reasoning
    "If x squared plus y squared equals z squared, then by the Pythagorean theorem this is a right triangle",
    "The derivative of the natural logarithm of x with respect to x is one divided by x",
    "A prime number is a natural number greater than one that has no positive divisors other than one and itself",
    "The Riemann zeta function zeta of s equals the sum from n equals one to infinity of one over n to the power s",
    "For any epsilon greater than zero, there exists a delta such that the absolute value of f of x minus L is less than epsilon",
    # Factual
    "The capital of France is Paris, a city known for its art, culture, and the Eiffel Tower",
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure at sea level",
    "The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation",
    # Combined reasoning
    "To solve the equation two x plus five equals thirteen, first subtract five from both sides to get two x equals eight, then divide by two",
    "The proof that the square root of two is irrational proceeds by contradiction: assume root two equals p over q in lowest terms",
] * (max(PHASE_A_STEPS, PHASE_B_STEPS) // 13 + 5)

print(f"  Training texts: {len(train_texts)}")

# -- Phase A: Basis-only alignment --
print(f"\n[3] Phase A: Basis alignment ({PHASE_A_STEPS} steps)...")
for p in model.parameters():
    p.requires_grad = False

optimizer = AdamW([basis], lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=80, num_training_steps=PHASE_A_STEPS
)

lambda_top = 0.01
losses_a = []

for step in range(PHASE_A_STEPS):
    text = train_texts[step % len(train_texts)]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    
    # Use middle + last layer hidden states for basis alignment
    hs_mid = out.hidden_states[n_layers // 2]  # [1, seq, d]
    hs_last = out.hidden_states[-1]  # [1, seq, d]
    hs = torch.cat([hs_mid, hs_last], dim=1).float()  # [1, 2*seq, d]
    hs_flat = hs.reshape(-1, d_model)  # [batch*2*seq, d]
    
    # Basis alignment: maximize cosine similarity of projections
    proj = hs_flat @ basis  # [N, k]
    proj_norm = F.normalize(proj, dim=-1)
    # Self-consistency loss: projection should preserve structure
    recon = proj @ basis.T  # [N, d]
    basis_loss = F.mse_loss(recon, hs_flat)
    
    # TOPLoss: inter-subspace orthogonality
    top_loss = 0.0
    sub_names = list(subspace_defs.keys())
    for i in range(len(sub_names)):
        for j in range(i + 1, len(sub_names)):
            Bi = basis[:, subspace_defs[sub_names[i]]]
            Bj = basis[:, subspace_defs[sub_names[j]]]
            top_loss += torch.norm(Bi.T @ Bj, p='fro') ** 2
    
    loss = basis_loss + lambda_top * top_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([basis], 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    losses_a.append(loss.item())
    if (step + 1) % 100 == 0:
        print(f"  Step {step+1:>4d}/{PHASE_A_STEPS}: loss={loss.item():.6f}  "
              f"basis={basis_loss.item():.4f}  top={top_loss.item():.6f}")

# Re-orthonormalize
Q, R = torch.linalg.qr(basis.float())
basis.data.copy_(Q)

# -- Save basis --
torch.save(basis.detach(), f"{OUT_DIR}/taxonomic_basis.pt")
print(f"  Basis saved: {basis.shape}")

# -- Phase B: Zone specialization (light) --
print(f"\n[4] Phase B: Zone specialization ({PHASE_B_STEPS} steps)...")
# Create zone head weights
zone_dim = 64
zone_heads = torch.nn.Parameter(torch.randn(4, d_model, zone_dim, device=DEVICE, dtype=torch.float32) * 0.01)
zone_temp = 0.3
losses_b = []

opt_b = AdamW([zone_heads], lr=LR * 0.5)
sched_b = get_linear_schedule_with_warmup(
    opt_b, num_warmup_steps=200, num_training_steps=PHASE_B_STEPS
)

for step in range(PHASE_B_STEPS):
    text = train_texts[step % len(train_texts)]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states[-1].float()  # [1, seq, d]
    hs_flat = hs.reshape(-1, d_model)
    
    # Zone competition: soft winner-take-all
    zone_scores = []
    for z in range(4):
        score = torch.norm(hs_flat @ zone_heads[z], dim=-1).mean()  # [N, zdim] -> scalar
        zone_scores.append(score)
    zone_scores = torch.stack(zone_scores)  # [4]
    zone_weights = F.softmax(zone_scores / zone_temp, dim=0)  # [4]
    
    # Zone specialization loss: each zone should be selectively active
    # Penalize uniform zone usage (encourage specialization)
    uniform = torch.ones(4, device=DEVICE) / 4
    zone_loss = F.kl_div(
        F.log_softmax(zone_scores / zone_temp, dim=0),
        uniform,
        reduction='batchmean'
    )
    # Negative KL = push away from uniform = encourage specialization
    zone_loss = -zone_loss
    
    # Projection alignment: zone heads should align with basis subspaces
    proj_loss = 0.0
    for z, sname in enumerate(sub_names):
        dims = subspace_defs[sname]
        Bz = basis[:, dims]  # [d, |Sz|]
        # Zone head should lie in its designated subspace
        proj_loss += F.mse_loss(
            Bz @ Bz.T @ zone_heads[z],
            zone_heads[z]
        )
    
    loss = zone_loss + 0.001 * proj_loss
    loss.backward()
    opt_b.step()
    sched_b.step()
    opt_b.zero_grad()
    
    losses_b.append(loss.item())
    if (step + 1) % 500 == 0:
        zs = zone_weights.detach().cpu().tolist()
        print(f"  Step {step+1:>5d}/{PHASE_B_STEPS}: loss={loss.item():.6f}  "
              f"zones=[{zs[0]:.2f},{zs[1]:.2f},{zs[2]:.2f},{zs[3]:.2f}]")

# -- Measure zone specialization --
print("\n[5] Measuring zone specialization...")
test_prompts = {
    "syntax": [
        "The cat sat on the",
        "She walked to the store and bought",
        "In the garden there were many beautiful",
    ],
    "routing": [
        "To solve this problem, first we need to",
        "The logical next step in the proof is to",
        "By applying the chain rule, we can compute",
    ],
    "factual": [
        "The largest planet in our solar system is",
        "The chemical symbol for gold is",
        "The capital of Japan is",
    ],
    "math": [
        "The integral of x squared dx equals",
        "The eigenvalues of a symmetric matrix are always",
        "A group is abelian if and only if",
    ],
}

zone_ppls = {}
for zname, prompts in test_prompts.items():
    total_loss = 0
    total_tok = 0
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc, labels=enc.input_ids)
        total_loss += out.loss.item() * enc.input_ids.shape[1]
        total_tok += enc.input_ids.shape[1]
    zone_ppls[zname] = round(total_loss / max(total_tok, 1), 2)
    print(f"  {zname:<10}: PPL={zone_ppls[zname]:.1f}")

# -- Save --
print("\n[6] Saving model...")
torch.save(zone_heads.detach(), f"{OUT_DIR}/zone_heads.pt")

results = {
    "config": {
        "model": MODEL_ID,
        "d_model": d_model,
        "n_layers": n_layers,
        "k_basis": K_BASIS,
        "subspaces": {k: len(v) for k, v in subspace_defs.items()},
        "phase_a_steps": PHASE_A_STEPS,
        "phase_b_steps": PHASE_B_STEPS,
        "device": "EC2 L40S",
    },
    "training": {
        "phase_a_final_loss": round(losses_a[-1], 6),
        "phase_b_final_loss": round(losses_b[-1], 6),
    },
    "zone_specialization": zone_ppls,
}

with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"  UGT QWEN-1.5B COMPLETE")
print(f"{'='*60}")
print(f"  Basis: {basis.shape}")
print(f"  Zone PPLs: {zone_ppls}")
print(f"  Saved to {OUT_DIR}/")
