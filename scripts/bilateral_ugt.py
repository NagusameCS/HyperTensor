"""Bilateral UGT Hot-Swap Demo.
Trains recipient model with Phase A UGT (basis alignment, ~200 steps),
then tests attention-head swap with donor UGT Phase 5 model.
Deploy to EC2."""
import torch, json, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F

DEVICE = "cuda"
D_MODEL = 576
K_BASIS = 256
STEPS = 300
BATCH_SIZE = 2
LR = 1e-4

print("=" * 60)
print("  BILATERAL UGT HOT-SWAP DEMO")
print("=" * 60)

# -- Load models --
print("\n[1/6] Loading models...")
donor_path = "/home/ubuntu/benchmarks/ugt_phase5/final"
donor = AutoModelForCausalLM.from_pretrained(donor_path, torch_dtype=torch.float16, device_map=DEVICE, local_files_only=True)
recipient = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float16, device_map=DEVICE)
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Load UGT basis from donor
basis = torch.load(f"{donor_path}/taxonomic_basis.pt", map_location=DEVICE)  # [576, 256]
print(f"  Donor: UGT Phase 5 (k={basis.shape[1]})")
print(f"  Recipient: SmolLM2-135M base (untrained for UGT)")
print(f"  Taxonomic basis: {basis.shape}")

# -- Baseline PPL (pre-training) --
print("\n[2/6] Measuring baseline PPL...")
test_texts = [
    "The capital of France is Paris, a city known for its",
    "The solution to the equation 2x + 5 = 13 is x =",
    "In quantum mechanics, the wave function describes the",
    "The three primary colors of light are red, green, and",
    "Machine learning is a subset of artificial intelligence that",
]
def measure_ppl(model, texts):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tok(text, return_tensors="pt").to(DEVICE)
            out = model(**enc, labels=enc.input_ids)
            total_loss += out.loss.item() * enc.input_ids.shape[1]
            total_tokens += enc.input_ids.shape[1]
    return total_loss / total_tokens

baseline_ppl_donor = measure_ppl(donor, test_texts)
baseline_ppl_recipient = measure_ppl(recipient, test_texts)
print(f"  Donor PPL (cross-entropy):    {baseline_ppl_donor:.4f}")
print(f"  Recipient PPL (cross-entropy): {baseline_ppl_recipient:.4f}")

# -- Phase A UGT Training on Recipient --
print(f"\n[3/6] Phase A UGT training ({STEPS} steps)...")

# Training data: mixed domain sentences
train_data = [
    "The quick brown fox jumps over the lazy dog near the river bank",
    "If x equals three then two x plus five equals eleven",
    "The mitochondria is the powerhouse of the cell",
    "Shakespeare wrote Hamlet and Romeo and Juliet",
    "The derivative of x squared is two x",
    "Paris is the capital of France and Berlin is the capital of Germany",
    "Water boils at one hundred degrees Celsius at sea level",
    "The Pythagorean theorem states that a squared plus b squared equals c squared",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen",
    "The first law of thermodynamics is the conservation of energy",
] * (STEPS // 10 + 1)

# TOPLoss hyperparameters
lambda_top = 0.01
# Subspace definitions (matching UGT taxonomy)
subspace_defs = {
    "syntax": list(range(0, 96)),      # dims 0-95
    "routing": list(range(96, 192)),    # dims 96-191
    "factual": list(range(192, 256)),   # dims 192-255
}

# Freeze model, only train basis projection
for p in recipient.parameters():
    p.requires_grad = False

# Create trainable basis alignment adapter
basis_target = basis.clone()  # target basis from donor
basis_current = torch.randn(D_MODEL, K_BASIS, device=DEVICE, dtype=torch.float32) * 0.01
basis_current.requires_grad = True

optimizer = AdamW([basis_current], lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=STEPS)

recipient.train()
losses = []
for step in range(STEPS):
    text = train_data[step % len(train_data)]
    enc = tok(text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        out = recipient(**enc, output_hidden_states=True)
        # Get last layer hidden states
        hs = out.hidden_states[-1]  # [batch, seq, d]
    
    # Project hidden states onto current basis and target basis
    # and minimize distance
    hs_flat = hs.reshape(-1, D_MODEL).float()  # [batch*seq, d]
    
    # Basis alignment loss: ||hs @ B_current - hs @ B_target||^2
    proj_current = hs_flat @ basis_current  # [N, k]
    proj_target = hs_flat @ basis_target.float()  # [N, k]
    
    # Normalize both projections
    proj_current = F.normalize(proj_current, dim=-1)
    proj_target = F.normalize(proj_target, dim=-1)
    
    # Cosine similarity loss (maximize similarity = minimize 1 - cos_sim)
    basis_loss = (1 - (proj_current * proj_target).sum(dim=-1)).mean()
    
    # TOPLoss: penalize inter-subspace overlap in current basis
    top_loss = 0.0
    for name_i, dims_i in subspace_defs.items():
        for name_j, dims_j in subspace_defs.items():
            if name_i >= name_j:
                continue
            # Sub-basis for subspaces
            Bi = basis_current[:, dims_i]  # [d, |Si|]
            Bj = basis_current[:, dims_j]  # [d, |Sj|]
            # Inter-subspace overlap: ||Bi^T Bj||_F^2
            overlap = torch.norm(Bi.T @ Bj, p='fro') ** 2
            top_loss = top_loss + overlap
    
    loss = basis_loss + lambda_top * top_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([basis_current], 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    losses.append(loss.item())
    if (step + 1) % 50 == 0:
        print(f"  Step {step+1:>4d}/{STEPS}: loss={loss.item():.6f}  basis={basis_loss.item():.4f}  top={top_loss.item():.6f}")

# -- Align recipient to UGT basis --
print("\n[4/6] Aligning recipient to UGT basis...")
# Orthogonalize the learned basis via QR
Q, R = torch.linalg.qr(basis_current.float())
basis_aligned = Q  # [d, k] orthonormal

# Create projection matrix for alignment
P_align = basis_aligned @ basis_aligned.T  # [d, d]

# Align recipient's weights via subspace projection
n_layers = len(recipient.model.layers)
for li in range(n_layers):
    layer = recipient.model.layers[li]
    # Align attention weights
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if hasattr(layer.self_attn, name):
            w = getattr(layer.self_attn, name).weight.data
            w_float = w.float()
            # q/k/v: [n_heads*head_dim, d_model] --- project input side (dim=1)
            # o_proj: [d_model, n_heads*head_dim] --- project output side (dim=0)
            if w_float.shape[1] == D_MODEL:  # input projection: [*, d_model]
                w_aligned = w_float @ P_align.float()
            elif w_float.shape[0] == D_MODEL:  # output projection: [d_model, *]
                w_aligned = P_align.float() @ w_float
            else:
                continue
            w.copy_(w_aligned.to(w.dtype))
    # Align FFN weights
    for name in ["gate_proj", "up_proj", "down_proj"]:
        if hasattr(layer.mlp, name):
            w = getattr(layer.mlp, name).weight.data
            w_float = w.float()
            if w_float.shape[1] == D_MODEL:  # input projection: [intermediate, d_model]
                w_aligned = w_float @ P_align.float()
            elif w_float.shape[0] == D_MODEL:  # output projection: [d_model, intermediate]
                w_aligned = P_align.float() @ w_float
            else:
                continue
            w.copy_(w_aligned.to(w.dtype))

recipient_ppl = measure_ppl(recipient, test_texts)
print(f"  Recipient PPL after alignment: {recipient_ppl:.4f}")
print(f"  PPL change: {recipient_ppl - baseline_ppl_recipient:+.4f}")

# -- Hot-Swap Test --
print("\n[5/6] Hot-swap attention heads between donor and recipient...")

# Copy original weights for recovery
def save_layer_weights(model, layer_idx):
    layer = model.model.layers[layer_idx]
    return {
        "q": layer.self_attn.q_proj.weight.data.clone(),
        "k": layer.self_attn.k_proj.weight.data.clone(),
        "v": layer.self_attn.v_proj.weight.data.clone(),
        "o": layer.self_attn.o_proj.weight.data.clone(),
    }

def swap_attention_weights(model_a, model_b, layer_idx):
    """Swap all attention weights between two models at given layer."""
    la = model_a.model.layers[layer_idx]
    lb = model_b.model.layers[layer_idx]
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        wa = getattr(la.self_attn, name).weight.data.clone()
        wb = getattr(lb.self_attn, name).weight.data.clone()
        getattr(la.self_attn, name).weight.data.copy_(wb)
        getattr(lb.self_attn, name).weight.data.copy_(wa)

# Test swaps at multiple layers
swap_results = []
swap_layers = [0, 5, 10, 15, 20, 25, 29]

for layer_idx in swap_layers:
    # Measure PPL before swap
    ppl_donor_before = measure_ppl(donor, test_texts)
    ppl_recipient_before = measure_ppl(recipient, test_texts)
    
    # Swap
    swap_attention_weights(donor, recipient, layer_idx)
    
    # Measure PPL after swap
    ppl_donor_after = measure_ppl(donor, test_texts)
    ppl_recipient_after = measure_ppl(recipient, test_texts)
    
    # Swap back (recovery)
    swap_attention_weights(donor, recipient, layer_idx)
    
    # Verify recovery
    ppl_donor_recovered = measure_ppl(donor, test_texts)
    ppl_recipient_recovered = measure_ppl(recipient, test_texts)
    
    delta_donor = ppl_donor_after - ppl_donor_before
    delta_recipient = ppl_recipient_after - ppl_recipient_before
    
    result = {
        "layer": layer_idx,
        "donor_ppl_before": round(ppl_donor_before, 4),
        "donor_ppl_after": round(ppl_donor_after, 4),
        "donor_delta": round(delta_donor, 4),
        "recipient_ppl_before": round(ppl_recipient_before, 4),
        "recipient_ppl_after": round(ppl_recipient_after, 4),
        "recipient_delta": round(delta_recipient, 4),
        "recovery_donor": round(abs(ppl_donor_recovered - ppl_donor_before), 6),
        "recovery_recipient": round(abs(ppl_recipient_recovered - ppl_recipient_before), 6),
    }
    swap_results.append(result)
    
    success = "PASS" if delta_donor < 0.5 else "FAIL"
    print(f"  Layer {layer_idx:>2d}: donor Δ={delta_donor:+.4f} recip Δ={delta_recipient:+.4f} [{success}]")

# -- Save results --
print("\n[6/6] Saving results...")
output = {
    "config": {
        "donor": "UGT Phase 5 (100K steps, k=256)",
        "recipient": "SmolLM2-135M + Phase A UGT (300 steps)",
        "basis_dim": K_BASIS,
        "top_lambda": lambda_top,
        "device": "EC2 L40S",
    },
    "baseline": {
        "donor_ppl": round(baseline_ppl_donor, 4),
        "recipient_ppl": round(baseline_ppl_recipient, 4),
        "recipient_aligned_ppl": round(recipient_ppl, 4),
    },
    "training": {
        "steps": STEPS,
        "final_loss": round(losses[-1], 6),
        "loss_history": [round(l, 6) for l in losses[::10]],
    },
    "swap_results": swap_results,
    "summary": {
        "swaps_tested": len(swap_layers),
        "swaps_passed": sum(1 for r in swap_results if abs(r["donor_delta"]) < 0.5),
        "mean_donor_delta": round(sum(r["donor_delta"] for r in swap_results) / len(swap_results), 4),
        "mean_recipient_delta": round(sum(r["recipient_delta"] for r in swap_results) / len(swap_results), 4),
    },
}

out_path = "/home/ubuntu/benchmarks/bilateral_ugt_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"  BILATERAL UGT RESULTS")
print(f"{'='*60}")
s = output["summary"]
print(f"  Swaps tested: {s['swaps_tested']}")
print(f"  Swaps passed (Δ<0.5): {s['swaps_passed']}/{s['swaps_tested']}")
print(f"  Mean donor Δ: {s['mean_donor_delta']:+.4f}")
print(f"  Mean recipient Δ: {s['mean_recipient_delta']:+.4f}")
print(f"  Final training loss: {losses[-1]:.6f}")
print(f"\nSaved to {out_path}")
