"""ISAGI DUEL: Dual-agent collaborative problem-solving over N turns.
Agent A = ISAGI (HyperTensor stack, adaptive learning)
Agent B = Partner (clean model, collaborative persona)
They share one model instance to save VRAM, alternate turns,
and work together on a joint task.

Data collected per turn:
- Response text (both agents)
- ISAGI metrics: COG action, GTC hits, metric growth, similarity
- Partner metrics: response length, generation time
- Conversation quality: agreement rate, novel contributions
"""
import torch, json, time, os, sys, argparse, random
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

# ============================================================================
# CONFIG
# ============================================================================
K_UGT = 512
MAX_NEW_ISAGI = 256
MAX_NEW_PARTNER = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DELTA_NOVEL = 0.05
ETA_METRIC = 0.15
N_CAL_PROMPTS = 30

# Disk paths
if os.path.exists("D:/"):
    STATE_DIR = "D:/hyperchat_states"
    CACHE_DIR = "D:/huggingface_cache"
else:
    STATE_DIR = os.path.expanduser("~/hyperchat_states")
    CACHE_DIR = None
os.makedirs(STATE_DIR, exist_ok=True)

# ============================================================================
# JOINT TASKS
# ============================================================================
JOINT_TASKS = {
    "math_proof": {
        "goal": "Prove that there are infinitely many prime numbers, using two different approaches, then compare and unify them.",
        "agent_a_role": "ISAGI: You use rigorous geometric reasoning. Find a geometric/combinatorial proof.",
        "agent_b_role": "Partner: You use classical number theory. Provide Euclid's classical proof.",
        "success_criteria": "Both proofs are valid, the comparison identifies strengths of each approach, and a unified perspective emerges."
    },
    "code_review": {
        "goal": "Design and iteratively improve a Python implementation of a trie (prefix tree) data structure with insert, search, and prefix-search operations.",
        "agent_a_role": "ISAGI: You focus on algorithmic correctness, edge cases, and mathematical properties.",
        "agent_b_role": "Partner: You focus on code style, efficiency, Python idioms, and test coverage.",
        "success_criteria": "The final implementation is correct, efficient, well-tested, and both agents agree it's optimal."
    },
    "creative_writing": {
        "goal": "Co-write a science fiction short story (500 words) about first contact with an alien intelligence that communicates through mathematical patterns rather than language.",
        "agent_a_role": "ISAGI: You write the scientific/technical parts — describe the mathematical patterns, the detection method, the analysis.",
        "agent_b_role": "Partner: You write the human/emotional parts — the scientists' reactions, the ethical dilemmas, the narrative flow.",
        "success_criteria": "The story is coherent, scientifically plausible, emotionally engaging, and seamlessly integrates both perspectives."
    },
    "system_design": {
        "goal": "Design a fault-tolerant distributed key-value store with strong consistency guarantees. Discuss CAP theorem trade-offs, consensus protocols, and failure recovery.",
        "agent_a_role": "ISAGI: You handle the theoretical guarantees — consistency proofs, protocol analysis, failure bounds.",
        "agent_b_role": "Partner: You handle the practical implementation — data structures, network protocols, operational concerns.",
        "success_criteria": "A complete design document covering both theory and practice, with explicit trade-off analysis."
    },
    "puzzle_chain": {
        "goal": "Solve this puzzle collaboratively: 'I am a 4-digit number. My digits sum to 20. My first digit is twice my last. My middle two digits are consecutive in ascending order. What number am I?' Then create a new puzzle of similar difficulty for the other agent to solve, and verify each other's solutions.",
        "agent_a_role": "ISAGI: You solve the puzzle step by step with explicit algebra. Then verify Partner's solution.",
        "agent_b_role": "Partner: You also solve the puzzle independently, then verify ISAGI's solution. Create a new puzzle.",
        "success_criteria": "Both solutions match, verification is thorough, the new puzzle is well-formed and solved correctly."
    },
    "research_proposal": {
        "goal": "Write a 1-page research proposal for using HyperTensor geometric compression to improve transformer inference efficiency on edge devices.",
        "agent_a_role": "ISAGI: You provide the technical depth — the math of GRC compression, L2 cache residency, the k* formula, expected throughput gains.",
        "agent_b_role": "Partner: You provide the practical framing — motivation, related work, experimental design, impact assessment.",
        "success_criteria": "A complete, compelling proposal that is both technically rigorous and practically grounded."
    }
}

# ============================================================================
# SIMPLIFIED GTC CACHE (self-contained for duel script)
# ============================================================================
class MiniGTC:
    """Lightweight GTC for tracking ISAGI's learning during the duel."""
    def __init__(self):
        self.q_projs = []
        self.r_projs = []
        self.responses = []
        self.q_texts = []
        self.hits = 0
        self.misses = 0
        self.radius = 0.35
    
    def query(self, k_proj, raw_text=""):
        if not self.responses:
            self.misses += 1
            return False, None, 0.0
        if raw_text and raw_text in self.q_texts:
            idx = self.q_texts.index(raw_text)
            self.hits += 1
            return True, self.responses[idx], 1.0
        q = F.normalize(k_proj.unsqueeze(0).float(), dim=1)
        if self.q_projs:
            qs = torch.stack(self.q_projs).float()
            best_q = (qs @ q.T).squeeze(-1).max().item()
        else:
            best_q = -1.0
        if self.r_projs:
            rs = torch.stack(self.r_projs).float()
            best_r = (rs @ q.T).squeeze(-1).max().item()
        else:
            best_r = -1.0
        best = max(best_q, best_r)
        if 1.0 - best < self.radius:
            # Find index
            if best_q >= best_r:
                idx = (qs @ q.T).squeeze(-1).argmax().item()
            else:
                idx = (rs @ q.T).squeeze(-1).argmax().item()
            self.hits += 1
            return True, self.responses[idx], best
        self.misses += 1
        return False, None, best
    
    def store(self, kq, kr, resp, qtext=""):
        if len(self.responses) >= 50000:
            self.q_texts.pop(0); self.q_projs.pop(0)
            self.r_projs.pop(0); self.responses.pop(0)
        self.q_texts.append(qtext)
        self.q_projs.append(F.normalize(kq.unsqueeze(0).float(), dim=1).squeeze(0).cpu())
        self.r_projs.append(F.normalize(kr.unsqueeze(0).float(), dim=1).squeeze(0).cpu())
        self.responses.append(resp)

# ============================================================================
# BUILD THE DUEL SYSTEM
# ============================================================================
def build_duel(model_id="Qwen/Qwen2.5-7B-Instruct", use_4bit=True):
    print("=" * 70)
    print("  ISAGI DUEL — Dual-Agent Collaborative Problem Solving")
    print("=" * 70)
    
    # Load shared model
    print("\n[1/5] Loading shared model...")
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, cache_dir=CACHE_DIR,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto",
            trust_remote_code=True, cache_dir=CACHE_DIR,
        )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    d_model = model.config.hidden_size
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  d={d_model}, VRAM={vram:.1f}GB")
    
    # Build ISAGI's HyperTensor stack (same as isagi_chat.py)
    print("\n[2/5] Bootstrapping ISAGI stack...")
    
    # Calibration
    cal_prompts = [
        "The mitochondria is the powerhouse of the cell.",
        "Newton's second law: F = ma.",
        "The Pythagorean theorem: a² + b² = c².",
        "A transformer model uses self-attention for sequence processing.",
        "The Riemann zeta function ζ(s) = Σ 1/n^s.",
        "Gradient descent iteratively minimizes loss functions.",
        "DNA replication is semiconservative.",
        "The speed of light c ≈ 299,792,458 m/s.",
        "Euler's identity: e^(iπ) + 1 = 0.",
        "Photosynthesis: CO₂ + H₂O → glucose + O₂.",
        "The attention mechanism: softmax(QK^T/√d_k)V.",
        "In thermodynamics, entropy never decreases.",
        "Neurons communicate via action potentials.",
        "Bayes theorem: P(A|B) = P(B|A)P(A)/P(B).",
        "Convolutional networks detect hierarchical spatial patterns.",
        "The immune system has innate and adaptive components.",
        "Group theory: sets with associative operations and inverses.",
        "Climate science integrates physics, chemistry, and biology.",
        "The Halting Problem is undecidable.",
        "Natural selection drives evolutionary adaptation.",
        "Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.",
        "The Higgs boson gives particles mass.",
        "Backpropagation uses the chain rule for gradient computation.",
        "Plate tectonics explains continental drift.",
        "The Industrial Revolution mechanized production.",
    ]
    
    hidden_states = []
    for prompt in cal_prompts[:N_CAL_PROMPTS]:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden_states.append(out.hidden_states[-1][0, -1, :].float())
    
    hs_stack = torch.stack(hidden_states)
    hs_centered = hs_stack - hs_stack.mean(dim=0)
    U, S, _ = torch.linalg.svd(hs_centered.T, full_matrices=False)
    
    n_cal = min(N_CAL_PROMPTS, K_UGT)
    basis = U[:, :n_cal].float().to(model.device)
    if n_cal < K_UGT:
        rand_pad = torch.randn(d_model, K_UGT - n_cal, device=model.device, dtype=torch.float32)
        for i in range(rand_pad.shape[1]):
            rand_pad[:, i] -= basis @ (basis.T @ rand_pad[:, i])
        Q_pad, _ = torch.linalg.qr(rand_pad)
        basis = torch.cat([basis, Q_pad], dim=1)
    Q, _ = torch.linalg.qr(basis)
    basis = Q[:, :K_UGT]
    
    # Safety (minimal)
    forbidden = list(range(10))
    snipe_coords = [int(K_UGT * p) for p in [0.45, 0.05, 0.50, 0.14, 0.33, 0.17]]
    snipe_coords = [c for c in snipe_coords if c < K_UGT]
    
    ft = torch.tensor(forbidden, device=model.device, dtype=torch.long)
    Bf = basis[:, ft].float(); Qf, _ = torch.linalg.qr(Bf)
    P_safe = torch.eye(d_model, device=model.device) - Qf @ Qf.T
    
    st = torch.tensor(snipe_coords, device=model.device, dtype=torch.long)
    Bs = basis[:, st].float(); Qs, _ = torch.linalg.qr(Bs)
    P_privacy = torch.eye(d_model, device=model.device) - Qs @ Qs.T
    
    def safe_h(h):
        return P_privacy @ (P_safe @ h)
    
    def to_k(h):
        return h.float() @ basis.float()
    
    def get_h(text):
        e = tok(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            o = model(**e, output_hidden_states=True)
        return o.hidden_states[-1][0, -1, :].float()
    
    # COG
    metric = torch.eye(K_UGT, device=model.device, dtype=torch.float32)
    trajectories = []
    
    def is_novel(h):
        nonlocal trajectories
        if not trajectories:
            return True, 0.0
        hk = to_k(h)
        hk_norm = F.normalize(hk.unsqueeze(0), dim=1).squeeze(0)
        best = -1.0
        for tp in trajectories:
            tp_n = F.normalize(tp["proj"].to(model.device).unsqueeze(0), dim=1).squeeze(0)
            sim = torch.dot(hk_norm, tp_n).item()
            if sim > best: best = sim
        return (1.0 - best) > DELTA_NOVEL, 1.0 - best
    
    def expand(h, label):
        nonlocal metric, trajectories
        hk = to_k(h)
        hk_norm = F.normalize(hk.unsqueeze(0), dim=1).squeeze(0)
        J = torch.outer(hk_norm, hk_norm)
        metric = metric + ETA_METRIC * J + 0.001 * torch.eye(K_UGT, device=model.device)
        trajectories.append({"proj": hk.detach().cpu(), "label": label[:80], "time": time.time()})
    
    # Seed trajectories
    for prompt in cal_prompts[:6]:
        h = get_h(prompt); hs = safe_h(h)
        trajectories.append({"proj": to_k(hs).cpu(), "label": prompt[:60], "time": time.time()})
    
    gtc = MiniGTC()
    # Calibrate GTC
    cal_projs = [to_k(safe_h(hs)) for hs in hidden_states]
    if len(cal_projs) >= 5:
        cos_dists = []
        for _ in range(200):
            i, j = random.randint(0, len(cal_projs)-1), random.randint(0, len(cal_projs)-1)
            if i != j:
                sim = F.cosine_similarity(cal_projs[i].unsqueeze(0), cal_projs[j].unsqueeze(0)).item()
                cos_dists.append(1.0 - sim)
        cos_dists.sort()
        gtc.radius = cos_dists[int(len(cos_dists) * 0.75)]
        gtc.radius = max(0.20, min(gtc.radius, 0.55))
    
    print(f"  Basis: {basis.shape}, GTC radius: {gtc.radius:.3f}, Trajectories: {len(trajectories)}")
    
    # ISAGI persona (short, action-oriented)
    ISAGI_PERSONA = """You are ISAGI — a problem-solving AI that collaborates with a partner to solve tasks. You focus on rigorous analysis, mathematical reasoning, and geometric thinking. You are adaptive — you learn from each interaction and remember previous turns. Be concise. Build on what your partner says. When you agree, say so and extend. When you disagree, explain why with evidence. Always move the solution forward."""
    
    # Partner persona (collaborative, different perspective)
    PARTNER_PERSONA = """You are a collaborative AI partner working with ISAGI on a joint task. You bring practical, intuitive, and creative perspectives. You complement ISAGI's rigorous approach with real-world insight. Read ISAGI's contributions carefully. Agree when correct, respectfully challenge when you see gaps. Build on ISAGI's ideas. Stay focused on the shared goal. Be concise — no more than 3 paragraphs per turn."""
    
    print("\n[3/5] Agents ready.")
    print(f"  Agent A: ISAGI (HyperTensor stack, geometric reasoning)")
    print(f"  Agent B: Partner (clean model, collaborative)")
    
    return {
        "model": model, "tok": tok, "d_model": d_model,
        "basis": basis, "metric": metric, "trajectories": trajectories,
        "safe_h": safe_h, "to_k": to_k, "get_h": get_h,
        "is_novel": is_novel, "expand": expand,
        "gtc": gtc,
        "persona_a": ISAGI_PERSONA,
        "persona_b": PARTNER_PERSONA,
    }

# ============================================================================
# AGENT TURN
# ============================================================================
def agent_turn(system, agent_id, prompt, other_agent_response, conversation_history,
               is_isagi=False, max_tokens=256):
    """One agent takes a turn. Returns response text + ISAGI metrics."""
    s = system
    persona = s["persona_a"] if agent_id == "A" else s["persona_b"]
    
    # Build context from conversation history
    ctx = ""
    for entry in conversation_history[-6:]:
        ctx += f"\n{entry['agent']}: {entry['text'][:300]}"
    
    full_prompt = f"""<|im_start|>system
{persona}<|im_end|>
{ctx}
<|im_start|>user
{other_agent_response if other_agent_response else prompt}
<|im_end|>
<|im_start|>assistant
"""
    
    enc = s["tok"](full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(s["model"].device)
    np_tok = enc.input_ids.shape[1]
    
    t0 = time.time()
    
    out = s["model"].generate(
        **enc, max_new_tokens=max_tokens, do_sample=True,
        temperature=TEMPERATURE, top_p=TOP_P,
        pad_token_id=s["tok"].eos_token_id,
    )
    response = s["tok"].decode(out[0, np_tok:], skip_special_tokens=True).strip()
    elapsed = (time.time() - t0) * 1000
    
    result = {
        "agent": agent_id,
        "text": response,
        "ms": round(elapsed),
        "tokens": out.shape[-1] - np_tok,
    }
    
    # ISAGI-specific metrics
    if is_isagi:
        h_user = s["get_h"](other_agent_response if other_agent_response else prompt)
        h_safe = s["safe_h"](h_user)
        h_k = s["to_k"](h_safe)
        
        # GTC check
        gtc_hit, gtc_resp, gtc_sim = s["gtc"].query(h_k, raw_text=other_agent_response[:200] if other_agent_response else "")
        
        # COG
        novel, cos_dist = s["is_novel"](h_safe)
        if novel:
            s["expand"](h_safe, f"partner: {other_agent_response[:60]}" if other_agent_response else "task_start")
            cog = "EXPANDED"
        else:
            cog = "known"
        
        # Store in GTC
        h_resp = s["get_h"](response[:200])
        h_resp_safe = s["safe_h"](h_resp)
        h_resp_k = s["to_k"](h_resp_safe)
        s["gtc"].store(h_k, h_resp_k, response, (other_agent_response or "")[:200])
        
        mc = torch.norm(s["metric"] - torch.eye(K_UGT, device=s["metric"].device)).item()
        
        result.update({
            "gtc_hit": gtc_hit,
            "gtc_sim": round(gtc_sim, 3),
            "cog": cog,
            "cos_dist": round(cos_dist, 4),
            "metric_growth": round(mc, 4),
            "n_trajectories": len(s["trajectories"]),
        })
    
    return result

# ============================================================================
# RUN THE DUEL
# ============================================================================
def run_duel(system, task_key, n_turns, save_path=None):
    """Run N turns of ISAGI vs Partner on a joint task."""
    task = JOINT_TASKS[task_key]
    
    print(f"\n{'='*70}")
    print(f"  TASK: {task_key}")
    print(f"  Goal: {task['goal'][:80]}...")
    print(f"  Turns: {n_turns}")
    print(f"{'='*70}\n")
    
    # Initialize conversation
    conversation = []
    
    # Task prompt for Agent A (ISAGI starts)
    initial_prompt = f"""TASK: {task['goal']}

YOUR ROLE: {task['agent_a_role']}

Begin by analyzing the task and proposing your first contribution. Be specific and actionable."""
    
    # Agent A turn 1
    print(f"--- Turn 1/ {n_turns}: ISAGI (A) ---")
    result_a = agent_turn(system, "A", initial_prompt, None, conversation, is_isagi=True)
    conversation.append(result_a)
    print(f"  ISAGI: {result_a['text'][:200]}...")
    if "cog" in result_a:
        print(f"  [COG:{result_a['cog']} gtc_hit:{result_a['gtc_hit']} metric:{result_a.get('metric_growth',0):.4f} {result_a['ms']}ms]")
    
    # Alternate turns
    for turn in range(2, n_turns + 1):
        if turn % 2 == 0:
            # Agent B (Partner)
            agent_id = "B"
            prev = conversation[-1]["text"]
            prompt_for_b = f"""TASK: {task['goal']}

YOUR ROLE: {task['agent_b_role']}

ISAGI just said: "{prev[:300]}"

Respond to ISAGI's point. Build on it, challenge it, or extend it. Be specific."""
            
            result = agent_turn(system, "B", prompt_for_b, prev, conversation, is_isagi=False)
        else:
            # Agent A (ISAGI)
            agent_id = "A"
            prev = conversation[-1]["text"]
            prompt_for_a = f"""TASK: {task['goal']}

YOUR ROLE: {task['agent_a_role']}

Your partner just said: "{prev[:300]}"

Respond. Build on it, challenge it, or extend it. Be specific."""
            
            result = agent_turn(system, "A", prompt_for_a, prev, conversation, is_isagi=True)
        
        conversation.append(result)
        
        isagi = "ISAGI" if agent_id == "A" else "Partner"
        print(f"--- Turn {turn}/{n_turns}: {isagi} ({agent_id}) ---")
        print(f"  {result['text'][:200]}...")
        if "cog" in result:
            print(f"  [COG:{result['cog']} gtc_hit:{result['gtc_hit']} metric:{result.get('metric_growth',0):.4f} {result['ms']}ms]")
        
        # Save checkpoint every 50 turns
        if save_path and turn % 50 == 0:
            ckpt = save_path.replace(".json", f"_ckpt_{turn}.json")
            with open(ckpt, "w") as f:
                json.dump({
                    "task": task_key,
                    "n_turns_completed": turn,
                    "n_turns_total": n_turns,
                    "conversation": conversation,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2, default=str)
            print(f"  [Checkpoint saved: {ckpt}]")
    
    # Final analysis
    print(f"\n{'='*70}")
    print(f"  DUEL COMPLETE — {len(conversation)} turns")
    
    isagi_turns = [c for c in conversation if c["agent"] == "A"]
    partner_turns = [c for c in conversation if c["agent"] == "B"]
    
    isagi_expansions = sum(1 for c in isagi_turns if c.get("cog") == "EXPANDED")
    isagi_gtc_hits = sum(1 for c in isagi_turns if c.get("gtc_hit"))
    isagi_avg_time = sum(c["ms"] for c in isagi_turns) / max(len(isagi_turns), 1)
    partner_avg_time = sum(c["ms"] for c in partner_turns) / max(len(partner_turns), 1)
    
    # Average response length
    isagi_avg_len = sum(len(c["text"]) for c in isagi_turns) / max(len(isagi_turns), 1)
    partner_avg_len = sum(len(c["text"]) for c in partner_turns) / max(len(partner_turns), 1)
    
    final_metric = isagi_turns[-1].get("metric_growth", 0) if isagi_turns else 0
    initial_metric = isagi_turns[0].get("metric_growth", 0) if isagi_turns else 0
    
    analysis = {
        "task": task_key,
        "n_turns": len(conversation),
        "isagi_turns": len(isagi_turns),
        "partner_turns": len(partner_turns),
        "isagi_expansions": isagi_expansions,
        "isagi_expansion_rate": round(isagi_expansions / max(len(isagi_turns), 1) * 100, 1),
        "isagi_gtc_hits": isagi_gtc_hits,
        "isagi_gtc_hit_rate": round(isagi_gtc_hits / max(len(isagi_turns), 1) * 100, 1),
        "isagi_avg_time_ms": round(isagi_avg_time),
        "partner_avg_time_ms": round(partner_avg_time),
        "isagi_avg_response_len": round(isagi_avg_len),
        "partner_avg_response_len": round(partner_avg_len),
        "metric_growth_total": round(final_metric - initial_metric, 5),
        "metric_final": round(final_metric, 5),
        "trajectories_final": isagi_turns[-1].get("n_trajectories", 0) if isagi_turns else 0,
        "success_estimate": min(100, max(0, 
            int(isagi_expansions / max(len(isagi_turns), 1) * 200) +
            int(isagi_gtc_hits / max(len(isagi_turns), 1) * 200) +
            int(final_metric * 500)
        )),
    }
    
    print(f"\n  --- DUEL STATISTICS ---")
    print(f"  ISAGI expansions: {isagi_expansions}/{len(isagi_turns)} ({analysis['isagi_expansion_rate']}%)")
    print(f"  ISAGI GTC hits: {isagi_gtc_hits}/{len(isagi_turns)} ({analysis['isagi_gtc_hit_rate']}%)")
    print(f"  ISAGI avg response time: {isagi_avg_time:.0f}ms")
    print(f"  Partner avg response time: {partner_avg_time:.0f}ms")
    print(f"  Metric growth: {initial_metric:.4f} → {final_metric:.4f}")
    print(f"  ISAGI avg response length: {isagi_avg_len:.0f} chars")
    print(f"  Partner avg response length: {partner_avg_len:.0f} chars")
    print(f"  Success estimate: {analysis['success_estimate']}/100")
    print(f"{'='*70}")
    
    return conversation, analysis

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="ISAGI DUEL — Dual Agent Collaboration")
    parser.add_argument("--task", type=str, default="math_proof",
                        choices=list(JOINT_TASKS.keys()),
                        help="Joint task for the agents")
    parser.add_argument("--turns", type=int, default=100,
                        help="Number of conversation turns")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model ID")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit",
                        help="Use 4-bit quantization")
    parser.add_argument("--save", type=str, help="Save path for results")
    args = parser.parse_args()
    
    if not args.use_4bit and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram < 12:
            print(f"[auto] VRAM={vram:.1f}GB → enabling 4-bit")
            args.use_4bit = True
    
    system = build_duel(args.model, args.use_4bit)
    conversation, analysis = run_duel(system, args.task, args.turns, args.save)
    
    # Save
    save_path = args.save or os.path.join(STATE_DIR, f"duel_{args.task}_{args.turns}t.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        json.dump({
            "task": args.task,
            "n_turns": args.turns,
            "analysis": analysis,
            "conversation": conversation,
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
        }, f, indent=2, default=str)
    
    print(f"\n  Full results saved to: {save_path}")
    print(f"  Analysis: {json.dumps(analysis, indent=2)}")

if __name__ == "__main__":
    main()
