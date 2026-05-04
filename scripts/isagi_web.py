#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║              ISAGI Web Chat Interface (Gradio)                  ║
║                                                                 ║
║  Simple web UI for chatting with ISAGI.                         ║
║  Opens in your browser at http://127.0.0.1:7860                 ║
║                                                                 ║
║  Usage:                                                         ║
║    python isagi_web.py --model Qwen/Qwen2.5-7B-Instruct --4bit  ║
║    python isagi_web.py --share  (public link via Gradio)        ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys, os, argparse, time

# Add parent to path for isagi_chat import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import torch

# Lazy import — ISAGI loads when first message arrives
isagi_system = None

def build_isagi_once(model_id, use_4bit, cpu_offload):
    """Build ISAGI system once and cache globally."""
    global isagi_system
    if isagi_system is None:
        from isagi_chat import build_isagi
        print(f"[ISAGI WEB] Building system with {model_id}...")
        isagi_system = build_isagi(model_id, use_4bit=use_4bit, cpu_offload=cpu_offload)
        print("[ISAGI WEB] Ready!")
    return isagi_system

def chat(message, history, model_id, use_4bit, cpu_offload):
    """Handle one chat turn. Called by Gradio on each message."""
    if not message.strip():
        return "", history
    
    system = build_isagi_once(model_id, use_4bit, cpu_offload)
    
    from isagi_chat import isagi_turn
    
    t0 = time.time()
    result = isagi_turn(system, message)
    elapsed = time.time() - t0
    
    response = result["response"]
    
    # Format status line
    status = (
        f"<-> {result['cog']} | "
        f"[CHART] sim={result['sim']:.2f} | "
        f" metric={result['metric']:.4f} | "
        f" {result['traj']} traj | "
        f" {elapsed:.1f}s"
    )
    
    # Add status as subtitle
    full_response = f"{response}\n\n---\n*{status}*"
    
    history.append((message, full_response))
    return "", history

def get_status(model_id):
    """Return system status string."""
    if isagi_system is None:
        return "ISAGI not yet initialized. Send a message to start."
    
    s = isagi_system
    mc = torch.norm(s["metric"] - torch.eye(512, device=s["metric"].device)).item()
    gtc_s = s["gtc"].stats()
    ott_s = s["ott"].stats()
    
    return f"""
**ISAGI System Status**
- Model: {model_id}
- COG Trajectories: {len(s['trajectories'])} | Metric: {mc:.4f}
- GTC Cache: {gtc_s['size']} entries | Hit Rate: {gtc_s['hit_rate']}%
- OTT Drafts: {ott_s['total_drafts']} | Acceptance: {ott_s['acceptance_rate']}%
- Safety: {len(s['forbidden'])} forbidden coords | {len(s['snipe_coords'])} snipe coords | 0% TEH
- Conversation Turns: {len(s['conv_log'])}
- GRC: k={s['grc'].k} | L2 Resident: {s['grc'].l2_resident}
"""

def save_state():
    """Save current ISAGI state to .miku file."""
    if isagi_system is None:
        return "ISAGI not yet initialized."
    
    from isagi_chat import save_isagi_state
    path = f"D:/hyperchat_states/isagi_web_{time.strftime('%Y%m%d_%H%M%S')}.miku"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_isagi_state(path, isagi_system)
    return f"State saved to {path}"

# ── Gradio UI ──
def create_ui(model_id, use_4bit, cpu_offload, share):
    """Build the Gradio chat interface."""
    
    with gr.Blocks(title="ISAGI — The Adaptive Living Model", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        #  ISAGI — The Adaptive Living Model
        
        *"No problem is unsolvable — only a question of time and depth."*
        
        **Stack:** GTC + OTT + GRC + UGT + Safe OGD + Snipe + COG+TEH
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Conversation")
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask ISAGI anything...",
                        label="Your message",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    status_btn = gr.Button("System Status")
                    save_btn = gr.Button("Save State (.miku)")
            
            with gr.Column(scale=1):
                status_box = gr.Markdown("ISAGI ready. Send a message to begin.")
                
                gr.Markdown("""
                ### About ISAGI
                
                ISAGI integrates all HyperTensor research:
                - **GTC** — Trajectory caching (15.5× vs RAG)
                - **OTT** — Speculative decoding
                - **GRC** — Attention compression
                - **UGT** — Taxonomic knowledge basis
                - **Safe OGD** — Geometric safety (0% TEH)
                - **Snipe** — Behavioral precision
                - **COG** — Living manifold (grows with you)
                
                ### Commands
                Type in chat:
                - `/status` — System stats
                - `/save` — Save .miku state
                """)
        
        # State variables
        state = gr.State([])
        
        # Hidden config
        model_id_state = gr.State(model_id)
        use_4bit_state = gr.State(use_4bit)
        offload_state = gr.State(cpu_offload)
        
        # Wire events
        def on_send(message, history, mid, q4, off):
            return chat(message, history, mid, q4, off)
        
        msg.submit(on_send, [msg, chatbot, model_id_state, use_4bit_state, offload_state], [msg, chatbot])
        send_btn.click(on_send, [msg, chatbot, model_id_state, use_4bit_state, offload_state], [msg, chatbot])
        
        clear_btn.click(lambda: ([], "Chat cleared."), None, [chatbot, status_box])
        status_btn.click(lambda mid: get_status(mid), [model_id_state], [status_box])
        save_btn.click(lambda: save_state(), None, [status_box])
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISAGI Web Chat Interface")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit")
    parser.add_argument("--offload", action="store_true", dest="cpu_offload")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    # Auto 4-bit for 32B
    if "32B" in args.model:
        args.use_4bit = True
    
    demo = create_ui(args.model, args.use_4bit, args.cpu_offload, args.share)
    demo.launch(share=args.share, server_port=args.port, inbrowser=True)
