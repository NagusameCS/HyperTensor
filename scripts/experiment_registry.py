#!/usr/bin/env python3
"""
MASTER EXPERIMENT REGISTRY — Every unproven prediction across HyperTensor Papers I-XV.
Persistent record of claims, experiments, and verification status.
"""
import json, os
from pathlib import Path

EXPERIMENTS = [
    # PAPER A: GRC Attention Compression
    {"id":"A1","paper":"I (A)","claim":"GRC preserves ≥95% signal at k≥256","status":"pending","script":"experiment_a1_ppl_vs_k.py","needs_gpu":True,"runtime_hrs":2},
    # PAPER E: GRC Light Distillation  
    {"id":"E1","paper":"V (E)","claim":"Per-matrix SVD beats shared GRC","status":"done","result":"+79.4% SmolLM2, +18.9% Llama-8B","needs_gpu":False},
    {"id":"E2","paper":"V (E)","claim":"LoRA distillation recovers PPL after GRC","status":"pending","needs_gpu":True,"runtime_hrs":3},
    # PAPER F: Task-Level Asymmetric Degradation
    {"id":"F1","paper":"VI (F)","claim":"MMLU survives <2% drop at k=1024","status":"pending","needs_gpu":True,"runtime_hrs":2},
    {"id":"F2","paper":"VI (F)","claim":"GSM8K collapses >5% at k=1024","status":"pending","needs_gpu":True,"runtime_hrs":3},
    {"id":"F3","paper":"VI (F)","claim":"HumanEval bimodal degradation","status":"pending","needs_gpu":True,"runtime_hrs":1},
    {"id":"F4","paper":"VI (F)","claim":"Safe frontier at k=1024","status":"pending","needs_gpu":False},
    # PAPER G: FFN Clustering
    {"id":"G1","paper":"VII (G)","claim":"Cluster compression recovers 21-25% error","status":"done","result":"Local reconstruction measured, 20.9-25.0% improvement","needs_gpu":False},
    {"id":"G2","paper":"VII (G)","claim":"End-to-end PPL validates cluster compression","status":"pending","needs_gpu":True,"runtime_hrs":2},
    {"id":"G3","paper":"VII (G)","claim":"Combined attn+FFN ~2.5 byte savings","status":"pending","needs_gpu":False},
    # PAPER H: GTC as Vector DB
    {"id":"H1","paper":"VIII (H)","claim":"GTC 15.5 faster than RAG","status":"done","result":"1121.8 speedup (claimed 15.5 was conservative)","needs_gpu":False},
    {"id":"H2","paper":"VIII (H)","claim":"Hybrid GTC+RAG Pareto-dominates","status":"pending","needs_gpu":False},
    # PAPER I: Super-Baseline Universality
    {"id":"I1","paper":"IX (I)","claim":"RTX 4090 peaks at k*=1536","status":"pending","needs_gpu":True,"requires":"RTX 4090"},
    {"id":"I2","paper":"IX (I)","claim":"A100 peaks at k*=1024","status":"pending","needs_gpu":True,"requires":"A100"},
    {"id":"I3","paper":"IX (I)","claim":"H100 peaks at k*=1280","status":"pending","needs_gpu":True,"requires":"H100"},
    {"id":"I4","paper":"IX (I)","claim":"KV-cache projection super-baseline","status":"pending","needs_gpu":False},
    {"id":"I5","paper":"IX (I)","claim":"LoRA FFN fusion super-baseline","status":"done","result":"VERIFIED at k=256 (+76.6%). REFINED: k* is kernel-specific.","needs_gpu":False},
    {"id":"I6","paper":"IX (I)","claim":"L40S follows prediction curve","status":"pending","needs_gpu":True,"requires":"EC2 L40S"},
    # PAPER J: CECI Chimeric Splicing
    {"id":"J1","paper":"X (J)","claim":"Within-model within-band splicing viable","status":"done","result":"FALSIFIED: 0/120 pairs viable at k=32","needs_gpu":False},
    {"id":"J2","paper":"X (J)","claim":"Cross-model shared-init splice viable","status":"running","needs_gpu":True,"runtime_hrs":3},
    # FUTURE PAPERS (XI-XV) — structural predictions, not yet measured
    {"id":"K1","paper":"XI","claim":"UGT taxonomic orthogonality penalty converges","status":"blueprint","needs_gpu":True,"runtime_hrs":"24-72"},
    {"id":"K2","paper":"XI","claim":"UGT models hot-swap attention heads seamlessly","status":"blueprint","needs_gpu":True},
    {"id":"L1","paper":"XII","claim":"Native k-space training matches ambient quality","status":"blueprint","needs_gpu":True,"runtime_hrs":"48-168"},
    {"id":"L2","paper":"XII","claim":"Riemannian AdamW beats standard AdamW at k-space","status":"blueprint","needs_gpu":True},
    {"id":"M1","paper":"XIII","claim":"OGD produces structurally valid novel concepts","status":"blueprint","needs_gpu":True},
    {"id":"M2","paper":"XIII","claim":"45% deviation generates coherent novel proofs","status":"blueprint","needs_gpu":True},
    {"id":"N1","paper":"XIV","claim":"Null-space projection permanently removes behaviors","status":"blueprint","needs_gpu":True},
    {"id":"N2","paper":"XIV","claim":"Ablated model resists jailbreak attempts","status":"blueprint","needs_gpu":True},
    {"id":"O1","paper":"XV","claim":"Topological event horizon blocks harmful generation","status":"blueprint","needs_gpu":True},
    {"id":"O2","paper":"XV","claim":"COG manifold expands without catastrophic forgetting","status":"blueprint","needs_gpu":True,"runtime_hrs":"ongoing"},
]

def save_registry():
    Path("benchmarks").mkdir(exist_ok=True)
    with open("benchmarks/experiment_registry.json","w") as f:
        json.dump(EXPERIMENTS, f, indent=2)
    
    done = sum(1 for e in EXPERIMENTS if e["status"]=="done")
    running = sum(1 for e in EXPERIMENTS if e["status"]=="running")
    pending = sum(1 for e in EXPERIMENTS if e["status"]=="pending")
    blueprint = sum(1 for e in EXPERIMENTS if e["status"]=="blueprint")
    
    print(f"Experiment Registry: {len(EXPERIMENTS)} total")
    print(f"   Done: {done} |  Running: {running} |  Pending: {pending} |  Blueprint: {blueprint}")
    print(f"  Saved: benchmarks/experiment_registry.json")

if __name__=="__main__":
    save_registry()
