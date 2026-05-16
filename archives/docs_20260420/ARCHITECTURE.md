<!-- :%#@=+-.                              .-=+@#%: -->
<!--  .:+#%@@@@@@@@@@@@@@@@@@@@@@@@#=...:=#@@@@@@@@@@@@@@@@@@@@@@@@%#+:. -->
<!--   .:=#@@@@@@@@@@@@@@@@@@@@@%-.:+%@@@@@#=:=%@@@@@@@@@@@@@@@@@%+-. -->
<!--      :-%@@@@@@@@@@@@@@@@@@@#=#@@@@@@@@@@@%=-@@@@@@@@@@@@@@@. -->
<!--         :+#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%+=*@@@@@@@@@@@#: -->
<!--            -*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#=#@@@@@@@@@+. -->
<!--              .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@@@@@@@*. -->
<!--                .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--                  -%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*. -->
<!--                   :%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%: -->
<!--                    .*@@@@@@@@@@@@@@@@@@@@@@@@@@@@%. -->
<!--                     :#@@@@@@@@@@@@@@@@@@@@@@@@@@@*. -->
<!--                      .#@@@@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                       .*@@@@@@@@@@@@@@@@@@@@@@@@+ -->
<!--                        .@@@@@@@@%+-::=@@@@@@@@: -->
<!--                        .*@@@@@@@.       .+@@@@@@: -->
<!--                        .*@@@@@@@:         :@@@@@@: -->
<!--                        .*@@@@@@@%+-::::-+#@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   HyperTensor -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   universal geometric tensor framework -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@: -->
<!--                        .*@@@@@@@@@@@@@@@@@@@@@@@:   Papers I-XXX -->
<!--                        :#@@@@@@@@@@@@@@@@@@@@@@@*.   15/18 at 100% -->
<!--                       .*@@@@@@@@@@@@@@@@@@@@@@@@%.   Jury-GTC @ 53x -->
<!--                      .+@@@@@@@@@@@@@@@@@@@@@@@@@@#:   AGT @ 50K primes -->
<!--                     -%@@@@@@@@@@@@@@@@@@@@@@@@@@@@*.   External verify 14/14 -->
<!--                   :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+.   COG 10K converged -->
<!--                 .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*.  Bilateral 1.5B 0.968 -->
<!--               :*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--            .-*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%. -->
<!--         .-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@. -->
<!--      .-*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#: -->
<!--   .:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%+: -->
<!-- .:+#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%#+:. -->

# ==============================================================================
# TensorOS --- Architecture Deep Dive
# ==============================================================================

## Boot Sequence

```
BIOS -> multibootstub.asm (Multiboot1) -> boot.asm -> entry64.asm -> kernelmain() -> AI Shell
```

### Phase 1: Hardware Init (multiboot_stub.asm + boot.asm)
1. Multiboot1 header validated (magic `0x1BADB002`)
2. `multiboot_stub.asm`: serial checkpoints, CPUID checks, builds 16 GB identity map
   with 2 MB huge pages (PML4 -> PDPT -> PD at `0x10000`, 18 pages)
3. Loads `kernel64.bin` at `0x200000`, jumps to 32-bit `boot.asm`
4. `boot.asm`: GDT loaded, PAE + long mode (IA32_EFER.LME) enabled
5. Jump to 64-bit `entry64.asm` -> `kernel_main`

### Phase 2: Kernel Subsystem Init (main.c)
1. `tensormminit()` --- physical bitmap, tensor heap, model cache, slab allocator
2. `smp_init()` --- LAPIC enable, AP bootstrap via INIT-SIPI-SIPI (up to 64 CPUs)
3. `tensorschedinit()` --- priority queues, GPU device state
4. `gpuinit()` / `tpuinit()` --- PCI bus scan, capability detection
5. `git_init()` --- kernel-level git object store
6. `tensorfs_init()` --- AI-aware virtual filesystem
7. `sandbox_init()` --- security subsystem
8. `tensoripcinit()` --- IPC channels
9. `virt_init()` --- VT-x/AMD-V detection, container support

### Phase 3: Runtime Init
1. `tensorengineinit()` --- eager ops, compute graph engine
2. `pseudoruntimeinit()` --- Pseudocode JIT (lexer, parser ready)
3. `modelpkg_init()` --- model registries

### Phase 4: Userland
1. `monitordaemonmain()` --- background monitoring (future: separate MEU)
2. `deployinit()` / `traininit()` --- services ready
3. `aishell_main()` --- interactive shell starts, banner displayed


## Memory Layout

Page tables are built by `multiboot_stub.asm` at physical address `0x10000` (18 pages).
The first 16 GB of physical memory is identity-mapped using 2 MB huge pages.

```
0x000000000000_0000  
                         Kernel code + data           
                         (loaded at 0x200000)         
0x000000000001_0000    Page tables (18 pages)       
                         PML4, PDPT, 8 PD            
                       
                         SMP trampoline (0x8000)      
                         AP stacks (65 KB each)       
                       
                         Identity-mapped first 16 GB  
                         (2 MB huge pages)            
                            
                          Tensor Heap               
                          (dynamic, 2MB pages)      
                            
                          Model Cache (LRU, 64)     
                            
                          JIT Code Pool (2 MB)      
                          W^X, max 64 buffers       
                            
                          Git Object Store          
                            
                          IPC Shared Buffers        
                            
0x000000040000_0000    (16 GB)
```

Actual heap/cache sizes depend on available RAM (detected from Multiboot1 mmap):
- 8 GB config: tensor heap ~4992 MB, model cache ~2976 MB
- 4 GB config: tensor heap ~256 MB, model cache ~512 MB

### Tensor Heap
- Bump allocator with free-list fallback
- Coalesces adjacent free blocks
- 2MB huge page alignment for GPU DMA

### Model Weight Cache
- LRU eviction with 64 entry slots
- Each entry: name hash -> physical address + reference count
- Cache hit = instant model load; cache miss = load from TensorFS

### Slab Allocator
8 size classes: 16, 32, 64, 128, 256, 512, 1024, 2048 bytes.
Used for kernel structures (MEU descriptors, scheduler queues, git objects).


## Scheduler Design

```
Priority Queues:
  [REALTIME]  -> Safety-critical inference (medical, autonomous)
  [CRITICAL]  -> Low-latency serving
  [HIGH]      -> Interactive inference
  [NORMAL]    -> Batch inference, training
  [LOW]       -> Background optimization
  [IDLE]      -> Model prefetching, cache warming
```

### Dispatch Policies
- THROUGHPUT: Maximize tensor ops/sec (batch-friendly)
- LATENCY: Minimize time-to-first-token
- EFFICIENCY: Minimize power consumption
- FAIR: Equal GPU time across MEUs

### GPU Scoring
When assigning an MEU to a GPU, the scheduler computes:
```
score = (available_VRAM  4)
      + ((100 - utilization%)  2)
      + ((100 - temperature°C)  1)
      + (weightlocalitybonus  8)
```
Weight locality bonus rewards GPUs that already have the model's weights cached.

### Batch Coalescing
If multiple MEUs request the same operation (e.g., matmul with same shapes),
the scheduler coalesces them into a single batched GPU dispatch for throughput.


## Tensor IR (TIR)

The intermediate representation used by the Pseudocode JIT and tensor engine:

| Opcode | Description |
|--------|-------------|
| TIR_LOAD | Load tensor from memory |
| TIR_STORE | Store tensor to memory |
| TIR_MATMUL | Matrix multiplication |
| TIR_ADD / MUL / DIV / SUB | Elementwise arithmetic |
| TIR_RELU / GELU / SILU / TANH | Activations |
| TIR_SOFTMAX | Softmax normalization |
| TIR_LAYERNORM | Layer normalization |
| TIR_ATTENTION | Fused multi-head attention |
| TIR_CONV2D | 2D convolution |
| TIR_POOL | Pooling (max/avg) |
| TIR_EMBEDDING | Embedding lookup |
| TIR_TRANSPOSE | Tensor transpose |
| TIR_RESHAPE | Tensor reshape |
| TIR_CONCAT | Tensor concatenation |
| TIR_SPLIT | Tensor split |
| TIR_REDUCE | Reduction (sum/mean/max) |
| TIR_CAST | Dtype conversion |
| TIR_ALLOC / FREE | Memory management |
| TIR_BRANCH / CALL / RET | Control flow |

### Optimization Passes
1. Op Fusion: MATMUL -> ADD (bias) -> RELU fused into single kernel
2. Precision Auto-Downgrade: FP32 -> FP16 for compute, FP32 for accumulation
3. Dead Code Elimination: Remove unused tensor computations
4. Memory Planning: Static allocation of tensor buffers, reuse across ops


## Security Model

Three sandbox policies:

| Policy | Tensor Ops | GPU | Network | Filesystem | IPC |
|--------|-----------|-----|---------|------------|-----|
| STRICT | Allowed | No | No | No | No |
| STANDARD | Allowed | Allowed | Read-only | Read-only | Allowed |
| PERMISSIVE | Allowed | Allowed | Allowed | Read/Write | Allowed |

- Every MEU has a permission bitmask (11 flags)
- Audit ring buffer logs all security-relevant operations
- Deterministic mode: fixed random seeds, no timing side channels
- Resource accounting: per-MEU memory and compute time tracking


## SMP Architecture

### Bootstrap
- BSP enables LAPIC, copies trampoline to physical `0x8000`
- Sends INIT-SIPI-SIPI to all APs (up to `MAX_CPUS=64`)
- Each AP: enters real mode at `0x8000`, transitions to protected -> long mode,
  gets a 65 KB stack, increments `smp.ap_started`, enters idle loop

### Work Dispatch
```
BSP                     AP 0                  AP 1                  AP 2
                                                                  
   smp_dispatch(fn)  wake via IPI 0xFE                       
                         execute fn(arg)                         
   smp_dispatch(fn)                       execute fn(arg)    
   smp_dispatch(fn)                                            execute fn(arg)
    (BSP does own share)                                          
   smpwaitall()  barrier  barrier 
                                                                  
```

### Parallel GEMV
When `ncpu > 1 && out_dim >= 64`, GEMV rows are partitioned across CPUs:
- CPU `c` processes rows `[c  rowspercpu, (c+1)  rowspercpu)`
- BSP executes its share directly, APs receive work via `smp_dispatch()`
- All CPUs join via `smpwaitall()` before the result is used
- Supports both Q40 and Q80 fused AVX2 GEMV paths


## JIT Compilation

### x8664 JIT Engine (`x86jit.c`)
- 2 MB executable code pool with W^X protection
- Max 64 concurrent JIT buffers
- Full x86_64 instruction encoder: REX, ModR/M, SIB, SSE2, AVX2 opcodes
- Register allocator using System V calling convention

### LLM Forward Kernels (`llm_jit.c`)
Lazy-compiled on first inference call. JIT pool: 2 MB W^X memory, up to 64 buffers.
Kernels are SSE2 4-wide (`v4f`). AVX2 is used in separate GEMV helper paths.

| Kernel | Operation | Used For |
|--------|-----------|----------|
| `fast_exp` | Fast scalar exp | Building block for silu/gelu |
| `silu` | x / (1 + e^−x) | FFN gate activation |
| `rmsnorm` | x / √(mean(x²) + ε) · w | Layer normalization |
| `rope` | Rotary position encoding | Q/K position embedding (non-Gemma4 only) |
| `elmul` | a[i]  b[i] | Element-wise multiply |
| `eladd` | a[i] + b[i] | Residual connections |
| `dot` | Σ a[i]·b[i] | Attention score computation |
| `axpy` | a[i] += α·b[i] | Attention value accumulation |
| `fusedsilumul` | silu(a[i]) · b[i] | Fused FFN gate ⊙ up projection |
| `gelu` | GELU activation | GELU-variant FFN |
| `layernorm` | (x−μ)/σ · w + b | LayerNorm (Phi-3 etc.) |
| `q80gemv` | Q80 quantized GEMV | CPU GEMV for Q80 weights |
| `q40q80gemv` | Q40  Q80 integer GEMV | CPU GEMV for Q4_0 weights |

Known JIT TODOs: softmax kernel, Gemma4-safe RoPE, LRU kernel cache, fused RMSNorm+scale, batched GEMV, AVX-512 variant.


## OTT / Axiom Beta Subsystem

The Organic Training Theory (OTT) research subsystem is integrated into the Geodessical
host runtime under `runtime/nn/axiom_*.{c,h}`. It operates as a parallel survey pipeline
that analyzes the geometric structure of loaded GGUF models.

### Component Map

```

               axiom_beta.c  (5-phase pipeline)               
  Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5   

                                               
                                               
axiomlinalg   llm.h/llm.c    axiomgeo.c    tensor_bridge
(PCA, TwoNN,   (embeddings,   (metric field, (hidden-state
 Jacobi eig,   hidden states, Christoffel,   capture/inject)
 dequant)      oracle calls)  RK4 geodesic)
```

### File Inventory

| File | Purpose |
|------|---------|
| `runtime/nn/axiomlinalg.h/c` | Dense matrix, PCA (Jacobi eigdecomp), TwoNN, Q40/Q80/Q6K/F16/BF16/F32 dequant |
| `runtime/nn/axiom_geo.h/c` | Riemannian metric field (IDW), Christoffel symbols, Ricci/scalar curvature, RK4 geodesic integrator |
| `runtime/nn/axiom_beta.h/c` | 5-phase pipeline driver, fast-mode clamp policy, knowledge injection, warp state persistence |
| `axiomwarpstate.dat` | Persistent warp accumulation state (survives restarts) |
| `axiombetareport.json` | Full JSON survey report (phases 1--5 + timings) |
| `ottreadinessreport.json` | Operational readiness flags for OTT subsystems |

### Five-Phase Pipeline Summary

| Phase | Operation | Key Output | Typical Time (fast) |
|-------|-----------|------------|---------------------|
| 1 | PCA + TwoNN on token embeddings | Intrinsic dim k≈14--41 | ~128 ms |
| 2 | Attention head weight fingerprints | Symmetry score, generators | ~1 ms |
| 3 | IDW metric field + Christoffel symbols + Ricci curvature | Scalar curvature field (warm: 0.17 s) | ~43 ms warm |
| 4 | Uncertainty-driven oracle axiom loop | Axiom set, consistency score | ~669 ms |
| 5 | RK4 geodesic pilot vs decode-aligned oracle targets | top1, MRR, speedup projection | ~43 ms |

### Tensor Bridge Integration

`tensor_bridge` provides the hidden-state capture API used by the axiom subsystem:
- `tensorbridgecapture(layer, hidden_state)` --- intercepts forward pass at layer N
- `tensorbridgeinject(layer, hidden_state)` --- injects modified hidden state
- Used by Phase 3 metric field construction and Phase 5 oracle target acquisition

### Knowledge Injection Design

```
  Phase 3 metric field (cached)
         
         
  Local Christoffel warp:
    Γ'ᵏᵢ(x) = Γᵏᵢ(x) + α · Δᵏᵢ · exp(−‖x − xᵢₙ‖² / 2σ²)
         
         
  RK4 geodesic integrator (Phase 5)
         
         
  Warp accumulation in axiomwarpstate.dat
         
  threshold exceeded?
    Yes -> full Phase 3+4 refresh (recalc_triggered = 1)
    No  -> continue accumulating
```

### OTT Readiness Gauges (April 2026)

| Component | Readiness |
|-----------|-----------|
| Geometry foundation (metric/Christoffel/curvature) | 70% |
| Axiom discovery (active learning + model oracle) | 65% |
| Geodesic inference replacement path | 35% |
| Knowledge injection (local curvature warp) | 55% |
| End-to-end OTT production replacement | 25% |
| Overall | ~70% |

