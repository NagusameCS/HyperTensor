# ==============================================================================
# TensorOS — Architecture Deep Dive
# ==============================================================================

## Boot Sequence

```
BIOS/UEFI → GRUB (Multiboot2) → boot.asm → kernel_main() → AI Shell
```

### Phase 1: Hardware Init (boot.asm)
1. Multiboot2 header validated by GRUB
2. GDT loaded (64-bit long mode segments)
3. Page tables: identity-map first 1GB + tensor memory region (4GB-8GB)
4. Enable PAE, long mode (IA32_EFER.LME), paging
5. Jump to 64-bit `kernel_main`

### Phase 2: Kernel Subsystem Init (main.c)
1. `tensor_mm_init()` — physical bitmap, tensor heap, model cache, slab allocator
2. `tensor_sched_init()` — priority queues, GPU device state
3. `gpu_init()` / `tpu_init()` — PCI bus scan, capability detection
4. `git_init()` — kernel-level git object store
5. `tensorfs_init()` — AI-aware virtual filesystem
6. `sandbox_init()` — security subsystem
7. `tensor_ipc_init()` — IPC channels
8. `virt_init()` — VT-x/AMD-V detection, container support

### Phase 3: Runtime Init
1. `tensor_engine_init()` — eager ops, compute graph engine
2. `pseudo_runtime_init()` — Pseudocode JIT (lexer, parser ready)
3. `modelpkg_init()` — model registries

### Phase 4: Userland
1. `monitor_daemon_main()` — background monitoring (future: separate MEU)
2. `deploy_init()` / `train_init()` — services ready
3. `aishell_main()` — interactive shell starts, banner displayed


## Memory Layout

```
0x0000_0000_0000_0000  ┌──────────────────────────────┐
                       │  Identity-mapped first 1GB    │
                       │  (kernel code, stack, data)   │
0x0000_0000_4000_0000  ├──────────────────────────────┤
                       │  ...                          │
0x0000_0001_0000_0000  ├──────────────────────────────┤  (4 GB)
                       │  Tensor Memory Region         │
                       │  (4 GB, mapped in boot.asm)   │
                       │  ┌────────────────────────┐   │
                       │  │ Tensor Heap (256 MB)   │   │
                       │  │ (2MB huge pages)       │   │
                       │  ├────────────────────────┤   │
                       │  │ Model Cache (512 MB)   │   │
                       │  │ (LRU, 64 entries)      │   │
                       │  ├────────────────────────┤   │
                       │  │ Git Object Store (64MB)│   │
                       │  ├────────────────────────┤   │
                       │  │ IPC Shared Buffers     │   │
                       │  └────────────────────────┘   │
0x0000_0002_0000_0000  └──────────────────────────────┘  (8 GB)
```

### Tensor Heap
- Bump allocator with free-list fallback
- Coalesces adjacent free blocks
- 2MB huge page alignment for GPU DMA

### Model Weight Cache
- LRU eviction with 64 entry slots
- Each entry: name hash → physical address + reference count
- Cache hit = instant model load; cache miss = load from TensorFS

### Slab Allocator
8 size classes: 16, 32, 64, 128, 256, 512, 1024, 2048 bytes.
Used for kernel structures (MEU descriptors, scheduler queues, git objects).


## Scheduler Design

```
Priority Queues:
  [REALTIME]  ──→ Safety-critical inference (medical, autonomous)
  [CRITICAL]  ──→ Low-latency serving
  [HIGH]      ──→ Interactive inference
  [NORMAL]    ──→ Batch inference, training
  [LOW]       ──→ Background optimization
  [IDLE]      ──→ Model prefetching, cache warming
```

### Dispatch Policies
- **THROUGHPUT**: Maximize tensor ops/sec (batch-friendly)
- **LATENCY**: Minimize time-to-first-token
- **EFFICIENCY**: Minimize power consumption
- **FAIR**: Equal GPU time across MEUs

### GPU Scoring
When assigning an MEU to a GPU, the scheduler computes:
```
score = (available_VRAM × 4)
      + ((100 - utilization%) × 2)
      + ((100 - temperature°C) × 1)
      + (weight_locality_bonus × 8)
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
1. **Op Fusion**: MATMUL → ADD (bias) → RELU fused into single kernel
2. **Precision Auto-Downgrade**: FP32 → FP16 for compute, FP32 for accumulation
3. **Dead Code Elimination**: Remove unused tensor computations
4. **Memory Planning**: Static allocation of tensor buffers, reuse across ops


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
