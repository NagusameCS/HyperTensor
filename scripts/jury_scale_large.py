import torch, time, math, random
import torch.nn.functional as F
torch.set_grad_enabled(False)
device = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print()

for K in [512]:
    for N in [10000, 50000, 100000, 200000, 500000]:
        try:
            pool = F.normalize(torch.randn(N, K, device=device), dim=1)
            queries = F.normalize(torch.randn(100, K, device=device), dim=1)
            
            # Warmup
            _ = queries @ pool.T
            torch.cuda.synchronize()
            
            # Naive (batched: 100 queries × N trajectories)
            t0 = time.perf_counter()
            for i in range(5):
                sims = queries @ pool.T
            torch.cuda.synchronize()
            naive_ms = (time.perf_counter()-t0)/500*1000  # per query
            
            # Simulated jury (search only ~N/3 trajectories)
            jury_pool_size = N // 3
            idx = torch.randint(0, N, (jury_pool_size,), device=device)
            t0 = time.perf_counter()
            for i in range(5):
                s2 = queries @ pool[idx].T
            torch.cuda.synchronize()
            jury_ms = (time.perf_counter()-t0)/500*1000
            
            speedup = naive_ms / max(jury_ms, 0.0001)
            ops_naive = N * K
            ops_jury = (20 + jury_pool_size) * K
            saved_pct = (1 - ops_jury/ops_naive) * 100
            
            print(f"N={N:>7d}: naive={naive_ms*1000:.1f}us, jury={jury_ms*1000:.1f}us, ops_saved={saved_pct:.0f}%, speedup={speedup:.1f}x")
        except RuntimeError as e:
            print(f"N={N:>7d}: OOM ({e})")
            break
