import torch, time, math, random
import torch.nn.functional as F
torch.set_grad_enabled(False)
device = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print()

for K in [128, 256, 512]:
    for N in [300, 1000, 3000, 10000]:
        pool = F.normalize(torch.randn(N, K, device=device), dim=1)
        queries = F.normalize(torch.randn(100, K, device=device), dim=1)
        t0 = time.perf_counter()
        sims = queries @ pool.T
        best_idx = torch.argmax(sims, dim=1)
        naive_ms = (time.perf_counter()-t0)/100*1000
        jury_comps = 20 + N//6 + 20
        t0 = time.perf_counter()
        sample = pool[torch.randint(0,N,(20,),device=device)]
        s = queries @ sample.T
        w = F.softmax(s*8, dim=1)
        search_n = N//3
        s2 = queries @ pool[torch.randint(0,N,(search_n,),device=device)].T
        jury_ms = (time.perf_counter()-t0)/100*1000
        saved = (1 - jury_comps/N)*100
        speedup = naive_ms/max(jury_ms,0.001)
        print(f"K={K:>3d} N={N:>6d}: naive={naive_ms:.2f}ms, jury={jury_ms:.2f}ms, saved={saved:.0f}%, speedup={speedup:.0f}x")
    print()
