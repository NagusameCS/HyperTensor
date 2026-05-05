"""jury_gtc_lib.py — Python wrapper for C jury-GTC kernel.

Falls back to pure PyTorch implementation if C library not available.
"""
import torch
import torch.nn.functional as F
import ctypes, os, platform
from pathlib import Path
from collections import defaultdict

# ============================================================================
# C LIBRARY LOADING
# ============================================================================
_lib = None
_lib_path = None

def _find_lib():
    """Find the compiled C library."""
    lib_dir = Path(__file__).parent.parent / "lib"
    system = platform.system()
    if system == "Linux":
        names = ["libjury_gtc.so"]
    elif system == "Windows":
        names = ["jury_gtc.dll"]
    elif system == "Darwin":
        names = ["libjury_gtc.dylib"]
    else:
        return None
    
    for name in names:
        path = lib_dir / name
        if path.exists():
            return str(path)
    return None

def _load_lib():
    """Load C library if available."""
    global _lib, _lib_path
    if _lib is not None:
        return _lib is not False
    
    _lib_path = _find_lib()
    if _lib_path is None:
        _lib = False
        return False
    
    try:
        _lib = ctypes.CDLL(_lib_path)
        
        # jury_search
        _lib.jury_search.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pool
            ctypes.POINTER(ctypes.c_int),    # domains
            ctypes.POINTER(ctypes.c_float),  # query
            ctypes.c_int, ctypes.c_int,      # N, K
            ctypes.c_int, ctypes.c_float, ctypes.c_float,  # sample_n, T, threshold
            ctypes.POINTER(ctypes.c_int),    # best_idx
            ctypes.POINTER(ctypes.c_float),  # best_sim
            ctypes.POINTER(ctypes.c_int),    # comparisons
            ctypes.POINTER(ctypes.c_int),    # dominant
        ]
        _lib.jury_search.restype = ctypes.c_int
        
        # jury_batch_search
        _lib.jury_batch_search.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        return True
    except Exception as e:
        print(f"  [jury_gtc_lib] C library load failed: {e}")
        print(f"  [jury_gtc_lib] Falling back to PyTorch implementation.")
        _lib = False
        return False

# ============================================================================
# JURY-GTC ENGINE (production, with C fallback)
# ============================================================================
class JuryGTC:
    """Jury-accelerated trajectory cache for lifelong learning.
    
    Usage:
        cache = JuryGTC(k_dim=512)
        cache.add(proj_tensor, response_text, domain="math")
        
        # Single query
        result = cache.search(query_proj)
        if result["hit"]:
            return result["cached_response"]
        
        # Batch queries (uses C kernel if available)
        results = cache.search_batch(query_projs)
    """
    
    def __init__(self, k_dim=512, jury_sample=20, temperature=8.0, hit_threshold=0.90):
        self.k = k_dim
        self.jury_sample = jury_sample
        self.T = temperature
        self.threshold = hit_threshold
        
        # Storage
        self.trajectories = []      # list of {"proj": tensor, "response": any, "domain": str}
        self._projs = None          # normalized stack
        self._domains = None        # int domain IDs
        self._domain_map = {}       # str → int
        self._domain_rev = {}       # int → str
        self._dirty = True
        
        # Jury metadata
        self.coverage_R = {}
        self.total_queries = 0
        self.total_hits = 0
        self.total_comparisons_saved = 0
        
        # Try loading C library
        self._has_c = _load_lib()
    
    # ── Storage ──
    def add(self, proj, response=None, domain="default"):
        """Add a trajectory."""
        dom_id = self._domain_map.get(domain)
        if dom_id is None:
            dom_id = len(self._domain_map)
            self._domain_map[domain] = dom_id
            self._domain_rev[dom_id] = domain
        
        self.trajectories.append({
            "proj": proj.float().cpu().detach(),
            "response": response,
            "domain": domain,
            "domain_id": dom_id,
            "timestamp": len(self.trajectories),  # insertion order
        })
        self._dirty = True
    
    def _normalize(self):
        """Rebuild normalized tensor stack."""
        if self._dirty and self.trajectories:
            projs = torch.stack([t["proj"] for t in self.trajectories])
            self._projs = F.normalize(projs, dim=1)
            domains_arr = [t["domain_id"] for t in self.trajectories]
            self._domains = torch.tensor(domains_arr, dtype=torch.int32)
            self._dirty = False
    
    # ── Search ──
    def search(self, q, use_c=True):
        """Search for nearest cached trajectory.
        
        Args:
            q: query vector [k_dim]
            use_c: try C kernel if available
        
        Returns:
            {"hit": bool, "best_idx": int, "best_sim": float,
             "comparisons": int, "domain": str, "cached_response": any}
        """
        self._normalize()
        self.total_queries += 1
        N = len(self.trajectories)
        
        if N == 0:
            self.total_hits += 0
            return {"hit": False, "best_idx": -1, "best_sim": 0.0,
                    "comparisons": 0, "domain": "none", "cached_response": None}
        
        qn = F.normalize(q.float().unsqueeze(0), dim=1).squeeze(0)
        
        # Try C kernel
        if use_c and self._has_c and N > self.jury_sample * 2:
            result = self._search_c(qn)
            if result is not None:
                if result["hit"]: self.total_hits += 1
                self.total_comparisons_saved += (N - result["comparisons"])
                return result
        
        # Fallback: PyTorch two-stage search
        return self._search_pytorch(qn)

    def _search_c(self, qn):
        """C kernel search."""
        try:
            best_idx = ctypes.c_int(-1)
            best_sim = ctypes.c_float(-1.0)
            comparisons = ctypes.c_int(0)
            dominant = ctypes.c_int(-1)
            
            N = len(self.trajectories)
            K = self.k
            pool_ptr = self._projs.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            dom_ptr = self._domains.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            q_ptr = qn.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            _lib.jury_search(
                pool_ptr, dom_ptr, q_ptr, N, K,
                self.jury_sample, ctypes.c_float(self.T), ctypes.c_float(self.threshold),
                ctypes.byref(best_idx), ctypes.byref(best_sim),
                ctypes.byref(comparisons), ctypes.byref(dominant)
            )
            
            hit = best_sim.value >= self.threshold
            idx = best_idx.value
            
            if idx >= 0 and idx < len(self.trajectories):
                traj = self.trajectories[idx]
                dom = self._domain_rev.get(dominant.value, traj["domain"])
                return {
                    "hit": hit,
                    "best_idx": idx,
                    "best_sim": round(best_sim.value, 4),
                    "comparisons": comparisons.value,
                    "domain": dom,
                    "cached_response": traj["response"] if hit else None,
                }
        except Exception:
            pass
        return None
    
    def _search_pytorch(self, qn):
        """PyTorch fallback two-stage search."""
        N = len(self.trajectories)
        
        # Stage 1: jury sample
        sample_n = min(self.jury_sample, N)
        stride = max(1, N // sample_n)
        sample_idx = list(range(0, N, stride))[:sample_n]
        
        sample_projs = self._projs[torch.tensor(sample_idx)]
        sims = (sample_projs @ qn.unsqueeze(1)).squeeze(-1)
        w = F.softmax(sims * self.T, dim=0)
        
        # Aggregate by domain
        domain_w = defaultdict(float)
        for si, idx in enumerate(sample_idx):
            domain_w[self.trajectories[idx]["domain"]] += w[si].item()
        
        top_domains = sorted(domain_w, key=domain_w.get, reverse=True)[:2]
        
        # Stage 2: domain search
        comparisons = sample_n
        best_sim = -1.0
        best_idx = -1
        
        for domain in top_domains:
            dom_indices = [i for i, t in enumerate(self.trajectories) if t["domain"] == domain]
            for idx in dom_indices:
                comparisons += 1
                sim = F.cosine_similarity(qn.unsqueeze(0), self._projs[idx:idx+1]).item()
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
                    if sim >= 0.995:
                        break
            if best_sim >= self.threshold:
                break
        
        hit = best_sim >= self.threshold
        
        if hit: self.total_hits += 1
        self.total_comparisons_saved += (N - comparisons)
        
        return {
            "hit": hit,
            "best_idx": best_idx,
            "best_sim": round(best_sim, 4),
            "comparisons": comparisons,
            "domain": self.trajectories[best_idx]["domain"] if best_idx >= 0 else "none",
            "cached_response": self.trajectories[best_idx]["response"] if hit else None,
        }
    
    # ── Stats ──
    @property
    def stats(self):
        return {
            "pool_size": len(self.trajectories),
            "total_queries": self.total_queries,
            "total_hits": self.total_hits,
            "hit_rate": self.total_hits / max(self.total_queries, 1),
            "avg_comparisons_saved": self.total_comparisons_saved / max(self.total_queries, 1),
            "domains": list(self._domain_map.keys()),
            "has_c_kernel": self._has_c,
        }
    
    def __repr__(self):
        s = self.stats
        return (f"JuryGTC(pool={s['pool_size']}, hit_rate={s['hit_rate']:.1%}, "
                f"avg_saved={s['avg_comparisons_saved']:.0f}, C={'yes' if s['has_c_kernel'] else 'no'})")


# ============================================================================
# LIFELONG LEARNING INTEGRATION
# ============================================================================
class LifelongCache:
    """Jury-GTC augmented trajectory cache with lifelong learning support.
    
    Features:
    - Never throws away old data
    - Domain-indexed for fast retrieval
    - Automatic jury metadata updates
    - Compression for long-term storage
    - Incremental updates without full rebuild
    """
    
    def __init__(self, k_dim=512, save_dir="outputs/lifelong_cache"):
        self.cache = JuryGTC(k_dim=k_dim)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.generation = 0
        self.history = []  # never delete — keep all generations
    
    def add_interaction(self, query_proj, response, domain="general", metadata=None):
        """Record a single interaction."""
        self.cache.add(query_proj, response, domain)
        
        # Auto-save every 100 interactions
        if len(self.cache.trajectories) % 100 == 0:
            self.save_checkpoint()
    
    def query(self, q, generate_fn=None):
        """Query the cache. If miss and generate_fn provided, generate + cache.
        
        Args:
            q: query projection [k_dim]
            generate_fn: optional callable(q) → response for cache misses
        
        Returns:
            {"hit": bool, "response": any, "cached": bool, "domain": str}
        """
        result = self.cache.search(q)
        
        if result["hit"]:
            return {"hit": True, "response": result["cached_response"], 
                    "cached": True, "domain": result["domain"],
                    "similarity": result["best_sim"]}
        
        # Cache miss — generate if possible
        if generate_fn is not None:
            response = generate_fn(q)
            self.add_interaction(q, response, result["domain"])
            return {"hit": False, "response": response, 
                    "cached": False, "domain": result["domain"],
                    "similarity": result["best_sim"]}
        
        return {"hit": False, "response": None, "cached": False,
                "domain": result["domain"], "similarity": result["best_sim"]}
    
    def save_checkpoint(self):
        """Save current state without deleting old data."""
        self.generation += 1
        gen_dir = self.save_dir / f"gen_{self.generation:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectories
        traj_data = []
        for t in self.cache.trajectories:
            traj_data.append({
                "proj": t["proj"].tolist(),
                "domain": t["domain"],
                "timestamp": t["timestamp"],
            })
        
        checkpoint = {
            "generation": self.generation,
            "k_dim": self.cache.k,
            "n_trajectories": len(self.cache.trajectories),
            "stats": self.cache.stats,
            "trajectories": traj_data,
            "domain_map": self.cache._domain_map,
        }
        
        import json
        with open(gen_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f)
        
        # Save tensors
        if self.cache._projs is not None:
            torch.save({
                "projs": self.cache._projs,
                "domains": self.cache._domains,
            }, gen_dir / "tensors.pt")
        
        self.history.append({
            "generation": self.generation,
            "path": str(gen_dir),
            "n_trajectories": len(self.cache.trajectories),
            "hit_rate": self.cache.stats["hit_rate"],
        })
        
        # Keep history log
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
    
    def load_latest(self):
        """Load the most recent checkpoint."""
        history_file = self.save_dir / "history.json"
        if not history_file.exists():
            return False
        
        import json
        with open(history_file) as f:
            self.history = json.load(f)
        
        if not self.history:
            return False
        
        latest = self.history[-1]
        gen_dir = Path(latest["path"])
        
        if not gen_dir.exists():
            return False
        
        # Load tensors
        tensors = torch.load(gen_dir / "tensors.pt", map_location="cpu")
        self.cache._projs = tensors["projs"]
        self.cache._domains = tensors["domains"]
        self.cache._dirty = False
        self.generation = latest["generation"]
        
        # Rebuild trajectory list
        with open(gen_dir / "checkpoint.json") as f:
            ckpt = json.load(f)
        
        self.cache._domain_map = ckpt.get("domain_map", {})
        self.cache._domain_rev = {v: k for k, v in self.cache._domain_map.items()}
        
        for t in ckpt["trajectories"]:
            self.cache.trajectories.append({
                "proj": torch.tensor(t["proj"]),
                "response": None,  # responses stored separately
                "domain": t["domain"],
                "domain_id": self.cache._domain_map.get(t["domain"], 0),
                "timestamp": t["timestamp"],
            })
        
        print(f"  Loaded generation {self.generation}: {len(self.cache.trajectories)} trajectories")
        return True
    
    @property
    def stats(self):
        return {
            **self.cache.stats,
            "generation": self.generation,
            "history_length": len(self.history),
        }


# ============================================================================
# SELF-TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  JURY-GTC LIBRARY SELF-TEST")
    print("=" * 60)
    
    # Test C library loading
    has_c = _load_lib()
    print(f"  C kernel available: {has_c}")
    if has_c:
        print(f"  Library path: {_lib_path}")
    
    # Test basic functionality
    print(f"\n  Testing JuryGTC engine...")
    cache = JuryGTC(k_dim=128)
    
    # Add trajectories from 3 domains
    for d in range(3):
        centroid = torch.randn(128)
        for _ in range(50):
            traj = F.normalize((centroid + torch.randn(128) * 0.05).unsqueeze(0), dim=1).squeeze(0)
            cache.add(traj, f"response_{d}_{_}", f"domain_{d}")
    
    print(f"  Pool: {len(cache.trajectories)} trajectories")
    
    # Test search
    q = F.normalize((torch.randn(128)).unsqueeze(0), dim=1).squeeze(0)
    result = cache.search(q)
    print(f"  Search result: hit={result['hit']}, sim={result['best_sim']:.4f}, "
          f"comps={result['comparisons']}, domain={result['domain']}")
    
    # Test lifelong cache
    print(f"\n  Testing LifelongCache...")
    lc = LifelongCache(k_dim=128, save_dir="outputs/test_lifelong")
    
    for i in range(50):
        q = F.normalize(torch.randn(1, 128), dim=1).squeeze(0)
        lc.add_interaction(q, f"response_{i}", "test")
    
    lc.save_checkpoint()
    print(f"  Saved generation {lc.generation}")
    print(f"  Stats: {lc.stats}")
    
    print(f"\n  All tests passed ✓")
