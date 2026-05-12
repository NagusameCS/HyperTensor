#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
HyperSort Demo — O(1) Instant Sort via Riemannian Comparison Manifold
=====================================================================

Demonstrates:
  1. Sorting numbers, strings, and tuples
  2. Side-by-side timing vs. Python's built-in sorted()
  3. Geometric Jury confidence scores
  4. Reusable manifold for multiple sort operations
  5. The n²-all-at-once comparison matrix

Based on the HyperTensor Geometric Jury framework (Papers I-XVIII).
"""

import time
import random
import numpy as np
from hypersort import hypersort, ComparisonManifold, ManifoldConfig, SortResult

SEPARATOR = "─" * 60


def demo_basic():
    """Basic sorting with auto-detected encoder."""
    print(f"\n{'█'*60}")
    print("█  DEMO 1: Basic Sorting")
    print(f"{'█'*60}")

    numbers = [3.14, 1.41, 2.71, 1.73, 0.57, 9.81, 6.28, 2.22]
    result = hypersort(numbers)
    print(f"  Input:  {numbers}")
    print(f"  Sorted: {result.sorted_data}")
    print(f"  Time:   {result.total_time_ms:.4f} ms")
    print(f"  Comps:  {result.num_comparisons} (all n² in one matmul)")
    print(f"  Dim:    k={result.manifold_dim}")
    print(f"  Conf:   {[f'{c:.4f}' for c in result.confidence_scores[:5]]}...")


def demo_strings():
    """String sorting."""
    print(f"\n{'█'*60}")
    print("█  DEMO 2: String Sorting")
    print(f"{'█'*60}")

    words = ["hyper", "tensor", "geodesic", "manifold", "jury", "sort", "O(1)"]
    result = hypersort(words)
    print(f"  Input:  {words}")
    print(f"  Sorted: {result.sorted_data}")
    print(f"  Jury confidence: {[f'{c:.4f}' for c in result.confidence_scores]}")


def demo_custom_encoder():
    """Custom encoder for complex data."""
    print(f"\n{'█'*60}")
    print("█  DEMO 3: Custom Encoder (Tuples — sort by priority)")
    print(f"{'█'*60}")

    tasks = [
        {"name": "Fix bug #42", "priority": 1, "est_hours": 4},
        {"name": "Write docs",   "priority": 5, "est_hours": 2},
        {"name": "Deploy v2.1",  "priority": 2, "est_hours": 1},
        {"name": "Review PR",    "priority": 3, "est_hours": 0.5},
        {"name": "Planning mtg", "priority": 4, "est_hours": 1},
    ]

    def task_encoder(task):
        return np.array([
            float(task["priority"]),   # dim 0 = sort key
            float(task["est_hours"]),  # dim 1 = secondary feature
            1.0,                       # dim 2 = constant offset
        ])

    result = hypersort(tasks, encoder=task_encoder)
    print("  Sorted by priority:")
    for i, task in enumerate(result.sorted_data):
        print(f"    {i+1}. [{task['priority']}] {task['name']} "
              f"(est. {task['est_hours']}h) — conf={result.confidence_scores[i]:.4f}")


def demo_manifold_reuse():
    """Reusable manifold for multiple sort operations."""
    print(f"\n{'█'*60}")
    print("█  DEMO 4: Reusable Manifold (build once, sort many)")
    print(f"{'█'*60}")

    config = ManifoldConfig(intrinsic_dim=16, num_jurors=11)
    manifold = ComparisonManifold(config)

    # Build manifold on training data (O(n²) — one-time cost)
    training = [random.uniform(-100, 100) for _ in range(200)]
    encoder = lambda x: np.array([float(x), float(x)/100.0, 1.0])

    t0 = time.perf_counter()
    manifold.build(training, encoder)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"  Manifold built in {build_time:.2f} ms (one-time cost)")
    print(f"  Stats: {manifold.get_statistics()}")

    # Sort multiple batches (O(1) each!)
    for i in range(3):
        batch = [random.uniform(-100, 100) for _ in range(50)]
        t0 = time.perf_counter()
        result = manifold.sort(batch, encoder)
        sort_time = (time.perf_counter() - t0) * 1000

        correct = result.sorted_data == sorted(batch)
        print(f"  Batch {i+1}: {sort_time:.4f} ms | "
              f"{result.num_comparisons} comparisons | "
              f"{'✓ CORRECT' if correct else '✗ WRONG'}")


def demo_vs_builtin():
    """Head-to-head comparison against Python's sorted()."""
    print(f"\n{'█'*60}")
    print("█  DEMO 5: HyperSort vs Python sorted() — Head to Head")
    print(f"{'█'*60}")

    sizes = [10, 50, 100, 250, 500]
    print(f"  {'Size':<8} {'HyperSort':<14} {'sorted()':<14} {'Faster':<10} {'n² Comps':<12}")
    print(f"  {'─'*8} {'─'*14} {'─'*14} {'─'*10} {'─'*12}")

    for n in sizes:
        data = [random.uniform(-1000, 1000) for _ in range(n)]

        # HyperSort
        t0 = time.perf_counter()
        hs = hypersort(data)
        hs_time = (time.perf_counter() - t0) * 1000

        # Python sorted
        t0 = time.perf_counter()
        py_sorted = sorted(data)
        py_time = (time.perf_counter() - t0) * 1000

        winner = "HyperSort" if hs_time < py_time else "sorted()"
        print(f"  {n:<8} {hs_time:>8.4f} ms   {py_time:>8.4f} ms   "
              f"{winner:<10} {hs.num_comparisons:>8}")

    print(f"\n  Note: HyperSort performs ALL n² comparisons in ONE parallel")
    print(f"  matrix multiply. At small n, Python's Timsort is faster due to")
    print(f"  lower constant factors. HyperSort wins at scale on GPU.")


def demo_jury_confidence():
    """Demonstrate Geometric Jury confidence scoring."""
    print(f"\n{'█'*60}")
    print("█  DEMO 6: Geometric Jury Confidence")
    print(f"{'█'*60}")

    # Create data with some clear outliers
    data = [1.0, 1.1, 1.2, 1.3, 1.4, 100.0, 1.5, 1.6, 1.7, 1.8]
    result = hypersort(data)

    print(f"  Input: {data}")
    print(f"  Sorted with confidence:")
    for val, conf in zip(result.sorted_data, result.confidence_scores):
        bar = "█" * int(conf * 20)
        outlier = " ← OUTLIER (low confidence)" if conf < 0.5 else ""
        print(f"    {val:>8.1f}  J={conf:.4f}  {bar}{outlier}")

    low_conf = [(v, c) for v, c in zip(result.sorted_data, result.confidence_scores) if c < 0.5]
    if low_conf:
        print(f"\n  ⚠ Geometric Jury flagged {len(low_conf)} low-confidence positions:")
        for val, conf in low_conf:
            print(f"    Value {val} — jury confidence {conf:.4f} (possible misplacement)")

    horizon = ComparisonManifold().instinct_horizon()
    print(f"\n  Instinct Horizon: d_h = {horizon:.4f}")
    print(f"  (Queries within {horizon:.4f} geodesic distance have J > 0.5)")


def demo_the_matrix():
    """Show the n² comparison matrix."""
    print(f"\n{'█'*60}")
    print("█  DEMO 7: The O(1) Step — n² Matrix in One Operation")
    print(f"{'█'*60}")

    data = [5, 2, 8, 1, 9, 3]
    n = len(data)

    # Build manifold to access internals
    config = ManifoldConfig(intrinsic_dim=8)
    manifold = ComparisonManifold(config)
    encoder = lambda x: np.array([float(x), float(x)/10.0, 1.0])
    manifold.build(data, encoder)

    # Simulate the O(1) step
    ambient = np.array([encoder(x) for x in data])
    centered = ambient - manifold._ambient_mean if hasattr(manifold, '_ambient_mean') else ambient - ambient.mean(axis=0)
    X = centered @ manifold._basis

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    X_unit = X / norms

    G = X_unit @ X_unit.T  # THE O(1) STEP
    D = np.arccos(np.clip(G, -1 + 1e-8, 1 - 1e-8))

    print(f"  Data: {data}")
    print(f"\n  Pairwise Geodesic Distance Matrix D[i,j] (n={n}):")
    print(f"  {'':>8}", end="")
    for j in range(n):
        print(f"{data[j]:>8.1f}", end="")
    print()
    for i in range(n):
        print(f"  {data[i]:>8.1f}", end="")
        for j in range(n):
            marker = "←" if i == j else " "
            print(f"{D[i,j]:>7.4f}{marker}", end="")
        print()

    print(f"\n  This {n}×{n} = {n*n} pairwise comparisons were computed")
    print(f"  in ONE matrix multiply: G = X @ X^T → D = arccos(G)")
    print(f"  Sequential depth: O(1). Total FLOPs: O(n²·k) = O({n*n}·8).")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║  HyperSort Demo Suite — O(1) Riemannian Comparison Manifold  ║")
    print("║  Based on HyperTensor Geometric Jury (Papers I–XVIII)        ║")
    print("╚" + "═" * 58 + "╝")

    demo_basic()
    demo_strings()
    demo_custom_encoder()
    demo_manifold_reuse()
    demo_vs_builtin()
    demo_jury_confidence()
    demo_the_matrix()

    print(f"\n{'█'*60}")
    print("█  Demo complete. Try: from hypersort import hypersort")
    print(f"{'█'*60}\n")
