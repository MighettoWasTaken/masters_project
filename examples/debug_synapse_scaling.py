#!/usr/bin/env python3
"""Quick diagnostic: verify synapse counts and timing for the scaling anomaly."""

import time
import numpy as np
from hodgkin_huxley import Network

# Import the benchmark helpers
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_network import _build_random_synapses, NumpyHHNetwork

N = 200
duration = 100.0
dt = 0.05
I_val = 10.0
num_steps = int(duration / dt)


def bench(label, synapses, trials=3):
    actual_s = len(synapses)
    net = Network(N)
    for pre, post, w, e, tau in synapses:
        net.add_synapse(pre, post, w, e, tau)
    I_ext = np.full((N, num_steps), I_val)

    times = []
    for _ in range(trials):
        net_t = Network(N)
        for pre, post, w, e, tau in synapses:
            net_t.add_synapse(pre, post, w, e, tau)
        start = time.perf_counter()
        net_t.simulate(duration, dt, I_ext)
        times.append(time.perf_counter() - start)

    print(f"  {label:>30s}  synapses={actual_s:>6d}  "
          f"times=[{', '.join(f'{t:.3f}' for t in times)}]  "
          f"mean={np.mean(times):.3f}s")


# --- Test 1: Original random synapses (set iteration order) ---
print("=== Test 1: Random synapses (set iteration order) ===")
for target_s in [10000, 15000, 20000, 25000, 30000, 35000, N*(N-1)]:
    synapses = _build_random_synapses(N, target_s)
    bench(f"random({target_s})", synapses)

# --- Test 2: Same random synapses but SORTED by (pre, post) ---
print("\n=== Test 2: Random synapses SORTED by (pre, post) ===")
for target_s in [10000, 20000, 30000, N*(N-1)]:
    synapses = _build_random_synapses(N, target_s)
    synapses_sorted = sorted(synapses, key=lambda s: (s[0], s[1]))
    bench(f"sorted({target_s})", synapses_sorted)

# --- Test 3: Fully-connected built sequentially (not via set) ---
print("\n=== Test 3: Fully-connected built sequentially ===")
synapses_seq = [(i, j, 0.5, 0.0, 2.0)
                for i in range(N) for j in range(N) if i != j]
bench("sequential_full", synapses_seq)
