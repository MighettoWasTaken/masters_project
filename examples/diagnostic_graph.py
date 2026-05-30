#!/usr/bin/env python3
"""
Diagnostic: verify CUDA graph capture is reducing per-step overhead.

If the graph is working, sparse recording (10 syncs total) should be
5-10x faster than dense recording (sync every step), because the only
remaining per-step cost is cudaStreamSynchronize + D2H copy for V.

If sparse ≈ dense, the graph is not firing and the old device-sync path
is still dominating.
"""

import sys
import time

import hodgkin_huxley as hh
from hodgkin_huxley import RegionalNetwork, SynapseSpec, RecordingConfig


def main():
    if not hh.cuda_is_available():
        print("No CUDA GPU — cannot run diagnostic.")
        sys.exit(0)

    print(f"GPU: {hh.cuda_device_name(0)}")
    print()

    rn = RegionalNetwork()
    rn.add_population("E", 1024)
    rn.connect(
        "E", "E",
        lambda n, m: [(i, (i + 1) % n) for i in range(n)],
        weight=0.3,
        synapse=SynapseSpec.ampa(),
    )
    rn.to(hh.Device.cuda(0))

    # warmup — also captures the graph for the first time
    rn.simulate(20.0, 0.05, {"E": 8.0})

    duration = 500.0   # ms
    dt = 0.05          # ms
    n_steps = int(duration / dt)  # 10 000 steps

    # Dense: sync + D2H every step
    t0 = time.perf_counter()
    rn.simulate(duration, dt, {"E": 8.0})
    t1 = time.perf_counter()
    dense_ms = (t1 - t0) * 1000

    # Sparse: sync + D2H only 10 times across 10 000 steps
    t2 = time.perf_counter()
    rn.simulate(duration, dt, {"E": 8.0},
                record=RecordingConfig(["V"], interval=1000))
    t3 = time.perf_counter()
    sparse_ms = (t3 - t2) * 1000

    ratio = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")

    print(f"Steps: {n_steps}")
    print(f"Dense  recording (sync every step):  {dense_ms:7.1f} ms  "
          f"({dense_ms / n_steps * 1000:.1f} µs/step)")
    print(f"Sparse recording (sync x10 total):   {sparse_ms:7.1f} ms  "
          f"({sparse_ms / n_steps * 1000:.1f} µs/step)")
    print(f"Ratio dense/sparse: {ratio:.1f}x")
    print()
    if ratio >= 4.0:
        print("GRAPH IS FIRING: sparse is significantly faster — "
              "per-step sync overhead is the bottleneck, not launch overhead.")
    else:
        print("GRAPH MAY NOT BE FIRING: ratio < 4x — "
              "per-step cost is similar with and without recording, "
              "suggesting device-sync overhead is dominating both paths.")


if __name__ == "__main__":
    main()
