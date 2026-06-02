"""
GPU vs CPU speedup by neuron model complexity.

Same tier structure as benchmark_parallel_scaling.py but compares
CPU serial vs CUDA cooperative kernel instead of CPU threading.

Five complexity tiers:
  0  Izhikevich RS   -- 2 state vars
  1  HH-default      -- 3 gates
  2  TH-like         -- 5 gates
  3  GPe-like        -- 6 gates + Ca2+ ODE
  4  STN-like        -- 11 gates + Ca2+ + Nernst

Output: examples/figs/parallel_complexity_sweep.png  (overwrites threading version)
        examples/figs/gpu_complexity_N_sweep.png
"""

from __future__ import annotations

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hodgkin_huxley import (
    Device, IzhikevichType, NeuronModelSpec,
    RecordingConfig, RegionalNetwork, SynapseSpec,
)
from benchmarks.ctxbgth_model import _make_gpe_spec, _make_stn_spec, _make_th_spec

# ---------------------------------------------------------------------------
# Configuration  (mirrors benchmark_parallel_scaling.py)
# ---------------------------------------------------------------------------
N_POP      = 8
N_NEURONS  = 200
N_SWEEP    = [50, 100, 200, 500, 1000, 2000, 4000]
DURATION   = 200.0
DT         = 0.05
N_REPEATS  = 2
DELAY_MS   = 5.0

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGURE_DIR, exist_ok=True)

SPIKE_CFG  = RecordingConfig(["spikes"], spike_threshold=-10.0)

TIERS = [
    ("Izhikevich\n(2 vars)",   2,  False),
    ("HH-default\n(3 gates)",  3,  False),
    ("TH-like\n(5 gates)",     5,  False),
    ("GPe-like\n(6 g + Ca)",   6,  True ),
    ("STN-like\n(11 g + Ca)", 11,  True ),
]
N_TIERS = len(TIERS)

# ---------------------------------------------------------------------------
# Network builders
# ---------------------------------------------------------------------------

def _add_izhikevich(rn, n_pop, n):
    types = [
        IzhikevichType.REGULAR_SPIKING, IzhikevichType.FAST_SPIKING,
        IzhikevichType.INTRINSICALLY_BURSTING, IzhikevichType.CHATTERING,
        IzhikevichType.LOW_THRESHOLD_SPIKING, IzhikevichType.REGULAR_SPIKING,
        IzhikevichType.FAST_SPIKING, IzhikevichType.INTRINSICALLY_BURSTING,
    ]
    for i in range(n_pop):
        rn.add_population(f"P{i}", n, model=NeuronModelSpec.izhikevich(types[i % len(types)]))

def _add_hh_default(rn, n_pop, n):
    for i in range(n_pop):
        spec = NeuronModelSpec.hh_default()
        spec.name = f"HH_{i}"
        rn.add_population(f"P{i}", n, model=spec)

def _add_th(rn, n_pop, n):
    for i in range(n_pop):
        spec = _make_th_spec()
        spec.name = f"TH_{i}"
        rn.add_population(f"P{i}", n, model=spec)

def _add_gpe(rn, n_pop, n):
    for i in range(n_pop):
        spec, ca = _make_gpe_spec()
        spec.name = f"GPe_{i}"
        rn.add_population(f"P{i}", n, model=spec)
        rn.add_intracellular(ca, f"P{i}")

def _add_stn(rn, n_pop, n):
    for i in range(n_pop):
        spec, ca = _make_stn_spec()
        spec.name = f"STN_{i}"
        rn.add_population(f"P{i}", n, model=spec)
        rn.add_intracellular(ca, f"P{i}")

_ADDERS = [_add_izhikevich, _add_hh_default, _add_th, _add_gpe, _add_stn]

def _build_network(tier, n_pop, n):
    rn = RegionalNetwork()
    _ADDERS[tier](rn, n_pop, n)
    for i in range(n_pop - 1):
        rn.connect(f"P{i}", f"P{i+1}", "one_to_one",
                   weight=0.4, synapse=SynapseSpec.ampa(), delay=DELAY_MS)
    return rn

def _I_ext(n_pop):
    return {"P0": 10.0, **{f"P{i}": 0.0 for i in range(1, n_pop)}}

def _time_one(rn, n_pop):
    t0 = time.perf_counter()
    rn.simulate(DURATION, DT, _I_ext(n_pop), record=SPIKE_CFG)
    return time.perf_counter() - t0

def _measure(tier, n_pop, n, on_gpu):
    rn = _build_network(tier, n_pop, n)
    if on_gpu:
        rn.to(Device.cuda(0))
    rn.simulate(min(DURATION, 20.0), DT, _I_ext(n_pop), record=SPIKE_CFG)  # warmup
    return min(_time_one(rn, n_pop) for _ in range(N_REPEATS))

# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_complexity():
    results = np.zeros((N_TIERS, 2))
    print(f"\n--- Complexity sweep (n_pop={N_POP}, n={N_NEURONS}, duration={DURATION}ms) ---")
    for ti, (label, n_gates, has_ca) in enumerate(TIERS):
        t_cpu = _measure(ti, N_POP, N_NEURONS, on_gpu=False)
        t_gpu = _measure(ti, N_POP, N_NEURONS, on_gpu=True)
        results[ti] = [t_cpu, t_gpu]
        ca = "+Ca" if has_ca else "   "
        print(f"  Tier {ti} {ca} ({n_gates:2d}g)  "
              f"CPU={t_cpu:.3f}s  GPU={t_gpu:.3f}s  "
              f"speedup={t_cpu/t_gpu:.2f}x")
    return results

def sweep_n():
    results = np.zeros((N_TIERS, len(N_SWEEP), 2))
    print(f"\n--- N sweep (n_pop={N_POP}, duration={DURATION}ms) ---")
    for ti, (label, n_gates, has_ca) in enumerate(TIERS):
        ca = "+Ca" if has_ca else "   "
        print(f"  Tier {ti} {ca} ({n_gates:2d}g):")
        for ni, n in enumerate(N_SWEEP):
            t_cpu = _measure(ti, N_POP, n, on_gpu=False)
            t_gpu = _measure(ti, N_POP, n, on_gpu=True)
            results[ti, ni] = [t_cpu, t_gpu]
            print(f"    n={n:4d}  CPU={t_cpu:.3f}s  GPU={t_gpu:.3f}s  "
                  f"speedup={t_cpu/t_gpu:.2f}x")
    return results

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TIER_LABELS = [t[0] for t in TIERS]

def plot_complexity(results):
    speedup = results[:, 0] / results[:, 1]
    colors  = ["#2ca02c" if s >= 1.0 else "#d62728" for s in speedup]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(N_TIERS)
    ax.bar(x, speedup, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(1.0, ls="--", color="black", lw=1.2)

    for xi, v in enumerate(speedup):
        ax.text(xi, v + 0.03, f"{v:.2f}x", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABELS, fontsize=9)
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"GPU vs CPU Speedup by Neuron Model Complexity\n"
        f"(n={N_NEURONS} neurons/pop, n_pop={N_POP}, duration={DURATION}ms)"
    )

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], ls="--", color="black", lw=1.2, label="Breakeven"),
        Patch(color="#2ca02c", label="GPU faster"),
        Patch(color="#d62728", label="GPU slower"),
    ], fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "parallel_complexity_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out}")

def plot_n_sweep(results):
    speedup = results[:, :, 0] / results[:, :, 1]
    colors  = ["#aaaaaa", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for ti, (label, _, _) in enumerate(TIERS):
        ax.plot(N_SWEEP, speedup[ti], color=colors[ti], marker=markers[ti],
                lw=2, ms=7, label=label.replace("\n", " "))
    ax.axhline(1.0, ls="--", color="black", lw=1.2, label="Breakeven")
    ax.set_xlabel("Neurons per population")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title(
        f"GPU Speedup vs Population Size by Model Complexity\n"
        f"(n_pop={N_POP}, duration={DURATION}ms)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "gpu_complexity_N_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import hodgkin_huxley as hh
    if not hh.cuda_is_available():
        print("No CUDA GPU — exiting.")
        return
    try:
        print(f"GPU: {hh.cuda_device_name(0)}")
    except Exception:
        pass

    cx = sweep_complexity()
    n  = sweep_n()
    plot_complexity(cx)
    plot_n_sweep(n)

if __name__ == "__main__":
    main()
