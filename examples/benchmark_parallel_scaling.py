"""
benchmark_parallel_scaling.py -- Serial vs Phase 1 (OpenMP) vs Phase 2
(delay-decomposition) speedup as a function of neuron model complexity and
population size.

Three modes are compared:
  serial  -- set_num_threads(1)
  phase1  -- OpenMP pool-level parallelism (set_num_threads(N_THREADS))
  phase2  -- delay-decomposition thread groups (one group per population)

Five complexity tiers are benchmarked:
  0 Izhikevich RS   -- 2 state vars; all populations share one IzPool
  1 HH-default      -- 3-gate composable; each population -> its own ComposablePool
  2 TH-like         -- 5 gates (including a derived gate); medium compute
  3 GPe-like        -- 6 gates + Ca2+ ODE; heavier per step
  4 STN-like        -- 11 gates + Ca2+ + Nernst update; heaviest

Two benchmark sweeps are produced:
  A) Complexity sweep (fixed N=200, n_pop=8):
     speedup vs complexity tier (phase1 and phase2 grouped bars)

  B) N sweep per tier (n_pop=8 throughout):
     speedup vs N neurons/pop — side-by-side phase1 / phase2 panels

Outputs to examples/figs/:
  parallel_complexity_sweep.png   -- speedup vs complexity (fixed N)
  parallel_N_sweep.png            -- speedup vs N, one panel per mode
  parallel_speedup_heatmap.png    -- speedup heatmap (tier x N), one panel per mode

Usage:
    python examples/benchmark_parallel_scaling.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hodgkin_huxley import (
    IzhikevichType,
    NeuronModelSpec,
    RecordingConfig,
    RegionalNetwork,
    SynapseSpec,
)

# Import real model factories from the CTX-BG-TH benchmark
from benchmarks.ctxbgth_model import (
    _make_th_spec,
    _make_gpe_spec,
    _make_stn_spec,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_POP       = 8                          # populations (one per ComposablePool)
N_NEURONS_COMPLEXITY = 200               # neurons/pop for the complexity sweep
N_NEURON_VALUES = [50, 100, 200, 500]    # neurons/pop for the N sweep
DURATION    = 200.0                      # ms
DT          = 0.05                       # ms -- coarser to keep benchmark fast
N_THREADS   = 4
N_REPEATS   = 2
DELAY_MS    = 5.0

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Complexity tier descriptors
# ---------------------------------------------------------------------------
# Each tier is (short_label, description, n_gates_approx, has_ca)
TIERS = [
    ("Izhikevich\n(2 vars)",   "Izhikevich RS -- all share IzPool",     2,  False),
    ("HH-default\n(3 gates)",  "HH-default composable -- 3 gates",      3,  False),
    ("TH-like\n(5 gates)",     "TH relay cell -- 5 gates",              5,  False),
    ("GPe-like\n(6 g + Ca)",   "GPe basal ganglia -- 6 gates + Ca2+",   6,  True ),
    ("STN-like\n(11 g + Ca)",  "STN -- 11 gates + Ca2+ + Nernst",      11, True ),
]
N_TIERS = len(TIERS)

# ---------------------------------------------------------------------------
# Per-tier network builders
# ---------------------------------------------------------------------------

def _add_populations_izhikevich(rn: RegionalNetwork, n_pop: int, n: int) -> None:
    """All populations share one IzPool -- worst case for Phase 1."""
    types = [
        IzhikevichType.REGULAR_SPIKING,
        IzhikevichType.FAST_SPIKING,
        IzhikevichType.INTRINSICALLY_BURSTING,
        IzhikevichType.CHATTERING,
        IzhikevichType.LOW_THRESHOLD_SPIKING,
        IzhikevichType.REGULAR_SPIKING,
        IzhikevichType.FAST_SPIKING,
        IzhikevichType.INTRINSICALLY_BURSTING,
    ]
    for i in range(n_pop):
        rn.add_population(
            f"P{i}", n,
            model=NeuronModelSpec.izhikevich(types[i % len(types)]),
        )


def _add_populations_hh_default(rn: RegionalNetwork, n_pop: int, n: int) -> None:
    """Each population gets a distinct ComposablePool via unique spec name."""
    base = NeuronModelSpec.hh_default()
    for i in range(n_pop):
        spec = NeuronModelSpec.hh_default()
        spec.name = f"HH_{i}"
        rn.add_population(f"P{i}", n, model=spec)


def _add_populations_th(rn: RegionalNetwork, n_pop: int, n: int) -> None:
    """TH-like: 5 gates, no Ca, one pool per pop."""
    for i in range(n_pop):
        spec = _make_th_spec()
        spec.name = f"TH_{i}"
        rn.add_population(f"P{i}", n, model=spec)


def _add_populations_gpe(rn: RegionalNetwork, n_pop: int, n: int) -> None:
    """GPe-like: 6 gates + Ca ODE, one pool per pop."""
    for i in range(n_pop):
        spec, ca_dyn = _make_gpe_spec()
        spec.name = f"GPe_{i}"
        rn.add_population(f"P{i}", n, model=spec)
        rn.add_intracellular(ca_dyn, f"P{i}")


def _add_populations_stn(rn: RegionalNetwork, n_pop: int, n: int) -> None:
    """STN-like: 11 gates + Ca ODE + Nernst, one pool per pop."""
    for i in range(n_pop):
        spec, ca_dyn = _make_stn_spec()
        spec.name = f"STN_{i}"
        rn.add_population(f"P{i}", n, model=spec)
        rn.add_intracellular(ca_dyn, f"P{i}")


_POP_ADDERS = [
    _add_populations_izhikevich,
    _add_populations_hh_default,
    _add_populations_th,
    _add_populations_gpe,
    _add_populations_stn,
]


def _build_network(tier_idx: int, n_pop: int, n: int) -> RegionalNetwork:
    """Chain topology: P0 -> P1 -> ... -> P_{n_pop-1}, one_to_one, AMPA."""
    rn = RegionalNetwork()
    _POP_ADDERS[tier_idx](rn, n_pop, n)
    pops = [f"P{i}" for i in range(n_pop)]
    for i in range(n_pop - 1):
        rn.connect(
            pops[i], pops[i + 1], "one_to_one",
            weight=0.4, synapse=SynapseSpec.ampa(), delay=DELAY_MS,
        )
    return rn


def _I_ext(n_pop: int) -> dict:
    pops = [f"P{i}" for i in range(n_pop)]
    return {pops[0]: 10.0, **{p: 0.0 for p in pops[1:]}}


def _time_one(rn: RegionalNetwork, n_pop: int) -> float:
    t0 = time.perf_counter()
    rn.simulate(DURATION, DT, _I_ext(n_pop), record=RecordingConfig(["V"]))
    return time.perf_counter() - t0


def _measure(tier_idx: int, n_pop: int, n: int, mode: str) -> float:
    """mode: 'serial' | 'phase1' | 'phase2'"""
    rn = _build_network(tier_idx, n_pop, n)
    if mode == "serial":
        rn.set_num_threads(1)
    elif mode == "phase1":
        rn.set_num_threads(N_THREADS)
    elif mode == "phase2":
        groups = {f"g{i}": [f"P{i}"] for i in range(n_pop)}
        rn.set_thread_groups(groups)
    return min(_time_one(rn, n_pop) for _ in range(N_REPEATS))


# ---------------------------------------------------------------------------
# Sweep A: complexity at fixed N
# ---------------------------------------------------------------------------

def sweep_complexity(n_pop: int = N_POP, n: int = N_NEURONS_COMPLEXITY) -> np.ndarray:
    """
    Returns shape (N_TIERS, 3) array: [:, 0]=serial, [:, 1]=phase1, [:, 2]=phase2 times.
    """
    results = np.zeros((N_TIERS, 3))
    print(f"\n--- Complexity sweep  (n_pop={n_pop}, n={n}, duration={DURATION}ms) ---")
    for ti, (label, desc, n_gates, has_ca) in enumerate(TIERS):
        t_ser = _measure(ti, n_pop, n, "serial")
        t_p1  = _measure(ti, n_pop, n, "phase1")
        t_p2  = _measure(ti, n_pop, n, "phase2")
        results[ti] = [t_ser, t_p1, t_p2]
        ca_str = "+Ca" if has_ca else "   "
        print(f"  Tier {ti} {ca_str} ({n_gates:2d} gates)  "
              f"serial={t_ser:.3f}s  phase1={t_p1:.3f}s ({t_ser/t_p1:.2f}x)  "
              f"phase2={t_p2:.3f}s ({t_ser/t_p2:.2f}x)")
    return results


# ---------------------------------------------------------------------------
# Sweep B: N at each tier
# ---------------------------------------------------------------------------

def sweep_n_per_tier(n_pop: int = N_POP) -> np.ndarray:
    """
    Returns shape (N_TIERS, len(N_NEURON_VALUES), 3) array: [..., 0]=serial, [..., 1]=phase1, [..., 2]=phase2.
    """
    results = np.zeros((N_TIERS, len(N_NEURON_VALUES), 3))
    print(f"\n--- N sweep  (n_pop={n_pop}, duration={DURATION}ms) ---")
    for ti, (label, desc, n_gates, has_ca) in enumerate(TIERS):
        ca_str = "+Ca" if has_ca else "   "
        print(f"  Tier {ti} {ca_str} ({n_gates:2d} gates):")
        for ni, n in enumerate(N_NEURON_VALUES):
            t_ser = _measure(ti, n_pop, n, "serial")
            t_p1  = _measure(ti, n_pop, n, "phase1")
            t_p2  = _measure(ti, n_pop, n, "phase2")
            results[ti, ni] = [t_ser, t_p1, t_p2]
            print(f"    n={n:4d}  serial={t_ser:.3f}s  "
                  f"phase1={t_p1:.3f}s ({t_ser/t_p1:.2f}x)  "
                  f"phase2={t_p2:.3f}s ({t_ser/t_p2:.2f}x)")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TIER_LABELS  = [t[0] for t in TIERS]
TIER_COLORS  = ["#aaaaaa", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
TIER_MARKERS = ["o", "s", "^", "D", "v"]


def plot_complexity_sweep(results: np.ndarray) -> None:
    sp1 = results[:, 0] / results[:, 1]
    sp2 = results[:, 0] / results[:, 2]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(N_TIERS)
    w = 0.35
    b1 = ax.bar(x - w/2, sp1, w, label=f"Phase 1 ({N_THREADS} threads)",
                color="#ff7f0e", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w/2, sp2, w, label="Phase 2 (delay decomp)",
                color="#2ca02c", edgecolor="black", linewidth=0.7)
    ax.axhline(1.0, ls="--", color="black", lw=1, label="Breakeven")
    ax.axhline(N_THREADS, ls=":", color="gray", lw=1, label=f"Ideal {N_THREADS}x")

    for xi, v in enumerate(sp1):
        ax.text(xi - w/2, v + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    for xi, v in enumerate(sp2):
        ax.text(xi + w/2, v + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABELS, fontsize=8)
    ax.set_ylabel("Speedup (serial / parallel)")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Parallel Speedup vs Neuron Model Complexity\n"
        f"(n={N_NEURONS_COMPLEXITY} neurons/pop, n_pop={N_POP}, duration={DURATION}ms)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "parallel_complexity_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out}")


def plot_n_sweep(results: np.ndarray) -> None:
    sp1 = results[:, :, 0] / results[:, :, 1]  # (N_TIERS, N_NEURON_VALUES)
    sp2 = results[:, :, 0] / results[:, :, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, speedup, phase_label in zip(
        axes,
        [sp1, sp2],
        [f"Phase 1 ({N_THREADS} OpenMP threads)", "Phase 2 (delay decomp)"],
    ):
        for ti, (label, _, n_gates, _) in enumerate(TIERS):
            ax.plot(
                N_NEURON_VALUES, speedup[ti],
                color=TIER_COLORS[ti], marker=TIER_MARKERS[ti],
                label=label.replace("\n", " "),
            )
        ax.axhline(1.0, ls="--", color="black", lw=1, label="Breakeven")
        ax.axhline(N_THREADS, ls=":", color="gray", lw=1, label=f"Ideal {N_THREADS}x")
        ax.set_xlabel("Neurons per population")
        ax.set_ylabel("Speedup (serial / parallel)")
        ax.set_title(phase_label)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Speedup vs Population Size by Model Complexity\n"
        f"(n_pop={N_POP}, duration={DURATION}ms)",
        y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "parallel_N_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_heatmap(results: np.ndarray) -> None:
    sp1 = results[:, :, 0] / results[:, :, 1]
    sp2 = results[:, :, 0] / results[:, :, 2]
    vmax = max(2.0, float(sp1.max()), float(sp2.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, speedup, title in zip(
        axes,
        [sp1, sp2],
        [f"Phase 1 ({N_THREADS} threads)", "Phase 2 (delay decomp)"],
    ):
        im = ax.imshow(speedup, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=vmax)
        ax.set_xticks(range(len(N_NEURON_VALUES)))
        ax.set_xticklabels(N_NEURON_VALUES)
        ax.set_yticks(range(N_TIERS))
        ax.set_yticklabels(TIER_LABELS, fontsize=8)
        ax.set_xlabel("Neurons per population")
        ax.set_title(title)
        for ti in range(N_TIERS):
            for ni in range(len(N_NEURON_VALUES)):
                v = speedup[ti, ni]
                ax.text(ni, ti, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.6 < v < 1.8 else "white")
        fig.colorbar(im, ax=ax, label="Speedup")

    fig.suptitle(
        f"Speedup Heatmap  (n_pop={N_POP}, duration={DURATION}ms)",
        y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "parallel_speedup_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Parallel scaling benchmark  (serial vs Phase 1 vs Phase 2)")
    print(f"  n_pop={N_POP}  |  N_NEURON_VALUES={N_NEURON_VALUES}")
    print(f"  duration={DURATION}ms  dt={DT}ms  threads={N_THREADS}  repeats={N_REPEATS}")
    print()
    for ti, (label, desc, n_gates, has_ca) in enumerate(TIERS):
        ca = " + Ca2+" if has_ca else ""
        print(f"  Tier {ti}: {desc} ({n_gates} gates{ca})")

    cx_results = sweep_complexity()
    n_results  = sweep_n_per_tier()

    plot_complexity_sweep(cx_results)
    plot_n_sweep(n_results)
    plot_heatmap(n_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
