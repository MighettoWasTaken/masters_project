#!/usr/bin/env python3
"""
Memory Usage Benchmark: Serial vs Delay-Parallel Threading

Uses the all-to-all multi-population network from benchmark_network_threading.py
so that synapse counts are visible and meaningful.

Network structure
-----------------
  N_POP = 4 populations, each of n_per_pop neurons.
  All-to-all inter-population connections with DELAY_MS axonal delay.
  Synapse count = N_POP*(N_POP-1) * n_per_pop^2  (quadratic in n_per_pop).

Backends compared
-----------------
  C++ serial   — RegionalNetwork, no thread groups
  C++ Phase-2  — RegionalNetwork, one thread group per population
  NumPy        — flat NumpyHHNetwork (identical synapse list)

Measurements
------------
  Peak RSS (MB) polled every 2 ms while simulate() runs.

Output
------
  examples/figs/benchmark_memory_threading_vs_synapses.png
  examples/figs/benchmark_memory_threading_per_synapse.png
  examples/figs/benchmark_memory_threading_vs_npp.png
"""

import sys
import os
import gc
import time
import threading
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_network_threading import (   # noqa: E402
    N_POP, DELAY_MS, WEIGHT, E_SYN, TAU_SYN,
    COLOR_SERIAL, COLOR_PHASE2, COLOR_NUMPY,
    TOPO_EDGES, TOPO_INTER_CONN,
    _build_cpp_net, _build_numpy_net, _n_synapses,
)
from benchmark_network import NumpyHHNetwork  # noqa: E402

from hodgkin_huxley import RegionalNetwork, SynapseSpec

# =============================================================================
# Configuration
# =============================================================================

# n_per_pop values to sweep.  Synapse count = 12 * n_per_pop^2.
#   n_per_pop=10  →   1,200 synapses
#   n_per_pop=25  →   7,500 synapses
#   n_per_pop=50  →  30,000 synapses
#   n_per_pop=75  →  67,500 synapses
#   n_per_pop=100 → 120,000 synapses
NPP_VALUES     = [10, 25, 50, 75, 100, 200, 400, 800, 2000]
NUMPY_MAX_NPP  = 75   # NumPy becomes slow beyond this; still measured for memory
DURATION_MS    = 50.0
DT             = 0.05
I_VAL          = 10.0
SAMPLE_MS      = 2     # RSS poll interval (ms)

EDGES      = TOPO_EDGES["All-to-All"]
INTER_CONN = TOPO_INTER_CONN["All-to-All"]   # "all_to_all"


# =============================================================================
# Memory measurement helper (mirrors benchmark_network.py)
# =============================================================================

def _peak_rss_mb_during(fn, *args, sample_ms: int = SAMPLE_MS, **kwargs) -> float:
    """Run fn(*args, **kwargs) in a thread; poll RSS; return peak (MB)."""
    proc = psutil.Process(os.getpid())
    peak = proc.memory_info().rss
    done = False
    exc = None

    def runner():
        nonlocal done, exc
        try:
            fn(*args, **kwargs)
        except Exception as e:
            exc = e
        finally:
            done = True

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    while not done:
        rss = proc.memory_info().rss
        if rss > peak:
            peak = rss
        time.sleep(sample_ms / 1000.0)
    t.join()
    if exc is not None:
        raise exc
    return peak / (1024 * 1024)


# =============================================================================
# Per-backend measurement helpers
# =============================================================================

def _measure_serial(n_per_pop: int, duration: float, dt: float,
                    I_val: float) -> float:
    gc.collect()
    rn = _build_cpp_net(n_per_pop, EDGES, INTER_CONN)
    rn.clear_thread_groups()
    I_ext = {f"P{i}": I_val for i in range(N_POP)}
    peak = _peak_rss_mb_during(rn.simulate, duration, dt, I_ext)
    del rn
    gc.collect()
    return peak


def _measure_phase2(n_per_pop: int, duration: float, dt: float,
                    I_val: float) -> float:
    gc.collect()
    rn = _build_cpp_net(n_per_pop, EDGES, INTER_CONN)
    rn.set_thread_groups({f"g{i}": [f"P{i}"] for i in range(N_POP)})
    I_ext = {f"P{i}": I_val for i in range(N_POP)}
    peak = _peak_rss_mb_during(rn.simulate, duration, dt, I_ext)
    del rn
    gc.collect()
    return peak


def _measure_numpy(n_per_pop: int, duration: float, dt: float,
                   I_val: float) -> float:
    gc.collect()
    N_total   = N_POP * n_per_pop
    num_steps = int(duration / dt)
    net = _build_numpy_net(n_per_pop, EDGES, INTER_CONN)
    I_arr = np.full((N_total, num_steps), I_val)
    peak = _peak_rss_mb_during(net.simulate, duration, dt, I_arr)
    del net, I_arr
    gc.collect()
    return peak


# =============================================================================
# Benchmark runner
# =============================================================================

def run_memory_benchmark(npp_values: List[int] = NPP_VALUES,
                         duration: float = DURATION_MS,
                         dt: float = DT,
                         I_val: float = I_VAL) -> Dict:
    serial_mem: List[float] = []
    phase2_mem: List[float] = []
    numpy_mem:  List[float] = []
    synapse_counts: List[int] = []
    numpy_npp: List[int] = []

    print(f"\n--- Memory benchmark (all-to-all, {N_POP} pops, delay={DELAY_MS} ms) ---")
    print(f"    duration={duration} ms,  dt={dt} ms")

    for npp in npp_values:
        n_syn    = _n_synapses(npp, EDGES, INTER_CONN)
        N_total  = N_POP * npp
        do_numpy = npp <= NUMPY_MAX_NPP
        synapse_counts.append(n_syn)

        print(f"  n_per_pop={npp:>4d}  N_total={N_total:>5d}  "
              f"synapses={n_syn:>8,d} ...", end="", flush=True)

        m_s = _measure_serial(npp, duration, dt, I_val)
        m_p = _measure_phase2(npp, duration, dt, I_val)
        serial_mem.append(m_s)
        phase2_mem.append(m_p)

        if do_numpy:
            m_n = _measure_numpy(npp, duration, dt, I_val)
            numpy_mem.append(m_n)
            numpy_npp.append(npp)
            print(f"  serial={m_s:.1f} MB  phase2={m_p:.1f} MB  numpy={m_n:.1f} MB")
        else:
            print(f"  serial={m_s:.1f} MB  phase2={m_p:.1f} MB  (NumPy skipped)")

    return {
        "npp_values":     npp_values,
        "synapse_counts": synapse_counts,
        "serial":         serial_mem,
        "phase2":         phase2_mem,
        "numpy":          numpy_mem,
        "numpy_npp":      numpy_npp,
        "numpy_syn":      [_n_synapses(n, EDGES, INTER_CONN) for n in numpy_npp],
        "N_POP":          N_POP,
        "duration":       duration,
        "dt":             dt,
    }


# =============================================================================
# Plots
# =============================================================================

def plot_memory_vs_synapses(data: Dict, figs_dir: Path):
    """Log-log: peak RSS vs synapse count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    syn  = data["synapse_counts"]
    ax.loglog(syn, data["serial"], "o-",  color=COLOR_SERIAL,
              linewidth=2, markersize=7, label="C++ serial")
    ax.loglog(syn, data["phase2"], "s-",  color=COLOR_PHASE2,
              linewidth=2, markersize=7, label="C++ delay-parallel")
    if data["numpy"]:
        ax.loglog(data["numpy_syn"], data["numpy"], "D--", color=COLOR_NUMPY,
                  linewidth=2, markersize=7, label="NumPy")

    # Annotate each point with synapse count
    for i, (s, m_s, m_p) in enumerate(zip(syn, data["serial"], data["phase2"])):
        npp = data["npp_values"][i]
        ax.annotate(f"npp={npp}", xy=(s, max(m_s, m_p)),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color="gray")

    ax.set_xlabel("Number of synapses", fontsize=12)
    ax.set_ylabel("Peak RSS (MB)", fontsize=12)
    ax.set_title(
        f"Peak Memory vs Synapse Count\n"
        f"(all-to-all, {N_POP} pops × n_per_pop, {DELAY_MS} ms delay)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = figs_dir / "benchmark_memory_threading_vs_synapses.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_memory_per_synapse(data: Dict, figs_dir: Path):
    """Memory per synapse (bytes) vs synapse count — shows amortisation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    syn      = data["synapse_counts"]
    mb_to_b  = 1024 * 1024
    bytes_s  = [m * mb_to_b / s for m, s in zip(data["serial"], syn)]
    bytes_p  = [m * mb_to_b / s for m, s in zip(data["phase2"], syn)]

    ax.semilogx(syn, bytes_s, "o-",  color=COLOR_SERIAL,
                linewidth=2, markersize=7, label="C++ serial")
    ax.semilogx(syn, bytes_p, "s-",  color=COLOR_PHASE2,
                linewidth=2, markersize=7, label="C++ delay-parallel")

    if data["numpy"]:
        bytes_n = [m * mb_to_b / s
                   for m, s in zip(data["numpy"], data["numpy_syn"])]
        ax.semilogx(data["numpy_syn"], bytes_n, "D--", color=COLOR_NUMPY,
                    linewidth=2, markersize=7, label="NumPy")

    ax.set_xlabel("Number of synapses", fontsize=12)
    ax.set_ylabel("Peak RSS per synapse (bytes)", fontsize=12)
    ax.set_title(
        "Peak Memory Per Synapse\n"
        "(lower = more efficient; fixed process overhead amortises at large N)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = figs_dir / "benchmark_memory_threading_per_synapse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_memory_vs_npp(data: Dict, figs_dir: Path):
    """
    Two-panel:
      Left  — peak RSS vs n_per_pop (both C++ paths + NumPy)
      Right — Phase-2 overhead vs serial (MB difference and %)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    npp = data["npp_values"]

    # --- Left: absolute memory ---
    ax = axes[0]
    ax.plot(npp, data["serial"], "o-",  color=COLOR_SERIAL,
            linewidth=2, markersize=7, label="C++ serial")
    ax.plot(npp, data["phase2"], "s-",  color=COLOR_PHASE2,
            linewidth=2, markersize=7, label="C++ delay-parallel")
    if data["numpy"]:
        ax.plot(data["numpy_npp"], data["numpy"], "D--", color=COLOR_NUMPY,
                linewidth=2, markersize=7, label="NumPy")

    # Secondary x-axis showing synapse count
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(npp)
    syn_labels = [f"{_n_synapses(n, EDGES, INTER_CONN) // 1000}k"
                  for n in npp]
    ax2.set_xticklabels(syn_labels, fontsize=8)
    ax2.set_xlabel("Synapse count", fontsize=10)

    ax.set_xlabel("n_per_pop (neurons per population)", fontsize=11)
    ax.set_ylabel("Peak RSS (MB)", fontsize=11)
    ax.set_title("Peak Memory vs n_per_pop", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right: Phase-2 overhead ---
    ax = axes[1]
    overhead_mb  = [p - s for s, p in zip(data["serial"], data["phase2"])]
    overhead_pct = [100.0 * (p - s) / s if s > 0 else 0.0
                    for s, p in zip(data["serial"], data["phase2"])]

    color_ovhd = "#d62728"
    ax.bar(range(len(npp)), overhead_mb, color=color_ovhd, alpha=0.7,
           label="Overhead (MB)")
    ax.set_xticks(range(len(npp)))
    ax.set_xticklabels([f"npp={n}" for n in npp], fontsize=9)
    ax.set_ylabel("Phase-2 overhead vs serial (MB)", fontsize=11)
    ax.set_title("Delay-Parallel Threading Memory Overhead",
                 fontsize=12, fontweight="bold")

    # Annotate with percentage
    for i, (mb, pct) in enumerate(zip(overhead_mb, overhead_pct)):
        ax.text(i, max(mb, 0) + 0.3, f"{pct:+.1f}%",
                ha="center", va="bottom", fontsize=8, color="black")

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Memory Benchmark: Serial vs Delay-Parallel  "
        f"({N_POP} pops, all-to-all, {DELAY_MS} ms delay)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = figs_dir / "benchmark_memory_threading_vs_npp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def setup_output_dir() -> Path:
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    return figs_dir


def main():
    print("=" * 65)
    print("Memory Benchmark: Serial vs Delay-Parallel Threading")
    print("=" * 65)
    print(f"Network:   {N_POP} populations, all-to-all inter-pop connections")
    print(f"Delay:     {DELAY_MS} ms  (activates Phase-2 ring-buffer path)")
    print(f"Duration:  {DURATION_MS} ms,  dt={DT} ms")
    print(f"Sizes:     n_per_pop in {NPP_VALUES}")
    print(f"           → synapses in "
          f"{[_n_synapses(n, EDGES, INTER_CONN) for n in NPP_VALUES]}")

    figs_dir = setup_output_dir()

    data = run_memory_benchmark()

    plot_memory_vs_synapses(data, figs_dir)
    plot_memory_per_synapse(data, figs_dir)
    plot_memory_vs_npp(data, figs_dir)

    print("\n" + "=" * 65)
    print("Memory benchmark complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
