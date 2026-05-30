#!/usr/bin/env python3
"""
GPU Performance Benchmarks: CUDA vs CPU Simulation

Measures wall-clock time of GPU-accelerated simulation (via rn.to(hh.Device.cuda(0)))
vs the CPU C++ backend, across neuron counts, synapse counts, and simulation durations.

Notes:
  - STDP/STP forces CPU synapse fallback; all tests use static weights.
  - Networks with no synapses isolate neuron-update throughput on the GPU.
  - Each configuration is run once with a warm-up step to avoid first-launch JIT noise.

Outputs:
  examples/figs/benchmark_gpu_time_vs_neurons.png   -- GPU vs CPU per topology (2x2 grid)
  examples/figs/benchmark_gpu_speedup.png           -- GPU speedup per topology
  examples/figs/benchmark_gpu_neuron_scaling.png
  examples/figs/benchmark_gpu_synapse_scaling.png
  examples/figs/benchmark_gpu_duration_scaling.png
  examples/figs/benchmark_gpu_neuron_types.png
"""

import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import hodgkin_huxley as hh
from hodgkin_huxley import RegionalNetwork, SynapseSpec

from benchmark_network import TOPOLOGIES

# GPU-specific topology caps — larger all-to-all than the CPU benchmark allows
# because the GPU benefits from the high synapse parallelism.
_GPU_TOPO_MAX_N = {
    "Chain": None,
    "Ring": None,
    "All-to-All": 2000,   # ~4M synapses
    "Star": None,
}


# =============================================================================
# Helpers
# =============================================================================

def _require_cuda():
    if not hh.cuda_is_available():
        print("No CUDA GPU detected — nothing to benchmark.")
        print("Re-run on a machine with a CUDA-capable GPU.")
        sys.exit(0)


def _build_rn(
    N: int,
    neuron_type: str = "HH",
    n_synapses_per_neuron: int = 0,
    seed: int = 42,
) -> RegionalNetwork:
    """Build a RegionalNetwork with a single population and optional random synapses."""
    rn = RegionalNetwork()
    rn.add_population("E", N, neuron_type=neuron_type)

    if n_synapses_per_neuron > 0 and N > 1:
        k = min(n_synapses_per_neuron, N - 1)
        rng = np.random.default_rng(seed)

        def _random_pairs(ns, nd):
            pre = np.arange(ns)
            pairs = []
            for p in pre:
                targets = rng.choice(
                    [j for j in range(ns) if j != p],
                    size=k,
                    replace=False,
                )
                pairs.extend((p, t) for t in targets)
            return pairs

        rn.connect(
            "E", "E", _random_pairs,
            weight=0.3,
            synapse=SynapseSpec.ampa(),
        )

    return rn


def _simulate_cpu(rn: RegionalNetwork, duration: float, dt: float, I_val: float) -> float:
    """Run on CPU and return wall time (s)."""
    rn.to(hh.Device.cpu())
    start = time.perf_counter()
    rn.simulate(duration, dt, {"E": I_val})
    return time.perf_counter() - start


def _simulate_gpu(rn: RegionalNetwork, duration: float, dt: float, I_val: float) -> float:
    """Run on GPU and return wall time (s)."""
    rn.to(hh.Device.cuda(0))
    start = time.perf_counter()
    rn.simulate(duration, dt, {"E": I_val})
    return time.perf_counter() - start


def _bench_pair(
    N: int,
    neuron_type: str,
    n_syn_per_neuron: int,
    duration: float,
    dt: float,
    I_val: float,
    warmup: bool = True,
) -> Tuple[float, float]:
    """
    Return (cpu_time, gpu_time) for a given configuration.
    Builds two separate networks (one stays on CPU, one goes to GPU)
    to avoid cross-contamination from device-state changes.
    """
    gc.collect()

    rn_cpu = _build_rn(N, neuron_type, n_syn_per_neuron)
    rn_gpu = _build_rn(N, neuron_type, n_syn_per_neuron)

    if warmup:
        _simulate_cpu(rn_cpu, min(duration, 20.0), dt, I_val)
        _simulate_gpu(rn_gpu, min(duration, 20.0), dt, I_val)

    t_cpu = _simulate_cpu(rn_cpu, duration, dt, I_val)
    t_gpu = _simulate_gpu(rn_gpu, duration, dt, I_val)

    del rn_cpu, rn_gpu
    gc.collect()

    return t_cpu, t_gpu


def _fmt_speedup(t_cpu: float, t_gpu: float) -> str:
    if t_gpu <= 0:
        return "inf"
    s = t_cpu / t_gpu
    return f"{s:.2f}x"


# =============================================================================
# Benchmark 1: Neuron scaling (fixed synapses-per-neuron, increasing N)
# =============================================================================

def run_neuron_scaling(
    neuron_counts: Optional[List[int]] = None,
    n_syn_per_neuron: int = 10,
    duration: float = 100.0,
    dt: float = 0.05,
    I_val: float = 8.0,
) -> Dict:
    if neuron_counts is None:
        neuron_counts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    cpu_times: List[float] = []
    gpu_times: List[float] = []
    syn_counts: List[int] = []

    print(f"\n--- Neuron scaling (HH, {n_syn_per_neuron} syn/neuron, "
          f"duration={duration} ms) ---")

    for N in neuron_counts:
        k = min(n_syn_per_neuron, N - 1)
        n_syn = N * k
        syn_counts.append(n_syn)
        print(f"  N={N:>5d}  synapses={n_syn:>8d} ... ", end="", flush=True)

        t_cpu, t_gpu = _bench_pair(N, "HH", n_syn_per_neuron, duration, dt, I_val)
        cpu_times.append(t_cpu)
        gpu_times.append(t_gpu)
        print(f"CPU={t_cpu:.3f}s  GPU={t_gpu:.3f}s  speedup={_fmt_speedup(t_cpu, t_gpu)}")

    return {
        "neuron_counts": neuron_counts,
        "syn_counts": syn_counts,
        "cpu": cpu_times,
        "gpu": gpu_times,
        "n_syn_per_neuron": n_syn_per_neuron,
        "duration": duration,
    }


# =============================================================================
# Benchmark 2: Synapse scaling (fixed N, increasing synapses-per-neuron)
# =============================================================================

def run_synapse_scaling(
    N: int = 2048,
    k_values: Optional[List[int]] = None,
    duration: float = 100.0,
    dt: float = 0.05,
    I_val: float = 8.0,
) -> Dict:
    if k_values is None:
        # 0 = no synapses (pure neuron update), then increasing k up to all-to-all
        k_values = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, min(2047, N - 1)]

    k_values = [k for k in k_values if k < N]

    cpu_times: List[float] = []
    gpu_times: List[float] = []
    syn_counts: List[int] = []

    print(f"\n--- Synapse scaling (HH, N={N}, duration={duration} ms) ---")

    for k in k_values:
        n_syn = N * k
        syn_counts.append(n_syn)
        print(f"  k={k:>4d}  synapses={n_syn:>8d} ... ", end="", flush=True)

        t_cpu, t_gpu = _bench_pair(N, "HH", k, duration, dt, I_val)
        cpu_times.append(t_cpu)
        gpu_times.append(t_gpu)
        print(f"CPU={t_cpu:.3f}s  GPU={t_gpu:.3f}s  speedup={_fmt_speedup(t_cpu, t_gpu)}")

    return {
        "N": N,
        "k_values": k_values,
        "syn_counts": syn_counts,
        "cpu": cpu_times,
        "gpu": gpu_times,
        "duration": duration,
    }


# =============================================================================
# Benchmark 3: Duration scaling — multiple network sizes
#
# Both CPU and GPU are O(steps), so the speedup ratio is flat vs duration.
# Running at several N values on the same axes makes this explicit: flat lines
# at different heights show that N (parallelism) drives speedup, not duration.
# =============================================================================

def run_duration_scaling(
    n_values: Optional[List[int]] = None,
    n_syn_per_neuron: int = 8,
    durations: Optional[List[float]] = None,
    dt: float = 0.05,
    I_val: float = 8.0,
) -> Dict:
    if n_values is None:
        n_values = [256, 1024, 4096, 16384]
    if durations is None:
        durations = [10, 25, 50, 100, 200, 500, 1000, 5000, 10000]

    print(f"\n--- Duration scaling (HH, {n_syn_per_neuron} syn/neuron, "
          f"N={n_values}) ---")

    per_n: Dict[int, Dict] = {}
    for N in n_values:
        n_syn = N * min(n_syn_per_neuron, N - 1)
        print(f"\n  N={N:>6d}  ({n_syn} synapses)")

        rn_cpu = _build_rn(N, "HH", n_syn_per_neuron)
        rn_gpu = _build_rn(N, "HH", n_syn_per_neuron)
        _simulate_cpu(rn_cpu, 10.0, dt, I_val)
        _simulate_gpu(rn_gpu, 10.0, dt, I_val)

        cpu_times: List[float] = []
        gpu_times: List[float] = []
        for dur in durations:
            steps = int(dur / dt)
            print(f"    dur={dur:>7.0f} ms  steps={steps:>7d} ... ",
                  end="", flush=True)
            t_cpu = _simulate_cpu(rn_cpu, dur, dt, I_val)
            t_gpu = _simulate_gpu(rn_gpu, dur, dt, I_val)
            cpu_times.append(t_cpu)
            gpu_times.append(t_gpu)
            print(f"CPU={t_cpu:.3f}s  GPU={t_gpu:.3f}s  "
                  f"speedup={_fmt_speedup(t_cpu, t_gpu)}")

        del rn_cpu, rn_gpu
        gc.collect()

        per_n[N] = {"cpu": cpu_times, "gpu": gpu_times, "n_syn": n_syn}

    return {"n_values": n_values, "durations": durations, "per_n": per_n}


# =============================================================================
# Benchmark 4: Neuron type comparison (HH vs Izhikevich)
# =============================================================================

def run_neuron_type_comparison(
    neuron_counts: Optional[List[int]] = None,
    n_syn_per_neuron: int = 8,
    duration: float = 100.0,
    dt: float = 0.05,
    I_val: float = 8.0,
) -> Dict:
    if neuron_counts is None:
        neuron_counts = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    results: Dict[str, Dict] = {}

    for neuron_type, label, i_val in [
        ("HH", "HH", 8.0),
        ("izhikevich_rs", "Izhikevich RS", 10.0),
    ]:
        cpu_times: List[float] = []
        gpu_times: List[float] = []

        print(f"\n--- Neuron type: {label} ({n_syn_per_neuron} syn/neuron, "
              f"duration={duration} ms) ---")

        for N in neuron_counts:
            k = min(n_syn_per_neuron, N - 1)
            n_syn = N * k
            print(f"  N={N:>5d}  synapses={n_syn:>8d} ... ", end="", flush=True)

            t_cpu, t_gpu = _bench_pair(N, neuron_type, n_syn_per_neuron,
                                       duration, dt, i_val)
            cpu_times.append(t_cpu)
            gpu_times.append(t_gpu)
            print(f"CPU={t_cpu:.3f}s  GPU={t_gpu:.3f}s  "
                  f"speedup={_fmt_speedup(t_cpu, t_gpu)}")

        results[label] = {
            "neuron_counts": neuron_counts,
            "cpu": cpu_times,
            "gpu": gpu_times,
        }

    return results


# =============================================================================
# Plotting
# =============================================================================

def _speedups(cpu_times, gpu_times) -> List[float]:
    return [c / g if g > 0 else 0.0 for c, g in zip(cpu_times, gpu_times)]


def plot_neuron_scaling(data: Dict, figs_dir: Path):
    ns = data["neuron_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.loglog(ns, data["cpu"], "o-", color="tab:blue", lw=2, ms=6, label="CPU (C++)")
    ax.loglog(ns, data["gpu"], "s-", color="tab:green", lw=2, ms=6, label="GPU (CUDA)")
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(
        f"Neuron Scaling — {data['n_syn_per_neuron']} syn/neuron, "
        f"{data['duration']:.0f} ms sim",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogx(ns, _speedups(data["cpu"], data["gpu"]), "D-",
                color="tab:purple", lw=2, ms=6)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Parity")
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("GPU Speedup vs Neuron Count", fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("GPU vs CPU: Neuron Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = figs_dir / "benchmark_gpu_neuron_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def plot_synapse_scaling(data: Dict, figs_dir: Path):
    ss = data["syn_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ss, data["cpu"], "o-", color="tab:blue", lw=2, ms=6, label="CPU (C++)")
    ax.plot(ss, data["gpu"], "s-", color="tab:green", lw=2, ms=6, label="GPU (CUDA)")
    if max(ss) > 0:
        ax.set_xscale("log")
    ax.set_xlabel("Number of synapses")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(
        f"Synapse Scaling — N={data['N']}, {data['duration']:.0f} ms sim",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    spd = _speedups(data["cpu"], data["gpu"])
    x = ss if ss[0] > 0 else ss[1:]
    y = spd if ss[0] > 0 else spd[1:]
    if x:
        ax.semilogx(x, y, "D-", color="tab:purple", lw=2, ms=6)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Parity")
    ax.set_xlabel("Number of synapses")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("GPU Speedup vs Synapse Count", fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("GPU vs CPU: Synapse Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = figs_dir / "benchmark_gpu_synapse_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_duration_scaling(data: Dict, figs_dir: Path):
    durations = data["durations"]
    n_values = data["n_values"]
    per_n = data["per_n"]

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(n_values)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: wall time vs duration (GPU solid, CPU dashed) per network size
    ax = axes[0]
    for i, N in enumerate(n_values):
        d = per_n[N]
        ax.loglog(durations, d["cpu"], "--", color=cmap[i], lw=1.5, alpha=0.6)
        ax.loglog(durations, d["gpu"], "o-", color=cmap[i], lw=2, ms=4,
                  label=f"N={N:,}")
    from matplotlib.lines import Line2D
    legend_entries = [
        Line2D([0], [0], color="gray", lw=2, label="GPU (solid)"),
        Line2D([0], [0], color="gray", lw=1.5, ls="--", alpha=0.6, label="CPU (dashed)"),
    ] + [Line2D([0], [0], color=cmap[i], lw=2, label=f"N={N:,}")
         for i, N in enumerate(n_values)]
    ax.legend(handles=legend_entries, fontsize=8)
    ax.set_xlabel("Simulation duration (ms)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Duration Scaling — GPU vs CPU per network size", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)

    # Right: speedup vs duration — flat lines confirm O(steps) for both backends;
    # height of each line is set by N (parallelism), not duration.
    ax = axes[1]
    for i, N in enumerate(n_values):
        d = per_n[N]
        spd = _speedups(d["cpu"], d["gpu"])
        ax.semilogx(durations, spd, "o-", color=cmap[i], lw=2, ms=5,
                    label=f"N={N:,}")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Parity")
    ax.set_xlabel("Simulation duration (ms)")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("Speedup vs Duration\n(flat = both backends O(steps); height set by N)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("GPU vs CPU: Duration Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = figs_dir / "benchmark_gpu_duration_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_neuron_type_comparison(data: Dict, figs_dir: Path):
    colors = {"HH": "tab:blue", "Izhikevich RS": "tab:orange"}
    markers = {"HH": "o", "Izhikevich RS": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for label, d in data.items():
        ax.loglog(d["neuron_counts"], d["cpu"], f"{markers[label]}--",
                  color=colors[label], lw=1.5, ms=5, alpha=0.6,
                  label=f"{label} CPU")
        ax.loglog(d["neuron_counts"], d["gpu"], f"{markers[label]}-",
                  color=colors[label], lw=2, ms=6,
                  label=f"{label} GPU")
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Timing by Neuron Type", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for label, d in data.items():
        spd = _speedups(d["cpu"], d["gpu"])
        ax.semilogx(d["neuron_counts"], spd, f"{markers[label]}-",
                    color=colors[label], lw=2, ms=6, label=label)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Parity")
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("GPU Speedup by Neuron Type", fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("GPU vs CPU: Neuron Type Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = figs_dir / "benchmark_gpu_neuron_types.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Benchmark 5: Topology comparison — GPU vs CPU (equivalent of benchmark_avg_time_vs_neurons)
# =============================================================================

def _make_topology_rn(N: int, synapses) -> RegionalNetwork:
    """Build a RegionalNetwork from a list of (pre, post, weight, E_syn, tau) tuples."""
    rn = RegionalNetwork()
    rn.add_population("E", N, neuron_type="HH")
    if synapses:
        pairs = [(s[0], s[1]) for s in synapses]
        w, e, tau = synapses[0][2], synapses[0][3], synapses[0][4]
        rn.connect("E", "E", lambda ns, nd: pairs, weight=w,
                   synapse=SynapseSpec.exponential(tau_S=tau, E_syn=e))
    return rn


def _bench_topology_pair(
    N: int,
    synapses,
    duration: float,
    dt: float,
    I_val: float,
    warmup: bool = True,
) -> Tuple[float, float]:
    """Return (cpu_time, gpu_time) for a topology-built network."""
    gc.collect()

    rn_cpu = _make_topology_rn(N, synapses)
    rn_gpu = _make_topology_rn(N, synapses)

    if warmup:
        _simulate_cpu(rn_cpu, min(duration, 20.0), dt, I_val)
        _simulate_gpu(rn_gpu, min(duration, 20.0), dt, I_val)

    t_cpu = _simulate_cpu(rn_cpu, duration, dt, I_val)
    t_gpu = _simulate_gpu(rn_gpu, duration, dt, I_val)

    del rn_cpu, rn_gpu
    gc.collect()

    return t_cpu, t_gpu


def run_topology_benchmarks(
    sizes: Optional[List[int]] = None,
    duration: float = 100.0,
    dt: float = 0.05,
    I_val: float = 8.0,
) -> Dict:
    """
    GPU vs CPU timing for Chain, Ring, All-to-All, and Star topologies.
    Mirrors run_benchmarks() in benchmark_network.py but compares GPU vs CPU.
    """
    if sizes is None:
        sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]

    results: Dict[str, Dict] = {}

    for topo_name, builder in TOPOLOGIES.items():
        cpu_times: List[float] = []
        gpu_times: List[float] = []
        max_n = _GPU_TOPO_MAX_N.get(topo_name)
        topo_sizes = [N for N in sizes if max_n is None or N <= max_n]

        if len(topo_sizes) < len(sizes):
            print(f"  ({topo_name}: capping at N={max_n} — {max_n * (max_n - 1):,} synapses)")

        print(f"\n  Topology: {topo_name}")
        for N in topo_sizes:
            synapses = builder(N)
            n_syn = len(synapses)
            print(f"    N={N:>4d}  synapses={n_syn:>7d} ... ", end="", flush=True)

            t_cpu, t_gpu = _bench_topology_pair(N, synapses, duration, dt, I_val)
            cpu_times.append(t_cpu)
            gpu_times.append(t_gpu)
            print(f"CPU={t_cpu:.3f}s  GPU={t_gpu:.3f}s  "
                  f"speedup={_fmt_speedup(t_cpu, t_gpu)}")

        results[topo_name] = {
            "sizes": topo_sizes,
            "cpu": cpu_times,
            "gpu": gpu_times,
        }

    return results


def plot_topology_timing(results: Dict, figs_dir: Path):
    """2x2 grid: one subplot per topology, GPU and CPU time vs N. Mirrors plot_timing()."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (topo_name, data) in enumerate(results.items()):
        ax = axes[idx]
        ns = data["sizes"]
        ax.loglog(ns, data["cpu"], "o-", color="tab:blue", lw=2, ms=6, label="CPU (C++)")
        ax.loglog(ns, data["gpu"], "s-", color="tab:green", lw=2, ms=6, label="GPU (CUDA)")
        ax.set_title(topo_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of neurons")
        ax.set_ylabel("Wall time (s)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("GPU vs CPU Timing by Topology", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = figs_dir / "benchmark_gpu_time_vs_neurons.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def plot_topology_speedup(results: Dict, figs_dir: Path):
    """GPU speedup (CPU time / GPU time) vs neuron count, one line per topology."""
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, (topo_name, data) in enumerate(results.items()):
        spd = _speedups(data["cpu"], data["gpu"])
        ax.plot(data["sizes"], spd, f"{markers[idx]}-", color=colors[idx],
                lw=2, ms=7, label=topo_name)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Parity")
    ax.set_xscale("log")
    ax.set_xlabel("Number of neurons", fontsize=12)
    ax.set_ylabel("Speedup (CPU time / GPU time)", fontsize=12)
    ax.set_title("GPU Speedup Over CPU by Topology", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = figs_dir / "benchmark_gpu_speedup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("GPU Benchmark: CUDA vs CPU Simulation")
    print("=" * 60)

    _require_cuda()

    try:
        dev_name = hh.cuda_device_name(0)
        print(f"\nGPU: {dev_name}")
    except Exception:
        print("\nGPU: (name unavailable)")

    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)

    dt = 0.05   # ms
    I_val = 8.0

    print(f"\ndt={dt} ms, I_ext={I_val} uA/cm^2")
    print(f"Output: {figs_dir}")
    print("\nNote: all networks use static weights (no STDP/STP) to stay on GPU synapse path.\n")

    # 0. Topology comparison (equivalent of benchmark_avg_time_vs_neurons.png)
    print("\n--- Topology benchmarks (GPU vs CPU) ---")
    topo_results = run_topology_benchmarks(
        sizes=[5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000],
        duration=100.0,
        dt=dt,
        I_val=I_val,
    )
    plot_topology_timing(topo_results, figs_dir)
    plot_topology_speedup(topo_results, figs_dir)

    # 1. Neuron scaling
    neuron_data = run_neuron_scaling(
        neuron_counts=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        n_syn_per_neuron=10,
        duration=100.0,
        dt=dt,
        I_val=I_val,
    )
    plot_neuron_scaling(neuron_data, figs_dir)

    # 2. Synapse scaling
    syn_data = run_synapse_scaling(
        N=2048,
        k_values=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2047],
        duration=100.0,
        dt=dt,
        I_val=I_val,
    )
    plot_synapse_scaling(syn_data, figs_dir)

    # 3. Duration scaling — multi-N to show speedup is flat vs duration,
    #    driven by N (parallelism), not number of steps.
    dur_data = run_duration_scaling(
        n_values=[256, 1024, 4096, 16384],
        n_syn_per_neuron=8,
        durations=[10, 25, 50, 100, 200, 500, 1000, 5000, 10000],
        dt=dt,
        I_val=I_val,
    )
    plot_duration_scaling(dur_data, figs_dir)

    # 4. Neuron type comparison
    type_data = run_neuron_type_comparison(
        neuron_counts=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        n_syn_per_neuron=8,
        duration=100.0,
        dt=dt,
    )
    plot_neuron_type_comparison(type_data, figs_dir)

    print("\n" + "=" * 60)
    print("GPU benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
