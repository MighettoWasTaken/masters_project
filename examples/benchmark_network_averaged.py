#!/usr/bin/env python3
"""
Averaged Network Benchmarks (N trials per setting)

Runs each benchmark from benchmark_network.py multiple times and averages
the results to reduce noise from background processes.

Output:
  examples/figs/benchmark_avg_time_vs_neurons.png
  examples/figs/benchmark_avg_speedup.png
  examples/figs/benchmark_avg_isolation.png
  examples/figs/benchmark_avg_duration.png
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

from benchmark_network import (
    NumpyHHNetwork,
    TOPOLOGIES,
    TOPO_MAX_N,
    _bench_cpp,
    _bench_numpy,
    _build_random_synapses,
    _build_sparse_ring,
    _build_all_to_all,
    _measure_memory_cpp,
    _measure_memory_numpy,
    validate_implementations,
)
from hodgkin_huxley import Network


# Number of trials per setting
N_TRIALS = 10


def setup_output_dir():
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    return figs_dir


# =============================================================================
# Averaged benchmark runners
# =============================================================================

def bench_averaged(bench_fn, n_trials: int = N_TRIALS, **kwargs) -> float:
    """Run a benchmark function n_trials times and return the mean time."""
    times = [bench_fn(**kwargs) for _ in range(n_trials)]
    return float(np.mean(times))


def run_benchmarks_avg(
    sizes: List[int],
    duration: float = 100.0,
    dt: float = 0.05,
    I_val: float = 10.0,
    n_trials: int = N_TRIALS,
) -> Dict[str, Dict[str, list]]:
    results: Dict[str, Dict[str, list]] = {}

    for topo_name, builder in TOPOLOGIES.items():
        cpp_times: List[float] = []
        numpy_times: List[float] = []
        max_n = TOPO_MAX_N.get(topo_name)
        topo_sizes = [N for N in sizes if max_n is None or N <= max_n]

        if len(topo_sizes) < len(sizes):
            print(f"  ({topo_name}: capping at N={max_n})")

        for N in topo_sizes:
            synapses = builder(N)
            n_syn = len(synapses)
            print(f"  {topo_name:>12s}  N={N:>4d}  syn={n_syn:>7d}  "
                  f"trials={n_trials} ... ", end="", flush=True)

            t_cpp = bench_averaged(
                _bench_cpp, n_trials, N=N, synapses=synapses,
                duration=duration, dt=dt, I_val=I_val)
            t_np = bench_averaged(
                _bench_numpy, n_trials, N=N, synapses=synapses,
                duration=duration, dt=dt, I_val=I_val)

            cpp_times.append(t_cpp)
            numpy_times.append(t_np)
            speedup = t_np / t_cpp if t_cpp > 0 else float('inf')
            print(f"C++={t_cpp:.4f}s  NumPy={t_np:.4f}s  "
                  f"speedup={speedup:.1f}x")

        results[topo_name] = {
            "cpp": cpp_times, "numpy": numpy_times, "sizes": topo_sizes,
        }

    return results


def run_synapse_scaling_avg(
    duration: float = 100.0, dt: float = 0.05,
    I_val: float = 10.0, n_trials: int = N_TRIALS,
) -> Dict:
    N = 200
    synapse_counts = [50, 200, 500, 1000, 2000, 5000, 10000, 20000,
                      N * (N - 1)]
    cpp_times = []
    numpy_times = []
    actual_syn_counts = []

    print(f"\n--- Synapse scaling (fixed N={N}, trials={n_trials}) ---")
    for target_s in synapse_counts:
        synapses = _build_random_synapses(N, target_s)
        actual_s = len(synapses)
        actual_syn_counts.append(actual_s)
        print(f"  N={N}  syn={actual_s:>6d} ... ", end="", flush=True)

        t_cpp = bench_averaged(
            _bench_cpp, n_trials, N=N, synapses=synapses,
            duration=duration, dt=dt, I_val=I_val)
        t_np = bench_averaged(
            _bench_numpy, n_trials, N=N, synapses=synapses,
            duration=duration, dt=dt, I_val=I_val)

        cpp_times.append(t_cpp)
        numpy_times.append(t_np)
        speedup = t_np / t_cpp if t_cpp > 0 else float('inf')
        print(f"C++={t_cpp:.4f}s  NumPy={t_np:.4f}s  speedup={speedup:.1f}x")

    return {"synapse_counts": actual_syn_counts,
            "cpp": cpp_times, "numpy": numpy_times, "N": N}


def run_neuron_scaling_avg(
    duration: float = 100.0, dt: float = 0.05,
    I_val: float = 10.0, n_trials: int = N_TRIALS,
) -> Dict:
    neuron_counts = [10, 20, 50, 100, 200, 500, 1000, 2000]
    k = 2
    cpp_times = []
    numpy_times = []
    syn_counts = []

    print(f"\n--- Neuron scaling (sparse ring k={k}, "
          f"trials={n_trials}) ---")
    for N in neuron_counts:
        synapses = _build_sparse_ring(N, k=k)
        actual_s = len(synapses)
        syn_counts.append(actual_s)
        print(f"  N={N:>5d}  syn={actual_s:>6d} ... ", end="", flush=True)

        t_cpp = bench_averaged(
            _bench_cpp, n_trials, N=N, synapses=synapses,
            duration=duration, dt=dt, I_val=I_val)
        t_np = bench_averaged(
            _bench_numpy, n_trials, N=N, synapses=synapses,
            duration=duration, dt=dt, I_val=I_val)

        cpp_times.append(t_cpp)
        numpy_times.append(t_np)
        speedup = t_np / t_cpp if t_cpp > 0 else float('inf')
        print(f"C++={t_cpp:.4f}s  NumPy={t_np:.4f}s  speedup={speedup:.1f}x")

    return {"neuron_counts": neuron_counts, "synapse_counts": syn_counts,
            "cpp": cpp_times, "numpy": numpy_times, "k": k}


def run_duration_scaling_avg(
    dt: float = 0.05, I_val: float = 10.0, n_trials: int = N_TRIALS,
) -> Dict:
    N = 100
    k = 5
    synapses = _build_sparse_ring(N, k=k)
    durations = [10, 25, 50, 100, 250, 500, 1000, 2000, 4000]
    cpp_times = []
    numpy_times = []

    print(f"\n--- Duration scaling (N={N}, k={k}, syn={len(synapses)}, "
          f"trials={n_trials}) ---")
    for dur in durations:
        num_steps = int(dur / dt)
        I_ext = np.full((N, num_steps), I_val)

        print(f"  dur={dur:>5.0f} ms  steps={num_steps:>6d} ... ",
              end="", flush=True)

        cpp_runs = []
        for _ in range(n_trials):
            cpp_net = Network(N)
            for pre, post, w, e, tau in synapses:
                cpp_net.add_synapse(pre, post, w, e, tau)
            start = time.perf_counter()
            cpp_net.simulate(dur, dt, I_ext)
            cpp_runs.append(time.perf_counter() - start)

        np_runs = []
        for _ in range(n_trials):
            np_net = NumpyHHNetwork(N)
            for pre, post, w, e, tau in synapses:
                np_net.add_synapse(pre, post, w, e, tau)
            start = time.perf_counter()
            np_net.simulate(dur, dt, I_ext)
            np_runs.append(time.perf_counter() - start)

        t_cpp = float(np.mean(cpp_runs))
        t_np = float(np.mean(np_runs))
        cpp_times.append(t_cpp)
        numpy_times.append(t_np)
        speedup = t_np / t_cpp if t_cpp > 0 else float('inf')
        print(f"C++={t_cpp:.4f}s  NumPy={t_np:.4f}s  speedup={speedup:.1f}x")

    return {"durations": durations, "cpp": cpp_times, "numpy": numpy_times,
            "N": N, "k": k, "S": len(synapses)}


# =============================================================================
# Plotting (same layout as benchmark_network.py but with N annotation)
# =============================================================================

def _subtitle(n_trials):
    return f"(averaged over N={n_trials} trials)"


def plot_timing_avg(results: Dict, figs_dir: Path, n_trials: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (topo_name, data) in enumerate(results.items()):
        ax = axes[idx]
        topo_sizes = data["sizes"]
        ax.loglog(topo_sizes, data["cpp"], 'o-', color='tab:blue',
                  linewidth=2, markersize=6, label='C++ backend')
        ax.loglog(topo_sizes, data["numpy"], 's--', color='tab:red',
                  linewidth=2, markersize=6, label='NumPy')
        ax.set_title(topo_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Number of neurons')
        ax.set_ylabel('Wall time (s)')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'Network Simulation: C++ vs NumPy Timing\n'
                 f'{_subtitle(n_trials)}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_avg_time_vs_neurons.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {figs_dir / 'benchmark_avg_time_vs_neurons.png'}")


def plot_speedup_avg(results: Dict, figs_dir: Path, n_trials: int):
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for idx, (topo_name, data) in enumerate(results.items()):
        topo_sizes = data["sizes"]
        speedups = [n / c if c > 0 else 0
                    for c, n in zip(data["cpp"], data["numpy"])]
        ax.plot(topo_sizes, speedups, f'{markers[idx]}-', color=colors[idx],
                linewidth=2, markersize=7, label=topo_name)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Parity')
    ax.set_xscale('log')
    ax.set_xlabel('Number of neurons', fontsize=12)
    ax.set_ylabel('Speedup (NumPy time / C++ time)', fontsize=12)
    ax.set_title(f'C++ Backend Speedup Over Pure NumPy\n'
                 f'{_subtitle(n_trials)}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_avg_speedup.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figs_dir / 'benchmark_avg_speedup.png'}")


def plot_isolation_avg(syn_data: Dict, neuron_data: Dict,
                       figs_dir: Path, n_trials: int):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: synapse scaling, time
    ax = axes[0, 0]
    ax.loglog(syn_data["synapse_counts"], syn_data["cpp"], 'o-',
              color='tab:blue', linewidth=2, markersize=6, label='C++')
    ax.loglog(syn_data["synapse_counts"], syn_data["numpy"], 's--',
              color='tab:red', linewidth=2, markersize=6, label='NumPy')
    ax.set_xlabel('Number of synapses')
    ax.set_ylabel('Wall time (s)')
    ax.set_title(f'Synapse Scaling (fixed N={syn_data["N"]})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # Top-right: synapse scaling, speedup
    ax = axes[0, 1]
    speedups = [n / c if c > 0 else 0
                for c, n in zip(syn_data["cpp"], syn_data["numpy"])]
    ax.semilogx(syn_data["synapse_counts"], speedups, 'D-',
                color='tab:purple', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Number of synapses')
    ax.set_ylabel('Speedup (NumPy / C++)')
    ax.set_title(f'Speedup vs Synapse Count (fixed N={syn_data["N"]})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    # Bottom-left: neuron scaling, time
    ax = axes[1, 0]
    ax.loglog(neuron_data["neuron_counts"], neuron_data["cpp"], 'o-',
              color='tab:blue', linewidth=2, markersize=6, label='C++')
    ax.loglog(neuron_data["neuron_counts"], neuron_data["numpy"], 's--',
              color='tab:red', linewidth=2, markersize=6, label='NumPy')
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Wall time (s)')
    ax.set_title(f'Neuron Scaling (sparse, {neuron_data["k"]} syn/neuron)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # Bottom-right: neuron scaling, speedup
    ax = axes[1, 1]
    speedups = [n / c if c > 0 else 0
                for c, n in zip(neuron_data["cpp"], neuron_data["numpy"])]
    ax.semilogx(neuron_data["neuron_counts"], speedups, 'D-',
                color='tab:purple', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Speedup (NumPy / C++)')
    ax.set_title(f'Speedup vs Neuron Count ({neuron_data["k"]} syn/neuron)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'Isolating Scaling: Neurons vs Synapses\n'
                 f'{_subtitle(n_trials)}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_avg_isolation.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figs_dir / 'benchmark_avg_isolation.png'}")


def plot_duration_avg(dur_data: Dict, figs_dir: Path, n_trials: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    durations = dur_data["durations"]
    N = dur_data["N"]
    S = dur_data["S"]

    ax = axes[0]
    ax.loglog(durations, dur_data["cpp"], 'o-', color='tab:blue',
              linewidth=2, markersize=6, label='C++')
    ax.loglog(durations, dur_data["numpy"], 's--', color='tab:red',
              linewidth=2, markersize=6, label='NumPy')
    ax.set_xlabel('Simulation duration (ms)')
    ax.set_ylabel('Wall time (s)')
    ax.set_title(f'Duration Scaling (N={N}, {S} synapses)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    speedups = [n / c if c > 0 else 0
                for c, n in zip(dur_data["cpp"], dur_data["numpy"])]
    ax.semilogx(durations, speedups, 'D-', color='tab:purple',
                linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Simulation duration (ms)')
    ax.set_ylabel('Speedup (NumPy / C++)')
    ax.set_title('Speedup vs Duration', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'Fixed Network, Scaling Simulation Time\n'
                 f'{_subtitle(n_trials)}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_avg_duration.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figs_dir / 'benchmark_avg_duration.png'}")


# =============================================================================
# Memory benchmark (averaged)
# =============================================================================

def run_memory_benchmark_avg(
    duration: float = 100.0, dt: float = 0.05,
    I_val: float = 10.0, n_trials: int = N_TRIALS,
) -> Dict:
    sizes = [5, 10, 20, 50, 100, 200, 500]
    cpp_mem = []
    numpy_mem = []

    print(f"\n--- Memory benchmark (all-to-all, trials={n_trials}) ---")
    for N in sizes:
        synapses = _build_all_to_all(N)
        n_syn = len(synapses)
        print(f"  N={N:>4d}  syn={n_syn:>7d} ... ", end="", flush=True)

        cpp_runs = [_measure_memory_cpp(N, synapses, duration, dt, I_val)
                    for _ in range(n_trials)]
        np_runs = [_measure_memory_numpy(N, synapses, duration, dt, I_val)
                   for _ in range(n_trials)]

        m_cpp = float(np.mean(cpp_runs))
        m_np = float(np.mean(np_runs))
        cpp_mem.append(m_cpp)
        numpy_mem.append(m_np)
        print(f"C++={m_cpp:.2f} MB  NumPy={m_np:.2f} MB")

    return {"sizes": sizes, "cpp": cpp_mem, "numpy": numpy_mem}


def plot_memory_avg(mem_data: Dict, figs_dir: Path, n_trials: int):
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = mem_data["sizes"]
    ax.plot(sizes, mem_data["cpp"], 'o-', color='tab:blue',
            linewidth=2, markersize=7, label='C++ backend')
    ax.plot(sizes, mem_data["numpy"], 's--', color='tab:red',
            linewidth=2, markersize=7, label='NumPy')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of neurons', fontsize=12)
    ax.set_ylabel('Peak memory (MB)', fontsize=12)
    ax.set_title(f'Peak Memory Usage (All-to-All Network)\n'
                 f'{_subtitle(n_trials)}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(figs_dir / "benchmark_avg_memory.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {figs_dir / 'benchmark_avg_memory.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    n_trials = N_TRIALS
    print("=" * 60)
    print(f"Averaged Network Benchmark (N={n_trials} trials per setting)")
    print("=" * 60)

    figs_dir = setup_output_dir()

    sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    duration = 100.0
    dt = 0.05
    I_val = 10.0

    print(f"\nSimulation: {duration} ms, dt={dt} ms, I_ext={I_val}")
    print(f"Network sizes: {sizes}")
    print(f"Trials per setting: {n_trials}")
    print(f"Output: {figs_dir}\n")

    validate_implementations(duration=50.0, dt=dt, I_val=I_val)

    # Topology scaling
    results = run_benchmarks_avg(sizes, duration, dt, I_val, n_trials)
    plot_timing_avg(results, figs_dir, n_trials)
    plot_speedup_avg(results, figs_dir, n_trials)

    # Isolation
    syn_data = run_synapse_scaling_avg(duration, dt, I_val, n_trials)
    neuron_data = run_neuron_scaling_avg(duration, dt, I_val, n_trials)
    plot_isolation_avg(syn_data, neuron_data, figs_dir, n_trials)

    # Duration scaling
    dur_data = run_duration_scaling_avg(dt, I_val, n_trials)
    plot_duration_avg(dur_data, figs_dir, n_trials)

    # Memory benchmark
    mem_data = run_memory_benchmark_avg(duration, dt, I_val, n_trials)
    plot_memory_avg(mem_data, figs_dir, n_trials)

    print("\n" + "=" * 60)
    print("Averaged benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
