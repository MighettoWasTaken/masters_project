#!/usr/bin/env python3
"""
Averaged Threading Benchmarks (N trials per setting)

Runs each benchmark from benchmark_network_threading.py multiple times and
averages the results to reduce noise from background processes.

Output:
  examples/figs/benchmark_threading_avg_time_vs_neurons.png
  examples/figs/benchmark_threading_avg_speedup.png
  examples/figs/benchmark_threading_avg_isolation.png
  examples/figs/benchmark_threading_avg_duration.png
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_network import NumpyHHNetwork
from benchmark_network_threading import (
    N_POP, DELAY_MS, WEIGHT, E_SYN, TAU_SYN,
    COLOR_SERIAL, COLOR_PHASE2, COLOR_NUMPY,
    TOPO_EDGES, TOPO_INTER_CONN, TOPO_MAX_N_CPP, TOPO_MAX_N_NUMPY,
    _bench_serial, _bench_phase2, _bench_numpy, _n_synapses,
    validate_implementations,
)
from hodgkin_huxley import RegionalNetwork, SynapseSpec

N_TRIALS = 10


def setup_output_dir() -> Path:
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    return figs_dir


def _subtitle(n_trials: int) -> str:
    return f"(averaged over {n_trials} trials)"


def _avg(fn, n_trials: int, **kwargs) -> float:
    return float(np.mean([fn(**kwargs) for _ in range(n_trials)]))


# =============================================================================
# Averaged benchmark runners
# =============================================================================

def run_benchmarks_avg(sizes: List[int], duration: float, dt: float,
                       I_val: float, n_trials: int) -> Dict:
    results: Dict[str, Dict] = {}

    for topo_name, edges in TOPO_EDGES.items():
        serial_times: List[float] = []
        phase2_times: List[float] = []
        numpy_times:  List[float] = []
        sizes_numpy:  List[int]   = []

        inter_conn = TOPO_INTER_CONN[topo_name]
        max_cpp    = TOPO_MAX_N_CPP[topo_name]
        max_numpy  = TOPO_MAX_N_NUMPY[topo_name]
        topo_sizes_cpp = [N for N in sizes if max_cpp is None or N <= max_cpp]

        if len(topo_sizes_cpp) < len(sizes):
            print(f"  ({topo_name}: C++ capped at N={max_cpp})")

        for N_total in topo_sizes_cpp:
            n_per_pop = N_total // N_POP
            n_syn = _n_synapses(n_per_pop, edges, inter_conn)
            do_numpy = (max_numpy is None or N_total <= max_numpy)

            print(f"  {topo_name:>12s}  N={N_total:>5d}  "
                  f"syn={n_syn:>8,d}  trials={n_trials} ... ",
                  end="", flush=True)

            t_s = _avg(_bench_serial, n_trials, n_per_pop=n_per_pop,
                       edges=edges, inter_conn=inter_conn,
                       duration=duration, dt=dt, I_val=I_val)
            t_p = _avg(_bench_phase2, n_trials, n_per_pop=n_per_pop,
                       edges=edges, inter_conn=inter_conn,
                       duration=duration, dt=dt, I_val=I_val)
            serial_times.append(t_s)
            phase2_times.append(t_p)

            if do_numpy:
                t_n = _avg(_bench_numpy, n_trials, n_per_pop=n_per_pop,
                           edges=edges, inter_conn=inter_conn,
                           duration=duration, dt=dt, I_val=I_val)
                numpy_times.append(t_n)
                sizes_numpy.append(N_total)
                print(f"serial={t_s:.3f}s  p2={t_p:.3f}s  numpy={t_n:.3f}s")
            else:
                print(f"serial={t_s:.3f}s  p2={t_p:.3f}s  (NumPy N/A)")

        results[topo_name] = {
            "serial":      serial_times,
            "phase2":      phase2_times,
            "numpy":       numpy_times,
            "sizes_cpp":   topo_sizes_cpp,
            "sizes_numpy": sizes_numpy,
        }

    return results


def run_synapse_scaling_avg(duration: float, dt: float,
                            I_val: float, n_trials: int) -> Dict:
    n_per_pop = 50
    N_total = N_POP * n_per_pop

    edge_sets = [
        ("1-edge",          [(0, 1)],                        "one_to_one"),
        ("2-edge",          [(0, 1), (1, 2)],                "one_to_one"),
        ("Chain (3)",       TOPO_EDGES["Chain"],             "one_to_one"),
        ("Ring (4)",        TOPO_EDGES["Ring"],              "one_to_one"),
        ("Star (6)",        TOPO_EDGES["Star"],              "one_to_one"),
        ("All-to-All (12)", TOPO_EDGES["All-to-All"],        "all_to_all"),
    ]

    serial_times:   List[float] = []
    phase2_times:   List[float] = []
    numpy_times:    List[float] = []
    synapse_counts: List[int]   = []

    print(f"\n--- Synapse scaling  "
          f"(n_per_pop={n_per_pop}, N_total={N_total}, trials={n_trials}) ---")
    for name, edges, inter_conn in edge_sets:
        n_syn = _n_synapses(n_per_pop, edges, inter_conn)
        synapse_counts.append(n_syn)
        print(f"  {name:>20s}  syn={n_syn:>6,d} ... ", end="", flush=True)

        t_s = _avg(_bench_serial, n_trials, n_per_pop=n_per_pop,
                   edges=edges, inter_conn=inter_conn,
                   duration=duration, dt=dt, I_val=I_val)
        t_p = _avg(_bench_phase2, n_trials, n_per_pop=n_per_pop,
                   edges=edges, inter_conn=inter_conn,
                   duration=duration, dt=dt, I_val=I_val)
        t_n = _avg(_bench_numpy, n_trials, n_per_pop=n_per_pop,
                   edges=edges, inter_conn=inter_conn,
                   duration=duration, dt=dt, I_val=I_val)

        serial_times.append(t_s)
        phase2_times.append(t_p)
        numpy_times.append(t_n)
        print(f"serial={t_s:.3f}s  p2={t_p:.3f}s  numpy={t_n:.3f}s")

    return {
        "synapse_counts": synapse_counts,
        "serial":   serial_times,
        "phase2":   phase2_times,
        "numpy":    numpy_times,
        "N_total":  N_total,
        "n_per_pop": n_per_pop,
    }


def run_neuron_scaling_avg(duration: float, dt: float,
                           I_val: float, n_trials: int) -> Dict:
    edges      = TOPO_EDGES["Chain"]
    inter_conn = TOPO_INTER_CONN["Chain"]
    n_per_pop_values = [2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000]
    numpy_max_npp    = TOPO_MAX_N_NUMPY["Chain"] // N_POP

    serial_times:  List[float] = []
    phase2_times:  List[float] = []
    numpy_times:   List[float] = []
    N_totals:      List[int]   = []
    synapse_counts: List[int]  = []
    numpy_N_totals: List[int]  = []

    print(f"\n--- Neuron scaling  (chain, 1:1 inter-pop, trials={n_trials}) ---")
    for npp in n_per_pop_values:
        N_total = N_POP * npp
        n_syn   = _n_synapses(npp, edges, inter_conn)
        N_totals.append(N_total)
        synapse_counts.append(n_syn)
        do_numpy = npp <= numpy_max_npp

        print(f"  n_per_pop={npp:>5d}  N_total={N_total:>5d}  "
              f"syn={n_syn:>7,d} ... ", end="", flush=True)

        t_s = _avg(_bench_serial, n_trials, n_per_pop=npp,
                   edges=edges, inter_conn=inter_conn,
                   duration=duration, dt=dt, I_val=I_val)
        t_p = _avg(_bench_phase2, n_trials, n_per_pop=npp,
                   edges=edges, inter_conn=inter_conn,
                   duration=duration, dt=dt, I_val=I_val)
        serial_times.append(t_s)
        phase2_times.append(t_p)

        if do_numpy:
            t_n = _avg(_bench_numpy, n_trials, n_per_pop=npp,
                       edges=edges, inter_conn=inter_conn,
                       duration=duration, dt=dt, I_val=I_val)
            numpy_times.append(t_n)
            numpy_N_totals.append(N_total)
            print(f"serial={t_s:.3f}s  p2={t_p:.3f}s  numpy={t_n:.3f}s")
        else:
            print(f"serial={t_s:.3f}s  p2={t_p:.3f}s  (NumPy N/A)")

    return {
        "N_totals":         N_totals,
        "synapse_counts":   synapse_counts,
        "serial":           serial_times,
        "phase2":           phase2_times,
        "numpy":            numpy_times,
        "numpy_N_totals":   numpy_N_totals,
        "n_per_pop_values": n_per_pop_values,
    }


def run_duration_scaling_avg(dt: float, I_val: float, n_trials: int) -> Dict:
    n_per_pop = 25
    k         = 5
    N_total   = N_POP * n_per_pop

    global_pairs = [
        (i, (i + off) % N_total)
        for i in range(N_total)
        for off in range(1, k + 1)
    ]
    n_syn = len(global_pairs)

    pop_edges: dict = defaultdict(list)
    for pre, post in global_pairs:
        sp = pre  // n_per_pop
        dp = post // n_per_pop
        pop_edges[(sp, dp)].append((pre % n_per_pop, post % n_per_pop))

    def _make_cpp(use_phase2: bool) -> RegionalNetwork:
        rn = RegionalNetwork()
        pops = [f"P{i}" for i in range(N_POP)]
        for p in pops:
            rn.add_population(p, n_per_pop)
        syn = SynapseSpec.exponential(TAU_SYN, E_syn=E_SYN)
        for (sp, dp), pairs in pop_edges.items():
            delay = dt if sp != dp else 0.0
            p = list(pairs)
            rn.connect(pops[sp], pops[dp],
                       lambda ns, nd, p=p: p,
                       weight=WEIGHT, synapse=syn, delay=delay)
        if use_phase2:
            rn.set_thread_groups({f"g{i}": [f"P{i}"] for i in range(N_POP)})
        else:
            rn.clear_thread_groups()
        return rn

    def _make_numpy() -> NumpyHHNetwork:
        net = NumpyHHNetwork(N_total)
        for pre, post in global_pairs:
            delay = dt if (pre // n_per_pop) != (post // n_per_pop) else 0.0
            net.add_synapse(pre, post, WEIGHT, E_SYN, TAU_SYN, delay)
        return net

    durations = [10, 25, 50, 100, 250, 500, 1000, 2000]
    serial_times: List[float] = []
    phase2_times: List[float] = []
    numpy_times:  List[float] = []

    print(f"\n--- Duration scaling  "
          f"(sparse ring, N_total={N_total}, k={k}, syn={n_syn}, "
          f"trials={n_trials}) ---")
    for dur in durations:
        num_steps = int(dur / dt)
        I_arr = np.full((N_total, num_steps), I_val)
        I_ext = {f"P{i}": I_val for i in range(N_POP)}

        print(f"  duration={dur:>5.0f} ms ... ", end="", flush=True)

        s_runs, p_runs, n_runs = [], [], []
        for _ in range(n_trials):
            rn_s = _make_cpp(False)
            t0 = time.perf_counter()
            rn_s.simulate(dur, dt, I_ext)
            s_runs.append(time.perf_counter() - t0)

            rn_p = _make_cpp(True)
            t0 = time.perf_counter()
            rn_p.simulate(dur, dt, I_ext)
            p_runs.append(time.perf_counter() - t0)

            net = _make_numpy()
            t0 = time.perf_counter()
            net.simulate(dur, dt, I_arr)
            n_runs.append(time.perf_counter() - t0)

        t_s = float(np.mean(s_runs))
        t_p = float(np.mean(p_runs))
        t_n = float(np.mean(n_runs))
        serial_times.append(t_s)
        phase2_times.append(t_p)
        numpy_times.append(t_n)
        print(f"serial={t_s:.3f}s  p2={t_p:.3f}s  numpy={t_n:.3f}s")

    return {
        "durations": durations,
        "serial":    serial_times,
        "phase2":    phase2_times,
        "numpy":     numpy_times,
        "N_total":   N_total,
        "n_syn":     n_syn,
    }


# =============================================================================
# Plotting (same layout as benchmark_network_threading.py + trial annotation)
# =============================================================================

def plot_timing_avg(results: Dict, figs_dir: Path, n_trials: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (topo_name, data) in enumerate(results.items()):
        ax = axes[idx]
        sc = data["sizes_cpp"]
        sn = data["sizes_numpy"]

        ax.loglog(sc, data["serial"], "o-",  color=COLOR_SERIAL,
                  linewidth=2, markersize=6, label="C++ serial")
        ax.loglog(sc, data["phase2"], "s-",  color=COLOR_PHASE2,
                  linewidth=2, markersize=6, label="C++ delay-parallel")
        if data["numpy"]:
            ax.loglog(sn, data["numpy"], "D--", color=COLOR_NUMPY,
                      linewidth=2, markersize=6, label="NumPy")

        ax.set_title(topo_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Total neurons")
        ax.set_ylabel("Wall time (s)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Network Simulation: Serial vs Delay-Parallel Threading vs NumPy\n"
        f"({N_POP} populations, {DELAY_MS} ms inter-group delay, "
        f"one thread group per population)  {_subtitle(n_trials)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figs_dir / "benchmark_threading_avg_time_vs_neurons.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def plot_speedup_avg(results: Dict, figs_dir: Path, n_trials: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    markers = ["o", "s", "^", "D"]
    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    ax_np_s = axes[0]
    ax_np_p = axes[1]

    # Collect all speedup values in a first pass so we can set a shared y range.
    all_speedups: List[float] = []
    plot_data = []

    for idx, (topo_name, data) in enumerate(results.items()):
        sn = data["sizes_numpy"]
        if data["numpy"] and sn:
            n_pts   = len(data["numpy"])
            sp_np_s = [n / s if s > 0 else 0.0
                       for n, s in zip(data["numpy"], data["serial"][:n_pts])]
            sp_np_p = [n / p if p > 0 else 0.0
                       for n, p in zip(data["numpy"], data["phase2"][:n_pts])]
            all_speedups.extend(sp_np_s)
            all_speedups.extend(sp_np_p)
            plot_data.append((idx, topo_name, sn, sp_np_s, sp_np_p))

    shared_ymax = max(all_speedups) * 1.12 if all_speedups else 10.0
    shared_ymin = max(0.5, min(v for v in all_speedups if v > 0) * 0.85) \
                  if all_speedups else 0.5

    for idx, topo_name, sn, sp_np_s, sp_np_p in plot_data:
        mk  = f"{markers[idx]}-"
        col = colors[idx]
        ax_np_s.plot(sn, sp_np_s, mk, color=col,
                     linewidth=2, markersize=7, label=topo_name)
        ax_np_p.plot(sn, sp_np_p, mk, color=col,
                     linewidth=2, markersize=7, label=topo_name)

    panel_cfg = [
        (ax_np_s, "NumPy vs C++ Serial",        "Speedup  (NumPy / serial)"),
        (ax_np_p, "NumPy vs C++ Delay-Parallel", "Speedup  (NumPy / delay-parallel)"),
    ]
    for ax, title, ylabel in panel_cfg:
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7, label="Parity")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(shared_ymin, shared_ymax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}×"))
        ax.set_xlabel("Total neurons", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Speedup Comparison: Delay-Parallel Threading vs Python Baseline\n"
        f"{_subtitle(n_trials)}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = figs_dir / "benchmark_threading_avg_speedup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_isolation_avg(syn_data: Dict, neuron_data: Dict,
                       figs_dir: Path, n_trials: int):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top-left: synapse scaling, absolute time ---
    ax = axes[0, 0]
    ax.loglog(syn_data["synapse_counts"], syn_data["serial"], "o-",
              color=COLOR_SERIAL, linewidth=2, markersize=6, label="C++ serial")
    ax.loglog(syn_data["synapse_counts"], syn_data["phase2"], "s-",
              color=COLOR_PHASE2,  linewidth=2, markersize=6, label="C++ delay-parallel")
    ax.loglog(syn_data["synapse_counts"], syn_data["numpy"],  "D--",
              color=COLOR_NUMPY,   linewidth=2, markersize=6, label="NumPy")
    ax.set_xlabel("Number of synapses")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"Synapse Scaling  (N_total={syn_data['N_total']})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # --- Top-right: synapse scaling, speedup (NumPy/serial and NumPy/delay-parallel) ---
    ax = axes[0, 1]
    sp_np_s = [n / s if s > 0 else 0.0
               for n, s in zip(syn_data["numpy"], syn_data["serial"])]
    sp_np_p = [n / p if p > 0 else 0.0
               for n, p in zip(syn_data["numpy"], syn_data["phase2"])]
    ax.semilogx(syn_data["synapse_counts"], sp_np_s, "D-",
                color="tab:purple", linewidth=2, markersize=6,
                label="NumPy / serial")
    ax.semilogx(syn_data["synapse_counts"], sp_np_p, "s-",
                color=COLOR_PHASE2, linewidth=2, markersize=6,
                label="NumPy / delay-parallel")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Number of synapses")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup vs Synapse Count  (N_total={syn_data['N_total']})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # --- Bottom-left: neuron scaling, absolute time ---
    ax = axes[1, 0]
    ax.loglog(neuron_data["N_totals"], neuron_data["serial"], "o-",
              color=COLOR_SERIAL, linewidth=2, markersize=6, label="C++ serial")
    ax.loglog(neuron_data["N_totals"], neuron_data["phase2"], "s-",
              color=COLOR_PHASE2,  linewidth=2, markersize=6, label="C++ delay-parallel")
    if neuron_data["numpy"]:
        ax.loglog(neuron_data["numpy_N_totals"], neuron_data["numpy"], "D--",
                  color=COLOR_NUMPY, linewidth=2, markersize=6, label="NumPy")
    ax.set_xlabel("Total neurons  (N_POP × n_per_pop)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Neuron Scaling  (chain topology)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # --- Bottom-right: neuron scaling, speedup (NumPy/serial and NumPy/delay-parallel) ---
    ax = axes[1, 1]
    if neuron_data["numpy"]:
        n_pts = len(neuron_data["numpy"])
        sp_np_s_n = [n / s if s > 0 else 0.0
                     for n, s in zip(neuron_data["numpy"],
                                     neuron_data["serial"][:n_pts])]
        sp_np_p_n = [n / p if p > 0 else 0.0
                     for n, p in zip(neuron_data["numpy"],
                                     neuron_data["phase2"][:n_pts])]
        ax.semilogx(neuron_data["numpy_N_totals"], sp_np_s_n, "D-",
                    color="tab:purple", linewidth=2, markersize=6,
                    label="NumPy / serial")
        ax.semilogx(neuron_data["numpy_N_totals"], sp_np_p_n, "s-",
                    color=COLOR_PHASE2, linewidth=2, markersize=6,
                    label="NumPy / delay-parallel")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Total neurons")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Neuron Count  (chain)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Isolating Scaling: Neurons vs Synapses  (Delay-Parallel Threading)\n"
        f"{_subtitle(n_trials)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = figs_dir / "benchmark_threading_avg_isolation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_duration_avg(dur_data: Dict, figs_dir: Path, n_trials: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    durations = dur_data["durations"]
    N_total   = dur_data["N_total"]
    n_syn     = dur_data["n_syn"]

    ax = axes[0]
    ax.loglog(durations, dur_data["serial"], "o-", color=COLOR_SERIAL,
              linewidth=2, markersize=6, label="C++ serial")
    ax.loglog(durations, dur_data["phase2"], "s-", color=COLOR_PHASE2,
              linewidth=2, markersize=6, label="C++ delay-parallel")
    ax.loglog(durations, dur_data["numpy"],  "D--", color=COLOR_NUMPY,
              linewidth=2, markersize=6, label="NumPy")
    ax.set_xlabel("Simulation duration (ms)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"Duration Scaling  (N={N_total}, {n_syn:,} synapses)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    sp_np_s = [n / s if s > 0 else 0.0
               for n, s in zip(dur_data["numpy"], dur_data["serial"])]
    sp_np_p = [n / p if p > 0 else 0.0
               for n, p in zip(dur_data["numpy"], dur_data["phase2"])]
    ax.semilogx(durations, sp_np_s, "D-", color="tab:purple",
                linewidth=2, markersize=6, label="NumPy / serial")
    ax.semilogx(durations, sp_np_p, "s-", color=COLOR_PHASE2,
                linewidth=2, markersize=6, label="NumPy / delay-parallel")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Simulation duration (ms)")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Duration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Fixed Network, Scaling Simulation Time\n{_subtitle(n_trials)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = figs_dir / "benchmark_threading_avg_duration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    n_trials = N_TRIALS
    print("=" * 65)
    print(f"Averaged Threading Benchmark ({n_trials} trials per setting)")
    print("=" * 65)

    figs_dir = setup_output_dir()

    sizes    = [8, 20, 40, 100, 200, 400, 800, 2000, 4000]
    duration = 100.0
    dt       = 0.05
    I_val    = 10.0

    print(f"\nSimulation:   {duration} ms,  dt={dt} ms,  I_ext={I_val} μA/cm²")
    print(f"Populations:  {N_POP},  inter-group delay={DELAY_MS} ms")
    print(f"Total-N sizes: {sizes}")
    print(f"Trials per setting: {n_trials}")
    print(f"Output:       {figs_dir}\n")

    validate_implementations(duration=20.0, dt=dt, I_val=I_val)

    print(f"\n--- Main benchmark  ({duration} ms, dt={dt} ms) ---")
    results = run_benchmarks_avg(sizes, duration, dt, I_val, n_trials)
    plot_timing_avg(results, figs_dir, n_trials)
    plot_speedup_avg(results, figs_dir, n_trials)

    syn_data    = run_synapse_scaling_avg(duration, dt, I_val, n_trials)
    neuron_data = run_neuron_scaling_avg(duration, dt, I_val, n_trials)
    plot_isolation_avg(syn_data, neuron_data, figs_dir, n_trials)

    dur_data = run_duration_scaling_avg(dt=dt, I_val=I_val, n_trials=n_trials)
    plot_duration_avg(dur_data, figs_dir, n_trials)

    print("\n" + "=" * 65)
    print("Averaged benchmarks complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
