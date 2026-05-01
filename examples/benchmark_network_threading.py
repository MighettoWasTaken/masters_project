#!/usr/bin/env python3
"""
Network Threading Benchmark: C++ Serial vs C++ Phase-2 Threading vs Pure NumPy

Identical in style to examples/benchmark_network.py, but uses multi-population
networks with inter-group axonal delays that activate Phase-2 delay-decomposition
parallelism.

Network structure
-----------------
  N_POP = 4 populations, each of n_per_pop neurons.
  Inter-population connections carry DELAY_MS axonal delay (required for Phase-2).
  No intra-population connections (keeps synapse counts and comparisons clean).

Topologies (directed edges between population indices 0..3):
  1. Chain       P0 → P1 → P2 → P3
  2. Ring        P0 → P1 → P2 → P3 → P0
  3. All-to-All  every Pi → Pj  (i ≠ j)
  4. Star        P0 ↔ P1, P0 ↔ P2, P0 ↔ P3

Backends compared
-----------------
  C++ serial   — RegionalNetwork, no thread groups, num_threads=1
  C++ Phase-2  — RegionalNetwork, one thread group per population
  Pure NumPy   — flat N_total-neuron NumpyHHNetwork (same synapse list, same delays)

Output
------
  examples/figs/benchmark_threading_time_vs_neurons.png
  examples/figs/benchmark_threading_speedup.png
  examples/figs/benchmark_threading_isolation.png
  examples/figs/benchmark_threading_duration.png
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Import the NumPy HH implementation from the companion benchmark
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_network import NumpyHHNetwork   # noqa: E402

from hodgkin_huxley import RegionalNetwork, SynapseSpec

# =============================================================================
# Configuration
# =============================================================================

N_POP    = 4     # populations / thread groups
DELAY_MS = 1.0   # axonal delay between populations (ms); triggers Phase-2 ring buffers
WEIGHT   = 0.5   # synaptic weight (μS)
E_SYN    = 0.0   # reversal potential (mV)
TAU_SYN  = 2.0   # synaptic time constant (ms)

COLOR_SERIAL = "tab:blue"
COLOR_PHASE2 = "#2ca02c"
COLOR_NUMPY  = "tab:red"

# Directed edges between population indices for each topology
TOPO_EDGES: Dict[str, List] = {
    "Chain":      [(i, i + 1) for i in range(N_POP - 1)],
    "Ring":       [(i, (i + 1) % N_POP) for i in range(N_POP)],
    "All-to-All": [(i, j) for i in range(N_POP)
                           for j in range(N_POP) if i != j],
    "Star":       [(0, i) for i in range(1, N_POP)]
                + [(i, 0) for i in range(1, N_POP)],
}

# Inter-population synapse density per topology.
#   "one_to_one"  — neuron j in src connects to neuron j in dst  (linear synapse count)
#   "all_to_all"  — every src neuron connects to every dst neuron (quadratic)
TOPO_INTER_CONN: Dict[str, str] = {
    "Chain":      "one_to_one",
    "Ring":       "one_to_one",
    "All-to-All": "all_to_all",
    "Star":       "one_to_one",
}

# Maximum total-N per topology
# one_to_one topologies have linear synapse counts so caps can be large
TOPO_MAX_N_CPP: Dict[str, Optional[int]] = {
    "Chain":      None,    # linear — no cap needed
    "Ring":       None,
    "All-to-All": 400,     # n_per_pop=100 → 12×10K = 120K synapses
    "Star":       None,
}
TOPO_MAX_N_NUMPY: Dict[str, Optional[int]] = {
    "Chain":      8000,    # 3×2K = 6K synapses — trivial for NumPy
    "Ring":       8000,
    "All-to-All": 200,     # n_per_pop=50 → 30K synapses
    "Star":       8000,
}


# =============================================================================
# Network builders
# =============================================================================

def _build_cpp_net(n_per_pop: int, edges, inter_conn: str) -> RegionalNetwork:
    """RegionalNetwork with N_POP populations connected over `edges`.

    inter_conn="one_to_one": neuron j in src → neuron j in dst (linear synapses)
    inter_conn="all_to_all": every src neuron → every dst neuron (quadratic)
    """
    rn = RegionalNetwork()
    pops = [f"P{i}" for i in range(N_POP)]
    for p in pops:
        rn.add_population(p, n_per_pop)
    syn = SynapseSpec.exponential(TAU_SYN, E_syn=E_SYN)
    for src, dst in edges:
        if inter_conn == "one_to_one":
            n = n_per_pop
            rn.connect(pops[src], pops[dst],
                       lambda ns, nd, n=n: [(j, j) for j in range(n)],
                       weight=WEIGHT, synapse=syn, delay=DELAY_MS)
        else:
            rn.connect(pops[src], pops[dst], "all_to_all",
                       weight=WEIGHT, synapse=syn, delay=DELAY_MS)
    return rn


def _build_numpy_net(n_per_pop: int, edges, inter_conn: str) -> NumpyHHNetwork:
    """Flat NumpyHHNetwork with the same inter-population synapses (global indices)."""
    N_total = N_POP * n_per_pop
    net = NumpyHHNetwork(N_total)
    for src_pop, dst_pop in edges:
        src_off = src_pop * n_per_pop
        dst_off = dst_pop * n_per_pop
        if inter_conn == "one_to_one":
            for j in range(n_per_pop):
                net.add_synapse(src_off + j, dst_off + j,
                                WEIGHT, E_SYN, TAU_SYN, DELAY_MS)
        else:
            for i in range(n_per_pop):
                for j in range(n_per_pop):
                    net.add_synapse(src_off + i, dst_off + j,
                                    WEIGHT, E_SYN, TAU_SYN, DELAY_MS)
    return net


def _n_synapses(n_per_pop: int, edges, inter_conn: str) -> int:
    if inter_conn == "one_to_one":
        return len(edges) * n_per_pop
    return len(edges) * n_per_pop * n_per_pop


# =============================================================================
# Timing helpers
# =============================================================================

def _bench_serial(n_per_pop: int, edges, inter_conn: str,
                  duration: float, dt: float, I_val: float) -> float:
    """C++ serial path — no thread groups."""
    rn = _build_cpp_net(n_per_pop, edges, inter_conn)
    rn.clear_thread_groups()
    I_ext = {f"P{i}": I_val for i in range(N_POP)}
    start = time.perf_counter()
    rn.simulate(duration, dt, I_ext)
    return time.perf_counter() - start


def _bench_phase2(n_per_pop: int, edges, inter_conn: str,
                  duration: float, dt: float, I_val: float) -> float:
    """C++ Phase-2 path — one thread group per population."""
    rn = _build_cpp_net(n_per_pop, edges, inter_conn)
    rn.set_thread_groups({f"g{i}": [f"P{i}"] for i in range(N_POP)})
    I_ext = {f"P{i}": I_val for i in range(N_POP)}
    start = time.perf_counter()
    rn.simulate(duration, dt, I_ext)
    return time.perf_counter() - start


def _bench_numpy(n_per_pop: int, edges, inter_conn: str,
                 duration: float, dt: float, I_val: float) -> float:
    """Pure NumPy — flat equivalent network with matching synapses and delays."""
    N_total = N_POP * n_per_pop
    net = _build_numpy_net(n_per_pop, edges, inter_conn)
    num_steps = int(duration / dt)
    I_arr = np.full((N_total, num_steps), I_val)
    start = time.perf_counter()
    net.simulate(duration, dt, I_arr)
    return time.perf_counter() - start


# =============================================================================
# Validation
# =============================================================================

def validate_implementations(duration: float = 20.0, dt: float = 0.05,
                              I_val: float = 10.0) -> bool:
    """
    Verify serial == Phase-2 (must be bit-identical) and NumPy is approximately
    consistent with the C++ result.
    """
    n_per_pop = 3
    edges = TOPO_EDGES["Chain"]
    inter_conn = TOPO_INTER_CONN["Chain"]

    # C++ serial
    rn_s = _build_cpp_net(n_per_pop, edges, inter_conn)
    rn_s.clear_thread_groups()
    traces_serial = rn_s.simulate(duration, dt, {f"P{i}": I_val for i in range(N_POP)})

    # C++ Phase-2
    rn_p = _build_cpp_net(n_per_pop, edges, inter_conn)
    rn_p.set_thread_groups({f"g{i}": [f"P{i}"] for i in range(N_POP)})
    traces_phase2 = rn_p.simulate(duration, dt, {f"P{i}": I_val for i in range(N_POP)})

    atol = 1e-10
    print(f"Validation: C++ serial vs Phase-2 (atol={atol:.0e})")
    all_ok = True
    for i in range(N_POP):
        key = f"P{i}"
        max_err = np.max(np.abs(traces_serial[key] - traces_phase2[key]))
        ok = max_err <= atol
        status = "PASS" if ok else "FAIL"
        print(f"  Pop {key}: max_err={max_err:.2e}  {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("  All pops: PASS\n")
    else:
        print("  WARNING: serial and Phase-2 differ beyond tolerance — threading bug?\n")

    # NumPy approximate comparison
    N_total = N_POP * n_per_pop
    net = _build_numpy_net(n_per_pop, edges, inter_conn)
    num_steps = int(duration / dt)
    np_traces = net.simulate(duration, dt, np.full((N_total, num_steps), I_val))

    cpp_P0 = traces_serial["P0"]
    np_P0  = np_traces[:n_per_pop, :]
    max_err = np.max(np.abs(cpp_P0 - np_P0))
    mean_err = np.mean(np.abs(cpp_P0 - np_P0))
    print(f"Validation: C++ serial vs NumPy (chain, {n_per_pop} neurons/pop)")
    print(f"  Pop P0: max_err={max_err:.4f} mV  mean_err={mean_err:.4f} mV")
    print(f"  {'PASS' if max_err < 5.0 else 'WARN: large discrepancy'}\n")

    return all_ok


# =============================================================================
# Main benchmark: all topologies × all sizes
# =============================================================================

def run_benchmarks(sizes: List[int], duration: float, dt: float,
                   I_val: float) -> Dict:
    """
    Sweep all topologies and neuron counts.

    Returns dict keyed by topology name:
      {
        "serial":      [times for sizes_cpp],
        "phase2":      [times for sizes_cpp],
        "numpy":       [times for sizes_numpy only],
        "sizes_cpp":   [...],
        "sizes_numpy": [...],
      }
    """
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
            n_syn_cap = _n_synapses(max_cpp // N_POP, edges, inter_conn)
            print(f"  ({topo_name}: C++ capped at N={max_cpp}, "
                  f"{n_syn_cap:,} synapses)")

        for N_total in topo_sizes_cpp:
            n_per_pop = N_total // N_POP
            n_syn = _n_synapses(n_per_pop, edges, inter_conn)
            do_numpy = (max_numpy is None or N_total <= max_numpy)

            print(f"  {topo_name:>12s}  N={N_total:>5d}  "
                  f"synapses={n_syn:>8,d} ... ", end="", flush=True)

            t_serial = _bench_serial(n_per_pop, edges, inter_conn, duration, dt, I_val)
            t_phase2 = _bench_phase2(n_per_pop, edges, inter_conn, duration, dt, I_val)
            serial_times.append(t_serial)
            phase2_times.append(t_phase2)

            sp_p2 = t_serial / t_phase2 if t_phase2 > 0 else float("inf")

            if do_numpy:
                t_numpy = _bench_numpy(n_per_pop, edges, inter_conn, duration, dt, I_val)
                numpy_times.append(t_numpy)
                sizes_numpy.append(N_total)
                sp_np_s = t_numpy / t_serial if t_serial > 0 else float("inf")
                sp_np_p = t_numpy / t_phase2 if t_phase2 > 0 else float("inf")
                print(f"serial={t_serial:.3f}s  phase2={t_phase2:.3f}s  "
                      f"numpy={t_numpy:.3f}s  "
                      f"phase2_su={sp_p2:.1f}x  "
                      f"np_vs_serial={sp_np_s:.1f}x  "
                      f"np_vs_phase2={sp_np_p:.1f}x")
            else:
                print(f"serial={t_serial:.3f}s  phase2={t_phase2:.3f}s  "
                      f"phase2_su={sp_p2:.1f}x  (NumPy N/A)")

        results[topo_name] = {
            "serial":      serial_times,
            "phase2":      phase2_times,
            "numpy":       numpy_times,
            "sizes_cpp":   topo_sizes_cpp,
            "sizes_numpy": sizes_numpy,
        }

    return results


# =============================================================================
# Isolation benchmarks
# =============================================================================

def run_synapse_scaling(duration: float = 100.0, dt: float = 0.05,
                        I_val: float = 10.0) -> Dict:
    """
    Fixed n_per_pop=50 (N_total=200), sweep synapse count by varying the
    number of directed inter-population edges (1 → 12).
    """
    n_per_pop = 50
    N_total = N_POP * n_per_pop

    # Each entry: (label, edges, inter_conn)
    # 1–5 edges use one_to_one; 12-edge all-to-all uses all_to_all
    edge_sets = [
        ("1-edge",          [(0, 1)],                        "one_to_one"),
        ("2-edge",          [(0, 1), (1, 2)],                "one_to_one"),
        ("Chain (3)",       TOPO_EDGES["Chain"],             "one_to_one"),
        ("Ring (4)",        TOPO_EDGES["Ring"],              "one_to_one"),
        ("Star (6)",        TOPO_EDGES["Star"],              "one_to_one"),
        ("All-to-All (12)", TOPO_EDGES["All-to-All"],        "all_to_all"),
    ]

    serial_times: List[float] = []
    phase2_times: List[float] = []
    numpy_times:  List[float] = []
    synapse_counts: List[int] = []

    print(f"\n--- Synapse scaling  "
          f"(fixed n_per_pop={n_per_pop}, N_total={N_total}) ---")
    for name, edges, inter_conn in edge_sets:
        n_syn = _n_synapses(n_per_pop, edges, inter_conn)
        synapse_counts.append(n_syn)
        print(f"  {name:>20s}  synapses={n_syn:>6,d} ... ", end="", flush=True)

        t_s = _bench_serial(n_per_pop, edges, inter_conn, duration, dt, I_val)
        t_p = _bench_phase2(n_per_pop, edges, inter_conn, duration, dt, I_val)
        t_n = _bench_numpy(n_per_pop, edges, inter_conn, duration, dt, I_val)

        serial_times.append(t_s)
        phase2_times.append(t_p)
        numpy_times.append(t_n)

        sp_p2 = t_s / t_p if t_p > 0 else 0.0
        sp_np = t_n / t_s if t_s > 0 else 0.0
        print(f"serial={t_s:.3f}s  phase2={t_p:.3f}s  numpy={t_n:.3f}s  "
              f"phase2_su={sp_p2:.1f}x  np_su={sp_np:.1f}x")

    return {
        "synapse_counts": synapse_counts,
        "serial": serial_times,
        "phase2": phase2_times,
        "numpy":  numpy_times,
        "N_total": N_total,
        "n_per_pop": n_per_pop,
    }


def run_neuron_scaling(duration: float = 100.0, dt: float = 0.05,
                       I_val: float = 10.0) -> Dict:
    """
    Chain topology (1:1 inter-pop), sweep n_per_pop.
    Synapses grow linearly: S = 3 × n_per_pop.
    """
    edges = TOPO_EDGES["Chain"]
    inter_conn = TOPO_INTER_CONN["Chain"]
    n_per_pop_values = [2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000]
    numpy_max_npp = TOPO_MAX_N_NUMPY["Chain"] // N_POP  # 2000

    serial_times: List[float] = []
    phase2_times: List[float] = []
    numpy_times:  List[float] = []
    N_totals: List[int] = []
    synapse_counts: List[int] = []
    numpy_N_totals: List[int] = []

    print(f"\n--- Neuron scaling  (chain, 1:1 inter-pop) ---")
    for npp in n_per_pop_values:
        N_total = N_POP * npp
        n_syn = _n_synapses(npp, edges, inter_conn)
        N_totals.append(N_total)
        synapse_counts.append(n_syn)
        do_numpy = npp <= numpy_max_npp

        print(f"  n_per_pop={npp:>5d}  N_total={N_total:>5d}  "
              f"synapses={n_syn:>7,d} ... ", end="", flush=True)

        t_s = _bench_serial(npp, edges, inter_conn, duration, dt, I_val)
        t_p = _bench_phase2(npp, edges, inter_conn, duration, dt, I_val)
        serial_times.append(t_s)
        phase2_times.append(t_p)

        sp_p2 = t_s / t_p if t_p > 0 else 0.0

        if do_numpy:
            t_n = _bench_numpy(npp, edges, inter_conn, duration, dt, I_val)
            numpy_times.append(t_n)
            numpy_N_totals.append(N_total)
            sp_np = t_n / t_s if t_s > 0 else 0.0
            print(f"serial={t_s:.3f}s  phase2={t_p:.3f}s  numpy={t_n:.3f}s  "
                  f"phase2_su={sp_p2:.1f}x  np_su={sp_np:.1f}x")
        else:
            print(f"serial={t_s:.3f}s  phase2={t_p:.3f}s  "
                  f"phase2_su={sp_p2:.1f}x  (NumPy N/A)")

    return {
        "N_totals":        N_totals,
        "synapse_counts":  synapse_counts,
        "serial":          serial_times,
        "phase2":          phase2_times,
        "numpy":           numpy_times,
        "numpy_N_totals":  numpy_N_totals,
        "n_per_pop_values": n_per_pop_values,
    }


# =============================================================================
# Duration scaling
# =============================================================================

def run_duration_scaling(dt: float = 0.05, I_val: float = 10.0) -> Dict:
    """
    Fixed sparse-ring network matching benchmark_network.py exactly:
      N_total=100  (n_per_pop=25 × N_POP=4)
      k=5 neighbors  →  500 synapses

    Each neuron i connects to (i+1..i+5) % N_total.
    Intra-population synapses use delay=0.
    Inter-population synapses use delay=dt (1 step — minimum for Phase-2 groups).
    """
    from collections import defaultdict

    n_per_pop = 25
    k         = 5
    N_total   = N_POP * n_per_pop

    # Generate all (pre_global, post_global) pairs — identical to _build_sparse_ring
    global_pairs = [
        (i, (i + off) % N_total)
        for i in range(N_total)
        for off in range(1, k + 1)
    ]
    n_syn = len(global_pairs)  # = 100 × 5 = 500

    # Group by (src_pop, dst_pop) for C++ connect() calls
    pop_edges: dict = defaultdict(list)
    for pre, post in global_pairs:
        sp = pre  // n_per_pop
        dp = post // n_per_pop
        pop_edges[(sp, dp)].append((pre % n_per_pop, post % n_per_pop))

    def _make_cpp_dur(use_phase2: bool) -> RegionalNetwork:
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

    def _make_numpy_dur() -> NumpyHHNetwork:
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
          f"(sparse ring, N_total={N_total}, k={k}, synapses={n_syn}) ---")
    for dur in durations:
        num_steps = int(dur / dt)
        I_arr = np.full((N_total, num_steps), I_val)
        I_ext = {f"P{i}": I_val for i in range(N_POP)}

        print(f"  duration={dur:>5.0f} ms ... ", end="", flush=True)

        rn_s = _make_cpp_dur(False)
        start = time.perf_counter()
        rn_s.simulate(dur, dt, I_ext)
        t_s = time.perf_counter() - start

        rn_p = _make_cpp_dur(True)
        start = time.perf_counter()
        rn_p.simulate(dur, dt, I_ext)
        t_p = time.perf_counter() - start

        net = _make_numpy_dur()
        start = time.perf_counter()
        net.simulate(dur, dt, I_arr)
        t_n = time.perf_counter() - start

        serial_times.append(t_s)
        phase2_times.append(t_p)
        numpy_times.append(t_n)

        sp_p2 = t_s / t_p if t_p > 0 else 0.0
        sp_np = t_n / t_s if t_s > 0 else 0.0
        print(f"serial={t_s:.3f}s  phase2={t_p:.3f}s  numpy={t_n:.3f}s  "
              f"phase2_su={sp_p2:.1f}x  np_su={sp_np:.1f}x")

    return {
        "durations":  durations,
        "serial":     serial_times,
        "phase2":     phase2_times,
        "numpy":      numpy_times,
        "N_total":    N_total,
        "n_syn":      n_syn,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_timing(results: Dict, figs_dir: Path):
    """2×2 log-log: wall time vs total neurons, one subplot per topology."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (topo_name, data) in enumerate(results.items()):
        ax = axes[idx]
        sc = data["sizes_cpp"]
        sn = data["sizes_numpy"]

        ax.loglog(sc, data["serial"], "o-",  color=COLOR_SERIAL,
                  linewidth=2, markersize=6, label="C++ serial")
        ax.loglog(sc, data["phase2"], "s-",  color=COLOR_PHASE2,
                  linewidth=2, markersize=6, label="C++ Phase-2")
        if data["numpy"]:
            ax.loglog(sn, data["numpy"], "D--", color=COLOR_NUMPY,
                      linewidth=2, markersize=6, label="NumPy")

        ax.set_title(topo_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Total neurons")
        ax.set_ylabel("Wall time (s)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Network Simulation: Serial vs Phase-2 Threading vs NumPy\n"
        f"({N_POP} populations, {DELAY_MS} ms inter-group delay)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = figs_dir / "benchmark_threading_time_vs_neurons.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def plot_speedup(results: Dict, figs_dir: Path):
    """
    1×3 speedup panels:
      [0] NumPy / serial
      [1] NumPy / Phase-2
      [2] serial / Phase-2  (threading gain)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    markers = ["o", "s", "^", "D"]
    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    ax_np_s  = axes[0]
    ax_np_p  = axes[1]
    ax_p2_s  = axes[2]

    for idx, (topo_name, data) in enumerate(results.items()):
        sc = data["sizes_cpp"]
        sn = data["sizes_numpy"]
        mk = f"{markers[idx]}-"
        col = colors[idx]

        # Phase-2 threading speedup (serial / phase2) — always available
        sp_p2 = [s / p if p > 0 else 0.0
                 for s, p in zip(data["serial"], data["phase2"])]
        ax_p2_s.plot(sc, sp_p2, mk, color=col,
                     linewidth=2, markersize=7, label=topo_name)

        if data["numpy"] and sn:
            n_pts = len(data["numpy"])
            # NumPy / serial
            sp_np_s = [n / s if s > 0 else 0.0
                       for n, s in zip(data["numpy"], data["serial"][:n_pts])]
            ax_np_s.plot(sn, sp_np_s, mk, color=col,
                         linewidth=2, markersize=7, label=topo_name)
            # NumPy / phase2
            sp_np_p = [n / p if p > 0 else 0.0
                       for n, p in zip(data["numpy"], data["phase2"][:n_pts])]
            ax_np_p.plot(sn, sp_np_p, mk, color=col,
                         linewidth=2, markersize=7, label=topo_name)

    panel_cfg = [
        (ax_np_s,  "NumPy vs C++ Serial",     "Speedup  (NumPy / serial)"),
        (ax_np_p,  "NumPy vs C++ Phase-2",    "Speedup  (NumPy / Phase-2)"),
        (ax_p2_s,  "Phase-2 Threading Gain",  "Speedup  (serial / Phase-2)"),
    ]
    for ax, title, ylabel in panel_cfg:
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7, label="Parity")
        ax.set_xscale("log")
        ax.set_xlabel("Total neurons", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Speedup Comparison: Threading vs Python Baseline",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = figs_dir / "benchmark_threading_speedup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_isolation(syn_data: Dict, neuron_data: Dict, figs_dir: Path):
    """2×2: synapse-count scaling (top) and neuron-count scaling (bottom)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top-left: synapse scaling, absolute time ---
    ax = axes[0, 0]
    ax.loglog(syn_data["synapse_counts"], syn_data["serial"], "o-",
              color=COLOR_SERIAL, linewidth=2, markersize=6, label="C++ serial")
    ax.loglog(syn_data["synapse_counts"], syn_data["phase2"], "s-",
              color=COLOR_PHASE2, linewidth=2, markersize=6, label="C++ Phase-2")
    ax.loglog(syn_data["synapse_counts"], syn_data["numpy"], "D--",
              color=COLOR_NUMPY,  linewidth=2, markersize=6, label="NumPy")
    ax.set_xlabel("Number of synapses")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"Synapse Scaling  (N_total={syn_data['N_total']})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # --- Top-right: synapse scaling, speedup (NumPy/serial and NumPy/Phase-2) ---
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
                label="NumPy / Phase-2")
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
              color=COLOR_PHASE2, linewidth=2, markersize=6, label="C++ Phase-2")
    if neuron_data["numpy"]:
        ax.loglog(neuron_data["numpy_N_totals"], neuron_data["numpy"], "D--",
                  color=COLOR_NUMPY, linewidth=2, markersize=6, label="NumPy")
    ax.set_xlabel("Total neurons  (N_POP × n_per_pop)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Neuron Scaling  (chain topology)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # --- Bottom-right: neuron scaling, speedup (NumPy/serial and NumPy/Phase-2) ---
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
                    label="NumPy / Phase-2")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Total neurons")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Neuron Count  (chain)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Isolating Scaling: Neurons vs Synapses  (Threading Benchmark)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = figs_dir / "benchmark_threading_isolation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_duration_scaling(dur_data: Dict, figs_dir: Path):
    """1×2: absolute time and speedup vs simulation duration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    durations = dur_data["durations"]
    N_total   = dur_data["N_total"]
    n_syn     = dur_data["n_syn"]

    ax = axes[0]
    ax.loglog(durations, dur_data["serial"], "o-", color=COLOR_SERIAL,
              linewidth=2, markersize=6, label="C++ serial")
    ax.loglog(durations, dur_data["phase2"], "s-", color=COLOR_PHASE2,
              linewidth=2, markersize=6, label="C++ Phase-2")
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
                linewidth=2, markersize=6, label="NumPy / Phase-2")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Simulation duration (ms)")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Duration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Fixed Network, Scaling Simulation Time",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = figs_dir / "benchmark_threading_duration.png"
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
    print("Network Threading Benchmark: Serial vs Phase-2 vs NumPy")
    print("=" * 65)

    figs_dir = setup_output_dir()

    # Total-N sizes — must be multiples of N_POP=4
    sizes    = [8, 20, 40, 100, 200, 400, 800, 2000, 4000]
    duration = 100.0   # ms
    dt       = 0.05    # ms  (RK4-safe for HH)
    I_val    = 10.0    # μA/cm²

    print(f"\nSimulation:   {duration} ms,  dt={dt} ms,  I_ext={I_val} μA/cm²")
    print(f"Populations:  {N_POP},  inter-group delay={DELAY_MS} ms")
    print(f"Total-N sizes: {sizes}")
    print(f"Output:       {figs_dir}\n")

    validate_implementations(duration=20.0, dt=dt, I_val=I_val)

    print(f"--- Main benchmark  ({duration} ms, dt={dt} ms) ---")
    results = run_benchmarks(sizes, duration, dt, I_val)
    plot_timing(results, figs_dir)
    plot_speedup(results, figs_dir)

    syn_data    = run_synapse_scaling(duration, dt, I_val)
    neuron_data = run_neuron_scaling(duration, dt, I_val)
    plot_isolation(syn_data, neuron_data, figs_dir)

    dur_data = run_duration_scaling(dt=dt, I_val=I_val)
    plot_duration_scaling(dur_data, figs_dir)

    print("\n" + "=" * 65)
    print("Benchmarks complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
