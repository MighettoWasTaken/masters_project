"""
benchmark_parallel_ctxbgth.py — Serial vs Phase 1 (OpenMP) vs Phase 2 (delay-decomposition)
speedup on the CTX-BG-TH model.

Three modes are compared:
  serial  — no parallelism (set_num_threads(1))
  phase1  — OpenMP pool-level parallelism (set_num_threads(N_THREADS))
  phase2  — delay-decomposition thread groups (one group per population)

Outputs to benchmarks/figures/:
  parallel_ctxbgth_time.png    — wall-clock time vs. duration, three modes
  parallel_ctxbgth_speedup.png — speedup (serial/phase) vs. duration, two curves

Usage:
    python benchmarks/benchmark_parallel_ctxbgth.py
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

from benchmarks.ctxbgth_model import build_network
from hodgkin_huxley import RecordingConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N          = 10                                      # neurons per population
DURATIONS  = [100.0, 250.0, 500.0, 1000.0, 2000.0]  # ms
DT         = 0.01                                    # ms
N_REPEATS  = 3
N_THREADS  = 4                                       # OpenMP threads for Phase 1

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

POPULATIONS = ["TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i"]

I_EXT = {
    "TH":  1.2, "GPe": 3.0, "GPi": 3.0,
    "STN": 0.0, "Str_D2": 0.0, "Str_D1": 0.0,
    "CTX_e": 0.0, "CTX_i": 0.0,
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_simulate(rn, duration: float) -> float:
    t0 = time.perf_counter()
    rn.simulate(duration, DT, I_EXT, record=RecordingConfig(["V"]))
    return time.perf_counter() - t0


def run_serial(duration: float) -> list[float]:
    rn = build_network(n=N, pd=0, seed=42)
    rn.set_num_threads(1)
    return [_time_simulate(rn, duration) for _ in range(N_REPEATS)]


def run_phase1(duration: float) -> list[float]:
    rn = build_network(n=N, pd=0, seed=42)
    rn.set_num_threads(N_THREADS)
    return [_time_simulate(rn, duration) for _ in range(N_REPEATS)]


def run_phase2(duration: float) -> list[float]:
    rn = build_network(n=N, pd=0, seed=42)
    groups = {f"g{i}": [name] for i, name in enumerate(POPULATIONS)}
    rn.set_thread_groups(groups)
    return [_time_simulate(rn, duration) for _ in range(N_REPEATS)]


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"CTX-BG-TH parallel benchmark  (n={N} neurons/pop, dt={DT}ms)")
    print(f"  Modes: serial (1 thread) | phase1 ({N_THREADS} OpenMP threads) | "
          f"phase2 (1 group/pop)")
    print(f"  Durations: {DURATIONS} ms  |  repeats: {N_REPEATS}\n")

    results: dict[str, list[float]] = {"serial": [], "phase1": [], "phase2": []}

    for dur in DURATIONS:
        print(f"  duration={dur:.0f}ms")

        t_ser = run_serial(dur)
        t_ser_min = min(t_ser)
        results["serial"].append(t_ser_min)
        print(f"    serial : {t_ser_min:.3f}s")

        t_p1 = run_phase1(dur)
        t_p1_min = min(t_p1)
        results["phase1"].append(t_p1_min)
        sp1 = t_ser_min / t_p1_min if t_p1_min > 0 else float("nan")
        print(f"    phase1 : {t_p1_min:.3f}s  speedup={sp1:.2f}x")

        t_p2 = run_phase2(dur)
        t_p2_min = min(t_p2)
        results["phase2"].append(t_p2_min)
        sp2 = t_ser_min / t_p2_min if t_p2_min > 0 else float("nan")
        print(f"    phase2 : {t_p2_min:.3f}s  speedup={sp2:.2f}x")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    dur_arr = np.array(DURATIONS)
    t_ser   = np.array(results["serial"])
    t_p1    = np.array(results["phase1"])
    t_p2    = np.array(results["phase2"])
    sp1     = t_ser / t_p1
    sp2     = t_ser / t_p2

    # --- Time plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dur_arr, t_ser, "o-",  color="#1f77b4", label="Serial (1 thread)")
    ax.plot(dur_arr, t_p1,  "s--", color="#ff7f0e",
            label=f"Phase 1 — OpenMP ({N_THREADS} threads)")
    ax.plot(dur_arr, t_p2,  "^:", color="#2ca02c",
            label=f"Phase 2 — delay decomp (1 group/pop)")
    ax.set_xlabel("Simulation duration (ms)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"CTX-BG-TH: Simulation Time vs. Duration\n"
                 f"(n={N} neurons/pop, dt={DT}ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_time = os.path.join(FIGURE_DIR, "parallel_ctxbgth_time.png")
    fig.savefig(out_time, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_time}")

    # --- Speedup plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dur_arr, sp1, "s--", color="#ff7f0e",
            label=f"Phase 1 / Serial  ({N_THREADS} threads)")
    ax.plot(dur_arr, sp2, "^:",  color="#2ca02c",
            label="Phase 2 / Serial  (delay decomp)")
    ax.axhline(1.0,       ls=":",  color="gray", lw=1)
    ax.axhline(N_THREADS, ls="--", color="lightgray", lw=1,
               label=f"Ideal {N_THREADS}× speedup")
    ax.set_xlabel("Simulation duration (ms)")
    ax.set_ylabel("Speedup (serial time / parallel time)")
    ax.set_title("CTX-BG-TH: Parallel Speedup vs. Duration")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_sp = os.path.join(FIGURE_DIR, "parallel_ctxbgth_speedup.png")
    fig.savefig(out_sp, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_sp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
