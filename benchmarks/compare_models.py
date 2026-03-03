"""
compare_models.py — side-by-side comparison of the benchmark and library models.

Runs both simulations with identical parameters and prints a table of:
  - Per-population firing rates (mean ± std Hz)
  - GPi β-band power (7–35 Hz, primary DBS efficacy metric)
  - Simulation runtime
  - Source lines of code

Usage
-----
    python benchmarks/compare_models.py [--n N] [--tmax MS] [--pd 0|1] [--seed S]

Defaults: n=10, tmax=1000ms, pd=0 (healthy), seed=42
"""

import sys
import os
import argparse
import numpy as np
from timeit import default_timer as timer

# Make sure we can import from this directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark vs library model comparison")
    p.add_argument("--n",    type=int,   default=10,     help="neurons per population")
    p.add_argument("--tmax", type=float, default=1000.0, help="duration in ms")
    p.add_argument("--pd",   type=int,   default=0,      choices=[0, 1], help="0=healthy, 1=PD")
    p.add_argument("--seed", type=int,   default=42)
    p.add_argument("--dt",   type=float, default=0.01)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rate_from_benchmark_aps(aps, duration_s):
    """aps: list of {'times': [...]} dicts (spike times in seconds)."""
    return np.array([len(a['times']) / duration_s for a in aps])


def _rate_from_lib_spikes(spikes_ms, duration_s):
    """spikes_ms: list of ndarrays (spike times in ms)."""
    return np.array([len(s) / duration_s for s in spikes_ms])


def _loc(filepath):
    """Count non-blank, non-comment source lines."""
    count = 0
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                count += 1
    return count


def _fmt(mean, std):
    return f"{mean:6.1f} ± {std:4.1f}"


# ---------------------------------------------------------------------------
# Run benchmark (calls network core directly, skipping .mat file save)
# ---------------------------------------------------------------------------

def run_benchmark(n, tmax, dt, pd, seed):
    from simulate_network_model import (
        CTX_BG_TH_network, make_Spectrum
    )

    np.random.seed(seed)
    t_arr = np.arange(0, tmax + dt, dt)
    Idbs    = np.zeros(len(t_arr))
    Iappco  = np.zeros(len(t_arr))

    t0 = timer()
    TH_APs, STN_APs, GPe_APs, GPi_APs, \
        Str_D2_APs, Str_D1_APs, Cor_APs = CTX_BG_TH_network(
            pd, 0, tmax, dt, n, Idbs, Iappco
        )
    elapsed = timer() - t0

    params = {
        "Fs": 1.0 / (dt * 1e-3),
        "fpass": [1, 100],
        "tapers": [3, 5],
        "trialave": 1,
    }
    beta_power, gpi_S, gpi_f = make_Spectrum(GPi_APs, params)

    duration_s = tmax / 1000.0
    print(Cor_APs)
    rates = {
        "TH":     _rate_from_benchmark_aps(TH_APs,    duration_s),
        "STN":    _rate_from_benchmark_aps(STN_APs,   duration_s),
        "GPe":    _rate_from_benchmark_aps(GPe_APs,   duration_s),
        "GPi":    _rate_from_benchmark_aps(GPi_APs,   duration_s),
        "Str_D2": _rate_from_benchmark_aps(Str_D2_APs, duration_s),
        "Str_D1": _rate_from_benchmark_aps(Str_D1_APs, duration_s),
        # Cor_APs has 2n neurons: first n = CTX_e, last n = CTX_i
        "CTX_e":  _rate_from_benchmark_aps(Cor_APs[:n],  duration_s),
        "CTX_i":  _rate_from_benchmark_aps(Cor_APs[n:],  duration_s),
    }
    return rates, beta_power, gpi_S, gpi_f, elapsed


# ---------------------------------------------------------------------------
# Run library model
# ---------------------------------------------------------------------------

def run_library(n, tmax, dt, pd, seed):
    from ctxbgth_model import simulate_ctxbgth

    beta_power, gpi_S, gpi_f, spike_times, _, elapsed, _ = simulate_ctxbgth(
        n=n, pd=pd, tmax=tmax, dt=dt, seed=seed
    )

    duration_s = tmax / 1000.0
    rates = {
        pop: _rate_from_lib_spikes(spike_times[pop], duration_s)
        for pop in ("TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i")
    }
    return rates, beta_power, gpi_S, gpi_f, elapsed


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison(bm_rates, bm_power, bm_t,
                     lib_rates, lib_power, lib_t,
                     n, tmax, pd):
    POPS = ["TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i"]

    bm_file  = os.path.join(_HERE, "simulate_network_model.py")
    lib_file = os.path.join(_HERE, "ctxbgth_model.py")
    bm_loc  = _loc(bm_file)
    lib_loc = _loc(lib_file)

    w = 72
    print("\n" + "=" * w)
    print(f"  CTX-BG-TH MODEL COMPARISON  (n={n}, tmax={tmax:.0f}ms, {'PD' if pd else 'healthy'})")
    print("=" * w)

    # Runtime
    print(f"\n{'RUNTIME':}")
    print(f"  {'Benchmark':30s}: {bm_t:7.2f} s")
    print(f"  {'Library':30s}: {lib_t:7.2f} s")
    speedup = bm_t / lib_t if lib_t > 0 else float("inf")
    print(f"  Speedup (bench / lib)          : {speedup:.1f}×")

    # LOC
    print(f"\n{'LINES OF CODE (non-blank, non-comment)':}")
    print(f"  {'Benchmark':30s}: {bm_loc:5d}")
    print(f"  {'Library model':30s}: {lib_loc:5d}")
    print(f"  Reduction                      : {(1 - lib_loc/bm_loc)*100:.0f}%")

    # Firing rates
    print(f"\n{'FIRING RATES (mean ± std  Hz)':}")
    hdr = f"  {'Pop':8s}  {'Benchmark':>18s}  {'Library':>18s}  {'Δ mean':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pop in POPS:
        bm_m, bm_s = bm_rates[pop].mean(),  bm_rates[pop].std()
        lb_m, lb_s = lib_rates[pop].mean(), lib_rates[pop].std()
        delta = lb_m - bm_m
        flag = "  <-- !" if abs(delta) > 5 else ""
        print(f"  {pop:8s}  {_fmt(bm_m, bm_s):>18s}  {_fmt(lb_m, lb_s):>18s}  {delta:+7.1f}{flag}")

    # GPi beta power
    print(f"\n{'GPi β-BAND POWER (7–35 Hz, multitaper)':}")
    print(f"  {'Benchmark':30s}: {bm_power:12.4f}")
    print(f"  {'Library':30s}: {lib_power:12.4f}")
    if bm_power != 0:
        print(f"  Relative difference            : {(lib_power - bm_power) / abs(bm_power) * 100:+.1f}%")

    print("\n" + "=" * w + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    print(f"\nRunning BENCHMARK  (n={args.n}, tmax={args.tmax}ms, pd={args.pd}) ...")
    bm_rates, bm_power, bm_S, bm_f, bm_t = run_benchmark(
        args.n, args.tmax, args.dt, args.pd, args.seed
    )

    print(f"\nRunning LIBRARY MODEL ...")
    lib_rates, lib_power, lib_S, lib_f, lib_t = run_library(
        args.n, args.tmax, args.dt, args.pd, args.seed
    )

    print_comparison(
        bm_rates, bm_power, bm_t,
        lib_rates, lib_power, lib_t,
        args.n, args.tmax, args.pd
    )
