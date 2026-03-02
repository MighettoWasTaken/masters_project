"""
compare_matlab.py — compare library model output against MATLAB reference data.

Loads one .mat file from TestDataSet_64, runs our hodgkin_huxley library model
with matching parameters, and prints a side-by-side comparison of firing rates
and GPi beta-band power.

Usage
-----
    python benchmarks/compare_matlab.py [--mat PATH] [--seed N]

Default mat file: benchmarks/TestDataSet_64/1con0.mat  (healthy, no DBS)
"""

import sys
import os
import argparse
import numpy as np
from timeit import default_timer as timer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mat",  default=os.path.join(_HERE, "TestDataSet_64", "1con0.mat"),
                   help="path to .mat reference file")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for library run (MATLAB used a random seed)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load .mat reference
# ---------------------------------------------------------------------------

def load_mat(path):
    m = loadmat(path, simplify_cells=True)

    duration_s = float(m["tmax"]) / 1000.0   # tmax stored in ms

    def rates(aps):
        def spike_count(a):
            t = a["times"]
            return len(t) if hasattr(t, "__len__") else 0
        return np.array([spike_count(a) / duration_s for a in aps])

    # Cor_APs combines CTX_e (first n) and CTX_i (last n)
    n = int(m["n"])
    cor = m["Cor_APs"]

    pop_rates = {
        "TH":     rates(m["TH_APs"]),
        "STN":    rates(m["STN_APs"]),
        "GPe":    rates(m["GPe_APs"]),
        "GPi":    rates(m["GPi_APs"]),
        "Str_D2": rates(m["Striat_APs_indr"]),
        "Str_D1": rates(m["Striat_APs_dr"]),
        "CTX_e":  rates(cor[:n]),
        "CTX_i":  rates(cor[n:]),
    }

    # Determine pd from filename
    fname = os.path.basename(path)
    pd = 1 if "pd" in fname else 0

    return {
        "rates":       pop_rates,
        "beta_power":  float(m["gpi_alpha_beta_area"]),
        "gpi_S":       m["gpi_S"],
        "gpi_f":       m["gpi_f"],
        "tmax":        float(m["tmax"]),
        "n":           n,
        "pd":          pd,
        "fname":       fname,
        "duration_s":  duration_s,
    }


# ---------------------------------------------------------------------------
# Run library model
# ---------------------------------------------------------------------------

def run_library(tmax, n, pd, seed):
    from ctxbgth_model import simulate_ctxbgth

    t0 = timer()
    beta_power, gpi_S, gpi_f, spike_times, _, _ = simulate_ctxbgth(
        n=n, pd=pd, tmax=tmax, dt=0.01, seed=seed
    )
    elapsed = timer() - t0

    duration_s = tmax / 1000.0
    rates = {
        pop: np.array([len(s) / duration_s for s in spike_times[pop]])
        for pop in ("TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i")
    }
    return rates, beta_power, gpi_S, gpi_f, elapsed


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_comparison(ref, lib_rates, lib_power, lib_S, lib_f, lib_t, seed):
    POPS = ["TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i"]

    w = 72
    condition = "PD" if ref["pd"] else "healthy"
    print("\n" + "=" * w)
    print(f"  MATLAB REFERENCE vs LIBRARY  "
          f"(n={ref['n']}, tmax={ref['tmax']:.0f}ms, {condition})")
    print(f"  Reference file : {ref['fname']}")
    print(f"  Library seed   : {seed}  "
          f"(MATLAB used random seed — single-trial comparison)")
    print("=" * w)

    print(f"\n{'RUNTIME':}")
    print(f"  Library simulation             : {lib_t:.2f} s")

    print(f"\n{'FIRING RATES (mean ± std  Hz)':}")
    hdr = f"  {'Pop':8s}  {'MATLAB ref':>18s}  {'Library':>18s}  {'Δ mean':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pop in POPS:
        rm, rs = ref["rates"][pop].mean(), ref["rates"][pop].std()
        lm, ls = lib_rates[pop].mean(),    lib_rates[pop].std()
        delta = lm - rm
        flag = "  !" if abs(delta) > 5 else ""
        print(f"  {pop:8s}  {rm:6.1f} ± {rs:4.1f}      {lm:6.1f} ± {ls:4.1f}    {delta:+7.1f}{flag}")

    print(f"\n{'GPi β-BAND POWER (7–35 Hz)':}")
    print(f"  {'MATLAB ref':30s}: {ref['beta_power']:12.4f}")
    print(f"  {'Library':30s}: {lib_power:12.4f}")
    if ref["beta_power"] != 0:
        rel = (lib_power - ref["beta_power"]) / abs(ref["beta_power"]) * 100
        print(f"  Relative difference            : {rel:+.1f}%")

    # Spectrum overlap: how similar are the shapes in the beta band?
    ref_f, ref_S = ref["gpi_f"], ref["gpi_S"]
    # Interpolate library spectrum onto reference frequency grid for comparison
    lib_f_min, lib_f_max = lib_f.min(), lib_f.max()
    common_mask = (ref_f >= max(lib_f_min, ref_f.min())) & \
                  (ref_f <= min(lib_f_max, ref_f.max()))
    if common_mask.sum() > 1:
        from scipy.interpolate import interp1d
        lib_interp = interp1d(lib_f, lib_S, bounds_error=False, fill_value=0.0)
        lib_S_common = lib_interp(ref_f[common_mask])
        ref_S_common = ref_S[common_mask]
        # Pearson correlation of spectral shapes
        if ref_S_common.std() > 0 and lib_S_common.std() > 0:
            corr = np.corrcoef(ref_S_common, lib_S_common)[0, 1]
            print(f"  Spectral shape correlation     : {corr:+.3f}  (1.0 = identical shape)")

    print("\n" + "=" * w + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse()

    print(f"\nLoading MATLAB reference: {args.mat}")
    ref = load_mat(args.mat)

    print(f"Running library model  (tmax={ref['tmax']:.0f}ms, n={ref['n']}, "
          f"pd={ref['pd']}, seed={args.seed}) ...")
    lib_rates, lib_power, lib_S, lib_f, lib_t = run_library(
        ref["tmax"], ref["n"], ref["pd"], args.seed
    )

    print_comparison(ref, lib_rates, lib_power, lib_S, lib_f, lib_t, args.seed)
