"""
benchmark_cpu_vs_gpu_scaling.py — CTX-BG-TH runtime vs network size.

Sweeps the per-population neuron count `n` and times one CTX-BG-TH simulation on:
    serial    — single-threaded CPU       (threaded=False, run_cuda=False)
    threaded  — one CPU thread per pop     (threaded=True,  run_cuda=False)
    cuda      — GPU cooperative kernel      (run_cuda=True)

Produces benchmarks/figures/cpu_vs_gpu_scaling.png:
    (left)  wall-clock sim time vs n   (log-log), with the GPU/serial crossover marked
    (right) speedup vs serial CPU      (>1 = faster than serial)

Only the simulate() call is timed (network construction and host->device transfer
are excluded — they are one-time costs). The GPU is warmed up once before timing
to exclude CUDA context / first-launch initialization.

Usage:
    python benchmarks/benchmark_cpu_vs_gpu_scaling.py
    python benchmarks/benchmark_cpu_vs_gpu_scaling.py --n 10 25 50 100 200 400 800 \
        --tmax 500 --reps 1 --max-cpu-seconds 180
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

FIG_DIR = os.path.join(_HERE, "figures")

# Distinct styling per mode.
MODES = [
    ("serial",   dict(threaded=False, run_cuda=False), "CPU (serial)",   "#1f77b4", "o"),
    ("threaded", dict(threaded=True,  run_cuda=False), "CPU (threaded)", "#2ca02c", "s"),
    ("cuda",     dict(threaded=False, run_cuda=True),  "GPU (CUDA)",     "#d62728", "^"),
]


def _time_sim(n, tmax, dt, seed, kwargs, reps):
    """Run simulate_ctxbgth `reps` times; return (mean, std) sim wall-time (s).

    Averaged (not min) because GPU timing is noisy run-to-run."""
    import ctxbgth_model as M
    times = []
    for _ in range(reps):
        # simulate_ctxbgth returns (..., spike_times, elapsed); elapsed = sim only.
        out = M.simulate_ctxbgth(n=n, pd=0, tmax=tmax, dt=dt, seed=seed, **kwargs)
        times.append(float(out[-1]))
    return float(np.mean(times)), float(np.std(times))


def run_sweep(n_values, tmax, dt, seed, reps, max_cpu_seconds, do_cuda):
    import ctxbgth_model as M

    have_cuda = False
    if do_cuda:
        try:
            import hodgkin_huxley as hh
            have_cuda = bool(hh.cuda_is_available())
        except Exception as e:
            print(f"  [cuda] unavailable: {e}")
        if have_cuda:
            # Warm up: first GPU run pays CUDA-context / first-launch init.
            print("  warming up GPU ...")
            _time_sim(min(n_values), tmax, dt, seed,
                      dict(threaded=False, run_cuda=True), reps=1)

    results = {m[0]: {} for m in MODES}
    # CPU modes are dropped for larger n once they blow past the time budget.
    cpu_capped = {"serial": False, "threaded": False}

    for n in n_values:
        print(f"\nn = {n}  ({n * 8} neurons total)")
        for key, kwargs, label, _c, _mk in MODES:
            if key == "cuda" and not have_cuda:
                continue
            if key in cpu_capped and cpu_capped[key]:
                print(f"  {label:14s}: skipped (exceeded {max_cpu_seconds}s budget)")
                continue
            try:
                t, sd = _time_sim(n, tmax, dt, seed, kwargs, reps)
                results[key][n] = t
                print(f"  {label:14s}: {t:8.3f} s  (+/- {sd:.3f}, n={reps})")
                if key in cpu_capped and t > max_cpu_seconds:
                    cpu_capped[key] = True
            except Exception as e:
                print(f"  {label:14s}: FAILED ({type(e).__name__}: {e})")
    return results, have_cuda


def crossover_n(serial, cuda):
    """First n where cuda <= serial (linear interp in log-log). None if no crossover."""
    common = sorted(set(serial) & set(cuda))
    if len(common) < 2:
        return None
    prev = None
    for n in common:
        faster = cuda[n] <= serial[n]
        if faster and prev is not None and not prev[1]:
            # crossover between prev_n and n — interpolate in log space
            n0, _ = prev
            r0 = np.log(serial[n0] / cuda[n0])  # <0 (gpu slower)
            r1 = np.log(serial[n] / cuda[n])    # >=0 (gpu faster)
            frac = -r0 / (r1 - r0) if (r1 - r0) != 0 else 0.0
            return float(np.exp(np.log(n0) + frac * (np.log(n) - np.log(n0))))
        if faster and prev is None:
            return float(n)  # GPU already faster at the smallest size
        prev = (n, faster)
    return None


def plot(results, have_cuda, tmax, out_path, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for key, _kw, label, color, mk in MODES:
        data = results[key]
        if not data:
            continue
        ns = sorted(data)
        ts = [data[n] for n in ns]
        ax1.plot(ns, ts, marker=mk, color=color, label=label, lw=2, ms=6)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("neurons per population (n)")
    ax1.set_ylabel("simulation wall-time (s)")
    ax1.set_title(f"CTX-BG-TH runtime vs network size (tmax={tmax:.0f} ms)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # Speedup vs serial CPU.
    serial = results.get("serial", {})
    if serial:
        for key, _kw, label, color, mk in MODES:
            if key == "serial":
                continue
            data = results[key]
            ns = sorted(set(data) & set(serial))
            if not ns:
                continue
            sp = [serial[n] / data[n] for n in ns]
            ax2.plot(ns, sp, marker=mk, color=color, label=f"{label} vs serial", lw=2, ms=6)
        ax2.axhline(1.0, color="gray", ls="--", lw=1, label="parity")

        xo = crossover_n(serial, results.get("cuda", {})) if have_cuda else None
        if xo is not None:
            for ax in (ax1, ax2):
                ax.axvline(xo, color="#d62728", ls=":", lw=1.5)
            ax2.annotate(f"GPU > CPU\nat n ≈ {xo:.0f}", xy=(xo, 1.0),
                         xytext=(xo, 1.5), color="#d62728",
                         ha="center", fontsize=9)

    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("neurons per population (n)")
    ax2.set_ylabel("speedup vs serial CPU  (>1 = faster)")
    ax2.set_title("Speedup relative to serial CPU")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"\nSaved figure -> {out_path}")


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, nargs="+",
                   default=[10, 25, 50, 100, 200, 400, 600, 1000],
                   help="per-population neuron counts to sweep")
    p.add_argument("--tmax", type=float, default=500.0, help="sim duration (ms)")
    p.add_argument("--dt", type=float, default=0.01, help="time step (ms)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reps", type=int, default=3,
                   help="reps per point (mean +/- std reported; GPU timing is noisy)")
    p.add_argument("--max-cpu-seconds", type=float, default=500.0,
                   help="drop a CPU mode for larger n once a run exceeds this")
    p.add_argument("--no-cuda", action="store_true", help="skip the GPU runs")
    p.add_argument("--out", default=os.path.join(FIG_DIR, "cpu_vs_gpu_scaling.png"))
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    n_values = sorted(set(args.n))
    print("=" * 64)
    print("  CTX-BG-TH scaling: serial CPU vs threaded CPU vs GPU")
    print(f"  n = {n_values}")
    print(f"  tmax = {args.tmax:.0f} ms, dt = {args.dt}, reps = {args.reps}")
    print("=" * 64)

    results, have_cuda = run_sweep(
        n_values, args.tmax, args.dt, args.seed, args.reps,
        args.max_cpu_seconds, do_cuda=not args.no_cuda)

    # Summary table.
    print("\n" + "=" * 64)
    hdr = f"  {'n':>6} | " + " | ".join(f"{m[2]:>14}" for m in MODES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for n in n_values:
        cells = []
        for key, _kw, _label, _c, _mk in MODES:
            t = results[key].get(n)
            cells.append(f"{t:14.3f}" if t is not None else f"{'-':>14}")
        print(f"  {n:>6} | " + " | ".join(cells))

    plot(results, have_cuda, args.tmax, args.out, args.dpi)
