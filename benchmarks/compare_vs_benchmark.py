"""
compare_vs_benchmark.py — Library model vs pure-Python benchmark, head-to-head.

Runs both models pairwise (library + benchmark simultaneously in threads),
collecting firing rates, GPi beta power, runtimes, and peak memory. Produces
box-and-whisker plots comparing the two implementations.

Results are cached per trial so runs can be resumed after interruption.

Usage
-----
    python benchmarks/compare_vs_benchmark.py [options]

Options
-------
    --n-control N       Healthy trials              (default 25)
    --n-pd N            PD trials                   (default 39)
    --tmax MS           Simulation duration ms       (default 2000)
    --corstim 0|1       Cortical stimulation         (default 0)
    --dbs-freq HZ       DBS frequency Hz; 0=off      (default 0)
    --dbs-pw MS         DBS pulse width ms           (default 0.3)
    --dbs-amplitude A   DBS amplitude µA/cm²         (default 300)
    --no-cache          Ignore cache, re-run all
    --dpi N             Figure DPI                   (default 150)

Outputs (benchmarks/figures/)
------------------------------
    firing_rates_vs_benchmark.png
    beta_power_vs_benchmark.png
    runtime_speedup.png
    memory_usage.png
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
import threading
from timeit import default_timer as timer

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import psutil

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

POPS       = ["TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i"]
CACHE_FILE = os.path.join(_HERE, "bench_compare_cache.pkl")
FIG_DIR    = os.path.join(_HERE, "figures")

# Okabe-Ito colorblind-friendly palette
C_BMK_CON = "#E69F00"   # amber      — Benchmark healthy
C_LIB_CON = "#56B4E9"   # sky blue   — Library healthy
C_BMK_PD  = "#D55E00"   # vermilion  — Benchmark PD
C_LIB_PD  = "#0072B2"   # blue       — Library PD

_COLORS = [C_BMK_CON, C_LIB_CON, C_BMK_PD, C_LIB_PD]
_LABELS = ["Benchmark (healthy)", "Library (healthy)",
           "Benchmark (PD)",      "Library (PD)"]


# =============================================================================
# Worker functions — run inside threads (shared process, no sys.path tricks)
# =============================================================================

def _run_lib_worker(pd: int, corstim: int, dbs_freq: float,
                    PW: float, amplitude: float,
                    n: int, tmax: float, dt: float) -> dict:
    """Library model worker."""
    from ctxbgth_model import simulate_ctxbgth

    seed = int(np.random.default_rng().integers(0, 2 ** 31))
    beta_power, _, _, spike_times, elapsed = simulate_ctxbgth(
        n=n, pd=pd, tmax=tmax, dt=dt,
        dbs_freq=dbs_freq, PW=PW, amplitude=amplitude,
        corstim=corstim, seed=seed,
    )
    duration_s = tmax / 1000.0
    rates = {
        pop: np.array([len(s) / duration_s for s in spike_times[pop]])
        for pop in POPS
    }
    return {"rates": rates, "beta_power": float(beta_power),
            "elapsed": float(elapsed), "pd": pd}


def _run_bench_worker(pd: int, corstim: int, dbs_freq: float,
                      PW: float, amplitude: float,
                      n: int, tmax: float, dt: float) -> dict:
    """Pure-Python benchmark worker."""
    from simulate_network_model import CTX_BG_TH_network, make_Spectrum, creatdbs

    np.random.seed()   # truly random (OS entropy), matching original benchmark

    t_arr  = np.arange(0, tmax + dt, dt)
    Idbs   = creatdbs(dbs_freq, tmax, dt, PW, amplitude) if dbs_freq > 0 \
             else np.zeros(len(t_arr))
    Iappco = np.zeros(len(t_arr))
    if corstim:
        Iappco[int(1000 / dt):int((1000 + 0.3) / dt)] = 350.0

    t0 = timer()
    with contextlib.redirect_stdout(io.StringIO()):
        TH_APs, STN_APs, GPe_APs, GPi_APs, Str_indr, Str_dr, Cor_APs = \
            CTX_BG_TH_network(pd, corstim, tmax, dt, n, Idbs, Iappco)
        dt1  = dt * 1e-3
        par  = {"Fs": 1 / dt1, "fpass": [1, 100], "tapers": [3, 5], "trialave": 1}
        beta_power, _, _ = make_Spectrum(GPi_APs, par)
    elapsed = timer() - t0

    duration_s = tmax / 1000.0

    def _ap_rates(aps):
        return np.array([len(a["times"]) / duration_s for a in aps])

    rates = {
        "TH":     _ap_rates(TH_APs),   "STN":    _ap_rates(STN_APs),
        "GPe":    _ap_rates(GPe_APs),   "GPi":    _ap_rates(GPi_APs),
        "Str_D2": _ap_rates(Str_indr),  "Str_D1": _ap_rates(Str_dr),
        "CTX_e":  _ap_rates(Cor_APs[:n]),
        "CTX_i":  _ap_rates(Cor_APs[n:]),
    }
    return {"rates": rates, "beta_power": float(beta_power),
            "elapsed": float(elapsed), "pd": pd}


# =============================================================================
# Memory poller (background thread, psutil RSS)
# =============================================================================

class _MemoryPoller:
    """Poll process RSS every `interval` seconds; report peak delta from baseline."""

    def __init__(self, interval: float = 0.1):
        self._proc     = psutil.Process()
        self._interval = interval
        self._stop     = threading.Event()
        self._baseline = self._proc.memory_info().rss
        self._peak     = self._baseline
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._baseline = self._proc.memory_info().rss
        self._peak     = self._baseline
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            self._stop.wait(self._interval)

    def stop_and_get_mb(self) -> float:
        self._stop.set()
        self._thread.join()
        return max(0, self._peak - self._baseline) / 1e6


# =============================================================================
# Cache helpers
# =============================================================================

def _cache_key(args) -> tuple:
    return (args.n_control, args.n_pd, args.tmax,
            args.corstim, args.dbs_freq, args.dbs_pw, args.dbs_amplitude)


def _load_cache(args) -> dict:
    if args.no_cache or not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "rb") as fh:
            cache = pickle.load(fh)
        if cache.get("key") == _cache_key(args):
            n = len(cache.get("trials", {}))
            print(f"Loaded cache: {n} completed trial pair(s).")
            return cache
        else:
            print("Cache key mismatch (parameters changed) — starting fresh.")
    except Exception as exc:
        print(f"Cache load error ({exc}) — starting fresh.")
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(cache, fh)


# =============================================================================
# Batch runner
# =============================================================================

def _warmup_imports() -> None:
    """
    Pre-import both model modules so their import cost is never charged to a
    measured pair.  simulate_network_model.py is ~1000 lines with heavy scipy
    imports; the first import can add 5-15 s of apparent 'simulation' time that
    inflates the first-pair speedup figure.
    """
    import importlib
    print("  Pre-importing models (one-time) ...", end="  ", flush=True)
    t0 = timer()
    importlib.import_module("ctxbgth_model")
    importlib.import_module("simulate_network_model")
    print(f"{timer() - t0:.1f}s", flush=True)


def collect_all(args):
    _warmup_imports()
    cache = _load_cache(args)
    if "key" not in cache:
        cache = {"key": _cache_key(args), "trials": {}}
    trials: dict = cache["trials"]

    work = []
    for i in range(args.n_control):
        if (0, i) not in trials:
            work.append((0, i))
    for i in range(args.n_pd):
        if (1, i) not in trials:
            work.append((1, i))

    grand_total = args.n_control + args.n_pd
    already     = len(trials)

    if not work:
        print(f"All {grand_total} trial pairs already cached.")
    else:
        print(f"\nRunning {len(work)} new pair(s)  "
              f"({already} cached / {grand_total} total)")
        print(f"  tmax={args.tmax}ms  n=10  dt=0.01ms  "
              f"corstim={args.corstim}  dbs={args.dbs_freq}Hz\n")

    worker_kwargs = dict(
        corstim=args.corstim, dbs_freq=args.dbs_freq,
        PW=args.dbs_pw, amplitude=args.dbs_amplitude,
        n=10, tmax=args.tmax, dt=0.01,
    )

    t_start = timer()

    for pair_num, (pd, trial_idx) in enumerate(work):
        done_count = already + pair_num
        cond       = "healthy" if pd == 0 else "PD"
        print(f"  [{done_count + 1}/{grand_total}] {cond} trial {trial_idx + 1}",
              flush=True)

        # Library
        lib_poller = _MemoryPoller()
        lib_poller.start()
        lib_res = _run_lib_worker(pd, **worker_kwargs)
        lib_peak_mb = lib_poller.stop_and_get_mb()
        print(f"    lib   {lib_res['elapsed']:.2f}s  mem={lib_peak_mb:.0f}MB",
              flush=True)

        # Benchmark
        bench_poller = _MemoryPoller()
        bench_poller.start()
        bench_res = _run_bench_worker(pd, **worker_kwargs)
        bench_peak_mb = bench_poller.stop_and_get_mb()
        speedup = bench_res["elapsed"] / max(lib_res["elapsed"], 1e-9)
        print(f"    bench {bench_res['elapsed']:.2f}s  mem={bench_peak_mb:.0f}MB"
              f"  speedup={speedup:.1f}×", flush=True)

        trials[(pd, trial_idx)] = {
            "lib":           lib_res,
            "bench":         bench_res,
            "lib_peak_mb":   lib_peak_mb,
            "bench_peak_mb": bench_peak_mb,
        }
        _save_cache(cache)

        pairs_done = pair_num + 1
        elapsed    = timer() - t_start
        remaining  = (elapsed / pairs_done) * (len(work) - pairs_done)
        if pairs_done < len(work):
            print(f"    ETA {_fmt_eta(remaining)}", flush=True)

    if work:
        print(f"\nDone in {_fmt_eta(timer() - t_start)}.\n")

    lib_con  = [trials[(0, i)]["lib"]   for i in range(args.n_control)]
    bmk_con  = [trials[(0, i)]["bench"] for i in range(args.n_control)]
    lib_pd   = [trials[(1, i)]["lib"]   for i in range(args.n_pd)]
    bmk_pd   = [trials[(1, i)]["bench"] for i in range(args.n_pd)]

    def _mem(t, key): return t.get(key) or 0.0
    lib_mem_con  = [_mem(trials[(0, i)], "lib_peak_mb")   for i in range(args.n_control)]
    bmk_mem_con  = [_mem(trials[(0, i)], "bench_peak_mb") for i in range(args.n_control)]
    lib_mem_pd   = [_mem(trials[(1, i)], "lib_peak_mb")   for i in range(args.n_pd)]
    bmk_mem_pd   = [_mem(trials[(1, i)], "bench_peak_mb") for i in range(args.n_pd)]

    return lib_con, bmk_con, lib_pd, bmk_pd, lib_mem_con, bmk_mem_con, lib_mem_pd, bmk_mem_pd


def _fmt_eta(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


# =============================================================================
# Shared plot helpers
# =============================================================================

def _draw_box(ax: plt.Axes, pos: float, data: np.ndarray,
              color: str, width: float = 0.32) -> None:
    if len(data) == 0:
        return
    ax.boxplot(
        data, positions=[pos], widths=width, patch_artist=True,
        notch=False, manage_ticks=False,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(facecolor=color, alpha=0.72, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.35,
                        markerfacecolor=color, markeredgecolor="none"),
    )
    rng    = np.random.default_rng(0)
    jitter = rng.uniform(-0.09, 0.09, size=len(data))
    ax.scatter(np.full(len(data), pos) + jitter, data,
               color=color, s=14, alpha=0.55, zorder=5, linewidths=0)


def _style_ax(ax: plt.Axes, title: str, ylabel: str,
              group_centers: list, group_labels: list,
              sep_x: float) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.axvline(sep_x, color="#999999", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles():
    return [mpatches.Patch(facecolor=c, alpha=0.72, label=l)
            for c, l in zip(_COLORS, _LABELS)]


def _annotate_median(ax, pos, data, fmt="{:.1f}"):
    if len(data):
        med = np.median(data)
        ax.text(pos, med, " " + fmt.format(med),
                va="center", ha="left", fontsize=7.5, alpha=0.8)


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# =============================================================================
# Figure 1 — Firing rates (2×4 grid)
# =============================================================================

def plot_firing_rates(bmk_con, lib_con, bmk_pd, lib_pd,
                      tmax: float, out_path: str, dpi: int = 150) -> None:
    p_bc, p_lc = 1.0, 1.5
    p_bp, p_lp = 2.8, 3.3
    sep_x = (p_lc + p_bp) / 2
    gc    = [(p_bc + p_lc) / 2, (p_bp + p_lp) / 2]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()

    for idx, pop in enumerate(POPS):
        ax = axes[idx]
        bc = np.array([d["rates"][pop].mean() for d in bmk_con])
        lc = np.array([d["rates"][pop].mean() for d in lib_con])
        bp = np.array([d["rates"][pop].mean() for d in bmk_pd])
        lp = np.array([d["rates"][pop].mean() for d in lib_pd])

        _draw_box(ax, p_bc, bc, C_BMK_CON)
        _draw_box(ax, p_lc, lc, C_LIB_CON)
        _draw_box(ax, p_bp, bp, C_BMK_PD)
        _draw_box(ax, p_lp, lp, C_LIB_PD)

        all_vals = np.concatenate([bc, lc, bp, lp])
        ymax = max(all_vals.max() * 1.12, 5.0) if len(all_vals) else 5.0
        ax.set_ylim(bottom=0, top=ymax)
        ax.set_xlim(0.55, 3.75)
        _style_ax(ax, pop, "Firing rate (Hz)", gc, ["Healthy", "PD"], sep_x)

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(
        "Population Firing Rates — Benchmark vs Library Model\n"
        f"(n=10 neurons/pop, tmax={tmax:.0f}ms, "
        f"{len(bmk_con)} healthy / {len(bmk_pd)} PD trials)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    _save(fig, out_path, dpi)


# =============================================================================
# Figure 2 — GPi β-band power
# =============================================================================

def plot_beta_power(bmk_con, lib_con, bmk_pd, lib_pd,
                    tmax: float, out_path: str, dpi: int = 150) -> None:
    p_bc, p_lc = 1.0, 1.55
    p_bp, p_lp = 2.95, 3.5
    sep_x = (p_lc + p_bp) / 2
    gc    = [(p_bc + p_lc) / 2, (p_bp + p_lp) / 2]

    bc = np.array([d["beta_power"] for d in bmk_con])
    lc = np.array([d["beta_power"] for d in lib_con])
    bp = np.array([d["beta_power"] for d in bmk_pd])
    lp = np.array([d["beta_power"] for d in lib_pd])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw_box(ax, p_bc, bc, C_BMK_CON, width=0.40)
    _draw_box(ax, p_lc, lc, C_LIB_CON, width=0.40)
    _draw_box(ax, p_bp, bp, C_BMK_PD,  width=0.40)
    _draw_box(ax, p_lp, lp, C_LIB_PD,  width=0.40)

    ax.set_xlim(0.45, 4.05)
    _style_ax(ax, "GPi β-band Oscillatory Power (7–35 Hz)",
              "Integrated power (a.u.)", gc, ["Healthy", "PD"], sep_x)
    ax.legend(handles=_legend_handles(), fontsize=9.5, loc="upper left",
              frameon=True, framealpha=0.9)

    for pos, data in [(p_bc, bc), (p_lc, lc), (p_bp, bp), (p_lp, lp)]:
        _annotate_median(ax, pos, data, fmt="{:.0f}")

    fig.suptitle(
        "GPi β-band Power — Benchmark vs Library Model\n"
        f"(tmax={tmax:.0f}ms, {len(bmk_con)} healthy / {len(bmk_pd)} PD trials)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_path, dpi)


# =============================================================================
# Figure 3 — Runtime & Speedup
# =============================================================================

def plot_runtime_speedup(bmk_con, lib_con, bmk_pd, lib_pd,
                         tmax: float, out_path: str, dpi: int = 150) -> None:
    bc_t = np.array([d["elapsed"] for d in bmk_con])
    lc_t = np.array([d["elapsed"] for d in lib_con])
    bp_t = np.array([d["elapsed"] for d in bmk_pd])
    lp_t = np.array([d["elapsed"] for d in lib_pd])

    sp_con = bc_t / np.maximum(lc_t, 1e-9)
    sp_pd  = bp_t / np.maximum(lp_t, 1e-9)

    fig, (ax_rt, ax_sp) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: absolute runtime
    p_bc, p_lc = 1.0, 1.5
    p_bp, p_lp = 2.8, 3.3
    sep_x = (p_lc + p_bp) / 2
    gc    = [(p_bc + p_lc) / 2, (p_bp + p_lp) / 2]

    _draw_box(ax_rt, p_bc, bc_t, C_BMK_CON)
    _draw_box(ax_rt, p_lc, lc_t, C_LIB_CON)
    _draw_box(ax_rt, p_bp, bp_t, C_BMK_PD)
    _draw_box(ax_rt, p_lp, lp_t, C_LIB_PD)
    ax_rt.set_xlim(0.55, 3.75)
    _style_ax(ax_rt, "Simulation Runtime", "Wall-clock time (s)",
              gc, ["Healthy", "PD"], sep_x)
    for pos, data in [(p_bc, bc_t), (p_lc, lc_t), (p_bp, bp_t), (p_lp, lp_t)]:
        _annotate_median(ax_rt, pos, data, fmt="{:.1f}s")

    # Right: speedup ratio
    p_sc, p_sp = 1.25, 2.75
    _draw_box(ax_sp, p_sc, sp_con, C_LIB_CON, width=0.50)
    _draw_box(ax_sp, p_sp, sp_pd,  C_LIB_PD,  width=0.50)
    ax_sp.axhline(1.0, color="#888888", linewidth=1.2,
                  linestyle="--", alpha=0.7, label="1× (no speedup)")
    ax_sp.set_xlim(0.55, 3.55)
    ax_sp.set_xticks([p_sc, p_sp])
    ax_sp.set_xticklabels(["Healthy", "PD"], fontsize=10)
    ax_sp.set_title("Speedup  (benchmark time / library time)",
                    fontsize=11, fontweight="bold", pad=4)
    ax_sp.set_ylabel("× faster than benchmark", fontsize=9)
    ax_sp.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax_sp.spines["top"].set_visible(False)
    ax_sp.spines["right"].set_visible(False)
    for pos, data in [(p_sc, sp_con), (p_sp, sp_pd)]:
        _annotate_median(ax_sp, pos, data, fmt="{:.1f}×")

    fig.suptitle(
        "Runtime & Speedup — Benchmark vs Library Model\n"
        f"(tmax={tmax:.0f}ms, n=10, "
        f"{len(bmk_con)} healthy / {len(bmk_pd)} PD trials)",
        fontsize=12, fontweight="bold",
    )
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    _save(fig, out_path, dpi)


# =============================================================================
# Figure 4 — Memory usage (combined pair peak RSS delta)
# =============================================================================

def plot_memory_usage(lib_mem_con: list, bmk_mem_con: list,
                      lib_mem_pd: list,  bmk_mem_pd: list,
                      tmax: float, out_path: str, dpi: int = 150) -> None:
    """
    Per-model peak RSS increase, measured sequentially so each model runs alone.
    """
    p_bc, p_lc = 1.0, 1.5
    p_bp, p_lp = 2.8, 3.3
    sep_x = (p_lc + p_bp) / 2
    gc    = [(p_bc + p_lc) / 2, (p_bp + p_lp) / 2]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _draw_box(ax, p_bc, np.array(bmk_mem_con), C_BMK_CON)
    _draw_box(ax, p_lc, np.array(lib_mem_con), C_LIB_CON)
    _draw_box(ax, p_bp, np.array(bmk_mem_pd),  C_BMK_PD)
    _draw_box(ax, p_lp, np.array(lib_mem_pd),  C_LIB_PD)

    ax.axvline(sep_x, color="#999999", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_xlim(0.55, 3.75)
    ax.set_xticks(gc)
    ax.set_xticklabels(["Healthy", "PD"], fontsize=10)
    ax.set_title("Peak Memory Usage — Benchmark vs Library",
                 fontsize=11, fontweight="bold", pad=4)
    ax.set_ylabel("Peak RSS increase (MB)", fontsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for pos, data in [(p_bc, bmk_mem_con), (p_lc, lib_mem_con),
                      (p_bp, bmk_mem_pd),  (p_lp, lib_mem_pd)]:
        _annotate_median(ax, pos, np.array(data), fmt="{:.0f} MB")

    fig.suptitle(
        f"Memory Usage — Benchmark vs Library  (tmax={tmax:.0f}ms, n=10)\n"
        f"({len(lib_mem_con)} healthy / {len(lib_mem_pd)} PD trials)",
        fontsize=12, fontweight="bold",
    )
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    _save(fig, out_path, dpi)


# =============================================================================
# Entry point
# =============================================================================

def _parse():
    p = argparse.ArgumentParser(
        description="Library vs benchmark head-to-head comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-control",     type=int,   default=25,
                   help="Number of healthy trials")
    p.add_argument("--n-pd",          type=int,   default=39,
                   help="Number of PD trials")
    p.add_argument("--tmax",          type=float, default=2000.0,
                   help="Simulation duration ms")
    p.add_argument("--corstim",       type=int,   default=0, choices=[0, 1],
                   help="Cortical stimulation: 0=off, 1=on")
    p.add_argument("--dbs-freq",      type=float, default=0.0,
                   help="DBS frequency Hz (0 = off)")
    p.add_argument("--dbs-pw",        type=float, default=0.3,
                   help="DBS pulse width ms")
    p.add_argument("--dbs-amplitude", type=float, default=300.0,
                   help="DBS amplitude µA/cm²")
    p.add_argument("--no-cache",      action="store_true",
                   help="Ignore cached results and re-run all simulations")
    p.add_argument("--dpi",           type=int,   default=150,
                   help="Figure DPI (300 for publication quality)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    print("=" * 60)
    print("  Library vs Benchmark — Head-to-Head Comparison")
    print("=" * 60)
    print(f"  Healthy trials : {args.n_control}")
    print(f"  PD trials      : {args.n_pd}")
    print(f"  Duration       : {args.tmax:.0f} ms")
    print(f"  Corstim        : {args.corstim}")
    dbs_str = f"{args.dbs_freq} Hz" if args.dbs_freq > 0 else "off"
    print(f"  DBS            : {dbs_str}")
    if args.dbs_freq > 0:
        print(f"  DBS PW         : {args.dbs_pw} ms")
        print(f"  DBS amplitude  : {args.dbs_amplitude} µA/cm²")
    print("=" * 60)

    lib_con, bmk_con, lib_pd, bmk_pd, \
        lib_mem_con, bmk_mem_con, lib_mem_pd, bmk_mem_pd = collect_all(args)

    print(f"Healthy: {len(lib_con)} trials | PD: {len(lib_pd)} trials\n")

    # Summary stats
    bc_t = np.array([d["elapsed"] for d in bmk_con + bmk_pd])
    lc_t = np.array([d["elapsed"] for d in lib_con + lib_pd])
    sp   = bc_t / np.maximum(lc_t, 1e-9)
    print("Runtime summary (all trials combined):")
    print(f"  Benchmark  : {np.median(bc_t):.1f}s median "
          f"({np.min(bc_t):.1f}–{np.max(bc_t):.1f}s)")
    print(f"  Library    : {np.median(lc_t):.1f}s median "
          f"({np.min(lc_t):.1f}–{np.max(lc_t):.1f}s)")
    print(f"  Speedup    : {np.median(sp):.1f}× median "
          f"({np.min(sp):.1f}–{np.max(sp):.1f}×)\n")

    plot_firing_rates(
        bmk_con, lib_con, bmk_pd, lib_pd, args.tmax,
        out_path=os.path.join(FIG_DIR, "firing_rates_vs_benchmark.png"),
        dpi=args.dpi,
    )
    plot_beta_power(
        bmk_con, lib_con, bmk_pd, lib_pd, args.tmax,
        out_path=os.path.join(FIG_DIR, "beta_power_vs_benchmark.png"),
        dpi=args.dpi,
    )
    plot_runtime_speedup(
        bmk_con, lib_con, bmk_pd, lib_pd, args.tmax,
        out_path=os.path.join(FIG_DIR, "runtime_speedup.png"),
        dpi=args.dpi,
    )
    plot_memory_usage(
        lib_mem_con, bmk_mem_con, lib_mem_pd, bmk_mem_pd, args.tmax,
        out_path=os.path.join(FIG_DIR, "memory_usage.png"),
        dpi=args.dpi,
    )
    print("\nDone.")
