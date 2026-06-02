"""
visualize_dataset.py — batch comparison of library model vs all .mat reference
files in TestDataSet_64.

Runs the hodgkin_huxley library against every .mat file (25 healthy, 39 PD),
collects per-population firing rates and GPi β-band power, then produces
publication-quality box-and-whisker plots comparing MATLAB reference vs library.

Results are cached to disk so subsequent runs skip the simulation step.

Usage
-----
    python benchmarks/visualize_dataset.py [--no-cache] [--seed-offset N] [--dpi N]

Outputs (saved to benchmarks/figures/)
-------
    firing_rates.png   — 2×4 grid of per-population firing rate distributions
    beta_power.png     — GPi β-band power (7–35 Hz) distributions
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from timeit import default_timer as timer

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats as _sps
except ImportError:
    _sps = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compare_matlab import load_mat, run_library   # reuse tested functions

DATA_DIR   = os.path.join(_HERE, "TestDataSet_64")
CACHE_FILE = os.path.join(_HERE, "viz_cache.pkl")
FIG_DIR    = os.path.join(_HERE, "figures")

POPS = ["TH", "STN", "GPe", "GPi", "Str_D2", "Str_D1", "CTX_e", "CTX_i"]

# Okabe-Ito colorblind-friendly palette
C_MAT_CON = "#E69F00"   # amber      — MATLAB healthy
C_LIB_CON = "#56B4E9"   # sky blue   — Library healthy
C_MAT_PD  = "#D55E00"   # vermilion  — MATLAB PD
C_LIB_PD  = "#0072B2"   # blue       — Library PD

_COLORS = [C_MAT_CON, C_LIB_CON, C_MAT_PD, C_LIB_PD]
_LABELS = ["MATLAB (healthy)", "Library (healthy)", "MATLAB (PD)", "Library (PD)"]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_mat_files() -> list[str]:
    """Return all .mat result files sorted numerically by trial number."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")
             and (re.search(r"\d+con\d", f) or re.search(r"\d+pd\d", f))]
    files.sort(key=lambda f: int(re.search(r"(\d+)", f).group(1)))
    return [os.path.join(DATA_DIR, f) for f in files]


# ---------------------------------------------------------------------------
# Batch data collection with caching
# ---------------------------------------------------------------------------

def collect_all(mat_paths: list[str], seed_offset: int,
                no_cache: bool) -> tuple[list[dict], list[dict]]:
    """
    Load all MATLAB reference files and run the library for each one.

    Returns
    -------
    mat_results : list of dicts  (from load_mat)
    lib_results : list of dicts  {rates, beta_power, pd}
    """
    print(f"Loading {len(mat_paths)} MATLAB reference files ...", flush=True)
    mat_results = [load_mat(p) for p in mat_paths]

    # Cache key: seed offset + ordered list of filenames
    cache_key = (seed_offset, [os.path.basename(p) for p in mat_paths])

    if not no_cache and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as fh:
            cache = pickle.load(fh)
        if cache.get("key") == cache_key:
            print("Using cached library results. "
                  "(Pass --no-cache to force re-run.)", flush=True)
            return mat_results, cache["lib_results"]

    lib_results = []
    total = len(mat_paths)
    t_total = timer()

    for i, (path, mat) in enumerate(zip(mat_paths, mat_results)):
        seed = seed_offset + i
        print(f"  [{i+1:2d}/{total}]  {mat['fname']}"
              f"  pd={mat['pd']}  seed={seed} ...",
              end="  ", flush=True)
        t0 = timer()
        rates, beta_power, _, _, _ = run_library(
            mat["tmax"], mat["n"], mat["pd"], seed
        )
        t_run = timer() - t0
        print(f"{t_run:.1f}s", flush=True)
        lib_results.append({"rates": rates, "beta_power": beta_power,
                             "pd": mat["pd"], "runtime": t_run})

    print(f"\nAll {total} simulations complete in "
          f"{timer() - t_total:.0f}s.", flush=True)

    with open(CACHE_FILE, "wb") as fh:
        pickle.dump({"key": cache_key, "lib_results": lib_results}, fh)
    print(f"Results cached to {CACHE_FILE}", flush=True)

    return mat_results, lib_results


def _split_by_condition(mat_results, lib_results):
    """Split both lists into (healthy, pd) pairs."""
    mat_con, mat_pd = [], []
    lib_con, lib_pd = [], []
    for m, l in zip(mat_results, lib_results):
        if m["pd"] == 0:
            mat_con.append(m); lib_con.append(l)
        else:
            mat_pd.append(m);  lib_pd.append(l)
    return mat_con, lib_con, mat_pd, lib_pd


# ---------------------------------------------------------------------------
# Low-level box-plot drawing
# ---------------------------------------------------------------------------

def _draw_box(ax: plt.Axes, pos: float, data: np.ndarray,
              color: str, width: float = 0.32) -> None:
    """Single box-and-whisker with jittered data points overlaid."""
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
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.09, 0.09, size=len(data))
    ax.scatter(np.full(len(data), pos) + jitter, data,
               color=color, s=14, alpha=0.55, zorder=5, linewidths=0)


def _style_ax(ax: plt.Axes, title: str, ylabel: str,
              group_centers: list[float], group_labels: list[str],
              sep_x: float) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.axvline(sep_x, color="#999999", linewidth=0.9,
               linestyle="--", alpha=0.55)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles():
    return [mpatches.Patch(facecolor=c, alpha=0.72, label=l)
            for c, l in zip(_COLORS, _LABELS)]


# ---------------------------------------------------------------------------
# Figure 1 — Firing rates (2 × 4 subplots)
# ---------------------------------------------------------------------------

def plot_firing_rates(mat_con, lib_con, mat_pd, lib_pd, out_path: str,
                      dpi: int = 150) -> None:
    """
    2×4 grid: one subplot per population.
    Each subplot has four box plots:
        Healthy group → [MATLAB | Library]
        PD group      → [MATLAB | Library]
    Data point = per-trial population-mean firing rate.
    """
    # Box positions within each subplot
    p_mc, p_lc = 1.0, 1.5    # healthy pair
    p_mp, p_lp = 2.8, 3.3    # PD pair
    sep_x = (p_lc + p_mp) / 2
    group_centers = [(p_mc + p_lc) / 2, (p_mp + p_lp) / 2]
    group_labels   = ["Healthy", "PD"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()

    for idx, pop in enumerate(POPS):
        ax = axes[idx]

        mc = np.array([d["rates"][pop].mean() for d in mat_con])
        lc = np.array([d["rates"][pop].mean() for d in lib_con])
        mp = np.array([d["rates"][pop].mean() for d in mat_pd])
        lp = np.array([d["rates"][pop].mean() for d in lib_pd])

        _draw_box(ax, p_mc, mc, C_MAT_CON)
        _draw_box(ax, p_lc, lc, C_LIB_CON)
        _draw_box(ax, p_mp, mp, C_MAT_PD)
        _draw_box(ax, p_lp, lp, C_LIB_PD)

        all_vals = np.concatenate([mc, lc, mp, lp])
        ymax = max(all_vals.max() * 1.12, 5.0) if len(all_vals) > 0 else 5.0
        ax.set_ylim(bottom=0, top=ymax)
        ax.set_xlim(0.55, 3.75)
        _style_ax(ax, pop, "Firing rate (Hz)", group_centers, group_labels, sep_x)

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(
        "Population Firing Rates — MATLAB Reference vs Library Model\n"
        f"(n=10 neurons/pop, tmax=2000 ms, "
        f"{len(mat_con)} healthy / {len(mat_pd)} PD trials)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — GPi β-band power
# ---------------------------------------------------------------------------

def plot_beta_power(mat_con, lib_con, mat_pd, lib_pd, out_path: str,
                    dpi: int = 150) -> None:
    """
    Single panel: GPi β-band (7–35 Hz) integrated power.
    Four box plots: MATLAB-healthy, Library-healthy, MATLAB-PD, Library-PD.
    """
    p_mc, p_lc = 1.0, 1.55
    p_mp, p_lp = 2.95, 3.5
    sep_x = (p_lc + p_mp) / 2
    group_centers = [(p_mc + p_lc) / 2, (p_mp + p_lp) / 2]
    group_labels   = ["Healthy", "PD"]

    mc = np.array([d["beta_power"] for d in mat_con])
    lc = np.array([d["beta_power"] for d in lib_con])
    mp = np.array([d["beta_power"] for d in mat_pd])
    lp = np.array([d["beta_power"] for d in lib_pd])

    fig, ax = plt.subplots(figsize=(7, 5.5))

    _draw_box(ax, p_mc, mc, C_MAT_CON, width=0.40)
    _draw_box(ax, p_lc, lc, C_LIB_CON, width=0.40)
    _draw_box(ax, p_mp, mp, C_MAT_PD,  width=0.40)
    _draw_box(ax, p_lp, lp, C_LIB_PD,  width=0.40)

    ax.set_xlim(0.45, 4.05)
    _style_ax(ax, "GPi β-band Oscillatory Power (7–35 Hz)",
              "Integrated power (a.u.)", group_centers, group_labels, sep_x)

    ax.legend(handles=_legend_handles(), fontsize=9.5, loc="upper left",
              frameon=True, framealpha=0.9)

    # Annotate medians
    for pos, data, color in [
        (p_mc, mc, C_MAT_CON), (p_lc, lc, C_LIB_CON),
        (p_mp, mp, C_MAT_PD),  (p_lp, lp, C_LIB_PD),
    ]:
        med = np.median(data)
        ax.text(pos, med, f" {med:.0f}", va="center", ha="left",
                fontsize=7.5, color="black", alpha=0.75)

    fig.suptitle(
        "GPi β-band Power — MATLAB Reference vs Library Model\n"
        f"({len(mat_con)} healthy / {len(mat_pd)} PD trials)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quantitative report — accompanies the plots
# ---------------------------------------------------------------------------

def _stats(vals):
    """(mean, min, max) of a 1-D sequence."""
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.min()), float(a.max())


def _lin_ccc(x, y) -> float:
    """Lin's concordance correlation coefficient (1 = perfect agreement)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    den = vx + vy + (mx - my) ** 2
    return float(2.0 * cov / den) if den > 0 else float("nan")


def _pearson(x, y) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    if _sps is not None:
        try:
            return float(_sps.pearsonr(x, y)[0])
        except Exception:
            pass
    return float(np.corrcoef(x, y)[0, 1])


def _wilcoxon_p(lib_vals, mat_vals) -> float:
    """Paired Wilcoxon signed-rank p on (Library - MATLAB); high p = no bias."""
    if _sps is None:
        return float("nan")
    d = np.asarray(lib_vals, dtype=float) - np.asarray(mat_vals, dtype=float)
    if len(d) < 1 or np.all(d == 0):
        return float("nan")
    try:
        return float(_sps.wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def comparison_report(mat_con, lib_con, mat_pd, lib_pd, out_path: str) -> None:
    """
    Quantify the MATLAB-vs-Library agreement that the box plots show visually:
    for every plotted metric (per-population firing rate and GPi β-band power),
    by condition, report each side's mean and [min, max] span, the delta
    (Library − MATLAB) of the means, and that delta as a percent of the MATLAB
    mean. Also summarizes the library simulation runtimes. Prints and saves.
    """
    lines: list[str] = []
    def w(s: str = "") -> None: lines.append(s)

    n_con, n_pd = len(mat_con), len(mat_pd)
    w("=" * 96)
    w("QUANTITATIVE COMPARISON - MATLAB reference vs Library model")
    w(f"({n_con} healthy / {n_pd} PD trials;  d = Library - MATLAB,  rel = d / MATLAB mean)")
    w("=" * 96)

    hdr = (f"{'Metric':<14}{'Cond':<9}{'MAT mean':>10}{'MAT [min,max]':>18}"
           f"{'LIB mean':>10}{'LIB [min,max]':>18}{'d(L-M)':>9}{'rel':>8}")

    # Per-population firing rates (per-trial population mean, Hz) — matches the plot.
    w("")
    w("Per-trial population-mean firing rate (Hz)")
    w("-" * 96)
    w(hdr)
    for pop in POPS:
        for label, mres, lres in [("Healthy", mat_con, lib_con), ("PD", mat_pd, lib_pd)]:
            if not mres:
                continue
            mm, mlo, mhi = _stats([d["rates"][pop].mean() for d in mres])
            lm, llo, lhi = _stats([d["rates"][pop].mean() for d in lres])
            delta = lm - mm
            rel = (delta / mm * 100.0) if mm != 0 else float("nan")
            w(f"{pop:<14}{label:<9}"
              f"{mm:>10.2f}{('[%.1f,%.1f]' % (mlo, mhi)):>18}"
              f"{lm:>10.2f}{('[%.1f,%.1f]' % (llo, lhi)):>18}"
              f"{delta:>+9.2f}{rel:>+7.1f}%")

    # GPi beta-band power.
    w("")
    w("GPi beta-band power (7-35 Hz, a.u.)")
    w("-" * 96)
    w(hdr)
    for label, mres, lres in [("Healthy", mat_con, lib_con), ("PD", mat_pd, lib_pd)]:
        if not mres:
            continue
        mm, mlo, mhi = _stats([d["beta_power"] for d in mres])
        lm, llo, lhi = _stats([d["beta_power"] for d in lres])
        delta = lm - mm
        rel = (delta / mm * 100.0) if mm != 0 else float("nan")
        w(f"{'GPi beta-pow':<14}{label:<9}"
          f"{mm:>10.1f}{('[%.0f,%.0f]' % (mlo, mhi)):>18}"
          f"{lm:>10.1f}{('[%.0f,%.0f]' % (llo, lhi)):>18}"
          f"{delta:>+9.1f}{rel:>+7.1f}%")

    # Per-trial agreement (paired by trial configuration). Library and MATLAB
    # trials share the same (n, pd, tmax) configuration but use different random
    # realizations, so within-condition scatter partly reflects intrinsic trial
    # variability, not model error. CCC / r are pooled across all trials (so the
    # healthy→PD range provides spread); the Wilcoxon test detects systematic
    # bias of the matched (Library − MATLAB) pairs within each condition.
    w("")
    w("Per-trial agreement: MATLAB vs Library  (paired by trial configuration)")
    w("  CCC = Lin's concordance (1 = perfect); r = Pearson, pooled over all trials.")
    w("  Wilcoxon p = systematic bias of paired (Library - MATLAB); high p = no bias.")
    w("-" * 96)
    w(f"{'Metric':<14}{'n':>5}{'Pearson r':>12}{'Lin CCC':>10}"
      f"{'Wilcoxon p (Healthy)':>22}{'Wilcoxon p (PD)':>18}")

    mat_all = mat_con + mat_pd
    lib_all = lib_con + lib_pd
    metrics = [(p, (lambda d, _p=p: float(d["rates"][_p].mean()))) for p in POPS]
    metrics.append(("GPi beta-pow", lambda d: float(d["beta_power"])))
    for label, fn in metrics:
        x = [fn(d) for d in mat_all]
        y = [fn(d) for d in lib_all]
        r = _pearson(x, y)
        ccc = _lin_ccc(x, y)
        p_con = _wilcoxon_p([fn(d) for d in lib_con], [fn(d) for d in mat_con])
        p_pd = _wilcoxon_p([fn(d) for d in lib_pd], [fn(d) for d in mat_pd])
        w(f"{label:<14}{len(x):>5}{r:>12.3f}{ccc:>10.3f}"
          f"{_fmt_p(p_con):>22}{_fmt_p(p_pd):>18}")
    if _sps is None:
        w("  (SciPy not installed — Pearson via numpy; Wilcoxon unavailable.)")

    # Library simulation runtimes.
    rt = [d["runtime"] for d in (lib_con + lib_pd) if d.get("runtime") is not None]
    w("")
    w("Library simulation runtime (s per trial)")
    w("-" * 96)
    if rt:
        a = np.asarray(rt, dtype=float)
        w(f"  mean={a.mean():.2f}s   span=[{a.min():.2f}, {a.max():.2f}]s   "
          f"range={a.max() - a.min():.2f}s   total={a.sum():.1f}s   n={len(a)}")
    else:
        w("  (unavailable — cached results predate runtime capture; re-run with --no-cache)")
    w("")

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="Batch library vs MATLAB comparison with box plots."
    )
    p.add_argument("--no-cache",    action="store_true",
                   help="Ignore cached results and re-run all simulations.")
    p.add_argument("--seed-offset", type=int, default=1000,
                   help="Base RNG seed; trial i uses seed_offset + i.")
    p.add_argument("--dpi",         type=int, default=150,
                   help="Figure DPI (use 300 for publication quality).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    mat_paths = _find_mat_files()
    if not mat_paths:
        raise FileNotFoundError(f"No .mat result files found in {DATA_DIR}")
    print(f"Found {len(mat_paths)} .mat files "
          f"(sorting: {os.path.basename(mat_paths[0])} … "
          f"{os.path.basename(mat_paths[-1])})")

    mat_results, lib_results = collect_all(
        mat_paths, seed_offset=args.seed_offset, no_cache=args.no_cache
    )

    mat_con, lib_con, mat_pd, lib_pd = _split_by_condition(mat_results, lib_results)
    print(f"\nHealthy: {len(mat_con)} trials | PD: {len(mat_pd)} trials")

    plot_firing_rates(
        mat_con, lib_con, mat_pd, lib_pd,
        out_path=os.path.join(FIG_DIR, "firing_rates.png"),
        dpi=args.dpi,
    )
    plot_beta_power(
        mat_con, lib_con, mat_pd, lib_pd,
        out_path=os.path.join(FIG_DIR, "beta_power.png"),
        dpi=args.dpi,
    )
    comparison_report(
        mat_con, lib_con, mat_pd, lib_pd,
        out_path=os.path.join(FIG_DIR, "comparison_report.txt"),
    )
    print("\nDone.")
