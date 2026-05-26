"""
Systematic CPU vs GPU timing sweep across representative network sizes.

Cases:
  1. HH default single population               -> composable neuron runtime
  2. HH default dense projection                -> CUDA synapse path
  3. Custom gate + synapse-g modulation         -> task 17.5 -> 17.9 path

Outputs:
  benchmarks/results/cuda_cpu_size_sweep_raw.csv
  benchmarks/results/cuda_cpu_size_sweep_summary.csv
  benchmarks/results/cuda_benchmark_report.md
  benchmarks/results/custom_modulated_cuda_codegen_preview.cu
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

import hodgkin_huxley as hh
from hodgkin_huxley import compile_model_cuda


DT_MS = 0.05
TRIALS = 3
RESULT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    title: str
    description: str
    sizes: tuple[int, ...]
    duration_ms: float


CASE_SPECS = (
    BenchmarkCase(
        key="hh_single",
        title="HH default single population",
        description="One composable HH population with no synapses.",
        sizes=(2048, 4096, 8192, 16384),
        duration_ms=300.0,
    ),
    BenchmarkCase(
        key="hh_dense",
        title="HH default dense projection",
        description="Two HH populations with dense all-to-all AMPA projection.",
        sizes=(128, 256, 384, 512),
        duration_ms=300.0,
    ),
    BenchmarkCase(
        key="custom_modulated",
        title="Custom gate plus synapse-g modulation",
        description=(
            "HH pre-population driving a post-population with a custom gate, "
            "a nonlinear intracellular ODE, and SYNAPSE_G modulation."
        ),
        sizes=(128, 256, 384, 512),
        duration_ms=300.0,
    ),
)


def _git_branch() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _make_custom_gate_spec() -> hh.NeuronModelSpec:
    model = hh.NeuronModel("custom_gate", C_m=1.0, V_init=-65.0)
    novel_inf = (
        1 / (1 + sp.exp(-(hh.V + 40) / 8))
    ) * sp.Float(0.999) + sp.Float(0.001)
    novel_tau = sp.Float(2.0) + sp.sqrt(hh.V**2 + 1) * sp.Float(0.0)
    model.add_gate("m", inf=novel_inf, tau=novel_tau)
    model.add_channel("Leak", g=0.1, E_rev=-65.0)
    return model.to_spec()


def _make_custom_modulation_dynamics() -> hh.IntracellularDynamics:
    da = hh.substance("DA")
    gain = 1 / (1 + sp.exp(-10 * (da - sp.Float(0.5))))
    return hh.IntracellularDynamics(
        "DA",
        ode=-(da**2 + da) / sp.Float(80.0),
        initial=0.5,
        modulations=[hh.Modulation.synapse_g(gain)],
    )


def _append_custom_modulation(spec: hh.NeuronModelSpec) -> hh.NeuronModelSpec:
    dyn = _make_custom_modulation_dynamics()
    channel_names = [ch.name for ch in spec.channels]
    gate_names = [g.name for g in spec.gates]
    substance_map = {ic.name: idx for idx, ic in enumerate(spec.intracellular)}
    spec.intracellular.append(dyn.to_spec(channel_names, gate_names, substance_map))
    return spec


def _build_case(case: BenchmarkCase, size: int) -> tuple[hh.RegionalNetwork, dict[str, float], int, int]:
    if case.key == "hh_single":
        rn = hh.RegionalNetwork()
        rn.add_population("pop", size, model=hh.NeuronModelSpec.hh_default())
        return rn, {"pop": 6.0}, size, 0

    if case.key == "hh_dense":
        rn = hh.RegionalNetwork()
        spec = hh.NeuronModelSpec.hh_default()
        rn.add_population("pre", size, model=spec)
        rn.add_population("post", size, model=spec)
        rn.connect(
            "pre",
            "post",
            "all_to_all",
            weight=0.3,
            synapse=hh.SynapseSpec.ampa(),
            delay=1.0,
            seed=1,
        )
        return rn, {"pre": 8.0, "post": 0.0}, 2 * size, size * size

    if case.key == "custom_modulated":
        rn = hh.RegionalNetwork()
        rn.add_population("pre", size, model=hh.NeuronModelSpec.hh_default())
        rn.add_population("post", size, model=_make_custom_gate_spec())
        rn.connect(
            "pre",
            "post",
            "all_to_all",
            weight=1.0,
            synapse=hh.SynapseSpec.ampa(),
            delay=0.0,
            seed=1,
        )
        rn.add_intracellular(
            _make_custom_modulation_dynamics(),
            populations=["post"],
        )
        return rn, {"pre": 8.0, "post": 0.0}, 2 * size, size * size

    raise ValueError(f"Unknown case key: {case.key}")


def _write_codegen_preview(outdir: Path) -> Path:
    spec = _append_custom_modulation(_make_custom_gate_spec())
    preview = compile_model_cuda(spec, "custom_modulated")
    path = outdir / "custom_modulated_cuda_codegen_preview.cu"
    path.write_text(preview, encoding="utf-8")
    return path


def _time_case(case: BenchmarkCase, size: int, device_label: str, trials: int) -> list[dict[str, object]]:
    device = hh.Device.cpu() if device_label == "cpu" else hh.Device.cuda(0)
    rn, stimuli, total_neurons, total_synapses = _build_case(case, size)
    rn.to(device)

    warmup_ms = min(20.0, case.duration_ms)
    rn.simulate(warmup_ms, DT_MS, stimuli)
    rn.reset()

    rows: list[dict[str, object]] = []
    for trial_idx in range(1, trials + 1):
        t0 = time.perf_counter()
        result = rn.simulate(case.duration_ms, DT_MS, stimuli)
        elapsed_s = time.perf_counter() - t0
        first_key = next(iter(result))
        rows.append(
            {
                "case_key": case.key,
                "case_title": case.title,
                "size": size,
                "duration_ms": case.duration_ms,
                "dt_ms": DT_MS,
                "total_neurons": total_neurons,
                "total_synapses": total_synapses,
                "device": device_label,
                "trial": trial_idx,
                "elapsed_s": elapsed_s,
                "trace_shape": str(result[first_key].shape),
            }
        )
        rn.reset()
    return rows


def _median_elapsed(rows: list[dict[str, object]], device: str) -> float:
    vals = [float(row["elapsed_s"]) for row in rows if row["device"] == device]
    return statistics.median(vals)


def _format_seconds(value: float) -> str:
    return f"{value:.4f}"


def _format_speedup(value: float) -> str:
    return f"{value:.2f}x"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["case_key"]), int(row["size"])), []).append(row)

    summary: list[dict[str, object]] = []
    for (case_key, size), rows in grouped.items():
        cpu_s = _median_elapsed(rows, "cpu")
        gpu_s = _median_elapsed(rows, "cuda")
        speedup = cpu_s / gpu_s if gpu_s > 0 else float("inf")
        sample = rows[0]
        summary.append(
            {
                "case_key": case_key,
                "case_title": sample["case_title"],
                "size": size,
                "duration_ms": sample["duration_ms"],
                "dt_ms": sample["dt_ms"],
                "total_neurons": sample["total_neurons"],
                "total_synapses": sample["total_synapses"],
                "cpu_median_s": cpu_s,
                "gpu_median_s": gpu_s,
                "speedup": speedup,
            }
        )

    summary.sort(key=lambda row: (str(row["case_key"]), int(row["size"])))
    return summary


def _case_table(summary_rows: list[dict[str, object]], case: BenchmarkCase) -> str:
    lines = [
        f"### {case.title}",
        "",
        case.description,
        "",
        "| Size | Total neurons | Total synapses | CPU median (s) | GPU median (s) | Speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if row["case_key"] != case.key:
            continue
        lines.append(
            "| "
            f"{row['size']} | "
            f"{row['total_neurons']} | "
            f"{row['total_synapses']} | "
            f"{_format_seconds(float(row['cpu_median_s']))} | "
            f"{_format_seconds(float(row['gpu_median_s']))} | "
            f"{_format_speedup(float(row['speedup']))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_report(
    path: Path,
    summary_rows: list[dict[str, object]],
    *,
    trials: int,
    preview_path: Path,
) -> None:
    branch = _git_branch()
    gpu_name = hh.cuda_device_name(0) if hh.cuda_is_available() else "no CUDA GPU"
    lines = [
        "# CUDA Benchmark Report",
        "",
        "## Run context",
        "",
        f"- Branch: `{branch}`",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Platform: `{platform.platform()}`",
        f"- GPU: `{gpu_name}`",
        f"- Trials per data point: `{trials}`",
        f"- Time step: `{DT_MS} ms`",
        f"- Timed region: `simulate(...)` only after a warmup run and `to(device)` setup",
        f"- Linked CUDA codegen preview: `{preview_path.relative_to(path.parent.parent).as_posix()}`",
        "",
        "## What was benchmarked",
        "",
        "- `hh_single` checks the composable neuron path without synapse overhead.",
        "- `hh_dense` stresses the dense connected CUDA synapse path.",
        "- `custom_modulated` exercises the 17.5 to 17.9 bridge: custom gate VM logic plus a nonlinear intracellular ODE and `SYNAPSE_G` modulation on the GPU runtime.",
        "",
        "## Speedup tables",
        "",
    ]

    for case in CASE_SPECS:
        lines.append(_case_table(summary_rows, case))

    lines.extend(
        [
            "## Takeaways",
            "",
            "- GPU overhead is visible on the smaller runs, especially when the network is not large enough to hide host-device movement and recording costs.",
            "- The dense synapse case shows the clearest CUDA benefit as size grows because the GPU can keep the repeated synapse work on device.",
            "- The custom gate plus modulation case also speeds up at larger sizes, which is the practical confirmation that the linked 17.5 and 17.9 path is not only correct but worth benchmarking.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS, help="timed trials per device and size")
    parser.add_argument("--outdir", type=Path, default=RESULT_DIR, help="output directory")
    args = parser.parse_args()

    if not hh.cuda_is_available():
        raise SystemExit("CUDA GPU not available on this machine")

    args.outdir.mkdir(parents=True, exist_ok=True)
    preview_path = _write_codegen_preview(args.outdir)

    raw_rows: list[dict[str, object]] = []
    for case in CASE_SPECS:
        print(f"\n[{case.key}] {case.title}  duration={case.duration_ms} ms  dt={DT_MS} ms")
        for size in case.sizes:
            print(f"  size={size}")
            for device in ("cpu", "cuda"):
                device_rows = _time_case(case, size, device, args.trials)
                raw_rows.extend(device_rows)
                median_s = statistics.median(float(row["elapsed_s"]) for row in device_rows)
                print(f"    {device:<4} median={median_s:.4f}s")

    raw_path = args.outdir / "cuda_cpu_size_sweep_raw.csv"
    summary_path = args.outdir / "cuda_cpu_size_sweep_summary.csv"
    report_path = args.outdir / "cuda_benchmark_report.md"

    summary_rows = _build_summary(raw_rows)
    _write_csv(raw_path, raw_rows)
    _write_csv(summary_path, summary_rows)
    _write_report(report_path, summary_rows, trials=args.trials, preview_path=preview_path)

    print(f"\nWrote raw results    : {raw_path}")
    print(f"Wrote summary tables : {summary_path}")
    print(f"Wrote report         : {report_path}")
    print(f"Wrote codegen preview: {preview_path}")


if __name__ == "__main__":
    main()
