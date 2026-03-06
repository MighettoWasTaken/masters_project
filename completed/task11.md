# Task 11: CTX-BG-TH Benchmark Model Reproduction and Validation

## Priority: 1 — COMPLETED

## Overview

Reproduce the rat Parkinson's disease cortex-basal ganglia-thalamus (CTX-BG-TH) network model (Hahn et al. 2019 / Kumaravelu et al. 2016) using only the hodgkin_huxley library API, and validate the reproduction against the original pure-Python benchmark (`simulate_network_model.py`) on firing rates, GPi beta-band power, simulation runtime, and source-code volume. This task serves as the primary end-to-end validation of the library's biological fidelity and as the benchmark for performance evaluation.

---

## 11.1 Network Architecture

**File:** `benchmarks/ctxbgth_model.py`

The network consists of eight populations arranged in the canonical basal ganglia-thalamo-cortical loop:

| Population | Count | Neuron type | Benchmark source |
|---|---|---|---|
| TH | n | Composable (custom) | Wang 1994 / Rubin-Terman |
| STN | n | Composable (custom) | Rubin & Terman 2004 |
| GPe | n | Composable (custom) | Rubin & Terman 2004 |
| GPi | n | Composable (GPe params) | Rubin & Terman 2004 |
| Str_D2 | n | Composable (custom) | Kumaravelu 2016 |
| Str_D1 | n | Composable (custom) | Kumaravelu 2016 |
| CTX_e | n | Izhikevich RS | Kumaravelu 2016 |
| CTX_i | n | Izhikevich FS | Kumaravelu 2016 |

Cortical populations use Izhikevich regular-spiking (`a=0.02, b=0.2, c=-65, d=8`) and fast-spiking (`a=0.10, b=0.2, c=-65, d=2`) parameters. All other populations are built with the `NeuronModel` composable builder — no modifications to the C++ library were required.

### Synaptic Connectivity

All projections replicate the benchmark's synapse weights and kinetics. Benchmark conductances are scaled by the synapse kernel peak amplitude (`gpeak=0.43` excitatory, `gpeak1=0.3` inhibitory) to account for the difference in how the two implementations normalise their synapse kernels:

| Projection | Pattern | Synapse type | Notes |
|---|---|---|---|
| GPi → TH | one-to-one | alpha (−85 mV) | delay 5 ms |
| GPe → STN | one-to-one + shifted | double-exp (−85 mV) | delay 4 ms, two incoming per neuron |
| STN → GPe | receiver-indexed sparse | double-exp AMPA + NMDA | 2 random active receivers; shared per-receiver weight |
| STN → GPi | receiver-indexed sparse | alpha (0 mV) | 5 random active receivers |
| GPe → GPi | shifted ×2 | alpha (−85 mV) | shifts +1 and −2 |
| GPe → GPe | receiver-indexed random | alpha (−85 mV) | per-receiver weight × PD scaling |
| Str_D2 → GPe | all-to-all | alpha (−85 mV) | delay 5 ms |
| Str_D1 → GPi | all-to-all | alpha (−85 mV) | delay 4 ms |
| CTX_e → Str_D2 | one-to-one | alpha (0 mV) | delay 5.1 ms |
| CTX_e → Str_D1 | one-to-one | alpha (0 mV) | uniform weight range; PD reduction |
| CTX_e → STN | one-to-one + shifted | double-exp AMPA + NMDA ×4 | delay 5.9 ms |
| TH → CTX_e | one-to-one | alpha (0 mV) | delay 5 ms |
| CTX_i → CTX_e | random permutation ×4 | alpha (−85 mV) | ~0 delay |
| CTX_e → CTX_i | random permutation ×4 | alpha (0 mV) | ~0 delay |
| Str_D2 → Str_D2 | random permutation ×4 | kinetic GABA-A | TANH_GATE, τ=13 ms |
| Str_D1 → Str_D1 | random permutation ×3 | kinetic GABA-A | TANH_GATE, τ=13 ms |

Connectivity patterns not natively supported by `RegionalNetwork.connect()` (receiver-indexed weights shared across two shifted inputs) are expressed using `add_connection()` in explicit per-neuron loops, matching the benchmark's indexing logic exactly.

### Stimulators

- **Tonic drive:** constant `I_ext` for TH (1.2 µA/cm²), GPe (3.0 µA/cm²), GPi (3.0 µA/cm²), matching `Iappth`, `Iappgpe`, `Iappgpi` in the benchmark. STN, striatum, and cortex receive zero tonic drive.
- **Cortical stimulus:** `PulseStimulator.single(onset=1000.0, duration=0.3, amplitude=350.0)` applied to CTX_e and CTX_i when `corstim=1`.
- **DBS:** `DBSStimulator` on STN with configurable frequency, pulse width, and amplitude.

---

## 11.2 Validation Metrics

### Firing Rates

Per-population mean firing rates are compared against the benchmark across both healthy and PD conditions. Both models are expected to produce the same pattern of rate changes between conditions — elevated STN/GPi activity and suppressed TH in PD — confirming that the library reproduces the network's disease-state dynamics, not just its healthy baseline.

### GPi Beta-Band Power

The primary DBS efficacy metric is the integrated multitaper power spectral density of the GPi population over 7–35 Hz. Computed using `analyze_beta_power()` in `spectral.py`, which implements the Chronux `mtspectrumpt` algorithm (Mitra & Bokil 2008) matching the benchmark's `make_Spectrum()` function. Beta power is integrated using the trapezoidal rule. Both models are expected to show significantly elevated GPi beta power in the PD condition relative to healthy.

---

## 11.3 Comparison Scripts

### `benchmarks/compare_models.py`

Side-by-side single-run comparison table. Runs benchmark and library models sequentially with identical parameters and prints:

- Per-population firing rates (mean ± std Hz) for both implementations and the delta
- GPi beta-band power and relative difference
- Wall-clock runtime for each
- Source lines of code (non-blank, non-comment) for each model file

### `benchmarks/compare_vs_benchmark.py`

Multi-trial statistical comparison. Runs `--n-control` healthy trials and `--n-pd` PD trials, accumulating distributions of per-population firing rates, GPi beta-band power, runtime speedup, and peak RSS memory. Results are cached to `bench_compare_cache.pkl` so runs can be resumed. Four figures are produced:

| Figure | Content |
|---|---|
| `firing_rates_vs_benchmark.png` | 8-panel rate box plots across all populations |
| `beta_power_vs_benchmark.png` | GPi beta power distributions, healthy vs PD |
| `runtime_speedup.png` | Speedup ratio (benchmark / library) by condition |
| `memory_usage.png` | Peak RSS per model per condition |

---

## 11.4 Performance Targets

At n=10 neurons per population (80 total), tmax=2000 ms, dt=0.01 ms:

- **Runtime:** >10× faster than the pure-Python benchmark
- **Memory:** Lower peak RSS than the benchmark
- **Lines of code:** <50% of `simulate_network_model.py` (non-blank, non-comment)

---

## Checklist

- [x] Implement `build_network()` in `benchmarks/ctxbgth_model.py` with all 8 populations
- [x] Replicate all 16 projection types with correct weights and synapse kinetics
- [x] Implement custom TH spec with `DERIVED` n_K gate and voltage-dependent tau_r
- [x] Implement STN spec with calcium dynamics (Nernst, AHP)
- [x] Implement GPe/GPi spec with calcium-dependent AHP channel
- [x] Implement striatum spec with M-type K and PD-dependent conductance
- [x] Implement `simulate_ctxbgth()` matching benchmark return signature
- [x] Add `DBSStimulator` and `PulseStimulator` support
- [x] Implement `compare_models.py` side-by-side table
- [x] Implement `compare_vs_benchmark.py` multi-trial statistical comparison
- [x] Produce `firing_rates_vs_benchmark.png`, `beta_power_vs_benchmark.png`, `runtime_speedup.png`, `memory_usage.png`
- [x] Verify both models produce matched rate changes between healthy and PD conditions
- [x] Verify both models show elevated GPi beta power in PD condition
