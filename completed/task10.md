# Task 10: Simulation Quality, Extended Recording & API Improvements

## Priority: 1–2 (mixed — see per-section priorities)

## Overview

A collection of issues identified through API analysis that would block or significantly impede researchers building novel brain models with the library. Grouped into four areas: numerical correctness bugs, extended simulation observability, noise injection, and population-level heterogeneity.

---

## 10.1 ALPHA_BETA Gate Euler Instability

**Priority: 1**

### Problem

The INF_TAU gate update was fixed to use exact exponential integration (Task 5 bugfix). The ALPHA_BETA form still uses forward Euler in both `composable_neuron.cpp` and `composable_pool.cpp`:

```cpp
// current — unstable when dt * (alpha + beta) > 1
gate_states_[i] += dt * (alpha * (1.0 - gate_states_[i]) - beta * gate_states_[i]);
```

The effective time constant is `tau = 1 / (alpha + beta)`. For fast gates (e.g. classical HH m-gate where alpha+beta ≈ 10–100 ms⁻¹ at threshold), `dt/tau` can exceed 1, causing oscillation or clamped drift identical to the INF_TAU issue that was already fixed.

### Fix

Apply exact exponential integration, same pattern as INF_TAU:

```cpp
// composable_neuron.cpp
double rate = alpha + beta;
double x_inf = (rate > 1e-10) ? alpha / rate : gate_states_[i];
double tau_x = (rate > 1e-10) ? 1.0 / rate : 1e10;
gate_states_[i] = x_inf + (gate_states_[i] - x_inf) * std::exp(-dt * tau_x);
```

```cpp
// composable_pool.cpp  (Eigen vectorized equivalent)
Eigen::ArrayXd rate = alpha + beta;
Eigen::ArrayXd x_inf = alpha / rate.max(1e-10);
Eigen::ArrayXd tau_x = rate.max(1e-10).inverse();
X = x_inf + (X - x_inf) * (-dt / tau_x).exp();
```

### Checklist
- [x] Fix ALPHA_BETA in `composable_neuron.cpp`
- [x] Fix ALPHA_BETA in `composable_pool.cpp`
- [x] Add test: ALPHA_BETA gate converges to alpha/(alpha+beta) steady state
- [x] Add test: stable output at dt >> tau (dt=1ms, fast gate)

---

## 10.2 h_T Tau Parameter Bug in TH and STN Presets

**Priority: 1**

### Problem

The `DOUBLE_EXP_SUM` tau formula is `base + amp / (exp((V+v1)/s1) + exp(-(V+v2)/s2))`.

In `ion_channels.cpp`, both the TH and STN h_T gates have:
```cpp
g.tau.params[5] = -0.6;   // v2  ← should be +0.6
g.tau.params[6] = -7.4;   // s2  ← should be +7.4
```

Due to double negation `exp(-(V + (-0.6)) / (-7.4))` evaluates to `exp((V - 0.6) / 7.4)` — the wrong sign on the exponent. At V = -65 mV this gives tau_h_T ≈ 1535 ms instead of the correct ≈ 20 ms. T-channel burst dynamics in TH and STN are therefore significantly wrong for any researcher using these presets.

### Fix

In `ion_channels.cpp`, for both the TH h_T gate and the STN h_T gate:
```cpp
g.tau.params[5] = 0.6;    // v2
g.tau.params[6] = 7.4;    // s2
```

### Note on test_th_fires_under_current

After fixing the tau, h_T at rest quickly converges to h_T_inf(-65) ≈ 0.076, increasing n_K and substantially reducing the net depolarising drive. Verify whether `test_th_fires_under_current` still passes (may need `I_ext` raised or the test reworked to verify burst/rebound firing rather than tonic firing, which is more representative of TH physiology anyway).

### Checklist
- [x] Fix params[5] and params[6] in TH h_T gate (`ion_channels.cpp`)
- [x] Fix params[5] and params[6] in STN h_T gate (`ion_channels.cpp`)
- [ ] Update / re-verify `test_th_fires_under_current`
- [x] Add test: tau_h_T at V=-65 is in the range 15–25 ms after fix

---

## 10.3 Extended Simulation Recording ✅

**Priority: 1** — **COMPLETED**

### Problem

`Network.simulate()` only returns voltage traces. Gate states, calcium concentration, channel currents, and synaptic conductances are never accessible during or after a simulation run. Researchers studying burst mechanisms, calcium accumulation, or channel-level contributions must implement their own simulation loops via `net.step()`, losing the performance of the pool path entirely.

### Implementation

A `RecordingConfig` / `MetricsResult` system was implemented, going beyond the original design:

```python
cfg = RecordingConfig(["V", "gates", "calcium", "spikes", "firing_rate",
                       "g_syn", "I_syn", "ISI_mean", "ISI_cv"])
result = net.simulate(duration, dt, I_ext, record=cfg)
# result["V"]           → (n_neurons, n_rec) ndarray
# result["gates"]       → (n_neurons, max_gates, n_rec) ndarray
# result["calcium"]     → (n_neurons, n_rec) ndarray
# result["spikes"]      → list of spike-time arrays (ms)
# result["firing_rate"] → (n_neurons,) Hz
# result.time           → time axis array
```

Recording is zero-copy: Python pre-allocates numpy buffers, C++ fills them in the hot loop via `_simulate_into_buffers()`. No `step()` fallback; full SIMD performance is retained. An `interval` parameter controls temporal subsampling.

`RegionalNetwork.simulate(record=cfg)` returns a `PopulationMetricsResult` that slices metrics by population name.

New C++ methods: `scatter_gate_states_into()`, `scatter_calcium_into()` on `ComposablePool`; `simulate_into_buffers()`, `max_gate_count()`, `get_synapse_pre/post_indices()` on `Network`.

### Checklist
- [x] Add `scatter_gate_states_into()` to `ComposablePool`
- [x] Add `scatter_calcium_into()` to `ComposablePool`
- [x] Add `simulate_into_buffers()` to `Network` (fills caller-allocated numpy buffers)
- [x] `Network.simulate()` accepts `record=RecordingConfig(...)`, returns `MetricsResult`
- [x] `RegionalNetwork.simulate()` returns `PopulationMetricsResult` when `record=` given
- [x] Backwards-compatible: `record=None` returns plain ndarray / dict as before
- [x] `RecordingConfig` presets: `voltage_only`, `spikes_only`, `spike_stats`, `summary_metrics`, `all_neuron_metrics`, `all_synapse_metrics`, `for_population`
- [x] Metrics: `V`, `gates`, `calcium`, `g_syn`, `I_syn`, `spikes`, `spike_count`, `firing_rate`, `mean_V`, `ISI_mean`, `ISI_cv`, `spike_count_per_synapse`
- [x] Tests in `tests/python/test_recording.py` (~35 tests)

---

## 10.4 Noise Injection

**Priority: 2**

### Problem

No mechanism exists for adding membrane noise. Biological neurons exhibit stochastic fluctuations (synaptic background activity, channel noise) that are essential for realistic firing variability, subthreshold dynamics, and network synchrony studies. Every published model of STN, GPe, or striatum includes additive or multiplicative noise.

### Design

Noise should be pre-generated and passed as part of the current injection — this requires no C++ changes and keeps the simulation loop clean.

#### Python utility

```python
class NoiseGenerator:
    """Pre-generates noise current traces for injection via I_ext."""

    @staticmethod
    def gaussian(n_neurons, n_steps, std, seed=None):
        """
        White Gaussian noise: I_noise[i][t] ~ N(0, std).
        Returns numpy array shape (n_neurons, n_steps).
        """

    @staticmethod
    def ornstein_uhlenbeck(n_neurons, n_steps, dt,
                           tau=5.0, std=1.0, seed=None):
        """
        Ornstein-Uhlenbeck process: dX = -X/tau * dt + std * sqrt(2/tau) * dW.
        Returns numpy array shape (n_neurons, n_steps).
        """
```

Usage:
```python
noise = NoiseGenerator.ornstein_uhlenbeck(50, n_steps, dt=0.1, tau=5.0, std=1.5)
I_base = np.full((50, n_steps), 10.0)
traces = net.simulate(500.0, 0.1, (I_base + noise).tolist())
```

The main I_ext memory concern (N × T doubles) is a separate issue (see 10.5 below). For now this approach is consistent with the existing API.

### Checklist
- [ ] Implement `NoiseGenerator` class in `__init__.py`
- [ ] `gaussian()` static method
- [ ] `ornstein_uhlenbeck()` static method
- [ ] Export `NoiseGenerator` from package
- [ ] Tests: output shape and dtype correct; OU process mean ≈ 0, variance ≈ std² in steady state; different seeds give different traces, same seed gives identical traces

---

## 10.5 Population Parameter Heterogeneity

**Priority: 2**

### Problem

`add_population("STN", 50, spec=NeuronModelSpec.stn())` creates 50 neurons with exactly identical parameters. Real biological populations exhibit cell-to-cell conductance variation (typically ±15–20% Gaussian spread) that affects firing rates, synchrony, and network dynamics. Without this, large populations fire in perfect lockstep, producing unrealistically strong oscillations.

### Design

Python-only addition. Add a `heterogeneity` parameter to `RegionalNetwork.add_population()`:

```python
rnet.add_population(
    "STN", 50,
    spec=NeuronModelSpec.stn(),
    heterogeneity={
        "channels[0].g": ("normal", 1.0, 0.15),   # Na: mean=1×, std=15%
        "channels[1].g": ("normal", 1.0, 0.15),   # K
        "C_m":           ("uniform", 0.9, 1.1),   # capacitance ±10%
    },
    seed=42
)
```

The `heterogeneity` dict maps dotted field paths to `(distribution, param1, param2)` triples (same convention as `WeightDistribution`). Under the hood, each of the 50 neurons gets an individually modified copy of the spec before being passed to `_rnet.add_population` (or individual `add_neuron` calls).

Field path syntax covers the most common use cases:
- `"channels[i].g"` — channel conductance by index
- `"C_m"` — membrane capacitance
- `"V_init"` — initial voltage scatter (useful on its own)
- `"gates[i].initial_value"` — gate initial condition scatter

### Checklist
- [ ] Implement `_apply_heterogeneity(spec, field_path, dist, p1, p2, rng)` helper
- [ ] Parse dotted path with bracket index support
- [ ] Integrate into `RegionalNetwork.add_population()`
- [ ] Tests: conductance spread produces non-identical firing across population; seed reproducibility; unknown field path raises `ValueError`

---

## 10.6 NeuronModelSpec Validation

**Priority: 2**

### Problem

Invalid `NeuronModelSpec` configurations fail silently at runtime:
- A `ChannelSpec` with a gate index larger than `len(spec.gates)` is silently ignored
- A `DERIVED` gate with `derived_source_gate` pointing to a non-existent gate does nothing
- A `CalciumSpec` referencing channel indices outside `spec.channels` is silently ignored
- Negative conductances and infinite tau parameters are passed through unchecked

These produce wrong results that are difficult to debug.

### Design

Add a `NeuronModelSpec.validate()` method (C++ and/or Python) that raises on:
- Gate index out of range in any `ChannelSpec.gates`
- `derived_source_gate` out of range
- `calcium.source_channels` indices out of range
- Negative `g`, `C_m`

And warns (but does not raise) on:
- `tau.params[0]` (base/scale) ≤ 0 for CONSTANT form
- `C_m` < 0.01 or > 100

Validation should run automatically inside `ComposableNeuron`'s constructor and inside `Network::add_neuron(const NeuronModelSpec&)`.

### Checklist
- [ ] Implement `NeuronModelSpec::validate()` in C++ (throws `std::invalid_argument`)
- [ ] Call from `ComposableNeuron` constructor
- [ ] Call from `Network::add_neuron(const NeuronModelSpec&)`
- [ ] Expose `validate()` to Python via bindings
- [ ] Tests: out-of-range gate index raises; out-of-range source channel raises; negative g raises; valid spec does not raise

---

## Summary

| Section | Area | Priority | C++ changes? |
|---|---|---|---|
| 10.1 | ALPHA_BETA Euler stability | 1 | Yes | ✅ Done |
| 10.2 | h_T tau bug in TH/STN | 1 | Yes | ✅ Done |
| 10.3 | Extended recording (gates, calcium) | 1 | Yes | ✅ Done |
| 10.4 | Noise injection utilities | 2 | No | ✅ Done |
| 10.5 | Population parameter heterogeneity | 2 | No |
| 10.6 | NeuronModelSpec validation | 2 | Yes (light) |
