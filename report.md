# Methodology Report: Neural Simulation Library

---

## 1. Speed

The library achieves a ~55× speedup over the pure-Python benchmark (`simulate_network_model.py`) on an 80-neuron CTX-BG-TH network. The speedup comes from several layered optimisations across the simulation hot loop, memory layout, arithmetic, and the Python–C++ boundary.

### 1.1 C++ Core with pybind11 Bindings

The entire simulation runs in a compiled C++ binary built with scikit-build-core and CMake. Python is only involved at setup (building `RecordingConfig`, allocating output buffers) and teardown (reading results). No Python bytecode executes during the main loop.

### 1.2 Batched Neuron Pools (Eigen SIMD)

Rather than updating neurons one at a time via virtual dispatch, neurons of each type are grouped into homogeneous *pools* backed by `Eigen::ArrayXd`:

- **`HHPool`** — holds all HH neurons; gate variables (`m`, `h`, `n`) and membrane potentials are stored as flat Eigen arrays and updated in a single vectorised pass.
- **`ComposablePool`** — one pool per unique `NeuronModelSpec`; gate states and calcium are laid out in parallel arrays so that a single loop over `N_` neurons applies all gate updates without branching on type.

Eigen maps directly to CPU SIMD registers (SSE/AVX on x86), so the hot loop executes as SIMD fused-multiply-add instructions over the full population in a single call rather than N separate scalar multiplications.

### 1.3 Polynomial `fast_exp` Approximation

The HH rate equations (`α_m`, `β_m`, etc.) and composable gate kinetics require `exp()` at every timestep for every neuron. The library replaces the standard `std::exp` with a fully vectorised polynomial approximation:

```
exp(x) = exp(x/32)^32
```

The reduced argument `x/32` is evaluated with a degree-7 Taylor polynomial in Horner form. Five squarings reconstruct `exp(x)`. This delivers ~8 significant digits while executing as pure Eigen SIMD operations — every coefficient multiplication is a SIMD FMA over the entire pool. The approximation activates only when `N > 64` (where throughput gain outweighs startup cost); smaller populations use Eigen's full-precision `exp`.

### 1.4 Structure-of-Arrays (SoA) Synapse Layout

Synapse data is stored in a `SynArrays` struct of parallel vectors:

```
pre[], post[], weight[], E_syn[], g[], type[], delay[], ...
```

All synapse updates iterate sequentially over a flat index range — no pointer chasing, no cache misses from traversing linked-list style synapse objects. The layout allows the compiler to auto-vectorise the conductance decay loop.

### 1.5 Type-Separated Synapse Groups

After all synapses are added, they are sorted into four `SynapseGroups` index lists (exponential, alpha, double-exponential, kinetic). The inner update loop for each group is therefore branch-free: `exp_groups` contains only `SYN_EXP` entries, so the `if (type == SYN_EXP)` branch is always taken and the CPU branch predictor is never stalled.

### 1.6 Pre-Allocated Scratch Buffers

All working memory is allocated once before the simulation starts:

- Neuron pools pre-allocate `tmp_am_`, `tmp_bm_`, `tmp_exp_r_`, etc. at construction time.
- The network pre-allocates `I_syn_buffer_` and `V_cache_` vectors.
- No heap allocation occurs inside the hot loop.

### 1.7 Cached Decay Factors

For fixed-step simulations, the synapse decay terms `exp(-dt/tau_rise)` and `exp(-dt/tau_decay)` are constant. They are computed once and stored in `dexp_rise_decay[]` / `dexp_fall_decay[]`, replacing a `std::exp` call per synapse per timestep with a single multiply.

### 1.8 Zero-Copy Recording Buffers

The `simulate_into_buffers()` method accepts pre-allocated numpy arrays (`V_buf`, `gate_buf`, `g_syn_buf`, etc.) and writes recorded values directly into those buffers during the hot loop at every `interval` steps. No intermediate copy or allocation occurs during simulation. The Python layer pre-allocates the numpy arrays once using `np.zeros`, and the C++ loop fills them in place.

### 1.9 GIL-Free Input Transfer

The external current array is passed as a contiguous numpy array. The pybind11 binding copies it into the C++ `vector<vector<double>>` via `memcpy` before releasing Python control, ensuring the GIL is not held during any part of the simulation — which was previously the dominant source of slowdown when simulations ran concurrently in benchmarking scripts.

---

## 2. Flexibility

The library is designed to model any conductance-based neuron expressible as a set of Hodgkin-Huxley-style gate variables and ionic currents, without requiring any C++ modifications.

### 2.1 Composable Gate System (`GateSpec`)

Each gate is specified independently with four update forms:

| Form | Dynamics | Use case |
|------|----------|----------|
| `INF_TAU` | `dx/dt = (x∞(V) − x) / τ(V)` | Most biological gates |
| `ALPHA_BETA` | `dx/dt = α(V)·(1−x) − β(V)·x` | Classic HH Na/K |
| `INSTANT` | `x = x∞(V)` | Fast-activating gates |
| `DERIVED` | `x = a·(b + c·x_source)` | Inactivation linked to activation |

Gates may depend on membrane voltage (`VOLTAGE`) or intracellular calcium (`CALCIUM`), enabling calcium-activated channels.

### 2.2 Six Tau Parameterisations (`TauParams`)

The time-constant function `τ(V)` supports six functional forms selected at model-specification time:

| Form | Function |
|------|----------|
| `CONSTANT` | `τ = k` |
| `BOLTZMANN` | `τ = base + amp / (1 + exp(−(V−v½)/k))` |
| `DOUBLE_EXP_SUM` | `τ = base + a / (exp((V+v1)/s1) + exp(−(V+v2)/s2))` |
| `OFFSET_DOUBLE_EXP` | `τ = base + a1·exp(−((V+v1)/s1)²) + a2·exp(−((V+v2)/s2)²)` |
| `SCALED_EXP` | `τ = scale / cosh((V−v½)/(2k))` |
| `COMPOUND_AB` | `τ = 1 / (α(V) + β(V))` |

These six forms span the full space of tau functions encountered in published HH-type models, including the voltage-dependent T-current inactivation kinetics in thalamic relay cells.

### 2.3 Four Rate-Function Forms (`RateFuncParams`)

Alpha/beta rate functions follow one of four standard shapes used in published models: `LINEAR_OVER_EXP` (classic HH), `EXP_DECAY`, `LINEAR_OVER_EXPM1`, and `SIGMOID`.

### 2.4 Calcium Dynamics

The `CalciumSpec` enables an intracellular calcium ODE:

```
dCa/dt = −ε·I_Ca − Ca / K_Ca
```

When `use_nernst = True`, the calcium reversal potential `E_Ca` is recomputed each timestep via the Nernst equation using the current intracellular and fixed extracellular concentrations. The AHP channel type (`is_ahp = True`) implements calcium-dependent after-hyperpolarisation via `g_AHP = g · Ca / (Ca + k1)`.

### 2.5 Mixed Neuron-Type Networks

A single `Network` or `RegionalNetwork` may contain any combination of:

- Classic Hodgkin-Huxley neurons
- Five Izhikevich presets (Regular Spiking, Fast Spiking, Intrinsically Bursting, Chattering, Low-Threshold Spiking) plus fully custom Izhikevich parameters
- `ComposableNeuron` neurons specified by `NeuronModelSpec`

### 2.6 Multiple Synapse Types

Four synapse kinetic models are available: single-exponential decay, alpha function, double-exponential (rise + decay), and kinetic gate synapses. Kinetic synapses support three update forms: `TANH_GATE` (used for the striatal GABA model), `ALPHA_BETA`, and `BOLTZMANN_GATE`, plus an `MG_BLOCK` current form for NMDA voltage-dependent magnesium blockade. Preset receptor specs (`SynapseSpec.ampa()`, `.nmda()`, `.gaba_a()`) encode published kinetic parameters.

### 2.7 RegionalNetwork Population API

`RegionalNetwork` manages named populations and inter-population connectivity. Supported wiring patterns cover most common network topologies:

| Pattern | Description |
|---------|-------------|
| `ALL_TO_ALL` | Every source to every target |
| `ONE_TO_ONE` | Matched-index pairs |
| `SHIFTED` | Wrapped offset mapping |
| `RANDOM_SPARSE` | Bernoulli sampling with probability `p` |
| `RANDOM_PERMUTATION` | Bijective random assignment |

Weight distributions (`CONSTANT`, `UNIFORM`, `NORMAL`) and optional axonal delays are set per projection. All wiring is executed in C++ for large-scale networks.

### 2.8 Stimulators

`DBSStimulator` delivers biphasic charge-balanced pulses replicating clinical deep-brain stimulation protocols. `PulseStimulator` provides general pulse-train injection for any population. `NoiseInjector` adds Ornstein-Uhlenbeck noise to specified neurons. All stimulators inject current before the hot loop, requiring no changes to the C++ simulation core.

---

## 3. Readability

### 3.1 Builder Pattern for Model Construction

The `NeuronModel` builder wraps the low-level `NeuronModelSpec` struct behind a chainable Python API:

```python
model = NeuronModel("TH", C_m=1.0, V_init=-65.0)
m_idx = model.add_gate("m", update_form="instant", inf=Boltzmann(-37, 7))
h_idx = model.add_gate("h", update_form="inf_tau",
                        inf=Boltzmann(-41, -4),
                        tau=Tau.double_exp_sum(4.2, 0.15, 25.0, 10.5, 1000.0, 1.0))
model.add_channel("I_T", g=0.2, E_rev=120.0, gates=[(m_idx, 2), (h_idx, 1)])
model.add_leak(g=0.05, E_rev=-70.0)
model.set_calcium(epsilon=1e-4, K_Ca=15.0)
spec = model.to_spec()
```

Every `add_gate` and `add_channel` call returns an integer index, so gate references in channel definitions are explicit and unambiguous without requiring named lookups.

### 3.2 Named Parameter Helpers

Rather than filling raw `TauParams` or `BoltzmannParams` structs directly, the library provides expression-level helper classes:

- **`Boltzmann(v_half, k)`** — constructs a Boltzmann steady-state parameter
- **`Tau.constant(k)`**, **`Tau.boltzmann(...)`**, **`Tau.double_exp_sum(...)`**, **`Tau.scaled_exp(...)`**, **`Tau.compound_ab(...)`** — construct `TauParams` by named form
- **`RateFunc.linear_over_exp(...)`**, **`RateFunc.exp_decay(...)`**, etc.

These helpers make the model code self-documenting: the function name encodes the mathematical form, so a reader unfamiliar with the internal enum values can understand the kinetics from context.

### 3.3 Declarative Recording Configuration

The `RecordingConfig` class describes *what* to record using string metric names rather than flags or positional arguments:

```python
cfg = RecordingConfig(["spikes", "firing_rate", "ISI_cv"])
result = rnet.simulate(1000.0, 0.01, I_ext, record=cfg)
```

Named presets cover common workflows without requiring users to enumerate metrics manually:

| Preset | Metrics recorded |
|--------|-----------------|
| `RecordingConfig.voltage_only()` | `"V"` |
| `RecordingConfig.spikes_only()` | `"spikes"`, `"spike_count"`, `"firing_rate"` |
| `RecordingConfig.spike_stats()` | Above + `"ISI_mean"`, `"ISI_cv"` |
| `RecordingConfig.summary_metrics()` | `"spike_count"`, `"firing_rate"`, `"mean_V"` (no traces) |
| `RecordingConfig.all_neuron_metrics()` | All per-neuron and gate metrics |

The `interval` parameter subsamples the voltage trace for memory efficiency without changing the simulation timestep.

### 3.4 Dict-like Result Access

`MetricsResult` supports key-based access, containment testing, and iteration, so downstream code reads like data retrieval rather than method chains:

```python
spike_times = result["spikes"]     # list of ndarrays (ms)
rates = result["firing_rate"]      # ndarray (Hz)
if "gates" in result:
    h_gate = result["gates"][:, 1, :]
```

`PopulationMetricsResult` extends this to a two-level structure keyed first by population name and then by metric:

```python
gpi_spikes = result["GPi"]["spikes"]
stn_rate   = result["STN"]["firing_rate"]
```

### 3.5 Population-Level Network API

`RegionalNetwork` exposes a vocabulary that matches how computational neuroscientists describe circuits, rather than exposing raw neuron indices:

```python
rnet = RegionalNetwork()
rnet.add_population("STN",  n, NeuronModel.stn().to_spec())
rnet.add_population("GPe",  n, NeuronModel.gpe().to_spec())
rnet.connect("STN", "GPe",
             pattern="all_to_all",
             synapse=SynapseSpec.ampa(),
             weight=WeightDistribution.constant(0.3))
```

Population names propagate through the results, so `result["GPe"]["firing_rate"]` refers unambiguously to the correct population regardless of neuron indexing order inside the flat network.

### 3.6 Backward Compatibility

`simulate()` without a `record=` argument returns the same type it always did — a plain `ndarray` for `Network`, a `dict` of arrays for `RegionalNetwork` — so existing scripts require no changes to continue working.
