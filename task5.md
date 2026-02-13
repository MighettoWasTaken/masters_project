# Task 5: Composable Neuron Models (Ion Channel System)

## Priority: 1 (Critical)

## Overview
Build a flexible, composable ion channel system that lets users construct arbitrary HH-variant neuron models by combining parameterized gating functions, ion channels, and calcium dynamics. The system must follow our 3 guiding principles: **speed** (C++ Eigen vectorized pools), **flexibility** (arbitrary channel/gate combinations), **ease of use** (presets for common cases, concise Python API).

The benchmark CTX-BG-TH network uses 5 distinct neuron types (TH, STN, GPe/GPi, Striatum D1/D2) — each should be expressible as a composition of channels using this system, not hardcoded.

---

## 5.1 Architecture

Every conductance-based neuron follows:
```
C_m * dV/dt = -sum(I_channels) + I_syn + I_ext
```

Where each channel contributes: `I = g * product(gate^power) * (V - E_rev)`

And each gate variable evolves via one of several kinetic forms.

The system is built from 4 composable abstractions:

```
GateVariable     — kinetics for a single gating variable
IonChannel       — conductance + reversal + gate references
CalciumDynamics  — optional [Ca] state with Nernst equation
NeuronModel      — complete specification: C_m + channels + gates + calcium
```

All parameters are **data, not code** — described by parameter structs, not function pointers. This allows Eigen vectorization across neurons in a pool (same model template, different V values per neuron).

---

## 5.2 Gating Variable Kinetics

A gating variable has a **state update form** and a **dependency** (voltage or calcium).

### Update Forms

| Form | Equation | Use Case |
|------|----------|----------|
| `INF_TAU` | `dX/dt = scale * (x_inf - X) / tau_x` | Most gates (TH, STN, GPe/GPi) |
| `ALPHA_BETA` | `dX/dt = alpha*(1-X) - beta*X` | Striatum gates |
| `INSTANT` | `X = x_inf(V)` each step (no dynamics) | TH m_inf, p_inf; GPe m_inf, a_inf, s_inf |
| `DERIVED` | `X = f(other_gate)` | TH K gate: `0.75*(1-H)` |

The `scale` factor in INF_TAU handles GPe's scaled updates (`dN/dt = 0.1*(n_inf-N)/tau_n`, `dH/dt = 0.05*(h_inf-H)/tau_h`).

### Steady-State Function (`x_inf`)

Nearly all `x_inf` in the benchmark are **Boltzmann sigmoids**:
```
x_inf(V) = 1 / (1 + exp(-(V - v_half) / k))
```
- `v_half`: half-activation voltage
- `k`: slope (positive = activating, negative = inactivating)

This single parameterized form covers: `th_minf`, `th_hinf`, `th_pinf`, `th_rinf`, `gpe_ainf`, `gpe_hinf`, `gpe_minf`, `gpe_ninf`, `gpe_rinf`, `gpe_sinf`, and all 11 STN `x_inf` functions.

### Time Constant Function (`tau_x`)

Six forms observed in the benchmark:

| Form | Equation | Example |
|------|----------|---------|
| `CONSTANT` | `tau = value` | STN D2 (tau=130), R (tau=2); GPe R (tau=30) |
| `BOLTZMANN` | `tau = a + b / (1 + exp((V - v_half) / k))` | GPe tau_h, tau_n; STN tau_m, tau_a |
| `DOUBLE_EXP_SUM` | `tau = a / (exp((V-c1)/k1) + exp((V-c2)/k2))` | STN tau_h, tau_n, tau_q |
| `OFFSET_DOUBLE_EXP` | `tau = a + b / (exp((V-c1)/k1) + exp((V-c2)/k2))` | STN tau_b, tau_c, tau_d1, tau_p |
| `SCALED_EXP` | `tau = a * (b + exp((V-c)/k))` | TH tau_r |
| `COMPOUND_AB` | `tau = 1 / (f1(V) + f2(V))` | TH tau_h (where f1=exp, f2=sigmoid) |

### Alpha-Beta Rate Functions (Striatum)

Three alpha/beta sub-forms observed:
```
Type 1: f(V) = A*(V+B) / (1 - exp(-(V+B)/C))    — alpham, alphan, alphap
Type 2: f(V) = A * exp((V+B) / C)                 — alphah, betan
Type 3: f(V) = A*(V+B) / (exp((V+B)/C) - 1)      — betam, betap
Type 4: f(V) = A / (1 + exp((V+B) / C))           — betah
```

### Dependency

Most gates depend on **voltage**. The STN CaK channel's R gate depends on **calcium concentration**:
```
stn_rinf([Ca]) = 1 / (1 + exp(-([Ca] - 0.17) / 0.08))
```
(Same Boltzmann form, but input is [Ca] not V.)

---

## 5.3 Ion Channel

An ion channel specifies:
- `g`: maximal conductance
- `E_rev`: reversal potential (constant, or `CALCIUM_NERNST` for dynamic E_Ca)
- `gates`: list of `(gate_name, power)` pairs

**Current:** `I = g * product(gate_i ^ power_i) * (V - E_rev)`

Special cases:
- **Leak channel**: no gates, just `I = g * (V - E)`
- **AHP channel**: uses calcium-dependent gate `Ca / (Ca + k1)` instead of voltage gate
- **Derived gate**: TH K channel uses `0.75*(1-H)` as a gate expression

---

## 5.4 Calcium Dynamics

Optional calcium tracking with two modes:

### Simple Decay (GPe/GPi)
```
d[Ca]/dt = epsilon * (-I_Ca - I_T - K_Ca * Ca)
```
- `epsilon`: scale factor (1e-4 for GPe/GPi)
- `K_Ca`: decay rate
- Source channels: list of channels contributing calcium current

### Nernst Equation (STN)
```
d[Ca]/dt = -alpha * (I_L + I_T) - K_Ca * [Ca]
E_Ca = (R*T)/(z*F) * ln(Ca_o / [Ca])
```
- `alpha = 1/(z*F)`, `z=2`, `F=96485`, `R=8314`, `T=298`
- `Ca_o = 2000`, `[Ca]_init = 0.005`
- E_Ca is recomputed each step and used by channels with `E_rev = CALCIUM_NERNST`

---

## 5.5 C++ Design

### Core Structs

```cpp
// Parameterized Boltzmann: 1 / (1 + exp(-(x - v_half) / k))
struct BoltzmannParams {
    double v_half;
    double k;  // positive = activating, negative = inactivating
};

// Tau function parameters (tagged union or variant)
struct TauParams {
    enum class Form {
        CONSTANT,          // tau = value
        BOLTZMANN,         // tau = a + b / (1 + exp((V - v_half) / k))
        DOUBLE_EXP_SUM,    // tau = a / (exp((V-c1)/k1) + exp((V-c2)/k2))
        OFFSET_DOUBLE_EXP, // tau = a + b / (exp((V-c1)/k1) + exp((V-c2)/k2))
        SCALED_EXP,        // tau = a * (b + exp((V-c)/k))
        COMPOUND_AB        // tau = 1 / (f1(V) + f2(V))
    };
    Form form;
    double params[8];  // interpretation depends on form
};

// Alpha/beta rate function parameters
struct RateFuncParams {
    enum class Form {
        LINEAR_OVER_EXP,   // A*(V+B) / (1 - exp(-(V+B)/C))
        EXP_DECAY,         // A * exp((V+B) / C)
        LINEAR_OVER_EXPM1, // A*(V+B) / (exp((V+B)/C) - 1)
        SIGMOID            // A / (1 + exp((V+B) / C))
    };
    Form form;
    double A, B, C;
};

// Gate variable specification
struct GateSpec {
    enum class UpdateForm { INF_TAU, ALPHA_BETA, INSTANT, DERIVED };
    enum class Dependency { VOLTAGE, CALCIUM };

    std::string name;
    UpdateForm update_form;
    Dependency dependency = Dependency::VOLTAGE;
    double scale = 1.0;        // scale factor for INF_TAU update
    double initial_value = 0.0;

    // INF_TAU / INSTANT: Boltzmann inf + tau params
    BoltzmannParams inf;
    TauParams tau;

    // ALPHA_BETA: rate function params
    RateFuncParams alpha;
    RateFuncParams beta;

    // DERIVED: expression index (references another gate)
    int derived_source_gate = -1;  // index of source gate
    double derived_a = 1.0;        // X = a * (b + c * source_gate)^power
    double derived_b = 0.0;
    double derived_c = 1.0;
};

// Channel specification
struct ChannelSpec {
    std::string name;
    double g;                              // maximal conductance
    double E_rev;                          // reversal potential
    bool use_calcium_nernst = false;       // use dynamic E_Ca instead
    std::vector<std::pair<int, int>> gates; // (gate_index, power)
    bool is_ahp = false;                   // uses Ca/(Ca+k1) gating
    double ahp_k1 = 0.0;                  // for AHP channels
};

// Calcium dynamics specification
struct CalciumSpec {
    bool enabled = false;
    bool use_nernst = false;              // compute E_Ca via Nernst
    double epsilon = 1e-4;                // scale factor
    double K_Ca = 15.0;                   // decay rate
    double Ca_init = 0.1;                 // initial [Ca]
    double Ca_o = 2000.0;                 // extracellular [Ca] (for Nernst)
    double z = 2.0, F = 96485.0, R = 8314.0, T = 298.0;  // Nernst constants
    std::vector<int> source_channels;     // indices of channels contributing to d[Ca]/dt
};

// Complete neuron model specification
struct NeuronModelSpec {
    std::string name;
    double C_m = 1.0;
    std::vector<GateSpec> gates;
    std::vector<ChannelSpec> channels;
    CalciumSpec calcium;
};
```

### CustomPool (Eigen Vectorized)

```cpp
class CustomPool {
public:
    CustomPool(const NeuronModelSpec& model, size_t num_neurons);

    void step(double dt,
              const Eigen::Ref<const Eigen::ArrayXd>& I_ext,
              const Eigen::Ref<const Eigen::ArrayXd>& I_syn);

    Eigen::ArrayXd& V() { return V_; }
    // ...

private:
    NeuronModelSpec model_;
    size_t n_;

    Eigen::ArrayXd V_;
    std::vector<Eigen::ArrayXd> gate_states_;  // one ArrayXd per gate
    Eigen::ArrayXd Ca_;                         // calcium (if enabled)
    Eigen::ArrayXd E_Ca_;                       // dynamic reversal (if Nernst)

    // Pre-allocated temporaries
    Eigen::ArrayXd I_total_;
    std::vector<Eigen::ArrayXd> inf_cache_;
    std::vector<Eigen::ArrayXd> tau_cache_;
};
```

**step() algorithm:**
1. For each gate: compute `x_inf(V)` and `tau_x(V)` (or `alpha(V)`, `beta(V)`) using parameterized form — all vectorized over N neurons via Eigen
2. Update gate states (Euler: `X += dt * scale * (x_inf - X) / tau` or `X += dt * (alpha*(1-X) - beta*X)`)
3. Compute derived gates (e.g., `0.75*(1-H)`)
4. For each channel: compute `g * product(gates^powers) * (V - E_rev)`, accumulate to I_total
5. Update V: `V += dt * (-I_total + I_ext + I_syn) / C_m`
6. Update calcium if enabled: `Ca += dt * calcium_rhs(...)`
7. Recompute E_Ca if Nernst enabled

The loop over gates/channels (5-10 iterations) is small; the inner vectorization over N neurons (100-1000) is where SIMD matters. This should be nearly as fast as the hardcoded HHPool.

### Integration with Network

`CustomPool` plugs into `Network::simulate()` alongside `HHPool` and `IzPool`. The `Network` detects neurons sharing the same `NeuronModelSpec` and groups them into a pool.

```cpp
// In Network:
enum class NeuronType {
    HH,
    IZHIKEVICH_RS, IZHIKEVICH_FS, IZHIKEVICH_IB, IZHIKEVICH_CH, IZHIKEVICH_LTS,
    IZHIKEVICH_CUSTOM,
    CUSTOM  // ← new: uses NeuronModelSpec
};

// New overload in Network and RegionalNetwork:
size_t add_neuron(const NeuronModelSpec& spec);
void add_population(const std::string& name, size_t count,
                    const NeuronModelSpec& spec);
```

---

## 5.6 Python API

### Building Custom Models

```python
from hodgkin_huxley import (
    NeuronModel, Gate, Tau, RateFunc, Channel, CalciumDynamics,
    Boltzmann
)

# --- Thalamic neuron ---
th_model = NeuronModel("TH", C_m=1.0)

# Gates (state variables with dynamics)
th_model.add_gate("H",
    inf=Boltzmann(v_half=-41, k=4),
    tau=Tau.compound_ab(
        f1=RateFunc.exp_decay(A=0.128, B=-46, C=18),   # ah
        f2=RateFunc.sigmoid(A=4, B=-23, C=5)            # bh
    ))
th_model.add_gate("R",
    inf=Boltzmann(v_half=-84, k=4),
    tau=Tau.scaled_exp(a=0.15, b=28, v0=-25, k=10.5))

# Channels
th_model.add_leak(g=0.05, E=-70)
th_model.add_channel("Na", g=3, E=50,
    gates=[("H", 1)],
    instant_gates=[(Boltzmann(-37, 7), 3)])  # m_inf^3
th_model.add_channel("K", g=5, E=-75,
    derived_gates=[("H", "0.75*(1-x)", 4)])  # (0.75*(1-H))^4
th_model.add_channel("T", g=5, E=0,
    gates=[("R", 1)],
    instant_gates=[(Boltzmann(-60, 6.2), 2)])  # p_inf^2
```

### Using Presets

```python
# One-liner presets for benchmark models
th_model  = NeuronModel.thalamic()
stn_model = NeuronModel.stn()
gpe_model = NeuronModel.gpe()
gpi_model = NeuronModel.gpi()       # same as gpe
str_model = NeuronModel.striatum(pd=0)  # pd=0 healthy, pd=1 PD

# Use in RegionalNetwork
rn = RegionalNetwork()
rn.add_population("TH",  10, model=th_model)
rn.add_population("STN", 10, model=stn_model)
rn.add_population("GPe", 10, model=gpe_model)
```

### Alpha-Beta Form (Striatum)

```python
str_model = NeuronModel("Striatum", C_m=1.0)

str_model.add_gate("m", form="alpha_beta",
    alpha=RateFunc.linear_over_exp(A=0.32,   B=54, C=4),
    beta =RateFunc.linear_over_expm1(A=0.28, B=27, C=5))
str_model.add_gate("h", form="alpha_beta",
    alpha=RateFunc.exp_decay(A=0.128, B=-50, C=18),
    beta =RateFunc.sigmoid(A=4, B=-27, C=5))
str_model.add_gate("n", form="alpha_beta",
    alpha=RateFunc.linear_over_exp(A=0.032, B=52, C=5),
    beta =RateFunc.exp_decay(A=0.5, B=-57, C=40))
str_model.add_gate("p", form="alpha_beta",
    alpha=RateFunc.linear_over_exp(A=3.209e-4, B=30, C=9),
    beta =RateFunc.linear_over_expm1(A=-3.209e-4, B=30, C=9))

str_model.add_leak(g=0.1, E=-67)
str_model.add_channel("Na", g=100, E=50,   gates=[("m", 3), ("h", 1)])
str_model.add_channel("K",  g=80,  E=-100, gates=[("n", 4)])
str_model.add_channel("M",  g=1*(2.6-1.1*pd), E=-100, gates=[("p", 1)])
```

### Calcium Dynamics (STN)

```python
stn_model = NeuronModel("STN", C_m=1.0)
# ... gates and channels ...

stn_model.set_calcium(
    mode="nernst",           # dynamic E_Ca via Nernst equation
    Ca_init=0.005,
    Ca_o=2000,
    K_Ca=2e-3,
    source_channels=["L_Ca", "T_Ca"]  # channels that contribute to d[Ca]/dt
)
# Channels using E_Ca:
stn_model.add_channel("L_Ca", g=15, E="calcium",  # uses dynamic E_Ca
    gates=[("C", 2), ("D1", 1), ("D2", 1)])
stn_model.add_channel("T_Ca", g=5,  E="calcium",
    gates=[("P", 2), ("Q", 1)])
# CaK channel with calcium-dependent gate:
stn_model.add_gate("R_ca", inf=Boltzmann(v_half=0.17, k=0.08),
    tau=Tau.constant(2), dependency="calcium")
stn_model.add_channel("CaK", g=1, E=-90, gates=[("R_ca", 2)])
```

---

## 5.7 Benchmark Model Reference

All 5 neuron types from the benchmark should be expressible (and provided as presets):

### TH — 2 dynamic gates, 4 channels
| Channel | g | E | Gates |
|---------|---|---|-------|
| Leak | 0.05 | -70 | — |
| Na | 3 | 50 | m_inf(V)^3 * H |
| K | 5 | -75 | (0.75*(1-H))^4 |
| T_Ca | 5 | 0 | p_inf(V)^2 * R |

### STN — 11 dynamic gates + calcium, 7 channels
| Channel | g | E | Gates |
|---------|---|---|-------|
| Leak | 0.35 | -60 | — |
| Na | 49 | 60 | M^3 * H |
| K | 57 | -90 | N^4 |
| A | 5 | -90 | A^2 * B |
| L_Ca | 15 | E_Ca(Nernst) | C^2 * D1 * D2 |
| T_Ca | 5 | E_Ca(Nernst) | P^2 * Q |
| CaK | 1 | -90 | R_ca^2 |

### GPe/GPi — 3 dynamic gates + calcium, 6 channels
| Channel | g | E | Gates |
|---------|---|---|-------|
| Leak | 0.1 | -65 | — |
| Na | 120 | 55 | m_inf(V)^3 * H |
| K | 30 | -80 | N^4 |
| T_Ca | 0.5 | 120 | a_inf(V)^3 * R |
| Ca | 0.15 | 120 | s_inf(V)^2 |
| AHP | 10 | -80 | Ca/(Ca+k1) |

GPe H: scale=0.05, N: scale=0.1 (slower kinetics)

### Striatum D1/D2 — 4 alpha-beta gates, 4 channels
| Channel | g | E | Gates |
|---------|---|---|-------|
| Leak | 0.1 | -67 | — |
| Na | 100 | 50 | m^3 * h |
| K | 80 | -100 | n^4 |
| M | g_M_eff | -100 | p |

Where `g_M_eff = (2.6 - 1.1 * pd) * g_M` (PD-dependent modulation).

---

## 5.8 Gating Function Reference

### All Boltzmann Parameters (x_inf)

| Function | v_half | k | Region |
|----------|--------|---|--------|
| th_minf | -37 | 7 | TH |
| th_hinf | -41 | -4 | TH |
| th_pinf | -60 | 6.2 | TH |
| th_rinf | -84 | -4 | TH |
| gpe_minf | -37 | 10 | GPe/GPi |
| gpe_ninf | -50 | 14 | GPe/GPi |
| gpe_hinf | -58 | -12 | GPe/GPi |
| gpe_ainf | -57 | 2 | GPe/GPi |
| gpe_rinf | -70 | -2 | GPe/GPi |
| gpe_sinf | -35 | 2 | GPe/GPi |
| stn_minf | -40 | 8 | STN |
| stn_hinf | -45.5 | -6.4 | STN |
| stn_ninf | -41 | 14 | STN |
| stn_ainf | -45 | 14.7 | STN |
| stn_binf | -90 | -7.5 | STN |
| stn_cinf | -30.6 | 5 | STN |
| stn_d1inf | -60 | -7.5 | STN |
| stn_d2inf | 0.1 | -0.02 | STN |
| stn_pinf | -56 | 6.7 | STN |
| stn_qinf | -85 | -5.8 | STN |
| stn_rinf | 0.17 | 0.08 | STN (Ca-dep) |

### All Tau Parameters

| Function | Form | Parameters |
|----------|------|------------|
| TH tau_h | COMPOUND_AB | f1: ExpDecay(0.128,-46,18), f2: Sigmoid(4,-23,5) |
| TH tau_r | SCALED_EXP | a=0.15, b=28, v0=-25, k=10.5 |
| GPe tau_h | BOLTZMANN | a=0.05, b=0.27, v_half=-40, k=12 |
| GPe tau_n | BOLTZMANN | a=0.05, b=0.27, v_half=-40, k=12 |
| GPe tau_r | CONSTANT | 30 |
| STN tau_m | BOLTZMANN | a=0.2, b=3, v_half=-53, k=0.7 |
| STN tau_h | DOUBLE_EXP_SUM | a=24.5, c1=-50, k1=15, c2=-50, k2=-16 |
| STN tau_n | DOUBLE_EXP_SUM | a=11, c1=-40, k1=40, c2=-40, k2=-50 |
| STN tau_a | BOLTZMANN | a=1, b=1, v_half=-40, k=0.5 |
| STN tau_b | OFFSET_DOUBLE_EXP | a=0, b=200, c1=-60, k1=30, c2=-40, k2=-10 |
| STN tau_c | OFFSET_DOUBLE_EXP | a=45, b=10, c1=-27, k1=20, c2=-50, k2=-15 |
| STN tau_d1 | OFFSET_DOUBLE_EXP | a=400, b=500, c1=-40, k1=15, c2=-20, k2=-20 |
| STN tau_d2 | CONSTANT | 130 |
| STN tau_p | OFFSET_DOUBLE_EXP | a=5, b=0.33, c1=-27, k1=10, c2=-102, k2=-15 |
| STN tau_q | DOUBLE_EXP_SUM | a=400, c1=-50, k1=15, c2=-50, k2=-16 |
| STN tau_r | CONSTANT | 2 |

### Striatum Alpha/Beta Parameters

| Function | Form | A | B | C |
|----------|------|---|---|---|
| alpha_m | LINEAR_OVER_EXP | 0.32 | 54 | 4 |
| beta_m | LINEAR_OVER_EXPM1 | 0.28 | 27 | 5 |
| alpha_h | EXP_DECAY | 0.128 | -50 | 18 |
| beta_h | SIGMOID | 4 | -27 | 5 |
| alpha_n | LINEAR_OVER_EXP | 0.032 | 52 | 5 |
| beta_n | EXP_DECAY | 0.5 | -57 | 40 |
| alpha_p | LINEAR_OVER_EXP | 3.209e-4 | 30 | 9 |
| beta_p | LINEAR_OVER_EXPM1 | -3.209e-4 | 30 | 9 |

---

## 5.9 Implementation Checklist

### C++ Core
- [ ] Implement `BoltzmannParams`, `TauParams`, `RateFuncParams` structs
- [ ] Implement `GateSpec`, `ChannelSpec`, `CalciumSpec`, `NeuronModelSpec` structs
- [ ] Implement parameterized Boltzmann evaluation (Eigen vectorized)
- [ ] Implement all 6 tau forms (Eigen vectorized)
- [ ] Implement all 4 alpha/beta rate function forms (Eigen vectorized)
- [ ] Implement `CustomPool` with vectorized step()
- [ ] Implement calcium dynamics (simple decay + Nernst modes)
- [ ] Implement derived gates and AHP channel support
- [ ] Register `CUSTOM` NeuronType in Network
- [ ] Add `add_neuron(NeuronModelSpec)` to Network and RegionalNetwork
- [ ] Integrate CustomPool into `Network::simulate()` pool dispatch

### Presets
- [ ] Implement `NeuronModelSpec::thalamic()` preset
- [ ] Implement `NeuronModelSpec::stn()` preset
- [ ] Implement `NeuronModelSpec::gpe()` preset
- [ ] Implement `NeuronModelSpec::striatum(double pd)` preset

### Python Bindings
- [ ] Bind all parameter structs (BoltzmannParams, TauParams, RateFuncParams, etc.)
- [ ] Bind NeuronModelSpec with factory methods
- [ ] Bind preset factories
- [ ] Python NeuronModel builder class with ergonomic API
- [ ] Python helper classes: Gate, Tau, RateFunc, Channel, Boltzmann

### Tests
- [ ] Unit test each gating function form against benchmark values
- [ ] Unit test each tau form against benchmark values
- [ ] Unit test alpha/beta forms against benchmark values
- [ ] Test CustomPool with TH model: correct resting state, spiking under current injection
- [ ] Test CustomPool with STN model: calcium dynamics, Nernst E_Ca
- [ ] Test CustomPool with GPe model: AHP current, scaled kinetics
- [ ] Test CustomPool with Striatum model: alpha-beta kinetics, M-current modulation
- [ ] Test preset factories produce correct parameters
- [ ] Test integration with RegionalNetwork
- [ ] Verification: compare firing patterns against benchmark output for each neuron type
