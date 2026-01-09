# Implementation Tasks for CTX-BG-TH Network Model

This document outlines the features required to recreate the rat basal ganglia-thalamo-cortical network model used for Parkinson's disease research (`simulate_network_model_CL_fast.py`).

---

## Priority 1: Core Neuron Models

### 1.1 Izhikevich Neuron Model
The cortical excitatory and inhibitory neurons use the Izhikevich model, not Hodgkin-Huxley.

**Equations:**
```
dv/dt = 0.04*v^2 + 5*v + 140 - u + I
du/dt = a*(b*v - u)

if v >= 30 mV:
    v = c
    u = u + d
```

**Parameters needed:**
- Regular spiking (RS): a=0.02, b=0.2, c=-65, d=8
- Fast spiking (FS): a=0.1, b=0.2, c=-65, d=2

**Implementation:**
- [ ] Create `IzhikevichNeuron` class in C++
- [ ] Implement `step()` method with spike reset logic
- [ ] Add Python bindings
- [ ] Unit tests

---

### 1.2 Additional Ion Channels

The benchmark uses several ion channels not present in standard HH. These should be implemented as modular components.

#### 1.2.1 T-type Calcium Current (I_T)
Low-threshold calcium current for thalamic and GPe/GPi neurons.

```
I_T = g_T * p^2 * r * (V - E_T)
```

- [ ] Implement T-type channel with p (activation) and r (inactivation) gates
- [ ] Different kinetics for thalamic vs GPe/GPi variants

#### 1.2.2 L-type Calcium Current (I_L)
High-voltage activated calcium current for STN neurons.

```
I_L = g_L * c^2 * d1 * d2 * (V - E_Ca)
```

- [ ] Implement L-type channel with c, d1, d2 gating variables
- [ ] E_Ca computed dynamically from Nernst equation

#### 1.2.3 A-type Potassium Current (I_A)
Fast transient potassium current for STN neurons.

```
I_A = g_A * a^2 * b * (V - E_K)
```

- [ ] Implement A-type channel with a (activation) and b (inactivation) gates

#### 1.2.4 M-type Potassium Current (I_M)
Slow muscarinic potassium current for striatal neurons.

```
I_M = g_M * p * (V - E_M)
```

- [ ] Implement M-current with single p gate
- [ ] E_M = -100 mV

#### 1.2.5 Calcium-dependent Potassium Currents (I_AHP, I_CaK)
After-hyperpolarization currents dependent on intracellular calcium.

```
I_AHP = g_AHP * (V - E_K) * (Ca / (Ca + k1))
I_CaK = g_CaK * r^2 * (V - E_K)
```

- [ ] Implement AHP current for GPe/GPi
- [ ] Implement CaK current for STN
- [ ] Both require calcium dynamics (see 1.3)

---

### 1.3 Calcium Dynamics

Intracellular calcium concentration tracking required for calcium-dependent currents.

**Equations:**
```
d[Ca]/dt = -alpha * (I_Ca + I_T) - K_Ca * [Ca]

E_Ca = (RT/zF) * ln([Ca]_o / [Ca]_i)  # Nernst equation
```

**Implementation:**
- [ ] Add `calcium_concentration` state variable to neurons
- [ ] Implement calcium accumulation from I_L and I_T
- [ ] Implement calcium decay (pump/buffer)
- [ ] Dynamic E_Ca calculation via Nernst equation
- [ ] Parameters: alpha, K_Ca, [Ca]_o (extracellular)

---

### 1.4 Configurable Neuron Parameters

Different brain regions use different kinetics. Need region-specific parameter sets.

| Region | Key Differences |
|--------|----------------|
| Thalamus | g_Na=3, g_K=5, g_L=0.05, T-type Ca |
| STN | g_Na=49, g_K=57, L-type Ca, A-current, 11 gating vars |
| GPe/GPi | g_Na=120, g_K=30, T-type Ca, AHP, Ca dynamics |
| Striatum | g_Na=100, g_K=80, M-current |

- [ ] Create preset parameter structures for each region
- [ ] Allow custom gating kinetics (alpha/beta or inf/tau functions)

---

## Priority 2: Synaptic Models

### 2.1 Alpha-function Synapse

Current implementation uses simple exponential decay. Benchmark uses alpha-function.

**Equation:**
```
g(t) = g_peak * (t/tau) * exp(1 - t/tau)   for t >= 0
```

**Implementation:**
- [ ] Add `SynapseType` enum: `EXPONENTIAL`, `ALPHA`, `DOUBLE_EXPONENTIAL`
- [ ] Implement alpha-function conductance calculation
- [ ] Spike time tracking per synapse

---

### 2.2 Double-exponential Synapse

Used for AMPA and NMDA synapses with distinct rise and decay times.

**Equation:**
```
g(t) = g_peak * f * (exp(-t/tau_d) - exp(-t/tau_r))

f = 1 / (exp(-t_peak/tau_d) - exp(-t_peak/tau_r))  # normalization
t_peak = (tau_d * tau_r)/(tau_d - tau_r) * ln(tau_d/tau_r)
```

**Parameters from benchmark:**
| Synapse | tau_r (ms) | tau_d (ms) |
|---------|-----------|-----------|
| STN-GPe AMPA | 0.4 | 2.5 |
| STN-GPe NMDA | 2.0 | 67 |
| GPe-STN | 0.4 | 7.7 |
| Cor-STN AMPA | 0.5 | 2.49 |
| Cor-STN NMDA | 2.0 | 90 |

- [ ] Implement double-exponential synapse model
- [ ] Precompute normalization factor

---

### 2.3 Synaptic Delays

Different pathways have different conduction delays.

| Pathway | Delay (ms) |
|---------|-----------|
| TH → Cortex | 5 |
| STN → GPe | 2 |
| STN → GPi | 1.5 |
| GPe → STN | 4 |
| GPe → GPi | 3 |
| GPe → GPe | 1 |
| GPi → TH | 5 |
| Striatum → GPe | 5 |
| Striatum → GPi | 4 |
| Cortex → Striatum | 5.1 |
| Cortex → STN | 5.9 |

- [ ] Add `delay` parameter to `Synapse` struct
- [ ] Implement delay buffer (circular buffer or spike queue)
- [ ] Delay resolution should match dt

---

### 2.4 Receptor-specific Synapses

Different receptor types with different reversal potentials and kinetics.

| Receptor | E_syn (mV) | Notes |
|----------|-----------|-------|
| GABA_A (GPe,GPi) | -85 | Inhibitory |
| GABA_A (Striatum) | -80 | Inhibitory |
| AMPA | 0 | Fast excitatory |
| NMDA | 0 | Slow excitatory, Mg²⁺ block (optional) |

- [ ] Add receptor type to synapse model
- [ ] Store E_syn per synapse (already exists)
- [ ] Consider NMDA voltage-dependent Mg²⁺ block for future

---

## Priority 3: Network Architecture

### 3.1 Population/Region Abstraction

Group neurons by brain region for easier connectivity specification.

```cpp
class Population {
    std::vector<NeuronBase*> neurons;
    std::string name;  // "STN", "GPe", etc.
};

class RegionalNetwork {
    std::map<std::string, Population> populations;
    void connect(string src, string dst, ConnectivityPattern, SynapseParams);
};
```

- [ ] Design population abstraction
- [ ] Implement bulk connectivity methods
- [ ] Python bindings for population-level operations

---

### 3.2 Connectivity Patterns

The benchmark uses various connectivity patterns.

**Patterns needed:**
- All-to-all (with optional self-connection exclusion)
- One-to-one (same index)
- Random sparse (probability-based)
- Shifted/rolled connections (lateral inhibition)
- Random permutation-based

- [ ] Implement `ConnectivityPattern` class/enum
- [ ] Support weight randomization (uniform, normal)
- [ ] Support sparse random connectivity with probability parameter

---

### 3.3 Heterogeneous Initial Conditions

Neurons start with randomized initial states.

```python
v1 = -62 + np.random.normal(loc=0, scale=5, size=(n, 1))
```

- [ ] Add method to randomize initial conditions
- [ ] Support setting distribution parameters (mean, std)

---

## Priority 4: Stimulation Module

### 4.1 DBS Pulse Generation

Deep brain stimulation with configurable parameters.

**Parameters:**
- Frequency (Hz): 0-200 Hz typical
- Amplitude (µA/cm²)
- Pulse width (ms)
- Target region (STN or GPi)

**Implementation:**
```cpp
class DBSStimulator {
    double frequency;
    double amplitude;
    double pulse_width;

    std::vector<double> generate(double duration, double dt);
};
```

- [ ] Implement DBS pulse train generator
- [ ] Support for biphasic pulses (future)
- [ ] Python bindings

---

### 4.2 Cortical Stimulation

Brief current pulses for cortical stimulation experiments.

```python
Iappco[int(1000/dt):int((1000+0.3)/dt)] = 350  # 350 µA/cm² for 0.3 ms at t=1000ms
```

- [ ] Implement pulse injection helper
- [ ] Support for arbitrary stimulus waveforms

---

## Priority 5: Analysis Tools (Optional)

### 5.1 Spike Detection
- [ ] Threshold-based spike detection
- [ ] Return spike times per neuron

### 5.2 Spectral Analysis
- [ ] Multi-taper spectrum for point processes (mtspectrumpt)
- [ ] Beta-band power calculation (7-35 Hz)

---

## Implementation Order Recommendation

1. **Phase 1 - Neuron Diversity**
   - Izhikevich neuron (required for cortex)
   - Modular ion channel system
   - Calcium dynamics

2. **Phase 2 - Synaptic Enhancements**
   - Alpha-function synapses
   - Double-exponential synapses
   - Synaptic delays

3. **Phase 3 - Network Structure**
   - Population abstraction
   - Connectivity patterns
   - DBS stimulation

4. **Phase 4 - Validation**
   - Reproduce benchmark results
   - Performance optimization

---

## Notes

- The benchmark uses **forward Euler** integration exclusively (dt=0.01 ms)
- Numba JIT is used for performance in the Python version
- State persistence between simulation runs is supported (for continuous simulations)
- The model has ~10 neurons per region (n=10 default)
