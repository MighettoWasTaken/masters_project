# Task 6: Voltage-Dependent GABA Kinetic Synapse

## Priority: 1 (Critical)

## Overview
Implement a voltage-dependent GABA kinetic synapse model used for intra-striatal inhibition in the benchmark. Unlike our existing spike-triggered synapses (exponential, alpha, double-exponential), this synapse has **continuously-varying** conductance driven by the presynaptic membrane potential — not by discrete spike events.

---

## 6.1 GABA Kinetic Model

The GABA synapse variable S evolves as:
```
dS/dt = Ggaba(V_pre) * (1 - S) - S / tau_i
```

Where:
- `S` is the synaptic gating variable (0 to 1)
- `V_pre` is the presynaptic neuron's membrane potential (continuous, not spike-triggered)
- `tau_i = 13 ms` is the decay time constant
- `Ggaba(V) = 2 * (1 + tanh(V / 4))` is the voltage-dependent opening rate

The resulting synaptic current on the postsynaptic neuron:
```
I_GABA = g_GABA * S * (V_post - E_GABA)
```
Where `E_GABA = -80 mV`.

### Key Differences from Existing Synapses
| Feature | Existing (Exp/Alpha/DblExp) | GABA Kinetic |
|---------|---------------------------|--------------|
| Trigger | Spike detection (threshold crossing) | Continuous V_pre |
| Conductance shape | Analytic waveform | ODE-driven (dS/dt) |
| State variable | g (decaying) | S (driven by V_pre) |
| Presynaptic access | Only needs spike times | Needs V_pre every step |

---

## 6.2 Benchmark Usage

In the benchmark, GABA synapses are used for:

### Cortex_I (inhibitory interneurons) → Striatum D1/D2
```python
# D2 striatum receives GABA from cortex inhibitory interneurons
S1c = S1c + dt * ((Ggaba(V5) * (1 - S1c)) - (S1c / tau_i))
# where V5 = vstr_indr (D2 striatum voltage)
# S1c is then permuted and used as:
Igaba5 = (ggaba / 4) * (V5 - Esyn[6]) * (S11cr + S12cr + S13cr + S14cr)
# S11cr = S1c[all], S12cr = S1c[bll], etc. (random permutation indices)
```

**Note:** In the benchmark, the GABA S variable is driven by the **postsynaptic** neuron's own voltage (V5/V6), not the presynaptic cortical interneuron voltage. This represents a simplified local inhibition model where nearby striatal neurons inhibit each other based on their own activity.

### Parameters
| Parameter | Value |
|-----------|-------|
| tau_i | 13 ms |
| g_GABA | 0.1 |
| E_GABA | -80 mV |
| Ggaba(V) | 2 * (1 + tanh(V/4)) |

---

## 6.3 C++ Implementation

### New Synapse Type

Add `SYN_GABA_KINETIC` to the existing SoA synapse system:

```cpp
// In Network:
enum SynType : uint8_t { SYN_EXP = 0, SYN_ALPHA = 1, SYN_DEXP = 2, SYN_GABA_KINETIC = 3 };

// New SoA fields for GABA kinetic synapses:
struct SynArrays {
    // ... existing fields ...

    // GABA kinetic-specific
    std::vector<double> gaba_S;       // synaptic gating variable
    std::vector<double> gaba_tau_i;   // decay time constant
};
```

### New Network Method

```cpp
void add_gaba_kinetic_synapse(size_t pre_idx, size_t post_idx, double weight,
                               double E_syn = -80.0, double tau_i = 13.0,
                               double delay = 0.0);
```

### Update Logic

In the synapse update loop, GABA kinetic synapses:
1. Read `V_pre` from the presynaptic neuron (or postsynaptic, depending on model variant)
2. Compute `Ggaba = 2 * (1 + tanh(V / 4))`
3. Update `S += dt * (Ggaba * (1 - S) - S / tau_i)`
4. Compute current: `I = weight * S * (V_post - E_syn)`

This must be integrated into the `update_synapses_grouped()` method with a new `SynapseGroups::gaba_kinetic` index list.

---

## 6.4 SynapseSpec Integration

Add GABA kinetic as a new SynapseSpec type for use with RegionalNetwork:

```cpp
struct SynapseSpec {
    enum class Type { EXPONENTIAL, ALPHA, DOUBLE_EXPONENTIAL, GABA_KINETIC };
    // ...
    double tau_i;  // GABA kinetic decay time

    static SynapseSpec gaba_kinetic(double E_syn = -80.0, double tau_i = 13.0);
};
```

### Python Usage
```python
# In RegionalNetwork
rn.connect("StrD2", "StrD2", "random_permutation",
           weight=0.025, delay=0.0,
           synapse=SynapseSpec.gaba_kinetic(E_syn=-80, tau_i=13))
```

---

## 6.5 Implementation Checklist
- [ ] Add `SYN_GABA_KINETIC` to SynType enum
- [ ] Add GABA-specific SoA fields (gaba_S, gaba_tau_i)
- [ ] Implement `Network::add_gaba_kinetic_synapse()`
- [ ] Implement GABA kinetic update in synapse grouped loop
- [ ] Add `gaba_kinetic` to SynapseGroups index list
- [ ] Add `GABA_KINETIC` to SynapseSpec::Type
- [ ] Implement `SynapseSpec::gaba_kinetic()` factory
- [ ] Update `add_synapse_from_spec()` in RegionalNetwork
- [ ] Python bindings for new synapse type
- [ ] Unit tests: S variable dynamics, steady-state at different V_pre, current sign/magnitude
- [ ] Integration test: GABA inhibition suppresses postsynaptic firing
