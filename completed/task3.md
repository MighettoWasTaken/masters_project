# Task 3: Synaptic Model Enhancements

## Priority: 2 (Synaptic)

## Overview
Enhance synapse models to support realistic kinetics and delays.

---

## 3.1 Synapse Types

### Current: Exponential Decay
```
g(t) = g * exp(-t/tau)
```

### Alpha-function
```
g(t) = g_peak * (t/tau) * exp(1 - t/tau)
```
Smooth rise and fall, peaks at t = tau.

### Double-exponential
```
g(t) = g_peak * f * (exp(-t/tau_d) - exp(-t/tau_r))
```
Used for AMPA/NMDA with distinct rise/decay times.

**Benchmark Parameters:**
| Synapse | tau_r (ms) | tau_d (ms) |
|---------|-----------|-----------|
| STN-GPe AMPA | 0.4 | 2.5 |
| STN-GPe NMDA | 2.0 | 67 |
| GPe-STN | 0.4 | 7.7 |
| Cor-STN AMPA | 0.5 | 2.49 |
| Cor-STN NMDA | 2.0 | 90 |

---

## 3.2 Synaptic Delays
Different pathways have different conduction delays.

| Pathway | Delay (ms) |
|---------|-----------|
| TH -> Cortex | 5 |
| STN -> GPe | 2 |
| STN -> GPi | 1.5 |
| GPe -> STN | 4 |
| GPe -> GPi | 3 |
| GPi -> TH | 5 |
| Cortex -> STN | 5.9 |

**Implementation:** Circular buffer or spike queue.

---

## 3.3 Receptor Types

| Receptor | E_syn (mV) | Notes |
|----------|-----------|-------|
| GABA_A | -80 to -85 | Inhibitory |
| AMPA | 0 | Fast excitatory |
| NMDA | 0 | Slow excitatory |

---

## Implementation Checklist
- [ ] Add `SynapseType` enum: EXPONENTIAL, ALPHA, DOUBLE_EXPONENTIAL
- [ ] Implement alpha-function conductance
- [ ] Implement double-exponential with tau_rise, tau_decay
- [ ] Add `delay` parameter to Synapse struct
- [ ] Implement delay buffer mechanism
- [ ] Python bindings for new synapse parameters
- [ ] Unit tests and verification plots
