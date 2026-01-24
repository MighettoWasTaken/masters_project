# Task 2: Neuron Model Diversity

## Priority: 1 (Core)

## Overview
Expand neuron capabilities to support the diverse cell types in the benchmark model.

---

## 2.1 Izhikevich Neuron Model
Cortical neurons use Izhikevich model, not HH.

**Equations:**
```
dv/dt = 0.04*v^2 + 5*v + 140 - u + I
du/dt = a*(b*v - u)

if v >= 30 mV: v = c, u = u + d
```

**Presets:**
| Type | a | b | c | d |
|------|---|---|---|---|
| Regular spiking (RS) | 0.02 | 0.2 | -65 | 8 |
| Fast spiking (FS) | 0.1 | 0.2 | -65 | 2 |

---

## 2.2 Additional Ion Channels
Modular channels to add to HH neurons:

| Channel | Equation | Used In |
|---------|----------|---------|
| T-type Ca (I_T) | g_T * p^2 * r * (V - E_T) | Thalamus, GPe/GPi |
| L-type Ca (I_L) | g_L * c^2 * d1 * d2 * (V - E_Ca) | STN |
| A-type K (I_A) | g_A * a^2 * b * (V - E_K) | STN |
| M-type K (I_M) | g_M * p * (V - E_M) | Striatum |
| AHP (I_AHP) | g_AHP * (V - E_K) * (Ca / (Ca + k1)) | GPe/GPi |
| CaK (I_CaK) | g_CaK * r^2 * (V - E_K) | STN |

---

## 2.3 Calcium Dynamics
Required for I_AHP and I_CaK currents.

```
d[Ca]/dt = -alpha * (I_Ca + I_T) - K_Ca * [Ca]
E_Ca = (RT/zF) * ln([Ca]_o / [Ca]_i)  # Nernst equation
```

---

## 2.4 Region-Specific Presets

| Region | g_Na | g_K | g_L | Special |
|--------|------|-----|-----|---------|
| Thalamus | 3 | 5 | 0.05 | T-type Ca |
| STN | 49 | 57 | - | L-type Ca, A-current |
| GPe/GPi | 120 | 30 | - | T-type Ca, AHP |
| Striatum | 100 | 80 | - | M-current |
| Cortex | - | - | - | Izhikevich (RS, FS) |

---

## Implementation Checklist
- [ ] Create `IzhikevichNeuron` class with Python bindings
- [ ] Design modular ion channel system
- [ ] Implement T-type, L-type, A-type, M-type channels
- [ ] Implement calcium dynamics and AHP/CaK currents
- [ ] Create factory functions for region presets
- [ ] Unit tests for each neuron type
