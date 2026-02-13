# Task 9: Cortical Pulse Stimulator

## Priority: 3 (Nice-to-have)

## Overview
Implement a cortical stimulus pulse generator used in the benchmark for evoking cortical responses. This is a simple rectangular current pulse injected into cortical excitatory neurons at a specified time and duration.

---

## 9.1 Cortical Stimulus

The benchmark applies a brief, strong current pulse to cortical excitatory neurons:

```python
if corstim == 1:
    Iappco = np.zeros(len(t))
    Iappco[int(1000/dt):int((1000+0.3)/dt)] = 350
else:
    Iappco = np.zeros(len(t))
```

### Parameters
| Parameter | Value |
|-----------|-------|
| onset | 1000 ms |
| duration | 0.3 ms |
| amplitude | 350 uA/cm^2 |

The pulse is applied to **both** cortical excitatory and inhibitory populations (both receive `Iappco[i]` in their voltage update equations).

### Effect on GPe Applied Current
When cortical stimulation is active AND the condition is healthy (not PD):
```python
Iappgpe = 3 - 2 * corstim * (not pd)
# corstim=1, pd=0 → Iappgpe = 1 (reduced from 3)
# corstim=0        → Iappgpe = 3 (normal)
# corstim=1, pd=1  → Iappgpe = 3 (PD: no modulation)
```

---

## 9.2 Implementation

This is simple enough to implement as a Python utility function, similar to the DBS stimulator but even simpler.

### Python API

```python
class PulseStimulator:
    """Simple rectangular pulse current generator."""

    def __init__(self, onset=1000.0, duration=0.3, amplitude=350.0):
        self.onset = onset        # ms
        self.duration = duration  # ms
        self.amplitude = amplitude  # uA/cm^2

    def generate(self, total_duration, dt):
        """Generate current trace as numpy array."""
        n_steps = int(total_duration / dt) + 1
        I = np.zeros(n_steps)
        start = int(self.onset / dt)
        end = int((self.onset + self.duration) / dt)
        I[start:end] = self.amplitude
        return I
```

### Usage with RegionalNetwork

```python
pulse = PulseStimulator(onset=1000, duration=0.3, amplitude=350)
Ipulse = pulse.generate(2000, 0.01)

traces = rn.simulate(2000, 0.01, {
    "Cortex_E": Ipulse,
    "Cortex_I": Ipulse,
    "STN": Idbs,
    "GPe": 3.0,  # constant background current
    "GPi": 3.0,
    "TH": 1.2,
})
```

---

## 9.3 Implementation Checklist
- [ ] Implement `PulseStimulator` Python class
- [ ] `generate()` method returning numpy array
- [ ] Support for multiple pulses (list of onset times) — nice to have
- [ ] Add to `__init__.py` exports
- [ ] Unit tests: correct pulse timing, amplitude, zero elsewhere
- [ ] Integration test: pulse evokes cortical response in RegionalNetwork
