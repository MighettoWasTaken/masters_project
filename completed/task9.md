# Task 9: Pulse Stimulator

## Priority: 3 (Nice-to-have) — COMPLETED

## Overview
A flexible rectangular / biphasic current-pulse generator for injecting
stimulation waveforms into any population. Covers the benchmark cortical
stimulus as a special case, but is designed for general use with any
waveform timing pattern.

---

## 9.1 Use Cases

| Pattern | Constructor |
|---|---|
| Single rectangular pulse (benchmark cortical) | `PulseStimulator.single(onset, duration, amplitude)` |
| Pulses at arbitrary irregular times | `PulseStimulator.from_onsets([t1, t2, ...], duration, amplitude)` |
| Regular pulse train | `PulseStimulator.train(frequency, duration, amplitude, total_duration=...)` |
| Single burst at high intra-burst frequency | `PulseStimulator.burst(n_pulses, intra_freq, duration, amplitude, onset=...)` |
| Any of the above with charge-balanced biphasic waveform | Add `waveform="biphasic"` to any constructor |

---

## 9.2 API

```python
from hodgkin_huxley import PulseStimulator

# --- Benchmark cortical stimulus ---
pulse = PulseStimulator.single(onset=1000.0, duration=0.3, amplitude=350.0)

# --- Regular 130 Hz DBS-style biphasic train ---
train = PulseStimulator.train(
    frequency=130, duration=0.06, amplitude=300,
    onset=0.0, total_duration=1000.0, waveform="biphasic"
)

# --- 5-pulse burst at 300 Hz starting at t=500 ms ---
burst = PulseStimulator.burst(
    n_pulses=5, intra_freq=300, duration=0.06, amplitude=200, onset=500.0
)

# --- Generate current array ---
I = pulse.generate(total_duration=2000.0, dt=0.01)   # ndarray, shape (n_steps,)

# --- Add pulse to a tonic background ---
I_net = pulse.apply_to(base_current=3.0, total_duration=2000.0, dt=0.01)

# --- Use with RegionalNetwork ---
rn.simulate(2000.0, 0.01, {
    "Cortex_E": pulse.generate(2000.0, 0.01),
    "Cortex_I": pulse.generate(2000.0, 0.01),
    "STN":      train.generate(2000.0, 0.01),
    "GPe":      3.0,
})
```

### Biphasic waveform

```
|← duration →|← gap →|← anodic_duration →|
 +amplitude                -anodic_amplitude
```

- Default: symmetric (`anodic_amplitude = amplitude`, `anodic_duration = duration`) — charge-balanced.
- Asymmetric: pass `anodic_amplitude` and/or `anodic_duration` explicitly.
- `p.is_charge_balanced` — checks equality of cathodic and anodic charges.
- `p.charge_per_phase` — `amplitude × duration` (µA·ms/cm²).

---

## 9.3 Implementation Checklist
- [x] `PulseStimulator` class in `src/hodgkin_huxley/pulse.py`
- [x] `single()` class method — single rectangular pulse
- [x] `from_onsets()` class method — pulses at arbitrary times
- [x] `train()` class method — regular pulse train (n_pulses or total_duration)
- [x] `burst()` class method — N pulses at intra-burst frequency
- [x] `waveform="biphasic"` with `interphase_gap`, `anodic_amplitude`, `anodic_duration`
- [x] `generate(total_duration, dt)` → 1-D ndarray compatible with `I_ext`
- [x] `apply_to(base_current, total_duration, dt)` → pulse added to background
- [x] Properties: `onsets`, `n_pulses`, `duration`, `amplitude`, `waveform`, `charge_per_phase`, `is_charge_balanced`
- [x] Input validation with `ValueError` on bad parameters
- [x] Export `PulseStimulator` from `__init__.py`
- [x] Unit tests: timing, amplitude, zeros, biphasic phases, gap, charge balance (`test_pulse_stimulator.py`)
- [x] Integration tests: evokes spikes in HH neuron; accepted by RegionalNetwork
- [x] Benchmark cortical stimulus pattern verified
