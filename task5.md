# Task 5: Stimulation & Analysis

## Priority: 4 (Tools)

## Overview
Stimulation protocols and analysis tools for experiments.

---

## 5.1 DBS Stimulation
Deep brain stimulation for STN or GPi.

**Parameters:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| frequency | 0-200 Hz | Stimulation frequency |
| amplitude | 0-500 uA/cm^2 | Pulse amplitude |
| pulse_width | 0.06-0.5 ms | Pulse duration |

```python
dbs = DBSStimulator(frequency=130, amplitude=300, pulse_width=0.1)
I_dbs = dbs.generate(duration=1000, dt=0.01)
```

---

## 5.2 Pulse Stimulation
Brief current pulses for cortical stimulation.

```python
# Single pulse: 350 uA/cm^2 for 0.3 ms at t=1000ms
I = PulseStimulator.pulse(duration=2000, dt=0.01,
                          onset=1000, width=0.3, amplitude=350)

# Pulse train
I = PulseStimulator.train(duration=2000, dt=0.01,
                          onset=500, width=0.3, amplitude=350,
                          interval=100, count=5)
```

---

## 5.3 Analysis Tools

### Spike Detection
```python
spike_times = detect_spikes(trace, dt, threshold=0.0)
```

### Firing Rate
```python
rate = firing_rate(spike_times, duration, bin_size=10.0)  # Hz
```

### Spectral Analysis
```python
# Beta-band power (7-35 Hz) - key metric for PD research
beta = beta_power(spike_times, duration)
```

---

## Implementation Checklist
- [ ] Implement DBSStimulator class
- [ ] Implement PulseStimulator utilities
- [ ] Implement spike detection function
- [ ] Implement firing rate calculation
- [ ] Implement spectral analysis (beta-band power)
- [ ] Python bindings / pure Python implementation
- [ ] Unit tests
