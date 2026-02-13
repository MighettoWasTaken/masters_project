# Task 7: DBS Stimulator

## Priority: 2 (Needed)

## Overview
Implement a Deep Brain Stimulation (DBS) current generator that produces periodic rectangular current pulses. In the benchmark, DBS is applied to the STN population as an external current injection. The stimulator needs to generate a pulse train with configurable frequency, pulse width, and amplitude.

---

## 7.1 DBS Pulse Train

The DBS stimulator generates a periodic rectangular pulse train:

```
Idbs(t) = amplitude    if t mod (1000/freq) < PW
           0            otherwise
```

### Parameters
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| frequency | Stimulation frequency (Hz) | 0-200 Hz (0 = off) |
| amplitude | Pulse amplitude (uA/cm^2) | Variable |
| PW | Pulse width (ms) | 0.06-0.5 ms |
| duration | Total stimulation time (ms) | Same as simulation |
| dt | Time step (ms) | 0.01 |

### Benchmark Implementation
```python
def creatdbs(pattern, tmax, dt, PW, amplitude):
    t = np.arange(0, tmax + dt, dt)
    Idbs = np.zeros_like(t)
    pulse = amplitude * np.ones(int(PW/dt))

    i = 0
    while i < len(t) - 1:
        pulse_len = int(PW/dt)
        if (i + pulse_len) > len(Idbs):
            pulse_len = len(Idbs) - i
        Idbs[i:i+pulse_len] = pulse[:pulse_len]
        isi = 1000 / pattern  # inter-stimulus interval in ms
        i += round(isi / dt)
    return Idbs
```

---

## 7.2 C++ Implementation

### Stimulator Class

```cpp
class DBSStimulator {
public:
    struct Parameters {
        double frequency = 130.0;   // Hz (0 = off)
        double amplitude = 0.0;     // uA/cm^2
        double pulse_width = 0.06;  // ms
    };

    DBSStimulator() = default;
    explicit DBSStimulator(const Parameters& params);

    // Generate full current trace for given duration/dt
    std::vector<double> generate(double duration, double dt) const;

    // Get current at specific time step index
    double current_at(size_t step_index, double dt) const;

    // Setters/Getters
    void set_parameters(const Parameters& params);
    const Parameters& parameters() const;

private:
    Parameters params_;
};
```

### Integration with RegionalNetwork

The DBS stimulator should be attachable to a population in RegionalNetwork:

```python
# Python API
dbs = DBSStimulator(frequency=130, amplitude=300, pulse_width=0.06)

# Option 1: Generate trace and pass as I_ext
Idbs = dbs.generate(duration=1000, dt=0.01)
traces = rn.simulate(1000, 0.01, {"STN": Idbs})

# Option 2: Attach to population (auto-added to I_ext)
rn.attach_stimulator("STN", dbs)
traces = rn.simulate(1000, 0.01, {})
```

Both options should work. Option 1 is simpler (just generates a numpy array). Option 2 is more convenient for complex setups.

---

## 7.3 Implementation Checklist
- [ ] Implement `DBSStimulator` C++ class
- [ ] Implement `generate()` — produces full current trace vector
- [ ] Implement `current_at()` — computes current for a single time step
- [ ] Python bindings for DBSStimulator and Parameters
- [ ] Python wrapper with ergonomic API
- [ ] Unit tests: zero frequency produces zero current, correct pulse width, correct frequency, correct amplitude
- [ ] Integration test: DBS applied to STN in RegionalNetwork
