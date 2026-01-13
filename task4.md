# Task 4: Network Architecture

## Priority: 3 (Network)

## Overview
Higher-level abstractions for building complex multi-region networks.

---

## 4.1 Population Abstraction
Group neurons by brain region.

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

**Python API:**
```python
net = RegionalNetwork()
net.add_population("STN", 10, NeuronType.STN)
net.add_population("GPe", 10, NeuronType.GPE)
net.connect("STN", "GPe", pattern="all_to_all", weight=0.1, delay=2.0)
```

---

## 4.2 Connectivity Patterns

| Pattern | Description |
|---------|-------------|
| all_to_all | Every source to every target |
| one_to_one | Same index mapping |
| random_sparse | Probability-based connections |
| shifted | Lateral inhibition with offset |
| random_permutation | Random one-to-one |

**Weight distributions:** uniform(min, max), normal(mean, std)

---

## 4.3 Heterogeneous Initial Conditions
Randomize initial states per population.

```python
net.randomize_population("STN", V_mean=-62, V_std=5)
```

---

## Implementation Checklist
- [ ] Design NeuronBase interface for polymorphism
- [ ] Implement Population class
- [ ] Implement RegionalNetwork class
- [ ] Implement connectivity pattern functions
- [ ] Support weight randomization
- [ ] Add initial condition randomization
- [ ] Python bindings
- [ ] Unit tests
