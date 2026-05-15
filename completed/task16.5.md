# Task 16.5: Active Synapse Set — Sparse Conductance Sweep

**Role:** C++ engineer  
**Status:** Not started  
**Depends on:** task16 (SynapseGroups SoA layout), task15 (STDP/STP already event-driven — no changes needed there)  
**Unlocks:** task17.13 (SynapseMatrix CSR mode builds on the active-set delivery path)

---

## Problem

`compute_synaptic_currents()` currently iterates over all S synapses every timestep regardless of conductance state:

```cpp
for (int i = 0; i < n_synapses_; ++i) {
    I_buf_[post_idx_[i]] += g_[i] * (E_syn_[i] - V_[post_idx_[i]]);
    g_[i] *= decay_factor_[i];
}
```

For biologically realistic spiking (1–10 Hz, tau ~5 ms, dt = 0.01 ms), >99% of synapses have `g ≈ 0` at any given step. This is pure wasted work.

The fix is an **active synapse set**: a compact list of synapse indices where `g > ε`. Only those entries are touched each step. Entries are added on spike delivery and removed when `g` decays below threshold.

---

## What to implement

### Data structures — one per `SynapseGroup`

Each `SynapseGroup` gains two new members:

```cpp
struct SynapseGroup {
    // --- existing fields ---
    std::vector<int>    post_idx;
    std::vector<double> g, A;           // conductance state
    std::vector<double> E_syn;
    std::vector<double> decay_g;        // exp(-dt/tau_S) per synapse
    std::vector<double> decay_A;        // exp(-dt/tau_A) per synapse (kinetic forms)
    // ... other existing fields ...

    // --- NEW ---
    std::vector<int>  active;           // indices into the above arrays with g > epsilon
    std::vector<bool> is_active;        // [n_synapses] O(1) membership test; prevents duplicate adds
    static constexpr double kEpsilon = 1e-9;  // conductance threshold for activation

    void activate(int synapse_idx);     // add to active set if not already present
    void deactivate_if_decayed(int pos); // swap-and-pop from active if decayed
};
```

`is_active` is a flat bool array parallel to the synapse arrays — cache-friendly for the membership check on spike delivery.

`active` is a `std::vector<int>` of currently live synapse indices. Iteration order doesn't matter; removal uses swap-and-pop:

```cpp
void SynapseGroup::deactivate_if_decayed(int pos) {
    // pos = position in active[], not synapse index
    if (g[active[pos]] < kEpsilon && A[active[pos]] < kEpsilon) {
        is_active[active[pos]] = false;
        active[pos] = active.back();
        active.pop_back();
    }
}
```

### `compute_synaptic_currents()` — iterate active only

```cpp
void Network::compute_synaptic_currents() {
    for (auto& grp : synapse_groups_) {
        int pos = 0;
        while (pos < static_cast<int>(grp.active.size())) {
            const int i = grp.active[pos];
            I_buf_[grp.post_idx[i]] +=
                grp.g[i] * (grp.E_syn[i] - V_[grp.post_idx[i]]);
            grp.g[i] *= grp.decay_g[i];
            grp.A[i] *= grp.decay_A[i];      // no-op for EXP_DECAY (decay_A = 0)
            grp.deactivate_if_decayed(pos);   // may swap this entry out
            if (grp.is_active[grp.active[pos]]) ++pos;  // only advance if entry survived
        }
    }
}
```

Note: after `deactivate_if_decayed(pos)`, `active[pos]` now holds the swapped-in entry (if any). The `if (is_active[...]) ++pos` pattern handles this correctly without a separate branch.

### Spike delivery — add to active set

When a spike arrives and `g[i] += delta` is applied, also call `activate(i)`:

```cpp
void SynapseGroup::activate(int i) {
    if (!is_active[i]) {
        is_active[i] = true;
        active.push_back(i);
    }
}
```

This is called from the existing spike delivery path (both Phase-1 and Phase-2 delay-decomposition). In Phase-2 (delay-decomposition threading), spikes arrive via SPSC ring buffers on the owning thread — no additional locking needed since `activate()` is only called by the same thread that owns the `SynapseGroup`.

### Removal condition per synapse type

| `SynapseUpdateForm` | Remove when |
|---|---|
| `EXP_DECAY` | `g[i] < ε` |
| `ALPHA_FUNC` | `g[i] < ε && A[i] < ε` |
| `DOUBLE_EXP` | `g[i] < ε && A[i] < ε` |
| `TANH_GATE` | `g[i] < ε` (voltage-gated; decay when V below threshold) |
| `BOLTZMANN_GATE` | `g[i] < ε` |
| `ALPHA_BETA` | `g[i] < ε` |
| `CUSTOM_EXPR` | `g[i] < ε && A[i] < ε` (conservative) |

For voltage-gated forms (`TANH_GATE`, `BOLTZMANN_GATE`, `ALPHA_BETA`), `g` can increase without a spike delivery event — conductance is driven by voltage. These must stay active as long as `g > ε`, regardless of spike history. The same removal condition applies; the activation path is just different (g rises during normal stepping rather than on spike delivery, so the entry is never removed while voltage-gating keeps it elevated).

### Initialisation

`SynapseGroup::add_synapse(...)` initialises `is_active[new_idx] = false`. `active` starts empty. The `g = 0` initial state means no synapse is active at t=0.

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/synapse_groups.hpp` | Add `active`, `is_active`, `activate()`, `deactivate_if_decayed()` to `SynapseGroup` |
| `src/cpp/src/network.cpp` | `compute_synaptic_currents()` iterates `active` instead of `0..S`; spike delivery calls `activate()` |

---

## Baseline tests (before PR to testing branch)

- [ ] `pytest tests/python/ -x -q` — all existing tests pass; V traces numerically identical to pre-patch (to double precision)
- [ ] Empty network (no spikes): `active.size() == 0` throughout; `compute_synaptic_currents()` is a no-op
- [ ] Single spike delivered at t=10 ms: `active.size()` rises to 1 immediately after delivery; decays back to 0 within 10 × tau ms
- [ ] `active.size()` at steady state ≈ `N * r * tau / dt` (analytical expectation); verify within 10% for 100-neuron network at 10 Hz
- [ ] CTX-BG-TH benchmark: V traces match pre-patch output to within 1e-12 mV; wall-clock time reduced for large networks
- [ ] Phase-2 threading: no race on `activate()` — verify by running thread-sanitizer on the CTX-BG-TH benchmark with 4 thread groups

---

## Expected performance gain

For a network with N neurons, mean firing rate r, synaptic time constant tau, and S total synapses:

```
Active fraction = r * tau  (in consistent units)
                = 10 Hz * 0.005 s = 0.05  (5% of synapses active per step)
```

`compute_synaptic_currents()` goes from O(S) to O(0.05 * S) — approximately 20× reduction in loop iterations at 10 Hz / 5 ms tau. Gains scale linearly with lower firing rates or shorter time constants.

The swap-and-pop removal adds one indirect write per deactivation event; amortised over the decay period this is negligible compared to the iterations saved.
