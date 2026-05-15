# Task 17.13: Sparse Connectivity — `SynapseMatrix`

**Role:** Memory / systems engineer  
**Status:** Not started  
**Depends on:** task16 (SynapseGroups SoA architecture), task13 (SynapseSpec)  
**Unlocks:** task17.6, task17.7 (CUDA pool/synapse kernels must not be built on dense SoA at scale); task19 (multi-GPU requires portable connectivity representation)

---

## Problem

Current `Network` stores all synapse fields as flat dense SoA arrays:

```cpp
std::vector<int>    post_idx_;     // S entries
std::vector<double> weight_;       // S entries
std::vector<double> E_syn_;        // S entries
std::vector<double> g_;            // S entries — live conductance state
std::vector<double> tau_decay_;    // S entries
// ... plus A_, delta_S_, etc.
```

At S = 10⁸ synapses (1000 neurons × 10⁵ synapses each): ~500 MB — acceptable.  
At S = 10⁹: ~5 GB for state alone, 40–80 GB including all fields — impossible to allocate.

The dense layout also prevents any skip of zero-conductance synapses in `compute_synaptic_currents()`.

This task replaces the flat arrays with a `SynapseMatrix` abstraction that has three internal representations, selected automatically:

| Mode | When used | Connectivity storage | State storage |
|------|-----------|---------------------|---------------|
| `DENSE` | S < threshold (default 10⁷) | flat SoA (current) | flat SoA (current) |
| `CSR` | S ≥ threshold, custom connectivity | CSR graph + flat state | flat SoA, float32 |
| `GENERATIVE` | standard patterns (RANDOM_SPARSE, ALL_TO_ALL, etc.) | none — regenerated on demand | none — regenerated on demand |

`DENSE` mode preserves 100% backward compatibility and zero performance overhead for existing use cases.

---

## What to implement

### `src/cpp/include/hodgkin_huxley/synapse_matrix.hpp` — new file

```cpp
#pragma once
#include "hodgkin_huxley/synapse_spec.hpp"
#include "hodgkin_huxley/connectivity.hpp"   // ConnectivityPattern, WeightDistribution
#include <vector>
#include <cstdint>

namespace hodgkin_huxley {

// ---------------------------------------------------------------------------
// CSR connectivity graph — stores only the static structure (no state).
// ---------------------------------------------------------------------------
struct CSRGraph {
    std::vector<int>     row_ptr;    // [N_pre + 1]; row_ptr[n]..row_ptr[n+1] = outgoing synapses of n
    std::vector<int>     col_idx;    // [S] post-synaptic neuron index
    std::vector<float>   weight;     // [S] synaptic weight (float32)
    std::vector<float>   E_syn;      // [S] reversal potential (float32)
    std::vector<uint16_t> delay_steps; // [S] delay in simulation steps (max ~65 ms at dt=0.01)
    std::vector<uint8_t>  spec_id;   // [S] index into SynapseMatrix::specs_

    int n_pre  = 0;
    int n_post = 0;
    int n_synapses() const { return static_cast<int>(col_idx.size()); }

    // Bytes consumed by this graph (for logging / assertion)
    size_t memory_bytes() const;
};

// ---------------------------------------------------------------------------
// Generative connectivity — standard patterns that need no storage.
// Outgoing targets for neuron n are regenerated deterministically from seed.
// ---------------------------------------------------------------------------
struct GenerativeConn {
    ConnectivityPattern pattern;
    int       N_pre, N_post;
    double    p         = 0.1;    // RANDOM_SPARSE connection probability
    float     weight    = 1.0f;
    float     E_syn     = 0.0f;
    uint16_t  delay_steps = 1;
    uint8_t   spec_id   = 0;
    uint64_t  seed      = 0;

    // Fill `out` with post-neuron indices for pre-neuron `n`.
    // Deterministic: same n + same seed always gives same result.
    void outgoing(int n, std::vector<int>& out) const;
};

// ---------------------------------------------------------------------------
// Per-synapse state (conductance, auxiliary variable for kinetic forms).
// Allocated as float32 flat arrays parallel to CSRGraph::col_idx.
// For DENSE mode this is the existing double arrays (kept as-is).
// ---------------------------------------------------------------------------
struct SynapseState {
    std::vector<float> g;    // [S] current conductance
    std::vector<float> A;    // [S] auxiliary (ALPHA_FUNC, DOUBLE_EXP only)
};

// ---------------------------------------------------------------------------
// SynapseMatrix — the unified interface used by Network, CUDA kernels, etc.
// ---------------------------------------------------------------------------
class SynapseMatrix {
public:
    enum class Mode { DENSE, CSR, GENERATIVE };

    // Construct from existing Network synapse arrays (DENSE mode — current behaviour).
    // Called by Network::connect() for backward compatibility.
    static SynapseMatrix from_dense(/* existing SoA args */);

    // Construct from a ConnectivityPattern spec.
    // Auto-selects GENERATIVE for standard patterns; CSR for CUSTOM connectivity.
    static SynapseMatrix from_pattern(
        ConnectivityPattern pattern,
        int N_pre, int N_post,
        const SynapseSpec& spec,
        const WeightDistribution& weights,
        double p,
        double delay_ms,
        double dt,
        uint64_t seed = 0
    );

    Mode mode() const { return mode_; }
    int  n_synapses_approx() const;  // exact for DENSE/CSR, estimated for GENERATIVE
    size_t memory_bytes() const;

    // ---------------------------------------------------------------------------
    // Iteration interface — used by Network hot loop and CUDA kernels.
    // For DENSE: iterates flat arrays (zero overhead vs current code).
    // For CSR: iterates row_ptr/col_idx.
    // For GENERATIVE: calls outgoing() per pre-neuron on spike.
    // ---------------------------------------------------------------------------

    // Called once per pre-neuron that fired this step.
    // Delivers spike to all post-synaptic targets: increments g[target] by weight.
    void deliver_spike(int pre_n, SynapseState& state, float* I_buf, const float* V) const;

    // Called every step for all active synapses (used with active-synapse list, task #1).
    // For DENSE mode: iterates all entries (backward compat).
    const int*   post_data()   const;  // raw pointer into col_idx (CSR) or post_idx (DENSE)
    const float* E_syn_data()  const;
    int          n_entries()   const;

    // ---------------------------------------------------------------------------
    // Migration
    // ---------------------------------------------------------------------------

    // Rebuild as CSR from current DENSE representation.
    // Called automatically when n_synapses_approx() > kDenseThreshold.
    void rebuild_as_csr();

    static constexpr int kDenseThreshold = 10'000'000;  // 10M synapses

private:
    Mode mode_ = Mode::DENSE;

    // DENSE mode fields (existing SoA — kept as raw pointers to avoid copy)
    // These point into the Network's existing arrays for zero-overhead DENSE mode.
    struct DenseView { /* raw pointer aliases into Network's existing arrays */ };

    CSRGraph         csr_;
    GenerativeConn   gen_;
    SynapseState     state_;
    std::vector<SynapseSpec> specs_;  // spec table (shared across all synapses)

    DenseView dense_;
};

} // namespace hodgkin_huxley
```

### `GenerativeConn::outgoing()` — deterministic sparse connectivity

```cpp
void GenerativeConn::outgoing(int n, std::vector<int>& out) const {
    out.clear();
    if (pattern == ConnectivityPattern::ALL_TO_ALL) {
        for (int j = 0; j < N_post; ++j)
            if (j != n) out.push_back(j);
        return;
    }
    if (pattern == ConnectivityPattern::RANDOM_SPARSE) {
        // PCG32 or xorshift64 seeded with (seed ^ (uint64_t)n * 2654435761ULL)
        // Bernoulli draw for each potential post-neuron — O(N_post) but allocation-free.
        // Alternatively: use binomial to draw k, then sample k indices (O(k log k)).
        uint64_t rng = seed ^ ((uint64_t)n * 6364136223846793005ULL);
        for (int j = 0; j < N_post; ++j) {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
            double u = (rng >> 11) * (1.0 / (1ULL << 53));
            if (u < p) out.push_back(j);
        }
        return;
    }
    // ONE_TO_ONE, SHIFTED, RANDOM_PERMUTATION — similar deterministic generators
}
```

The seed is fixed at `Network::connect()` time from a global RNG, making connectivity deterministic and reproducible without storage.

### `Network` changes

`Network::connect()` calls `SynapseMatrix::from_pattern()` instead of appending to flat vectors. After construction, if `n_synapses_approx() > kDenseThreshold`, automatically calls `rebuild_as_csr()`.

`compute_synaptic_currents()` becomes:

```cpp
void Network::compute_synaptic_currents() {
    // GENERATIVE: on-spike delivery only — no per-step sweep needed.
    // CSR / DENSE: iterate active synapses (integrates with task #1 active list).
    syn_matrix_.deliver_pending(spike_buffer_, g_state_, I_buf_.data(), V_.data());
}
```

### `float32` state arrays

State (`g`, `A`) uses `float32` in CSR and GENERATIVE modes:
- Halves memory vs `double` with negligible accuracy impact (conductances are O(1) nS–µS; float32 has 7 significant digits)
- DENSE mode keeps `double` for backward compatibility
- Unit test confirms `float32` vs `double` conductance produces < 0.01 mV difference in V over 1000 ms

### CUDA impact (task17.7)

task17.7's `accumulate_isyn` kernel receives a `SynapseMatrix` device view:
- `DENSE`: existing coalesced SoA access (no change)
- `CSR`: CSR graph in device memory; coalesced col_idx access per row
- `GENERATIVE`: kernel generates connectivity on-the-fly per warp — avoids device-side storage entirely. For `RANDOM_SPARSE` with large N, this is the only viable CUDA path.

`SynapseMatrix::to_device()` allocates and fills device-side CSR arrays; `GenerativeConn` parameters are passed as kernel constants (no allocation).

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/synapse_matrix.hpp` | New — `SynapseMatrix`, `CSRGraph`, `GenerativeConn`, `SynapseState` |
| `src/cpp/src/synapse_matrix.cpp` | New — `from_pattern()`, `rebuild_as_csr()`, `deliver_spike()`, `outgoing()` |
| `src/cpp/include/hodgkin_huxley/network.hpp` | Replace flat SoA synapse fields with `SynapseMatrix syn_matrix_` |
| `src/cpp/src/network.cpp` | `connect()` → `from_pattern()`; `compute_synaptic_currents()` uses matrix interface |
| `src/cpp/CMakeLists.txt` | Add `src/synapse_matrix.cpp` |
| `src/python/bindings.cpp` | Bind `SynapseMatrix::mode()`, `memory_bytes()`, `n_synapses_approx()` for introspection |

---

## Baseline tests (before PR to testing branch)

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass (DENSE mode is 100% backward compatible)
- [ ] `SynapseMatrix::from_pattern(RANDOM_SPARSE, ...)` with N=100, p=0.1: mode is GENERATIVE; `memory_bytes() == 0` (no connectivity stored)
- [ ] Two calls to `GenerativeConn::outgoing(n, ...)` with same seed produce identical output
- [ ] `SynapseMatrix` in GENERATIVE mode: network simulation with 100 neurons produces identical V traces to DENSE mode with same seed
- [ ] `rebuild_as_csr()`: DENSE matrix rebuilt as CSR; post-rebuild simulation matches pre-rebuild to within float32 precision
- [ ] `n_synapses_approx() > kDenseThreshold` triggers automatic CSR rebuild at construction
- [ ] `SynapseMatrix::memory_bytes()` reports < 10% of equivalent DENSE for GENERATIVE mode; < 60% for CSR with float32
- [ ] Network with S = 10⁸ (e.g., 10,000 neurons, p=1.0, RANDOM_SPARSE): construction completes without OOM; GENERATIVE mode selected automatically
- [ ] `SynapseMatrix::to_device()` (stub — tested in task17.7): device-side CSR arrays allocated correctly

---

## Notes

- `GENERATIVE` mode is the correct default for all `RANDOM_SPARSE` and `ALL_TO_ALL` patterns regardless of size — it's never wrong to use it, only potentially slower than DENSE for very small networks due to RNG overhead. The `kDenseThreshold` guards the automatic switch; `from_pattern()` can be forced to GENERATIVE explicitly.
- `uint16_t` for delay steps supports delays up to 655 ms at dt=0.01 ms — sufficient for all biologically realistic values. If > 655 ms is needed, promote to `uint32_t`.
- The `DENSE` mode raw-pointer aliases into the `Network`'s existing arrays mean zero allocation overhead and zero copy on construction — backward compatibility is not just API-level but performance-level.
- Cross-population synapses in `RegionalNetwork` use one `SynapseMatrix` per directed population pair. The GENERATIVE mode's per-call `outgoing()` is called only when a pre-neuron fires — O(N_post * p) per spike, identical to the DENSE spike delivery cost.
