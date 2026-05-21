# Task 27: Synapse Storage Compression

**Depends on:** task16 (SynArrays layout), task26 (spike buffer refactor removes spike_buf/buf_head)  
**Unlocks:** task19 (smaller per-synapse footprint improves multi-GPU transfer bandwidth)

---

## Problem

The current `SynArrays` layout stores ~192 bytes per synapse, the majority of which is either
redundant across synapses of the same type or structurally compressible. At 10M synapses
this is ~1.9 GB for synapse data alone, with poor cache utilisation because spec-derived
constants (identical for all AMPA synapses, etc.) are fetched from per-synapse arrays on every
simulation step rather than from a small shared table.

Two independent, composable optimisations address this:

- **27.1 Spec deduplication** — move spec-derived constants out of per-synapse arrays into the
  existing per-spec table. Small change, high impact, low risk.
- **27.2 Connectivity compression** — compress the structural connectivity data (post indices,
  weights, delays) using sorted delta-encoding. Moderate change, significant memory win for
  large networks.

Both are motivated by the same principle: data shared across a synapse population should live
at one reference point, not be replicated N_synapses times.

---

## 27.1 Spec Deduplication

### What moves

The following fields are currently stored per-synapse but are **identical for all synapses
sharing the same `spec_idx`**:

| Field | Bytes | Why it's spec-derived |
|---|---|---|
| `E_syn` | 8 | Reversal potential — property of receptor type |
| `delta_S`, `delta_A` | 16 | On-spike increment — quantal size from spec |
| `tau_S`, `tau_A` | 16 | Time constants from spec |
| `inv_tau_A` | 8 | `1/tau_A` — cached from spec |
| `norm` | 8 | Double-exp normalisation — derived from tau_S, tau_A |
| `decay_S`, `decay_A` | 16 | `exp(-dt/tau)` — derived from spec + dt |
| **Total** | **72 bytes** | Eliminated from per-synapse storage |

These are added to a `SynapseSpecCache` struct that is computed from `SynapseSpec` + `dt`
and stored in `synapse_spec_caches_` (parallel to the existing `synapse_specs_` vector).

```cpp
struct SynapseSpecCache {
    double E_syn;
    double delta_S,  delta_A;
    double tau_S,    tau_A;
    double inv_tau_A;
    double norm;
    double decay_S,  decay_A;
};
// Computed when dt changes (already tracked via SynArrays::cached_dt).
// For 4-8 spec types this is ~300-600 bytes — permanently in L1 cache.
```

### Hot loop change

Before:
```cpp
if (spike_detected_[k]) sa_.S[k] += sa_.delta_S[k];  // per-synapse fetch
sa_.S[k] *= sa_.decay_S[k];                            // per-synapse fetch
const auto& spec = synapse_specs_[sa_.spec_idx[k]];
sa_.g[k] = spec.g * sa_.weight[k] * sa_.S[k];
```

After:
```cpp
const auto& spec  = synapse_specs_[sa_.spec_idx[k]];
const auto& cache = synapse_spec_caches_[sa_.spec_idx[k]];
if (spike_detected_[k]) sa_.S[k] += cache.delta_S;  // L1 hit
sa_.S[k] *= cache.decay_S;                            // L1 hit
sa_.g[k] = spec.g * sa_.weight[k] * sa_.S[k];
```

With 4-8 spec types, the entire cache table fits in a few cache lines and stays resident for
the full simulation. The per-synapse array reads for constants disappear entirely.

The `E_syn` access in the I_syn scatter step changes identically:
```cpp
// Before: sa_.E_syn[k]
// After:  synapse_spec_caches_[sa_.spec_idx[k]].E_syn
```

### Files changed

| File | Change |
|---|---|
| `network.hpp` | Add `SynapseSpecCache`; add `synapse_spec_caches_` to Network; remove 9 vectors from `SynArrays` |
| `network.cpp` | Update `ensure_synapse_state()` to build cache table; update all sim loop phases to use cache; update `push_defaults()` |

### Complexity assessment

**Small-to-moderate.** The spec lookup (`synapse_specs_[sa_.spec_idx[k]]`) already exists in
every loop body for `spec.g` — the change is moving it one line earlier and replacing 9
per-synapse array reads with fields on an already-loaded struct. Approximately 200–300 lines
changed, almost all mechanical substitutions. Low correctness risk — easy to verify outputs are
identical before/after.

---

## 27.2 Connectivity Compression

### What gets compressed

The structural connectivity arrays — `pre`, `post`, `weight`, `delay` — are currently stored
as flat 8-byte values. For large networks most of this is compressible:

**Post indices (8 bytes → ~1 byte average):**  
Sort outgoing synapses by post-synaptic index within each pre-neuron's list and store
differences. For a uniformly connected network of 100K neurons with 10K synapses per neuron,
the average difference is ~10 — easily fits in a `uint8_t`. A separate overflow list handles
rare differences > 255.

**Delays (8 bytes → 1–2 bytes):**  
Most networks use a small number of distinct delay values, or delays drawn from a narrow
distribution. Store a per-neuron reference delay and encode each synapse's delay as a
`uint8_t` offset from the reference. Values outside range go to an overflow list.

**Weights (8 bytes → 0–8 bytes):**  
If all outgoing synapses from a neuron share the same weight (common in biological models),
store it once per neuron. Otherwise keep per-synapse. Detected automatically at
`add_connection()` time.

### Storage layout

Connectivity is restructured into a CSR-like format per pre-neuron:

```cpp
struct NeuronConnectivity {
    // Compressed post indices
    std::vector<uint8_t>  post_deltas;     // delta-encoded, sorted ascending
    std::vector<uint32_t> post_overflow;   // (local_idx, full_post) for deltas > 255

    // Delay encoding
    double                delay_ref;       // reference delay for this neuron
    std::vector<uint8_t>  delay_offsets;   // offsets from delay_ref (in dt steps)
    std::vector<uint32_t> delay_overflow;  // (local_idx, actual_delay_steps)

    // Weight encoding
    bool                  uniform_weight;
    double                weight_value;    // if uniform_weight
    std::vector<double>   weights;         // if !uniform_weight

    // Flat synapse index range (into state arrays S, A, g, is_active)
    size_t syn_start;
    size_t syn_count;
};

std::vector<NeuronConnectivity> neuron_connectivity_;  // size = N_neurons
```

The state arrays (`S`, `A`, `g`, `is_active`) remain flat and dense — they are the hot path
accessed every step. The connectivity structure is accessed only at spike delivery time.

### Decode path

At spike delivery (when `spike_detected_[n]` is true), decode the pre-neuron's connectivity
to find post indices and activate targets. This replaces the current flat scatter loop:

```cpp
// Current: flat loop checking pre index
if (spike_detected_[sa_.pre[k]]) { ... }

// New: spike-driven — only fires for neurons that actually spiked
for (size_t n = 0; n < N_neurons; ++n) {
    if (!spike_detected_[n]) continue;
    const auto& conn = neuron_connectivity_[n];
    uint32_t post = 0;
    for (size_t j = 0; j < conn.syn_count; ++j) {
        post += conn.post_deltas[j];  // decode delta
        // look up overflow if delta was sentinel value
        size_t k = conn.syn_start + j;
        sa_.S[k] += cache.delta_S;   // state array access
    }
}
```

### Sorting requirement

Post indices within each pre-neuron's list must be in ascending order for delta encoding.
An existing `permute()` function already handles synapse reordering. A sort pass at
`ensure_buffers()` time (once per simulation, not per step) produces the required order.

### STDP compatibility

Plasticity weight updates access synapses by flat index `k` — this is unchanged since the
state arrays remain flat. Weight readback (`get_synapse_weights()`) needs to decode the
compressed weight representation, but this is only called outside the simulation loop.

### Files changed

| File | Change |
|---|---|
| `network.hpp` | Add `NeuronConnectivity`; add `neuron_connectivity_` to Network; add flat `post_decoded_` cache array for hot-loop access |
| `network.cpp` | Build `neuron_connectivity_` in `ensure_buffers()`; replace flat scatter loop with spike-driven decode loop; update `get_synapse_weights()`, `reset()` |

### Complexity assessment

**Moderate.** The simulation loop restructuring — from iterating all synapses checking pre-index
to a spike-driven outer loop over fired neurons — is the most involved change, touching the
spike detection phase and all synapse type update phases. The sort + delta-encode build step
is new but self-contained. STDP and `get_synapse_weights()` need minor updates. Approximately
400–500 lines changed with moderate correctness risk; spike delivery timing tests are the
critical verification path.

---

## Memory Comparison

| Category | Current | After 27.1 | After 27.1 + 27.2 |
|---|---|---|---|
| Spec-derived constants | 72 bytes/syn | 0 (in cache table) | 0 |
| Post index | 8 bytes/syn | 8 bytes/syn | ~1 byte/syn avg |
| Weight | 8 bytes/syn | 8 bytes/syn | 0–8 bytes/syn |
| Delay | 8 bytes/syn | 8 bytes/syn | ~1 byte/syn avg |
| Pre index | 8 bytes/syn | 8 bytes/syn | implicit in CSR |
| State (S, A, g) | 24 bytes/syn | 24 bytes/syn | 24 bytes/syn |
| Spike buf metadata | 24 bytes/syn | 0 (task26) | 0 |
| Plasticity metadata | 16 bytes/syn | 16 bytes/syn | 16 bytes/syn |
| **Total** | **~192 bytes** | **~88 bytes** | **~42–50 bytes** |

At 10M synapses: 1.9 GB → 880 MB → 420–500 MB.

---

## Implementation Checklist

### 27.1 Spec deduplication
- [ ] Add `SynapseSpecCache` struct to `network.hpp`
- [ ] Add `synapse_spec_caches_` vector to `Network`
- [ ] Implement `rebuild_spec_caches(double dt)` — called when `cached_dt` changes
- [ ] Remove `delta_S`, `delta_A`, `tau_S`, `tau_A`, `inv_tau_A`, `norm`, `decay_S`, `decay_A`, `E_syn` from `SynArrays`
- [ ] Update `push_defaults()` — remove the 9 fields
- [ ] Update `ensure_synapse_state()` — replace per-synapse constant population with cache build
- [ ] Update all simulation loop phases (2a–2d) to use `cache.*` instead of `sa_.*`
- [ ] Update I_syn scatter step for `E_syn`
- [ ] Verify: all existing tests pass, outputs bit-identical before/after

### 27.2 Connectivity compression
- [ ] Add `NeuronConnectivity` struct to `network.hpp`
- [ ] Add `neuron_connectivity_` to `Network`
- [ ] Implement sort + delta-encode build in `ensure_buffers()`
- [ ] Implement delta-decode helper (inline, handles overflow list)
- [ ] Restructure spike detection + scatter to spike-driven outer loop
- [ ] Update `get_synapse_weights()` to decode compressed weights
- [ ] Update `reset()` to clear state arrays (connectivity unchanged)
- [ ] Update STDP weight update path — verify flat synapse index still valid
- [ ] Verify: spike delivery timing correct at delay = 1×dt, 2×dt, max delay
- [ ] Verify: outputs match 27.1 baseline for all existing test networks
- [ ] Benchmark: memory usage and step throughput at N_synapses = 1M, 10M
