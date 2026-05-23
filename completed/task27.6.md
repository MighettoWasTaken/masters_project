# Task 27.6: Memory Compression — Post-Mortem and Measurements

**Role:** Documentation  
**Status:** Completed  
**Covers:** task27.1 through task27.5

---

## Background

Task 27 addressed the per-synapse memory footprint of `SynArrays`. The driver was the
benchmark at `examples/benchmark_memory_threading.py`, which was extended to reach
`npp=2000` (~48M synapses) and measured ~7.2 GB peak RSS in serial mode.

The work was split into five sub-tasks, each targeting a different category of waste.

---

## What each task changed

### 27.1 — Spec deduplication (SynapseSpecCache)

Before 27.1, `SynArrays` stored per-synapse copies of:
`delta_S`, `delta_A`, `tau_S`, `tau_A`, `inv_tau_A`, `norm`, `decay_S`, `decay_A`

These are **identical for every synapse sharing the same spec** (AMPA, NMDA, GABA_A, etc.).
Task 27.1 moved them into a `SynapseSpecCache` struct indexed by `spec_idx`, shrinking
`SynArrays` by ~64 bytes/synapse. At 48M synapses: **~3 GB freed**.

The hot loop accesses the cache as `synapse_spec_caches_[sa_.spec_idx[k]]`, which for a
network with 3–8 spec types fits entirely in L1 cache — better for performance too.

### 27.2 — Structural compression (pre/post/delay narrowing + connectivity)

`sa_.pre` and `sa_.post` were `size_t` (8 bytes each). Narrowed to `uint32_t` (4 bytes),
then **cleared entirely** after the first `build_injection_tables()` call. The hot loop
reads post indices from `post_decoded_` (flat `uint32_t` array, 4 bytes/syn).
`sa_.delay` narrowed from `double` to `float` (4 bytes). `delay_steps_` narrowed from
`size_t` to `uint32_t`.

`NeuronConnectivity` was added to delta-encode sorted post indices (~1 byte/syn), and
a uniform-weight flag was added per pre-neuron for future use.

Net: **~15 bytes/syn** (pre/post freed = 16, replaced by decoded arrays = 8, delay halved = 4).

### 27.3 — SynapseRef padding removal

`SynapseRef` held `size_t syn_idx` (8 bytes) + `uint32_t delay_steps` (4 bytes),
leaving a 4-byte compiler padding hole → `sizeof(SynapseRef)` = 16.

Narrowing `syn_idx` to `uint32_t` eliminates the padding: `sizeof(SynapseRef)` = 8.
`post_from_` holds one `SynapseRef` per synapse, so this halves its cost.

Net: **8 bytes/syn** (~384 MB at 48M synapses).

### 27.4 — SynArrays remaining size_t and E_syn

`spec_idx` and `plast_spec_idx_arr` were `size_t` (8 bytes each). Both narrowed to
`uint32_t` (4 bytes). Neither will exceed 4 billion entries on any realistic network.

`E_syn` was `double` (8 bytes). Narrowed to `float` (4 bytes). Biological reversal
potentials (-80 mV for GABA_A, 0 mV for glutamate) have at most 3 significant figures —
float32's 7 significant figures are more than sufficient. In the hot loop,
`g * (E_syn - V)` promotes the float `E_syn` to double for the subtraction because
`V` is double — no arithmetic precision change.

Net: **12 bytes/syn** (~576 MB at 48M synapses).

### 27.5 — Lazy SynapseBase construction

`synapses_` (`vector<SynapseBase>`, 16 bytes/entry) was built eagerly on every
`add_synapse()` call and rebuilt on every `sort_synapses_by_pre()`. For
simulation-only code (benchmarks, RegionalNetwork.simulate()) that never calls
`Network::synapse(idx)`, this allocated ~768 MB at 48M synapses for pure API surface.

Making it lazy: `add_synapse()` and `sort_synapses_by_pre()` set `synapses_dirty_ = true`
instead. `synapse(idx)` calls `ensure_synapses_built()` which does the allocation only
on first access. Once built the vector persists (Python `reference_internal` requires
stable references).

Side fix: `SynapseBase::pre_idx()` and `post_idx()` accessed `syn_arrays().pre/post`
directly, which are cleared by task27.2. Fixed via new `Network::pre_at(idx)` /
`post_at(idx)` accessors that prefer `pre_decoded_`/`post_decoded_` with fallback to
`sa_.pre`/`sa_.post` for the pre-simulation case.

Net: **0 bytes/syn for simulation-only paths; 16 bytes/syn when synapse() is called**
(~768 MB at 48M synapses for benchmarks).

---

## Memory accounting at 48M synapses (after all tasks)

| Component | Bytes/syn | Notes |
|---|---|---|
| weight, g | 16 | double — must stay (STDP writes, precision-critical) |
| S, A | 16 | double — must stay (accumulated state, small-value threshold) |
| E_syn | 4 | float (task27.4) |
| delay | 4 | float (task27.2) |
| spec_idx | 4 | uint32_t (task27.4) |
| plast_type + plast_state_idx | 8 | enum + int32_t |
| plast_spec_idx_arr | 4 | uint32_t (task27.4) |
| is_active | ~0 | vector<bool> bit-packed |
| **SynArrays total** | **~56** | |
| post_decoded_ + pre_decoded_ | 8 | uint32_t each (task27.2) |
| post_from_ SynapseRef | 8 | uint32_t each (task27.3) |
| delay_steps_ | 4 | uint32_t (task27.2) |
| spike_detected_ | 1 | uint8_t |
| NeuronConnectivity.post_deltas | ~1 | uint8_t avg |
| synapses_ (SynapseBase) | 0–16 | lazy (task27.5); 0 for simulation-only |
| **Total (sim-only)** | **~78** | |
| **Total (with synapse API)** | **~94** | |

For comparison, the layout before task27.1 was ~192 bytes/syn.

---

## What cannot be compressed further

- **S and A**: accumulated double-precision state. Across thousands of simulation steps
  with multiplicative decay (e.g. `S *= 0.99`) and additive spikes, float32 precision
  loss compounds. The deactivation threshold (`S < 1e-9`) also requires double.
- **weight**: modified by STDP in small additive increments over many steps. Float would
  lose precision for weights after ~1000 updates.
- **g**: computed each step as `spec.g * weight * S`; stored because `compute_synaptic_currents()`
  runs as a separate pass. Could be eliminated by merging passes (future task).

---

## Remaining opportunities

1. **Eliminate `g` storage**: If `compute_synaptic_currents()` and `update_synapses_grouped()`
   are merged into a single pass, `g` (8 bytes/syn = 384 MB at 48M) can be computed inline
   and never stored. Requires restructuring the simulation loop order.

2. **Narrow `weight` to float when STDP is absent**: For static networks, `weight` is
   set once and never updated — float32 is sufficient. Could be selected at network
   construction time via a `static_weights` flag.

3. **Compress `post_from_` further**: The `SynapseRef` vector stores one entry per synapse.
   Since `syn_start`/`syn_count` are already encoded in `NeuronConnectivity`, `post_from_`
   is redundant for the sorted-by-pre layout. Future task: remove `post_from_` entirely
   and iterate `pre_decoded_`/`post_decoded_` slices directly.
