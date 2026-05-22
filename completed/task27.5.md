# Task 27.5: Lazy SynapseBase Construction

**Role:** Team lead  
**Status:** Completed  
**Depends on:** task27.2 (post_decoded_/pre_decoded_ available after first build)

---

## What to implement

`synapses_` (vector<SynapseBase>) is currently built eagerly in `add_synapse()` and
rebuilt in `sort_synapses_by_pre()`. For simulation-only code that never calls
`Network::synapse(idx)`, this wastes 16 bytes × N_synapses.

Make the build lazy:
- Remove `synapses_.emplace_back()` from `add_synapse()`
- Remove the rebuild loop from `sort_synapses_by_pre()`; set `synapses_dirty_ = true` instead
- Add `ensure_synapses_built() const` (builds on first call to `synapse(idx)`)
- Make `synapses_` and `synapses_dirty_` `mutable`
- Fix `num_synapses()` to return `sa_.size()` (source of truth) instead of `synapses_.size()`

Also fixes a pre-existing bug from task27.2: `SynapseBase::pre_idx()` and `post_idx()`
accessed `syn_arrays().pre/post` which are cleared after the first build. Fixed via
new `Network::pre_at(idx)` / `post_at(idx)` accessors that prefer `pre_decoded_`/
`post_decoded_` with fallback to `sa_.pre`/`sa_.post`.

## Key files

- `src/cpp/include/hodgkin_huxley/network.hpp` — mutable synapses_, synapses_dirty_,
  ensure_synapses_built() decl, pre_at/post_at accessors
- `src/cpp/src/network.cpp` — ensure_synapses_built() impl, synapse() accessor,
  add_synapse(), sort_synapses_by_pre()
- `src/cpp/src/synapse_base.cpp` — pre_idx/post_idx use pre_at/post_at

## Contract

- For code that never calls `synapse(idx)` or `get_synapse_weights()`: `synapses_` is
  never allocated → 16 bytes × N_synapses saved
- Once built, the vector persists (Python `reference_internal` references remain valid)
- All tests pass unchanged; `test_regional_network.py` synapse-access tests trigger
  the lazy build on first access
