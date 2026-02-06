# Plan: Vectorize C++ with Eigen (Batched Neuron Pools)

## Problem
Currently every neuron is stepped one-at-a-time through `unique_ptr<NeuronBase>` virtual dispatch.
Each HH neuron does 24 scalar `exp()` calls per timestep (6 rate functions x 4 RK4 stages).
NumPy batches all neurons into array ops with SIMD-vectorized `np.exp()` — that's why it keeps up.

## Solution
Add **Eigen** (header-only, SIMD-vectorized math) and create **batched neuron pools** that
process all neurons of the same type simultaneously using array operations — exactly like NumPy
does, but in C++ with less overhead.

## Architecture

```
Network (simulate hot loop)
  ├── HHPool         (NEW) — Eigen::ArrayXd for V,m,h,n × all HH neurons
  │   └── batched RK4 with vectorized exp() across entire population
  ├── IzhikevichPool (NEW) — Eigen::ArrayXd for v,u × all Iz neurons
  │   └── batched Euler with vectorized spike detection
  ├── SynArrays      (existing) — SoA synapse storage
  │   └── type-separated loops (no switch in inner loop)
  └── neurons_       (kept) — polymorphic objects for API access, lazy sync
```

## Implementation Steps

### 1. Add Eigen dependency
- CMakeLists.txt: `FetchContent` Eigen 3.4 (header-only, no build step)
- src/cpp/CMakeLists.txt: link `Eigen3::Eigen` to `hodgkin_huxley_core`

### 2. HHPool — batched HH neurons (biggest win)
- **New files**: `hh_pool.hpp`, `hh_pool.cpp`
- SoA state: `Eigen::ArrayXd V, m, h, n` (length = number of HH neurons)
- SoA params: `Eigen::ArrayXd C_m, g_Na, g_K, g_L, E_Na, E_K, E_L`
- `compute_derivatives()`: vectorized across all neurons simultaneously
  - `I_Na = g_Na * m.cube() * h * (V - E_Na)` — all N neurons at once
  - `(-dV_m * 0.1).exp()` — Eigen's SIMD-vectorized exp over N values
  - Singularity handling via `select()` (branchless, like `np.where`)
- `step_rk4()`: 4 stages using batched derivatives
- Index mapping: pool_idx → network_idx for voltage/current scatter-gather

### 3. IzhikevichPool — batched Izhikevich neurons (simpler)
- **New files**: `iz_pool.hpp`, `iz_pool.cpp`
- SoA state: `Eigen::ArrayXd v, u`
- Batched Euler step + branchless spike reset via mask operations
- No transcendentals — wins come from eliminating virtual dispatch

### 4. Type-separated synapse loops
- Add `SynapseGroups` struct to Network with per-type index lists
- Replace single `update_synapses()` loop (with switch) → three tight loops
- Each loop has uniform ops — compiler can auto-vectorize

### 5. Integrate into simulate()
- `build_pools()` at simulation start: classify neurons via `dynamic_cast`
- Hot loop: set I_total on pools → `hh_pool.step_rk4(dt)` → read V back
- Sync to API objects once at end (existing lazy pattern)
- Python bindings: **zero changes needed**

## Files Changed

| File | Action | What |
|------|--------|------|
| `CMakeLists.txt` | Modify | Add Eigen FetchContent |
| `src/cpp/CMakeLists.txt` | Modify | Add new sources, link Eigen |
| `src/cpp/include/hodgkin_huxley/hh_pool.hpp` | **New** | HHPool class |
| `src/cpp/src/hh_pool.cpp` | **New** | Batched HH RK4 |
| `src/cpp/include/hodgkin_huxley/iz_pool.hpp` | **New** | IzhikevichPool class |
| `src/cpp/src/iz_pool.cpp` | **New** | Batched Iz Euler |
| `src/cpp/include/hodgkin_huxley/network.hpp` | Modify | Add pool members, SynapseGroups |
| `src/cpp/src/network.cpp` | Modify | simulate() uses pools, type-separated synapse loops |
| `src/python/bindings.cpp` | No change | API preserved |

## Why This Should Beat NumPy
1. **Fewer temporaries**: Eigen expression templates fuse operations; NumPy allocates per-op
2. **No Python loop**: NumPy still has `for t in range(num_steps)` in Python; ours is native C++
3. **Same SIMD math**: Eigen's `pexp` uses SSE2/AVX for batched exp(), matching NumPy's approach
4. **Already-fast synapses**: SoA layout already cache-friendly, type separation removes last branch
