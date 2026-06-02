# CUDA Simulation Architecture

This document describes how `RegionalNetwork.simulate()` runs on the GPU: the
cooperative on-device kernel, how composable (custom) neuron models are
specialized, the model-layout fusion that keeps the SM busy, and the
performance characteristics and known limits. It is the reference for anyone
touching `src/cpp/src/cuda_sim_all.cu` or the CUDA pools.

---

## 1. The big picture

A GPU simulation of `num_steps` timesteps runs as **one kernel launch** — the
whole time loop lives on the device (`simulate_all_kernel_impl` in
`cuda_sim_all.cu`). There is no per-step return to the CPU. Threads cooperate
across the timestep loop with barriers (`grid.sync()` for multi-block, or
`__syncthreads()` for single-block — see §4).

This "cooperative kernel" design exists because on Windows WDDM every per-step
kernel launch / CPU sync costs several microseconds; over 100k steps that
dominates. Keeping the entire loop on-device eliminates that overhead.

**Host entry point:** `simulate_all_steps()` (`cuda_sim_all.cu`) — collects pool
descriptors, builds the neuron layout, uploads everything, picks single- vs
multi-block, and launches. Called from `Network::simulate_with_descriptors()`
(`network.cpp`) on the GPU path.

---

## 2. When the cooperative path is taken (fallback hierarchy)

`Network::simulate_with_descriptors()` uses the cooperative kernel only when
**all** of these hold (otherwise it falls back to the CPU-orchestrated per-step
loop, which is correct but slower):

- `pool_mgr_.on_cuda()` — network is on a CUDA device.
- `!has_stdp_ && !has_stp_` — no plastic synapses.
- `pool_mgr_.has_coop_capable_cuda_pools()` — every composable pool is
  `coop_fast_eligible()` (see §5.3).
- No `gate_buf / calcium_buf / u_buf / g_syn_buf / I_syn_buf / spike_event_buf`
  requested — i.e. the recording config only needs V and/or spikes. (Rich
  per-gate/substance recording uses the per-step path.)

The descriptor path (compact `StimPlan` instead of a dense I_ext matrix) is
itself selected by `RegionalNetwork.simulate()` when all I_ext values are scalars
and all stimulators are `DBSStimulator`.

---

## 3. Per-step phase structure (the fused loop)

Each timestep runs these phases inside the kernel:

```
P1  compute stim → d_I_syn ; scatter pool V → d_V_cache ; record V        [barrier]
P2  accumulate synaptic currents (atomicAdd → d_I_syn)                     [barrier]
P3+P4+P5  FUSED per-neuron: gather I_syn → step → scatter V → detect spike [barrier]
P6  write spike ring (d_neuron_spiked[pre] → d_spike_ring[s][step%R])      [barrier IF min_delay_steps==0]
P7  update synapse state (S, A, g)                                         [barrier, end of step]
```

Key facts:

- **P3/P4/P5 are fused** into one per-neuron pass — each thread steps its own
  neuron, writes V, and detects its own spike, with no barriers between, because
  there is no cross-thread dependency within a neuron.
- **The P6→P7 barrier is elided** when `min_delay_steps >= 1` (the common case):
  the ring write slot is `step % R` and the read slot is `(step − delay) % R`,
  which differ whenever delay ≥ 1 step. Networks with a zero-delay synapse keep
  the barrier.
- Barrier count is therefore **4 per step** for typical networks (5 with a
  zero-delay synapse), down from 7 in the original design.

The fused P3-P5 pass is itself a **single flat loop over all neurons** — see §6.

---

## 4. Single-block vs multi-block

Two launch modes, chosen by total work:

```cpp
const int single_block_max_work = 256;            // tunable (cuda_sim_all.cu)
int work_items = max(n_neurons, n_synapses, stim.n_neurons);
const bool use_single_block = (work_items <= single_block_max_work);
```

- **Single-block** (`<<<1, sb_threads>>>`, regular launch): all threads in one
  block on one SM; barriers are `__syncthreads()` (~ns). No `grid.sync()`.
  Used for small total-work networks.
- **Multi-block** (`cudaLaunchCooperativeKernel`): blocks spread across SMs;
  barriers are `grid.sync()`. Used once total work justifies multiple SMs.

The decision is on **total work** (`work_items`), not neuron count, on purpose:
a synapse-heavy network (e.g. all-to-all) with few neurons would overwhelm a
single SM, so it must take the multi-block path even though neuron count is low.

`single_block_max_work` is a tuned constant — raising it pushes larger networks
onto single-block (avoids `grid.sync` but loses SM parallelism); lowering it
does the opposite. 256 was tuned to give the best results across the
500–4000-neuron range.

> Note: on the RTX 4080 / WDDM measured here, `grid.sync()` turned out to cost
> only ~0.3s total for a 100k-step / 3-block run — far less than feared. Small
> heterogeneous models like CTX-BG-TH are **compute/occupancy-bound, not
> barrier-bound**, so the single/multi-block choice is a minor lever for them.

---

## 5. Composable neuron specialization

Composable (SymPy-built) neuron models are the performance-sensitive part. Three
changes took them from a generic interpreter (~25× slower than the hand-written
HH kernel) to near-`hh_step_single` speed for the pattern-matched standard forms.

### 5.1 Flat state layout (no pointer chasing)

`CudaComposablePool` formerly stored gate/substance state as a **jagged
array-of-pointers** (`double** d_gate_ptrs_`, one `cudaMalloc` per gate), so
every read was a two-level pointer chase. It now uses a **single flat block** per
quantity with a stride:

```
gate g of neuron i  →  d_gate_state_[g * capacity_ + i]   // single indirection, coalesced
```

Fields: `d_gate_state_`, `d_substance_state_`, `d_nernst_state_` (all `double*`,
stride = `capacity_`). This mirrors the HH pool's flat `d_V/d_m/d_h/d_n`. See
`cuda_composable_pool.cu` (`allocate_device`, `upload_state`, `download_state`,
`fill_coop_desc`) and `CudaComposableDesc` in `cuda_sim_all.hpp`.

### 5.2 Per-bucket unrolled kernels

`composable_step_unrolled<NG, NC, NS>` (`cuda_sim_all.cu`) is templated on the
**exact** gate / channel / substance counts. Loops are compile-time-bounded with
`#pragma unroll`, scratch arrays are exactly sized, and **all modulation and VM
code is gone** — that removed footprint is what raises occupancy enough to hide
the fp64 transcendental latency. It reuses the per-form evaluators
`sa_boltz` / `sa_tau` / `sa_rate` (which mirror the CPU's `compute_*_vec`).

`composable_step_dispatch()` selects the bucket by a warp-uniform `switch` on
`(n_gates, n_channels, n_intracellulars)`. Pre-instantiated buckets:

| Bucket (NG, NC, NS) | Model |
|---|---|
| (3, 3, 0)  | HH-default |
| (4, 4, 0)  | Striatum |
| (5, 4, 0)  | TH (thalamus) |
| (6, 6, 1)  | GPe / GPi (Ca²⁺) |
| (11, 7, 1) | STN (Ca²⁺ + Nernst) |

### 5.3 Routing: only pattern-matched, known-bucket pools

The cooperative kernel deliberately contains **only** the lean bucket code — no
generic interpreter — so its register footprint (and thus occupancy) is not
dragged down by a worst-case path. `CudaComposablePool::coop_fast_eligible()`
returns true only when a pool:

- has **no VM programs** (`!needs_vm_programs()` — custom SymPy expressions), and
- has **no modulations**, and
- matches a pre-instantiated bucket above.

`PoolManager::has_coop_capable_cuda_pools()` requires this of *every* composable
pool. A network with any VM / modulation / odd-bucket pool routes to the
per-step CPU-orchestrated path instead (still correct, just slower). Arbitrary-
SymPy GPU codegen — which would remove this restriction — is intentionally out of
scope here (a separate codegen track).

---

## 6. Model-layout fusion (flat neuron pass)

The fused P3-P5 step is **one flat strided loop over every neuron**, not a loop
per pool:

```cpp
for (int k = tid; k < n_neurons; k += total) {
    const NeuronSlot sl = layout[k];      // {kind, pool, local}
    // dispatch to hh_step_single / iz_step_single / composable_step_dispatch
    ...
}
```

`layout` is a `NeuronSlot[]` built on the host in `simulate_all_steps()`, ordered
**HH → Izhikevich → composable-sorted-by-bucket** so same-type neurons are
contiguous (minimizes warp divergence).

**Why this matters:** the old code looped *per pool* (`for p: for i in pool[p]`).
For a model like CTX-BG-TH — 8 pools × 10 neurons — each pool pass activated only
~10 of 256 threads (**less than one warp**) and the pools ran sequentially, so
the SM had no concurrent warps to hide latency. Measured cost was ~3.9 µs per
neuron per step at 10 neurons/pool versus ~0.34 µs at 500/pool — an ~11×
starvation tax. The flat loop keeps all neurons' threads active in one pass, so
latency overlaps across warps. This was the single biggest win for small
heterogeneous models (CTX-BG-TH 10s → 7s).

---

## 7. Performance characteristics & limits

- **GPU wins at scale.** Large, homogeneous populations (hundreds–thousands of
  same-type neurons) parallelize well and beat the CPU. See
  `examples/benchmark_complexity_sweep.py`.
- **Small heterogeneous models are hard.** CTX-BG-TH (80 neurons, ~600 synapses)
  is compute/occupancy-bound on roughly one SM. Fusion removed the warp
  starvation, but two ceilings remain:
  - **Consumer fp64 throughput.** The RTX 4080 runs fp64 at ~1/64 of fp32, and
    the gate kinetics are fp64-transcendental-heavy. This sets a hard floor
    (~0.6s for CTX-BG-TH) independent of any kernel restructuring.
  - **Kernel occupancy** is capped by the largest bucket's register count
    (the 11-gate STN path), since all buckets coexist in one kernel.
- **Determinism.** Synaptic accumulation uses `atomicAdd`, whose ordering is
  non-deterministic. Combined with the chaotic dynamics, GPU runs are not
  bit-reproducible run-to-run, and won't match the CPU spike-for-spike. Validate
  with aggregate statistics (firing rates, beta power) or with synapse-free
  isolated populations (which match the CPU to ~1e-13).

**Remaining levers (deliberately not taken here):**
- fp32 / mixed precision for the gate math — ~32× the fp64 transcendental
  throughput on consumer cards; would break exact CPU agreement.
- Whole-model GPU codegen — generate one kernel specialized to the exact model,
  removing per-bucket dispatch and the occupancy ceiling. Belongs to the
  arbitrary-SymPy codegen track.

---

## 8. Extending: adding a new composable bucket

To make a new model shape run on the cooperative kernel:

1. Add its `(NG, NC, NS)` case to `composable_step_dispatch()` in
   `cuda_sim_all.cu` (calls `composable_step_unrolled<NG, NC, NS>`).
2. Add the **same** `(NG, NC, NS)` to `CudaComposablePool::coop_fast_eligible()`
   in `cuda_composable_pool.cu` (keep the two lists in sync).

That's it — the routing and dispatch pick it up automatically. The model must use
only pattern-matched standard forms (no VM expressions, no modulations);
otherwise it correctly falls back to the per-step path.

---

## 9. Key files

| File | Role |
|---|---|
| `src/cpp/src/cuda_sim_all.cu` | Cooperative kernel, fused loop, bucket dispatch, host launcher, neuron layout |
| `src/cpp/include/hodgkin_huxley/cuda_sim_all.hpp` | `CudaComposableDesc`, `DeviceSynapseRaw`, `simulate_all_steps` decl |
| `src/cpp/src/cuda_composable_pool.cu` / `.hpp` | Flat state allocation/upload, `fill_coop_desc`, `coop_fast_eligible` |
| `src/cpp/src/cuda_hh_pool.cu`, `cuda_iz_pool.cu` | Hand-written HH / Izhikevich device steps |
| `src/cpp/src/network.cpp` | `simulate_with_descriptors` — coop-path gating + per-step fallback |
| `src/cpp/src/network/pool_manager.cpp` | `has_coop_capable_cuda_pools`, desc collection |
| `benchmarks/ctxbgth_model.py` | The heterogeneous small-model stress test |
| `examples/benchmark_complexity_sweep.py` | GPU-vs-CPU by model complexity and population size |
