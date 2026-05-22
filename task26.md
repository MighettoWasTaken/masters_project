# Task 26: Forward-Injection Spike Delivery

**Depends on:** task16 (SynArrays layout, simulation loop structure)  
**Unlocks:** task19 (multi-GPU SpikeTransport benefits from reduced per-neuron buffer overhead)

---

## Problem

The current spike delivery system allocates one `std::vector<bool>` ring buffer **per synapse**, sized to `ceil(delay / dt)` steps. This has two compounding problems:

**Memory:** `O(N_synapses × max_delay / dt)` bits, plus 24 bytes `vector` metadata per synapse regardless of size. At 10M synapses, delay=10ms, dt=0.01ms: ~1.2 GB for the buffers alone, ~240 MB for metadata.

**Write bandwidth:** Phase 1 of the simulation loop iterates every synapse every step to push/pop the ring buffer — `O(N_synapses)` work per step regardless of firing activity. The Phase 2 activation scan (checking `spike_detected_[k]` to promote newly spiked synapses to the active list) is also O(N_synapses). The conductance update itself already runs only on active synapses, but these two passes dominate in a silent network.

The root cause is that spike history is replicated: every synapse from the same pre-synaptic neuron stores an independent copy of identical data. The fix inverts the direction of data flow.

---

## Design

### 26.1 Core: Forward Injection

Instead of each synapse pulling from a ring buffer every step, spikes are **pushed forward** into a circular event buffer only when a spike actually occurs.

**`post_from_` lookup table** — built once at `ensure_buffers()` time:

```cpp
struct SynapseRef {
    size_t   syn_idx;       // index into SynArrays (S, A, g, etc.)
    uint32_t delay_steps;   // precomputed: round(delay[i] / dt)
};

// post_from_[pre_neuron_idx] = all local synapses receiving from that neuron
std::vector<std::vector<SynapseRef>> post_from_;  // size = N_neurons
```

`post_from_` stores connectivity on the **receiving side**. Each `SynapseRef` is 12 bytes; total memory is O(N_synapses), the same as the current `pre[]` array. Construction is one pass over SynArrays.

**Circular event buffer** — one per thread, size `max_delay_steps + 1`:

```cpp
// event_slots_[step % capacity] = syn_indices that activate at that step
std::vector<std::vector<size_t>> event_slots_;  // size = max_delay_steps + 1
```

Each slot holds the indices of synapses whose delayed spike arrives at that step. Slots are cleared after processing and reused.

### 26.2 Simulation Loop Changes

**Phase 2 — Forward injection** (replaces per-synapse ring buffer push):

```cpp
// O(1) per silent step; O(N_spikes × avg_fanout) when spikes occur
for (size_t n = 0; n < N; ++n) {
    if (!spike_detected_[n]) continue;
    for (const SynapseRef& syn : post_from_[n]) {
        size_t target = (current_step + syn.delay_steps) % event_slots_.size();
        event_slots_[target].push_back(syn.syn_idx);
    }
}
```

**Phase 3 — Event dispatch** (replaces per-synapse ring buffer read):

```cpp
// Process synapses due this step — only those with a pending delivery
size_t slot = current_step % event_slots_.size();
for (size_t k : event_slots_[slot]) {
    sa_.S[k] += cache.delta_S;     // conductance increment
    // ... STDP trace updates etc.
}
event_slots_[slot].clear();        // reset slot for reuse in max_delay steps
```

This replaces the existing loop that iterated all N_synapses every step checking `spike_buf`.

### 26.3 Thread-Pair Queue Extension (multi-thread)

For OpenMP thread groups (task16), threads own disjoint subsets of neurons. When thread A detects a spike in its neurons and thread B owns the target synapses, A must notify B. The thread-pair queue carries this notification cheaply.

Each ordered pair `(src_thread, dst_thread)` has one queue. Entries are tagged:

```cpp
// A step with no spikes from this thread: one QUIET entry (compressed)
// A step with spikes: one SPIKE entry listing fired pre-neuron indices

enum class QueueTag : uint8_t { QUIET, SPIKE };

struct QueueEntry {
    QueueTag             tag;
    int32_t              quiet_steps;  // used when tag == QUIET
    std::vector<uint32_t> indices;     // used when tag == SPIKE
};
```

**Producer (thread A, once per step):** If any neurons in A's group fired, push `SPIKE([fired_indices])`. Otherwise push `QUIET(1)` or increment the tail QUIET entry. This is O(1) per silent step regardless of fan-out.

**Consumer (thread B, once per step):** Pop the head entry. For each SPIKE index, look up `post_from_[idx]` (which is restricted to B's local synapses) and inject into `event_slots_`. QUIET entries are skipped — no synapse work at all.

The key property: the queue carries **neuron indices** (one `uint32_t` per fired neuron), not booleans replicated per synapse. For a neuron with fan-out 10,000 firing at 100 Hz into a remote thread, the current system writes 10,000 booleans per step to that thread's synapses; the queue sends one 4-byte index every ~100 steps.

For single-thread builds (or when all synapses are thread-local), the thread-pair queues are absent and the forward injection loop above runs directly.

### 26.4 Event Buffer Sizing

`event_slots_` must be large enough to hold the maximum delay:

```cpp
size_t max_delay_steps = *std::max_element(delay_steps_.begin(), delay_steps_.end());
event_slots_.assign(max_delay_steps + 1, {});  // +1 avoids off-by-one at wrap
```

Memory cost: `(max_delay_steps + 1)` empty `vector` objects (each 24 bytes on 64-bit = 24 KB for 1000 slots). Populated slots hold only in-flight spike deliveries — at steady state this is approximately `N_spikes_in_flight = N_neurons × firing_rate × max_delay × avg_fanout`. For 100K neurons, 10 Hz, 10ms delay, fanout 1000: ~10M entries in flight — 80 MB of `size_t`. This is proportional to actual network activity, not the static worst case.

For networks with very high delay/dt ratios, the `event_slots_` vector of vectors can be replaced with a flat arena allocator to amortise per-slot allocation overhead.

---

## Memory Comparison

| | Current | Task 26 |
|---|---|---|
| Ring buffers | N_synapses × vector<bool>(delay/dt) | — eliminated — |
| Per-synapse metadata | 24 bytes × N_synapses | — eliminated — |
| Per-synapse consumer state | ring buffer index + head ptr | — eliminated — |
| post_from_ lookup | — | 12 bytes × N_synapses (same as current pre[] array) |
| event_slots_ | — | 24 bytes × (max_delay/dt) + in-flight entries |
| Phase 1 ring buffer | O(N_synapses) per step | eliminated |
| Phase 2 activation scan | O(N_synapses) per step | eliminated (event_slots delivers directly) |
| Writes per spike | O(fan_out) | O(fan_out) [same — forward injection] |
| Memory (10M syn, delay=10ms, dt=0.01ms) | ~1.5 GB | ~120 MB post_from_ + ~few MB event_slots_ |

---

## Changes Required

### `src/cpp/include/hodgkin_huxley/network.hpp`

Remove from `SynArrays`:
```cpp
std::vector<std::vector<bool>> spike_buf;
std::vector<size_t>            buf_head;
std::vector<bool>              delay_init;
```

`SynArrays` gains no new fields — `delay` already exists and `delay_steps` is derived locally during buffer construction.

Add to `Network` private members:
```cpp
std::vector<std::vector<SynapseRef>> post_from_;    // [pre_idx] → local synapses
std::vector<std::vector<size_t>>     event_slots_;  // [step % capacity] → syn_indices due
std::vector<size_t>                  delay_steps_;  // precomputed per-synapse delay/dt
```

### `src/cpp/src/network.cpp`

**Initialisation** (`ensure_buffers`):
```cpp
// Precompute delay_steps_ once
delay_steps_.resize(N_synapses);
size_t max_delay_steps = 0;
for (size_t i = 0; i < N_synapses; ++i) {
    delay_steps_[i] = static_cast<size_t>(std::round(sa_.delay[i] / dt));
    max_delay_steps = std::max(max_delay_steps, delay_steps_[i]);
}

// Build post_from_ lookup table
post_from_.assign(N_neurons, {});
for (size_t i = 0; i < N_synapses; ++i)
    post_from_[sa_.pre[i]].push_back({i, static_cast<uint32_t>(delay_steps_[i])});

// Allocate event buffer (cleared on each simulate() call)
event_slots_.assign(max_delay_steps + 1, {});
```

**Simulation loop — forward injection** (replaces ring buffer write in phase 2):
```cpp
for (size_t n = 0; n < N; ++n) {
    if (!spike_detected_[n]) continue;
    for (const SynapseRef& syn : post_from_[n]) {
        size_t slot = (step + syn.delay_steps) % event_slots_.size();
        event_slots_[slot].push_back(syn.syn_idx);
    }
}
```

**Simulation loop — event dispatch** (replaces ring buffer read in phase 3):
```cpp
size_t slot = step % event_slots_.size();
for (size_t k : event_slots_[slot]) {
    const auto& cache = synapse_spec_caches_[sa_.spec_idx[k]];  // task27 path
    sa_.S[k] += cache.delta_S;
    // STDP trace updates if applicable
}
event_slots_[slot].clear();
```

**`reset()`** — clear all event slots (connectivity unchanged):
```cpp
for (auto& slot : event_slots_) slot.clear();
```

---

## Implementation Checklist

### Data structures
- [ ] Add `SynapseRef` struct to `network.hpp`
- [ ] Add `post_from_`, `event_slots_`, `delay_steps_` to `Network`
- [ ] Remove `spike_buf`, `buf_head`, `delay_init` from `SynArrays`

### Network integration
- [ ] Build `post_from_` and `event_slots_` in `ensure_buffers()`
- [ ] Replace ring-buffer write loop with forward injection (phase 2)
- [ ] Replace ring-buffer read loop with event dispatch (phase 3)
- [ ] Update `reset()` to clear event slots

### Correctness
- [ ] All existing tests pass with no behaviour change
- [ ] Spike delivery timing verified at boundary delays (delay = 1×dt, delay = 2×dt)
- [ ] Multi-synapse fan-out test: one pre-neuron, N post-synaptic neurons at varied delays — all receive spikes at correct times
- [ ] Verify STDP weight updates still receive spike signals at correct timing

### Performance
- [ ] Memory usage benchmark: before vs after at N_synapses = 1M, 10M
- [ ] Step throughput: verify O(1) per step in silent network (no spike activity)
- [ ] Step throughput: verify improvement vs baseline on physiological firing rates (~10–100 Hz)
