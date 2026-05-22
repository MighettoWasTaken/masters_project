# Task 19: Multi-GPU Parallelism

## Priority: 3 — Depends on task17 (CUDA device API), task16 (SpikeTransport abstraction)

## Overview

Extend the single-GPU CUDA backend (task17) to support distributed simulation across multiple GPUs. The primary use case is extremely large networks (N > 50,000 neurons or full-scale cortical column models) that exceed single-GPU memory or compute capacity. Population-to-device assignment follows PyTorch's device model (`hh.device("cuda:0")`); inter-device spike transfer uses the `SpikeTransport` abstraction introduced in task16 with a CUDA P2P or NCCL backend.

Expected scaling: near-linear with GPU count for networks where inter-population synaptic delays exceed inter-device transfer latency (~1–4 μs on NVLink).

---

## 19.1 Device Model

Following PyTorch conventions, devices are first-class objects:

```python
import hodgkin_huxley as hh

# Device specification
cpu  = hh.device("cpu")
gpu0 = hh.device("cuda:0")
gpu1 = hh.device("cuda:1")
gpu  = hh.device("cuda")      # default GPU (index 0)

# Device utilities
hh.device_count()              # number of available CUDA devices
hh.is_available("cuda")        # bool — is any CUDA device present?
hh.device_properties(0)        # name, total_memory, compute_capability, ...
```

The `Device` struct is introduced in task17 and extended here:

```cpp
// Defined in device.hpp (task17)
struct Device {
    enum class Type { CPU, CUDA };
    Type type  = Type::CPU;
    int  index = 0;         // CUDA device index; ignored for CPU

    static Device cpu()             { return {Type::CPU,  0}; }
    static Device cuda(int idx = 0) { return {Type::CUDA, idx}; }
    bool operator==(const Device&) const = default;
};
```

---

## 19.2 Population Device Assignment

Following PyTorch's `.to(device)` convention:

```python
rn = hh.RegionalNetwork()
# ... define populations ...

# Move entire network to one GPU (task17 single-GPU path — unchanged)
rn.to(hh.device("cuda:0"))

# Per-population assignment for multi-GPU
rn.assign("CTX_e", device=hh.device("cuda:0"))
rn.assign("CTX_i", device=hh.device("cuda:0"))
rn.assign("STN",   device=hh.device("cuda:1"))
rn.assign("GPe",   device=hh.device("cuda:1"))
rn.assign("GPi",   device=hh.device("cuda:1"))

# Query assignment
rn.device("CTX_e")     # hh.device("cuda:0")
rn.device_map()        # dict[str, hh.device] — all populations

# Automatic load-balanced assignment
rn.auto_assign_devices()   # distributes populations by neuron count across all available GPUs
```

**Constraint:** Populations with self-connections at zero synaptic delay must reside on the same device. `auto_assign_devices()` respects this automatically; manual `assign()` raises `DeviceConstraintError` if violated.

C++ additions to `RegionalNetwork`:

```cpp
void assign(const std::string& population, Device d);
Device device(const std::string& population) const;
std::unordered_map<std::string, Device> device_map() const;
void auto_assign_devices();
```

---

## 19.3 Inter-Device Communication

Cross-device synaptic current contributions require transferring spike vectors between GPUs at each synchronisation point. The `SpikeTransport` abstraction from task16 §16.2 provides the interface; task19 adds two CUDA-capable implementations.

### CUDA P2P Transport (2–4 GPUs, NVLink / PCIe peer access)

Uses `cudaMemcpyPeer` for direct device-to-device transfer without routing through host memory. Falls back to pinned staging buffers if peer access is unavailable.

```cpp
class CUDAP2PTransport : public SpikeTransport {
public:
    explicit CUDAP2PTransport(const std::vector<int>& device_indices);

    void send(int src_device, int dst_device,
              const SpikeEvent* events, size_t count) override;
    size_t recv(int src_device, SpikeEvent* buffer, size_t capacity) override;
    void flush() override;   // cudaStreamSynchronize per device

private:
    std::vector<cudaStream_t> streams_;   // one per device
    std::vector<SpikeEvent*>  staging_;   // pinned host buffers (fallback)
    bool peer_access_enabled_[16][16];    // device-pair capability matrix
};
```

### NCCL Transport (> 4 GPUs or no peer access)

For systems with many GPUs or without direct peer access. Uses `ncclAllGather` of the spike vector at each synchronisation point.

```cpp
class NCCLTransport : public SpikeTransport {
public:
    explicit NCCLTransport(const std::vector<int>& device_indices);

    void send(int src_device, int dst_device,
              const SpikeEvent* events, size_t count) override;
    size_t recv(int src_device, SpikeEvent* buffer, size_t capacity) override;
    void flush() override;   // ncclGroupEnd + synchronize

private:
    ncclComm_t comm_;
    std::vector<cudaStream_t> streams_;
};
```

**Automatic transport selection:**
- 1 GPU: `LocalTransport` (no-op transfer — task17 path unchanged)
- 2–4 GPUs with peer access: `CUDAP2PTransport`
- > 4 GPUs or no peer access: `NCCLTransport`

---

## 19.4 Synchronisation Protocol

The synchronisation schedule is determined by the delay decomposition algorithm from task16 §16.2. The minimum inter-population delay `D` across a device boundary defines how many steps can advance before a spike transfer is required:

```
sync_interval = floor(min_cross_device_delay / dt)   # steps between transfers
```

For typical inter-area delays of 4–6 ms at dt=0.01 ms: sync every 400–600 steps. A PCIe spike-vector transfer of ~10,000 spike events takes ~10 μs — well within the 4 ms compute budget.

**Per-step algorithm:**
1. Each device advances its populations independently for `sync_interval` steps
2. At the sync boundary, each device sends its accumulated spike vector to downstream devices via `SpikeTransport::send`
3. `flush()` synchronises all streams / NCCL collectives
4. Receiving devices inject the delayed spikes into their synapse ring buffers
5. Repeat

---

## 19.5 Memory Layout

Each device holds only the populations assigned to it:

- Neuron pool state (`V`, gate arrays, intracellular substances)
- Local synapses (both endpoints on the same device)
- Cross-device synapse arrays: stored on the **receiving** device; spike events from the sending device drive conductance updates

```cpp
class MultiDeviceSimContext {
public:
    MultiDeviceSimContext(const std::unordered_map<std::string, Device>& device_map,
                          size_t n_neurons, size_t n_synapses);
    ~MultiDeviceSimContext();

    CUDASimContext& context(int device_index);
    SpikeTransport& transport();

    void sync_spike_vectors();       // execute cross-device transfers
    void upload_all();
    void download_all();

private:
    std::vector<CUDASimContext>       contexts_;   // one per CUDA device
    std::unique_ptr<SpikeTransport>   transport_;
    std::vector<CrossDeviceSynArrays> cross_device_synapses_;
};
```

---

## 19.6 Limitations

- Requires CUDA Toolkit ≥ 11.0; NCCL path additionally requires NCCL 2.x
- P2P transport requires peer access between device pairs; automatically falls back to staged host-memory copy if unavailable
- Zero-delay recurrent connections must be co-located on the same device
- Plasticity rules (task15) that cross device boundaries are not supported in the first implementation — plastic synapses must have both endpoints on the same device
- Mixed-precision (float32 pools, float64 accumulation) remains out of scope

---

## 19.7 Extension Path: Multi-Machine (Hybrid MPI+OpenMP)

The current architecture is not a dead end for multi-machine scaling. The natural extension is **hybrid MPI+OpenMP**: shared memory within each node (existing OpenMP path), MPI for inter-node spike exchange. This is the same model used by NEST and other production simulators.

The current codebase already has most of the required building blocks:

- **`set_thread_groups` + inter-group delay constraint** — the same invariant (`delay >= dt` at partition boundaries) applies at the MPI boundary, just with a larger enforced d_min matched to MPI round-trip latency rather than dt. No new synchronisation concept is needed.
- **`SpikeTransport` abstraction (task16 §16.2)** — the seam is already there. Adding an `MPITransport` backend behind it is an implementation addition, not a redesign. Local delivery stays in shared memory; inter-node delivery goes through MPI.
- **`assign()` / `device_map()`** — population-to-device assignment introduced here maps cleanly to population-to-rank assignment. Extending `Device` with a `rank` field or adding a parallel `MachineAssignment` struct covers it.
- **Spike event buffers** — already compact integer arrays (indices + timestamps). Trivially serialisable for `MPI_Send`/`MPI_Recv` without additional packing.

What would actually need to be added:
1. `MPITransport` backend implementing `SpikeTransport::send/recv/flush` via MPI collective or CPEX pairwise exchange
2. Rank-aware population partitioner (extends `auto_assign_devices()`)
3. d_min enforcement at MPI boundaries (larger than dt — typically 0.1–1 ms for InfiniBand)
4. Optional: replace blocking MPI exchange with non-blocking (`MPI_Isend`/`MPI_Irecv`) once d_min window is large enough to hide latency

At the scales tested so far (tens of millions of synapses, single workstation), shared memory is strictly faster than any MPI approach. The crossover where MPI overhead is justified is roughly hundreds of millions of synapses — the equivalent of a dense cortical column model or larger.

---

## 19.8 Implementation Checklist

### CMake / Build
- [ ] Add `USE_NCCL` CMake option; `find_package(NCCL)` when enabled
- [ ] Guard NCCL-specific code behind `HH_USE_NCCL` preprocessor define
- [ ] CI: add multi-GPU job (skip on single-GPU runners)

### Transport Layer
- [ ] Implement `CUDAP2PTransport` with `cudaDeviceCanAccessPeer` capability check and pinned-buffer fallback
- [ ] Implement `NCCLTransport` using `ncclAllGather` spike vector broadcast
- [ ] Implement auto-transport selection in `MultiDeviceSimContext` constructor

### Population Assignment
- [ ] Add `device_map_` to `RegionalNetwork`
- [ ] Implement `RegionalNetwork::assign()` with zero-delay constraint validation
- [ ] Implement `RegionalNetwork::auto_assign_devices()` (balance by neuron count, respect zero-delay constraints)
- [ ] Bind `assign()`, `device_map()`, `auto_assign_devices()` in Python

### Simulation Loop
- [ ] Implement `MultiDeviceSimContext`: per-device `CUDASimContext` pool + transport
- [ ] Implement `CrossDeviceSynArrays`: receiving-device storage for inter-device projections
- [ ] Implement `sync_spike_vectors()` using `SpikeTransport::send`/`recv`/`flush`
- [ ] Extend `RegionalNetwork::simulate()` to dispatch to multi-device path when `device_map_` contains > 1 unique device

### Tests
- [ ] Verify multi-GPU simulation produces byte-identical output to single-GPU (2-GPU test, small network)
- [ ] Verify spike delivery accuracy across device boundaries at varying delays
- [ ] Benchmark: scaling efficiency at 1, 2, 4 GPUs for N = 10,000, 50,000, 100,000
