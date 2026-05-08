# Task 17.3: PoolManager + Network CUDA Routing

**Role:** Team lead  
**Status:** Not started  
**Depends on:** 17.1 (PoolBase virtual methods), 17.2 (HH_USE_CUDA define)  
**Unlocks:** 17.9 (recording needs pinned memory), 17.10 (Python `.to(device)` needs routing)

---

## What to implement

### `src/cpp/include/hodgkin_huxley/network/pool_manager.hpp`

Add CUDA pool storage and routing to `PoolManager`:

```cpp
#ifdef HH_USE_CUDA
#include "hodgkin_huxley/cuda_hh_pool.hpp"
#include "hodgkin_huxley/cuda_iz_pool.hpp"
#include "hodgkin_huxley/cuda_composable_pool.hpp"
#endif

// New private members:
#ifdef HH_USE_CUDA
    std::optional<CudaHHPool>                         cuda_hh_pool_;
    std::optional<CudaIzPool>                         cuda_iz_pool_;
    std::map<std::string, CudaComposablePool>         cuda_comp_pools_;
    bool                                               use_cuda_ = false;
    int                                                device_id_ = 0;
#endif

// New public methods:
void assign_to_device(int device_id);  // migrate all pools to CUDA device
bool on_cuda() const;                  // true if use_cuda_
void synchronize_cuda() const;         // call synchronize() on all CUDA pools
```

`scatter_all_voltages`, `gather_all_currents`, `step_all`, `sync_all_to_neurons` already delegate to `PoolBase::step()` etc. — no changes needed since CUDA pools override those. The routing point is `build_from_neurons`: when `use_cuda_` is true, construct CUDA pool variants instead of CPU variants.

### `pool_manager.cpp` — `assign_to_device()` implementation

```cpp
void PoolManager::assign_to_device(int device_id) {
#ifdef HH_USE_CUDA
    use_cuda_ = true;
    device_id_ = device_id;
    // migrate existing pools if already built
    if (!hh_pool_.empty())  hh_pool_.migrate_to_device(device_id);
    if (!iz_pool_.empty())  iz_pool_.migrate_to_device(device_id);
    for (auto& kv : comp_pools_) kv.second.migrate_to_device(device_id);
#endif
}

void PoolManager::synchronize_cuda() const {
#ifdef HH_USE_CUDA
    hh_pool_.synchronize();
    iz_pool_.synchronize();
    for (const auto& kv : comp_pools_) kv.second.synchronize();
#endif
}
```

### Network-level: pinned memory

`Network` holds `V_cache_` and `I_syn_buffer_` as `std::vector<double>`. When any pool returns `requires_pinned_memory() == true`, these must be allocated as CUDA pinned memory instead.

Add to `network.hpp` private section:

```cpp
bool use_pinned_memory_ = false;
double* V_cache_pinned_  = nullptr;  // cudaMallocHost
double* I_syn_pinned_    = nullptr;
```

In `Network::simulate_with_descriptors()` (beginning, after pools built), check:

```cpp
bool needs_pinned = false;
#ifdef HH_USE_CUDA
for each pool: if (pool.requires_pinned_memory()) needs_pinned = true;
#endif
if (needs_pinned && !use_pinned_memory_) reallocate_pinned_buffers(N);
```

Use `V_cache_pinned_` / `I_syn_pinned_` in the hot loop when set; fall back to `V_cache_.data()` otherwise.

Add `free_pinned_buffers()` called in destructor and before rebuild.

### `RegionalNetwork` — `.to(device)` routing

Add to `regional_network.hpp` public interface:

```cpp
// Assign all populations to the given CUDA device (or back to CPU).
// Rebuilds pool memory on the target device. Must be called before simulate().
void to(const Device& device);
Device current_device() const;
```

In `regional_network.cpp`:

```cpp
void RegionalNetwork::to(const Device& device) {
    if (device.type == Device::Type::CPU) {
        net_.pool_manager().assign_to_cpu();
    } else {
        net_.pool_manager().assign_to_device(device.index);
    }
    current_device_ = device;
}
```

Add `Device current_device_ = Device::cpu()` to private members.

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/network/pool_manager.hpp` | CUDA pool storage + routing methods |
| `src/cpp/src/network/pool_manager.cpp` | `assign_to_device`, `synchronize_cuda` |
| `src/cpp/include/hodgkin_huxley/network.hpp` | Pinned memory members |
| `src/cpp/src/network.cpp` | Pinned memory allocation/free in hot loop |
| `src/cpp/include/hodgkin_huxley/regional_network.hpp` | `to(Device)`, `current_device()` |
| `src/cpp/src/regional_network.cpp` | `to()` implementation |

---

## Contract for downstream tasks

- After `assign_to_device(id)`, all subsequent `step_all()` calls route to CUDA pools.
- `synchronize_cuda()` must be called by the hot loop after each `step_all()` and before reading `V_cache_pinned_`. Task 17.9 (recording) depends on this timing.
- `requires_pinned_memory()` check + pinned allocation must be complete before 17.9 attempts async recording.
- Python `.to(device)` bindings in task 17.10 call `RegionalNetwork::to()`.
