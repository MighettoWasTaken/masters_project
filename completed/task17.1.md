# Task 17.1: PoolBase CUDA Interface + Device Struct

**Role:** Team lead  
**Status:** Not started  
**Depends on:** Nothing — this is the first task, all others block on it  
**Unlocks:** 17.3, 17.5, 17.6, 17.7, 17.9

---

## What to implement

### `src/cpp/include/hodgkin_huxley/pool/pool_base.hpp`

Add five virtual methods to `PoolBase` with default no-op implementations. Existing CPU pools require no changes.

```cpp
// Device identity — used by PoolManager and Network for routing decisions
virtual bool is_cuda()   const { return false; }
virtual int  device_id() const { return -1; }  // -1 = host CPU

// After step() (which may be async), caller must synchronize before reading V_cache.
// CPU pools: no-op. CUDA pools: cudaStreamSynchronize(compute_stream_).
virtual void synchronize() {}

// If true, Network must allocate V_cache and I_syn_buffer as pinned host memory
// (cudaMallocHost) for async transfers. Called once at simulation build time.
virtual bool requires_pinned_memory() const { return false; }

// Migrate all device state to new_device_id. Called by rn.to(device).
// No-op for CPU pools. CUDA pools: free old device memory, alloc on new device, copy.
virtual void migrate_to_device(int /*new_device_id*/) {}
```

### `src/cpp/include/hodgkin_huxley/device.hpp` (new file)

```cpp
#pragma once
#include <string>

namespace hodgkin_huxley {

struct Device {
    enum class Type { CPU, CUDA };
    Type type  = Type::CPU;
    int  index = 0;

    static Device cpu()             { return {Type::CPU,  0}; }
    static Device cuda(int idx = 0) { return {Type::CUDA, idx}; }
    bool operator==(const Device&) const = default;
    std::string str() const;  // "cpu", "cuda:0", etc.
};

int  cuda_device_count();   // returns 0 if not built with HH_USE_CUDA
bool cuda_is_available();

} // namespace hodgkin_huxley
```

Add `device.cpp` under `src/cpp/src/` implementing `str()`, `cuda_device_count()`, `cuda_is_available()`. Behind `HH_USE_CUDA` guard for the CUDA calls; fallback returns 0/false.

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/pool/pool_base.hpp` | Add 5 virtual methods |
| `src/cpp/include/hodgkin_huxley/device.hpp` | New file |
| `src/cpp/src/device.cpp` | New file |
| `src/cpp/CMakeLists.txt` | Add `device.cpp` to library sources |

---

## Contract for downstream tasks

- All CUDA pool implementations (17.5, 17.7, 17.9) override `is_cuda()→true`, `device_id()`, `synchronize()`, `requires_pinned_memory()→true`, `migrate_to_device()`.
- `PoolManager` (17.3) reads `is_cuda()` and `requires_pinned_memory()` to decide memory allocation and synchronization strategy.
- Python bindings (17.10) wrap the `Device` struct.
