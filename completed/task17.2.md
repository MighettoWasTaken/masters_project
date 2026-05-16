# Task 17.2: CMake CUDA Build Configuration

**Role:** Team lead  
**Status:** Not started  
**Depends on:** 17.1 (device.hpp/device.cpp must exist to compile)  
**Unlocks:** 17.5, 17.6, 17.7 (all CUDA pool implementations)

---

## What to implement

### Root `CMakeLists.txt`

After the existing `find_package(OpenMP)` block, add CUDA detection:

```cmake
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    # sm_60 minimum: required for atomicAdd(double*) used in synapse kernels
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 CACHE STRING "CUDA architectures")
endif()
```

### `src/cpp/CMakeLists.txt`

Add `device.cpp` to the library sources (from 17.1).

Below the existing OpenMP block, add CUDA sources conditionally:

```cmake
if(CUDAToolkit_FOUND)
    target_sources(hodgkin_huxley_core PRIVATE
        src/cuda_hh_pool.cu
        src/cuda_iz_pool.cu
        src/cuda_composable_pool.cu
        src/cuda_synapse.cu
    )
    target_link_libraries(hodgkin_huxley_core PUBLIC CUDA::cudart)
    target_compile_definitions(hodgkin_huxley_core PRIVATE HH_USE_CUDA)
    set_target_properties(hodgkin_huxley_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()
```

The `.cu` files will be stubs (empty or minimal) when first added — they're populated by 17.5–17.8. Add them as empty files now so the build system is wired up and the `HH_USE_CUDA` define propagates.

### Stub `.cu` files to create

Create empty stubs (one `// placeholder` comment) so CMake doesn't error:

- `src/cpp/src/cuda_hh_pool.cu`
- `src/cpp/src/cuda_iz_pool.cu`
- `src/cpp/src/cuda_composable_pool.cu`
- `src/cpp/src/cuda_synapse.cu`

---

## Key files

| File | Change |
|---|---|
| `CMakeLists.txt` | Add `find_package(CUDAToolkit)` + `enable_language(CUDA)` |
| `src/cpp/CMakeLists.txt` | Add `device.cpp` + conditional `.cu` sources + `HH_USE_CUDA` define |
| `src/cpp/src/cuda_hh_pool.cu` | New stub |
| `src/cpp/src/cuda_iz_pool.cu` | New stub |
| `src/cpp/src/cuda_composable_pool.cu` | New stub |
| `src/cpp/src/cuda_synapse.cu` | New stub |

---

## Contract for downstream tasks

- `HH_USE_CUDA` preprocessor define gates all CUDA code in 17.3, 17.5–17.9. CPU-only builds must compile cleanly with none of it.
- `CUDA::cudart` is the only required CUDA library link at this stage. `CUDA::cublas` / `CUDA::cufft` are not needed.
- `CMAKE_CUDA_ARCHITECTURES` must include `60` as minimum (atomicAdd double).
