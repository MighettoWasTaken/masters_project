#pragma once

// =============================================================================
// cuda_hh_pool.hpp — CUDA-routing stub for Hodgkin-Huxley neuron pool
//
// Inherits HHPool so all neuron-state management (add, scatter, gather, step)
// runs on CPU for now.  Overrides PoolBase CUDA metadata so PoolManager routes
// correctly and Network allocates pinned memory.
//
// Actual GPU kernels (step, scatter_voltages, gather_currents) will override
// the parent methods in task 17.6.
// =============================================================================

#ifdef HH_USE_CUDA

#include "hodgkin_huxley/hh_pool.hpp"

namespace hodgkin_huxley {

class CudaHHPool : public HHPool {
public:
    CudaHHPool(int device_id, size_t capacity, bool fast_math)
        : HHPool(capacity, fast_math), device_id_(device_id) {}

    bool is_cuda()                const override { return true; }
    int  device_id()              const override { return device_id_; }
    bool requires_pinned_memory() const override { return true; }
    void synchronize()                  override;
    void migrate_to_device(int id)      override { device_id_ = id; }

private:
    int device_id_;
};

} // namespace hodgkin_huxley

#endif // HH_USE_CUDA
