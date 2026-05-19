#pragma once

// =============================================================================
// cuda_iz_pool.hpp — CUDA-routing stub for Izhikevich neuron pool
//
// See cuda_hh_pool.hpp for design rationale.  GPU kernels in task 17.7.
// =============================================================================

#ifdef HH_USE_CUDA

#include "hodgkin_huxley/iz_pool.hpp"

namespace hodgkin_huxley {

class CudaIzPool : public IzPool {
public:
    CudaIzPool(int device_id, size_t capacity)
        : IzPool(capacity), device_id_(device_id) {}

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
