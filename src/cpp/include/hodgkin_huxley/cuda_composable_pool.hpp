#pragma once

// =============================================================================
// cuda_composable_pool.hpp — CUDA-routing stub for composable neuron pool
//
// See cuda_hh_pool.hpp for design rationale.  GPU kernels in task 17.8.
// =============================================================================

#ifdef HH_USE_CUDA

#include "hodgkin_huxley/composable_pool.hpp"

namespace hodgkin_huxley {

class CudaComposablePool : public ComposablePool {
public:
    CudaComposablePool(int device_id, const NeuronModelSpec& model,
                       size_t capacity, bool fast_math)
        : ComposablePool(model, capacity, fast_math), device_id_(device_id) {}

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
