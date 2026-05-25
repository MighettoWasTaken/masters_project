#pragma once

#ifdef HH_USE_CUDA

#include "hodgkin_huxley/iz_pool.hpp"
#include <cuda_runtime.h>

namespace hodgkin_huxley {

class CudaIzPool : public IzPool {
public:
    CudaIzPool(int device_id, size_t capacity);
    ~CudaIzPool() override;

    void scatter_voltages(double* V_buf) const override;
    void gather_currents(const double* I_buf) override;
    void step(double dt) override;
    void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const override;
    void scatter_voltages_device(double* d_V_buf) const;
    void gather_currents_device(const double* d_I_buf);

    bool is_cuda()                const override { return true; }
    int  device_id()              const override { return device_id_; }
    bool requires_pinned_memory() const override { return true; }
    void synchronize()                  override;
    void migrate_to_device(int id)      override;

private:
    int device_id_ = 0;
    size_t capacity_ = 0;
    cudaStream_t stream_ = nullptr;

    mutable bool host_state_dirty_ = false;
    bool device_ready_ = false;

    double* d_v_ = nullptr;
    double* d_u_ = nullptr;
    double* d_I_ext_ = nullptr;
    size_t* d_net_idx_ = nullptr;
    double* d_a_ = nullptr;
    double* d_b_ = nullptr;
    double* d_c_ = nullptr;
    double* d_d_ = nullptr;

    void allocate_device();
    void free_device();
    void ensure_device_state();
    void upload_static_parameters();
    void upload_state();
    void upload_currents();
    void upload_index_map();
    void download_state() const;
    void ensure_host_state() const;
};

} // namespace hodgkin_huxley

#endif // HH_USE_CUDA
