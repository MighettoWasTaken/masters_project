#pragma once

#ifdef HH_USE_CUDA

#include "hodgkin_huxley/composable_pool.hpp"
#include "hodgkin_huxley/cuda_vm.hpp"
#include <cuda_runtime.h>

namespace hodgkin_huxley {

struct CudaGateDesc {
    int update_form = 0;
    int dependency = 0;
    int intracellular_idx = 0;
    double scale = 1.0;

    BoltzmannParams inf;
    TauParams tau;
    RateFuncParams alpha;
    RateFuncParams beta;

    int derived_source_gate = -1;
    double derived_a = 1.0;
    double derived_b = 0.0;
    double derived_c = 1.0;

    int inf_vm_idx = -1;
    int tau_vm_idx = -1;
    int alpha_vm_idx = -1;
    int beta_vm_idx = -1;
    int dxdt_vm_idx = -1;
};

struct CudaChannelGateRef {
    int gate_idx = 0;
    int power = 1;
};

struct CudaChannelDesc {
    double g = 0.0;
    double E_rev = 0.0;
    int nernst_substance_idx = -1;
    int is_ahp = 0;
    double ahp_k1 = 0.0;
    int ahp_substance_idx = 0;

    int gate_ref_start = 0;
    int gate_ref_count = 0;
    int gate_product_vm_idx = -1;
};

struct CudaIntracellularDesc {
    int update_form = 0;
    double initial = 0.0;
    double epsilon = 1e-4;
    double k_decay = 15.0;

    int source_start = 0;
    int source_count = 0;

    int nernst_enabled = 0;
    double nernst_Ca_o = 2000.0;
    double nernst_z = 2.0;
    double nernst_R = 8314.0;
    double nernst_F = 96485.0;
    double nernst_T = 298.0;

    int nernst_vm_idx = -1;
    int ode_vm_idx = -1;

    int mod_start = 0;
    int mod_count = 0;
};

struct CudaIntracellularModDesc {
    int target = 0;
    int target_idx = -1;
    int substance_idx = 0;
    int vm_idx = -1;
    double shift_scale = 0.0;
};

class CudaComposablePool : public ComposablePool {
public:
    static constexpr int kMaxGates = 32;
    static constexpr int kMaxChannels = 32;
    static constexpr int kMaxSubstances = 16;
    static constexpr int kMaxMods = 64;

    CudaComposablePool(int device_id, const NeuronModelSpec& model,
                       size_t capacity, bool fast_math);
    ~CudaComposablePool() override;

    void scatter_voltages(double* V_buf) const override;
    void gather_currents(const double* I_buf) override;
    void step(double dt) override;
    void sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const override;

    void scatter_gate_states_into(double* buf, size_t max_gates,
                                  size_t n_rec, size_t t_rec) const override;
    void scatter_substance_into(size_t subst_idx, double* buf,
                                size_t n_rec, size_t t_rec) const override;
    void scatter_synapse_g_scale(double* buf) const override;

    void scatter_voltages_device(double* d_V_buf) const;
    void gather_currents_device(const double* d_I_buf);
    void scatter_synapse_g_scale_device(double* d_buf) const;

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

    double* d_V_ = nullptr;
    double* d_I_ext_ = nullptr;
    double* d_synapse_g_scale_ = nullptr;
    size_t* d_net_idx_ = nullptr;

    std::vector<double*> gate_device_ptrs_;
    std::vector<double*> substance_device_ptrs_;
    std::vector<double*> nernst_device_ptrs_;
    double** d_gate_ptrs_ = nullptr;
    double** d_substance_ptrs_ = nullptr;
    double** d_nernst_ptrs_ = nullptr;

    CudaGateDesc* d_gate_descs_ = nullptr;
    CudaChannelDesc* d_channel_descs_ = nullptr;
    CudaChannelGateRef* d_channel_gate_refs_ = nullptr;
    CudaIntracellularDesc* d_intr_descs_ = nullptr;
    int* d_intr_source_channels_ = nullptr;
    CudaIntracellularModDesc* d_mod_descs_ = nullptr;

    std::vector<DeviceVmProgram> vm_programs_;
    DeviceVmProgram* d_vm_programs_ = nullptr;

    void allocate_device();
    void free_device();
    void upload_state();
    void upload_currents();
    void upload_index_map();
    void upload_pointer_tables();
    void upload_model_metadata();
    void download_state() const;
    void ensure_device_state();
    void ensure_host_state() const;

    int register_vm_program(const VmExpr& expr);
};

} // namespace hodgkin_huxley

#endif // HH_USE_CUDA
