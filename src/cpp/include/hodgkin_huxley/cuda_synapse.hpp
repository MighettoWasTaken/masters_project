#pragma once

#ifdef HH_USE_CUDA

#include "hodgkin_huxley/cuda_vm.hpp"
#include "hodgkin_huxley/model/synapse_spec.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace hodgkin_huxley {

struct DeviceSynapseSpecDesc {
    int update_form = 0;
    int current_form = 0;
    int power = 1;

    double spec_g = 0.0;
    double delta_S = 0.0;
    double delta_A = 0.0;
    double tau_S = 1.0;
    double tau_A = 1.0;
    double norm_factor = 1.0;

    double tanh_amp = 0.0;
    double tanh_vh = 0.0;
    double tanh_k = 1.0;
    double tau_decay = 1.0;

    BoltzmannParams s_inf;
    TauParams tau;
    RateFuncParams alpha;
    RateFuncParams beta;

    double mg_conc = 1.0;
    double mg_scale = 0.062;
    double mg_denom = 3.57;

    int dS_vm_idx = -1;
    int dA_vm_idx = -1;
    int current_vm_idx = -1;
};

struct DeviceSynapseArrays {
    int device_id = 0;
    double dt = -1.0;
    double spike_threshold = 0.0;

    size_t neuron_count = 0;
    size_t synapse_count = 0;
    size_t ring_size = 0;
    size_t spec_count = 0;
    // Minimum delay across all synapses, in steps. If >= 1, the GPU kernel
    // can skip the P6→P7 within-step barrier (since write_slot != read_slot).
    uint32_t min_delay_steps = 0;

    double* d_weight = nullptr;
    double* d_g = nullptr;
    double* d_S = nullptr;
    double* d_A = nullptr;
    double* d_E_syn = nullptr;

    uint32_t* d_pre = nullptr;
    uint32_t* d_post = nullptr;
    uint32_t* d_delay_steps = nullptr;
    uint32_t* d_spec_idx = nullptr;

    DeviceSynapseSpecDesc* d_spec_descs = nullptr;

    uint8_t* d_spike_ring = nullptr;
    uint8_t* d_spike_arrived = nullptr;
    uint8_t* d_neuron_spiked = nullptr;
    double* d_V_prev = nullptr;

    std::vector<DeviceVmProgram> vm_programs;
    DeviceVmProgram* d_vm_programs = nullptr;
};

void allocate_device_synapses(DeviceSynapseArrays& dev,
                              size_t neuron_count,
                              size_t synapse_count,
                              size_t ring_size,
                              size_t spec_count,
                              int device_id);

void free_device_synapses(DeviceSynapseArrays& dev);

void upload_device_synapse_state(
    DeviceSynapseArrays& dev,
    const std::vector<double>& weight,
    const std::vector<double>& g,
    const std::vector<double>& S,
    const std::vector<double>& A,
    const std::vector<float>& E_syn,
    const std::vector<uint32_t>& pre,
    const std::vector<uint32_t>& post,
    const std::vector<uint32_t>& delay_steps,
    const std::vector<uint32_t>& spec_idx,
    const std::vector<DeviceSynapseSpecDesc>& spec_descs,
    const std::vector<VmExpr>& vm_exprs,
    const std::vector<double>& V_prev,
    const std::vector<uint8_t>& spike_ring,
    double dt,
    double spike_threshold);

void download_device_synapse_state(
    const DeviceSynapseArrays& dev,
    std::vector<double>& g,
    std::vector<double>& S,
    std::vector<double>& A,
    std::vector<double>& V_prev,
    std::vector<uint8_t>* spike_ring = nullptr,
    std::vector<uint8_t>* spike_arrived = nullptr);

void fill_device_buffer(double* d_buf, double value, size_t n);

void accumulate_cuda_synaptic_currents(
    const DeviceSynapseArrays& dev,
    const double* d_V_cache,
    double* d_I_syn,
    const double* d_mod_factor);

void update_cuda_synapses(
    DeviceSynapseArrays& dev,
    const double* d_V_cache,
    size_t current_step);

} // namespace hodgkin_huxley

#endif // HH_USE_CUDA
