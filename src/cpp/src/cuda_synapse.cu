#include "hodgkin_huxley/cuda_synapse.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

__device__ inline double cuda_boltzmann_scalar(
    double x, const hodgkin_huxley::BoltzmannParams& p)
{
    double arg = -(x - p.v_half) / p.k;
    if (arg > 500.0) return 0.0;
    if (arg < -500.0) return 1.0;
    return 1.0 / (1.0 + exp(arg));
}

__device__ inline double cuda_compute_tau_scalar(
    double V, const hodgkin_huxley::TauParams& p)
{
    switch (p.form) {
        case hodgkin_huxley::TauParams::Form::CONSTANT:
            return p.params[0];
        case hodgkin_huxley::TauParams::Form::BOLTZMANN: {
            double arg = -(V - p.params[2]) / p.params[3];
            arg = fmax(-500.0, fmin(500.0, arg));
            return p.params[0] + p.params[1] / (1.0 + exp(arg));
        }
        case hodgkin_huxley::TauParams::Form::DOUBLE_EXP_SUM: {
            double e1 = exp((V + p.params[2]) / p.params[3]);
            double e2 = exp(-(V + p.params[5]) / p.params[6]);
            double denom = e1 + e2;
            if (denom < 1e-10) denom = 1e-10;
            return p.params[0] + p.params[1] / denom;
        }
        case hodgkin_huxley::TauParams::Form::OFFSET_DOUBLE_EXP: {
            double x1 = (V + p.params[2]) / p.params[3];
            double x2 = (V + p.params[5]) / p.params[6];
            return p.params[0] + p.params[1] * exp(-x1 * x1)
                               + p.params[4] * exp(-x2 * x2);
        }
        case hodgkin_huxley::TauParams::Form::SCALED_EXP: {
            double arg = (V - p.params[1]) / (2.0 * p.params[2]);
            arg = fmax(-500.0, fmin(500.0, arg));
            double ch = cosh(arg);
            if (ch < 1e-10) ch = 1e-10;
            return p.params[0] / ch;
        }
        case hodgkin_huxley::TauParams::Form::COMPOUND_AB: {
            double alpha = p.params[0] * exp((V + p.params[1]) / p.params[2]);
            double beta = p.params[3] * exp((V + p.params[4]) / p.params[5]);
            double sum = alpha + beta;
            if (sum < 1e-10) sum = 1e-10;
            return 1.0 / sum;
        }
    }
    return 1.0;
}

__device__ inline double cuda_compute_rate_scalar(
    double V, const hodgkin_huxley::RateFuncParams& p)
{
    switch (p.form) {
        case hodgkin_huxley::RateFuncParams::Form::LINEAR_OVER_EXP: {
            double x = V + p.B;
            double xc = x / p.C;
            if (fabs(xc) < 1e-6) return p.A * p.C * (1.0 + xc * 0.5);
            return p.A * x / (exp(xc) - 1.0);
        }
        case hodgkin_huxley::RateFuncParams::Form::EXP_DECAY: {
            double arg = (V + p.B) / p.C;
            arg = fmax(-500.0, fmin(500.0, arg));
            return p.A * exp(arg);
        }
        case hodgkin_huxley::RateFuncParams::Form::LINEAR_OVER_EXPM1: {
            double x = V + p.B;
            double xc = x / p.C;
            if (fabs(xc) < 1e-6) return p.A * p.C * (1.0 + xc * 0.5);
            return p.A * x / (1.0 - exp(-xc));
        }
        case hodgkin_huxley::RateFuncParams::Form::SIGMOID: {
            double arg = (V + p.B) / p.C;
            arg = fmax(-500.0, fmin(500.0, arg));
            return p.A / (1.0 + exp(arg));
        }
    }
    return 0.0;
}

__global__ void fill_buffer_kernel(double* buf, double value, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = value;
}

__global__ void detect_neuron_spikes_kernel(
    const double* V,
    double* V_prev,
    uint8_t* neuron_spiked,
    int n_neurons,
    double threshold)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    const double v = V[i];
    neuron_spiked[i] = (v > threshold && V_prev[i] <= threshold) ? 1u : 0u;
    V_prev[i] = v;
}

__global__ void update_spike_ring_kernel(
    const uint8_t* neuron_spiked,
    uint8_t* spike_ring,
    int n_neurons,
    int write_slot,
    int ring_size)
{
    // Per-NEURON ring: one entry per neuron, not per synapse.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    spike_ring[i * ring_size + write_slot] = neuron_spiked[i] ? 1u : 0u;
}

__global__ void update_synapse_state_kernel(
    double* g,
    double* S,
    double* A,
    const double* E_syn,
    const double* weight,
    const uint32_t* pre,
    const uint32_t* post,
    const uint32_t* delay_steps,
    const uint32_t* spec_idx,
    const double* V,
    const uint8_t* spike_ring,
    uint8_t* spike_arrived,
    const hodgkin_huxley::DeviceSynapseSpecDesc* spec_descs,
    const hodgkin_huxley::DeviceVmProgram* vm_programs,
    int n_synapses,
    int ring_size,
    size_t current_step,
    double dt)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_synapses) return;

    const int read_slot = static_cast<int>(
        (current_step + ring_size - (delay_steps[k] % static_cast<uint32_t>(ring_size))) % ring_size);
    // Per-neuron ring: read this synapse's PREsynaptic neuron history at its delay.
    const bool arrived = spike_ring[pre[k] * ring_size + read_slot] != 0;
    spike_arrived[k] = arrived ? 1u : 0u;

    const auto& desc = spec_descs[spec_idx[k]];
    const double v_pre = V[pre[k]];
    const double v_post = V[post[k]];

    double s = S[k];
    double a = A[k];

    switch (desc.update_form) {
        case 0: { // EXP_DECAY
            if (arrived) s += desc.delta_S;
            s *= exp(-dt / desc.tau_S);
            if (s < 0.0) s = 0.0;
            a = 0.0;
            break;
        }
        case 1: { // ALPHA_FUNC
            if (arrived) a += desc.delta_A;
            const double dS = (a - s) / desc.tau_A;
            const double dA = -a / desc.tau_A;
            s += dt * dS;
            a += dt * dA;
            if (s < 0.0) s = 0.0;
            break;
        }
        case 2: { // DOUBLE_EXP
            if (arrived) {
                s += desc.delta_S;
                a += desc.delta_A;
            }
            s *= exp(-dt / desc.tau_S);
            a *= exp(-dt / desc.tau_A);
            break;
        }
        case 3: { // TANH_GATE
            const double rate_open = desc.tanh_amp
                * (1.0 + tanh((v_pre - desc.tanh_vh) / desc.tanh_k));
            const double rate = rate_open + 1.0 / desc.tau_decay;
            const double s_inf = rate_open / rate;
            s = s_inf + (s - s_inf) * exp(-dt * rate);
            break;
        }
        case 4: { // BOLTZMANN_GATE
            const double s_inf = cuda_boltzmann_scalar(v_pre, desc.s_inf);
            double tau_s = cuda_compute_tau_scalar(v_pre, desc.tau);
            if (tau_s < 1e-10) tau_s = 1e-10;
            s = s_inf + (s - s_inf) * exp(-dt / tau_s);
            break;
        }
        case 5: { // ALPHA_BETA
            const double alpha = cuda_compute_rate_scalar(v_pre, desc.alpha);
            const double beta = cuda_compute_rate_scalar(v_pre, desc.beta);
            const double rate = alpha + beta;
            const double s_inf = (rate > 1e-10) ? alpha / rate : s;
            s = s_inf + (s - s_inf) * exp(-dt * rate);
            break;
        }
        case 6: { // CUSTOM_EXPR
            if (desc.dS_vm_idx >= 0) {
                if (desc.dA_vm_idx >= 0) {
                    const double dS_dt = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.dS_vm_idx], v_pre, s, a, nullptr, 0, nullptr, 0);
                    const double dA_dt = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.dA_vm_idx], v_pre, s, a, nullptr, 0, nullptr, 0);
                    s = fmax(0.0, fmin(1.0, s + dt * dS_dt));
                    a += dt * dA_dt;
                } else {
                    const double dS_dt = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.dS_vm_idx], v_pre, s, a, nullptr, 0, nullptr, 0);
                    s = fmax(0.0, fmin(1.0, s + dt * dS_dt));
                }
            }
            break;
        }
    }

    double g_eff = 0.0;
    if (desc.update_form == 0) {
        g_eff = desc.spec_g * weight[k] * s;
    } else if (desc.update_form == 1) {
        g_eff = desc.spec_g * weight[k] * s;
    } else if (desc.update_form == 2) {
        double shape = desc.norm_factor * (s - a);
        if (shape < 0.0) shape = 0.0;
        g_eff = desc.spec_g * weight[k] * shape;
    } else if (desc.current_form == 2 && desc.current_vm_idx >= 0) { // CUSTOM_EXPR current
        g_eff = hodgkin_huxley::eval_vm_program(
                    vm_programs[desc.current_vm_idx], v_post, s, a, nullptr, 0, nullptr, 0)
                * weight[k];
    } else {
        double gate = desc.spec_g * weight[k];
        for (int p = 0; p < desc.power; ++p) gate *= s;
        if (desc.current_form == 1) { // MG_BLOCK
            const double mg = 1.0 + desc.mg_conc * exp(-desc.mg_scale * v_post) / desc.mg_denom;
            gate /= mg;
        }
        g_eff = gate;
    }

    S[k] = s;
    A[k] = a;
    g[k] = g_eff;
    (void)E_syn;
}

__global__ void accumulate_isyn_kernel(
    double* I_syn,
    const double* g,
    const double* E_syn,
    const double* V,
    const uint32_t* post,
    const double* mod_factor,
    int n_synapses)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_synapses) return;

    const uint32_t post_idx = post[k];
    double current = g[k] * (E_syn[k] - V[post_idx]);
    if (mod_factor) current *= mod_factor[post_idx];
    atomicAdd(&I_syn[post_idx], current);
}

} // namespace

namespace hodgkin_huxley {

void allocate_device_synapses(DeviceSynapseArrays& dev,
                              size_t neuron_count,
                              size_t synapse_count,
                              size_t ring_size,
                              size_t spec_count,
                              int device_id)
{
    free_device_synapses(dev);

    dev.device_id = device_id;
    dev.neuron_count = neuron_count;
    dev.synapse_count = synapse_count;
    dev.ring_size = ring_size;
    dev.spec_count = spec_count;

    check_cuda(cudaSetDevice(device_id), "allocate_device_synapses cudaSetDevice");

    auto alloc_doubles = [&](double*& ptr, size_t count, const char* what) {
        if (count == 0) return;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(double)), what);
    };
    auto alloc_u32 = [&](uint32_t*& ptr, size_t count, const char* what) {
        if (count == 0) return;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(uint32_t)), what);
    };
    auto alloc_u8 = [&](uint8_t*& ptr, size_t count, const char* what) {
        if (count == 0) return;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(uint8_t)), what);
    };

    alloc_doubles(dev.d_weight, synapse_count, "cudaMalloc weight");
    alloc_doubles(dev.d_g, synapse_count, "cudaMalloc g");
    alloc_doubles(dev.d_S, synapse_count, "cudaMalloc S");
    alloc_doubles(dev.d_A, synapse_count, "cudaMalloc A");
    alloc_doubles(dev.d_E_syn, synapse_count, "cudaMalloc E_syn");

    alloc_u32(dev.d_pre, synapse_count, "cudaMalloc pre");
    alloc_u32(dev.d_post, synapse_count, "cudaMalloc post");
    alloc_u32(dev.d_delay_steps, synapse_count, "cudaMalloc delay_steps");
    alloc_u32(dev.d_spec_idx, synapse_count, "cudaMalloc spec_idx");

    if (spec_count > 0) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dev.d_spec_descs),
                              spec_count * sizeof(DeviceSynapseSpecDesc)),
                   "cudaMalloc spec_descs");
    }

    alloc_u8(dev.d_spike_ring, neuron_count * ring_size, "cudaMalloc spike_ring"); // per-neuron ring
    alloc_u8(dev.d_spike_arrived, synapse_count, "cudaMalloc spike_arrived");
    alloc_u8(dev.d_neuron_spiked, neuron_count, "cudaMalloc neuron_spiked");
    alloc_doubles(dev.d_V_prev, neuron_count, "cudaMalloc V_prev");
}

void free_device_synapses(DeviceSynapseArrays& dev) {
    if (dev.d_weight) cudaFree(dev.d_weight);
    if (dev.d_g) cudaFree(dev.d_g);
    if (dev.d_S) cudaFree(dev.d_S);
    if (dev.d_A) cudaFree(dev.d_A);
    if (dev.d_E_syn) cudaFree(dev.d_E_syn);
    if (dev.d_pre) cudaFree(dev.d_pre);
    if (dev.d_post) cudaFree(dev.d_post);
    if (dev.d_delay_steps) cudaFree(dev.d_delay_steps);
    if (dev.d_spec_idx) cudaFree(dev.d_spec_idx);
    if (dev.d_spec_descs) cudaFree(dev.d_spec_descs);
    if (dev.d_spike_ring) cudaFree(dev.d_spike_ring);
    if (dev.d_spike_arrived) cudaFree(dev.d_spike_arrived);
    if (dev.d_neuron_spiked) cudaFree(dev.d_neuron_spiked);
    if (dev.d_V_prev) cudaFree(dev.d_V_prev);
    if (dev.d_vm_programs) cudaFree(dev.d_vm_programs);

    for (auto& prog : dev.vm_programs) free_vm_program(prog);
    dev.vm_programs.clear();

    dev.d_weight = nullptr;
    dev.d_g = nullptr;
    dev.d_S = nullptr;
    dev.d_A = nullptr;
    dev.d_E_syn = nullptr;
    dev.d_pre = nullptr;
    dev.d_post = nullptr;
    dev.d_delay_steps = nullptr;
    dev.d_spec_idx = nullptr;
    dev.d_spec_descs = nullptr;
    dev.d_spike_ring = nullptr;
    dev.d_spike_arrived = nullptr;
    dev.d_neuron_spiked = nullptr;
    dev.d_V_prev = nullptr;
    dev.d_vm_programs = nullptr;
    dev.neuron_count = 0;
    dev.synapse_count = 0;
    dev.ring_size = 0;
    dev.spec_count = 0;
    dev.dt = -1.0;
    dev.spike_threshold = 0.0;
}

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
    double spike_threshold)
{
    dev.dt = dt;
    dev.spike_threshold = spike_threshold;

    check_cuda(cudaSetDevice(dev.device_id), "upload_device_synapse_state cudaSetDevice");

    for (auto& prog : dev.vm_programs) free_vm_program(prog);
    dev.vm_programs.clear();
    if (dev.d_vm_programs) {
        cudaFree(dev.d_vm_programs);
        dev.d_vm_programs = nullptr;
    }

    if (!vm_exprs.empty()) {
        dev.vm_programs.reserve(vm_exprs.size());
        for (const auto& expr : vm_exprs)
            dev.vm_programs.push_back(upload_vm_program(expr));

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dev.d_vm_programs),
                              dev.vm_programs.size() * sizeof(DeviceVmProgram)),
                   "cudaMalloc d_vm_programs");
        check_cuda(cudaMemcpy(dev.d_vm_programs, dev.vm_programs.data(),
                              dev.vm_programs.size() * sizeof(DeviceVmProgram),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_vm_programs");
    }

    auto copy_vec = [&](auto* dst, const auto& src, const char* what) {
        if (src.empty()) return;
        check_cuda(cudaMemcpy(dst, src.data(), src.size() * sizeof(src[0]),
                              cudaMemcpyHostToDevice),
                   what);
    };

    std::vector<double> E_syn_d(E_syn.begin(), E_syn.end());
    copy_vec(dev.d_weight, weight, "cudaMemcpy weight");
    copy_vec(dev.d_g, g, "cudaMemcpy g");
    copy_vec(dev.d_S, S, "cudaMemcpy S");
    copy_vec(dev.d_A, A, "cudaMemcpy A");
    copy_vec(dev.d_E_syn, E_syn_d, "cudaMemcpy E_syn");
    copy_vec(dev.d_pre, pre, "cudaMemcpy pre");
    copy_vec(dev.d_post, post, "cudaMemcpy post");
    copy_vec(dev.d_delay_steps, delay_steps, "cudaMemcpy delay_steps");
    copy_vec(dev.d_spec_idx, spec_idx, "cudaMemcpy spec_idx");
    copy_vec(dev.d_V_prev, V_prev, "cudaMemcpy V_prev");
    copy_vec(dev.d_spike_ring, spike_ring, "cudaMemcpy spike_ring");

    if (!spec_descs.empty()) {
        check_cuda(cudaMemcpy(dev.d_spec_descs, spec_descs.data(),
                              spec_descs.size() * sizeof(DeviceSynapseSpecDesc),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy spec_descs");
    }
    if (dev.synapse_count > 0) {
        check_cuda(cudaMemset(dev.d_spike_arrived, 0, dev.synapse_count * sizeof(uint8_t)),
                   "cudaMemset spike_arrived");
    }
    if (dev.neuron_count > 0) {
        check_cuda(cudaMemset(dev.d_neuron_spiked, 0, dev.neuron_count * sizeof(uint8_t)),
                   "cudaMemset neuron_spiked");
    }
}

void download_device_synapse_state(
    const DeviceSynapseArrays& dev,
    std::vector<double>& g,
    std::vector<double>& S,
    std::vector<double>& A,
    std::vector<double>& V_prev,
    std::vector<uint8_t>* spike_ring,
    std::vector<uint8_t>* spike_arrived)
{
    check_cuda(cudaSetDevice(dev.device_id), "download_device_synapse_state cudaSetDevice");

    auto copy_back = [&](auto& dst, const auto* src, size_t count, const char* what) {
        if (count == 0) {
            dst.clear();
            return;
        }
        dst.resize(count);
        check_cuda(cudaMemcpy(dst.data(), src, count * sizeof(dst[0]),
                              cudaMemcpyDeviceToHost),
                   what);
    };

    copy_back(g, dev.d_g, dev.synapse_count, "cudaMemcpy g D2H");
    copy_back(S, dev.d_S, dev.synapse_count, "cudaMemcpy S D2H");
    copy_back(A, dev.d_A, dev.synapse_count, "cudaMemcpy A D2H");
    copy_back(V_prev, dev.d_V_prev, dev.neuron_count, "cudaMemcpy V_prev D2H");

    if (spike_ring) {
        spike_ring->resize(dev.neuron_count * dev.ring_size); // per-neuron ring
        if (!spike_ring->empty()) {
            check_cuda(cudaMemcpy(spike_ring->data(), dev.d_spike_ring,
                                  spike_ring->size() * sizeof(uint8_t),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy spike_ring D2H");
        }
    }
    if (spike_arrived) {
        spike_arrived->resize(dev.synapse_count);
        if (!spike_arrived->empty()) {
            check_cuda(cudaMemcpy(spike_arrived->data(), dev.d_spike_arrived,
                                  spike_arrived->size() * sizeof(uint8_t),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy spike_arrived D2H");
        }
    }
}

void fill_device_buffer(double* d_buf, double value, size_t n) {
    if (n == 0) return;
    const int block = 128;
    const int grid = static_cast<int>((n + block - 1) / block);
    fill_buffer_kernel<<<grid, block>>>(d_buf, value, static_cast<int>(n));
    check_cuda(cudaGetLastError(), "fill_buffer_kernel");
}

void accumulate_cuda_synaptic_currents(
    const DeviceSynapseArrays& dev,
    const double* d_V_cache,
    double* d_I_syn,
    const double* d_mod_factor)
{
    if (dev.synapse_count == 0) return;
    const int block = 128;
    const int grid = static_cast<int>((dev.synapse_count + block - 1) / block);
    accumulate_isyn_kernel<<<grid, block>>>(
        d_I_syn,
        dev.d_g,
        dev.d_E_syn,
        d_V_cache,
        dev.d_post,
        d_mod_factor,
        static_cast<int>(dev.synapse_count));
    check_cuda(cudaGetLastError(), "accumulate_isyn_kernel");
}

void update_cuda_synapses(
    DeviceSynapseArrays& dev,
    const double* d_V_cache,
    size_t current_step)
{
    if (dev.synapse_count == 0) return;
    const int block_neurons = 128;
    const int grid_neurons = static_cast<int>((dev.neuron_count + block_neurons - 1) / block_neurons);
    detect_neuron_spikes_kernel<<<grid_neurons, block_neurons>>>(
        d_V_cache,
        dev.d_V_prev,
        dev.d_neuron_spiked,
        static_cast<int>(dev.neuron_count),
        dev.spike_threshold);
    check_cuda(cudaGetLastError(), "detect_neuron_spikes_kernel");

    const int block_syn = 128;
    const int grid_syn = static_cast<int>((dev.synapse_count + block_syn - 1) / block_syn);
    const int write_slot = static_cast<int>(current_step % dev.ring_size);
    // Per-neuron ring: write one entry per neuron (grid sized over neurons).
    const int grid_neu = static_cast<int>((dev.neuron_count + block_syn - 1) / block_syn);
    update_spike_ring_kernel<<<grid_neu, block_syn>>>(
        dev.d_neuron_spiked,
        dev.d_spike_ring,
        static_cast<int>(dev.neuron_count),
        write_slot,
        static_cast<int>(dev.ring_size));
    check_cuda(cudaGetLastError(), "update_spike_ring_kernel");

    update_synapse_state_kernel<<<grid_syn, block_syn>>>(
        dev.d_g,
        dev.d_S,
        dev.d_A,
        dev.d_E_syn,
        dev.d_weight,
        dev.d_pre,
        dev.d_post,
        dev.d_delay_steps,
        dev.d_spec_idx,
        d_V_cache,
        dev.d_spike_ring,
        dev.d_spike_arrived,
        dev.d_spec_descs,
        dev.d_vm_programs,
        static_cast<int>(dev.synapse_count),
        static_cast<int>(dev.ring_size),
        current_step,
        dev.dt);
    check_cuda(cudaGetLastError(), "update_synapse_state_kernel");
}

} // namespace hodgkin_huxley
