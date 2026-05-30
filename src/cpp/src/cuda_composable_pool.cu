#include "hodgkin_huxley/cuda_composable_pool.hpp"
#include "hodgkin_huxley/cuda_sim_all.hpp"

#include <algorithm>
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

__device__ inline double clamp01(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

__device__ inline double cuda_boltzmann_scalar(double x, const hodgkin_huxley::BoltzmannParams& p) {
    double arg = -(x - p.v_half) / p.k;
    if (arg > 500.0) return 0.0;
    if (arg < -500.0) return 1.0;
    return 1.0 / (1.0 + exp(arg));
}

__device__ inline double cuda_compute_tau_scalar(double V, const hodgkin_huxley::TauParams& p) {
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

__device__ inline double cuda_compute_rate_scalar(double V, const hodgkin_huxley::RateFuncParams& p) {
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

__global__ void scatter_by_index_kernel(
    const size_t* net_idx, const double* src, double* dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[net_idx[i]] = src[i];
}

__global__ void gather_by_index_kernel(
    const size_t* net_idx, const double* src, double* dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[net_idx[i]];
}

__global__ void composable_step_kernel(
    double* V,
    double* I_ext,
    double* synapse_g_scale,
    double* const* gate_ptrs,
    double* const* substance_ptrs,
    double* const* nernst_ptrs,
    const hodgkin_huxley::CudaGateDesc* gate_descs,
    const hodgkin_huxley::CudaChannelDesc* channel_descs,
    const hodgkin_huxley::CudaChannelGateRef* channel_gate_refs,
    const hodgkin_huxley::CudaIntracellularDesc* intr_descs,
    const int* intr_source_channels,
    const hodgkin_huxley::CudaIntracellularModDesc* mod_descs,
    const hodgkin_huxley::DeviceVmProgram* vm_programs,
    int n_gates,
    int n_channels,
    int n_substances,
    double C_m,
    double dt,
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double v = V[i];
    double current = I_ext[i];

    double gate_vals[hodgkin_huxley::CudaComposablePool::kMaxGates];
    double x_vals[hodgkin_huxley::CudaComposablePool::kMaxSubstances];
    double e_nernst[hodgkin_huxley::CudaComposablePool::kMaxSubstances];
    double channel_currents[hodgkin_huxley::CudaComposablePool::kMaxChannels];
    double mod_ch_g[hodgkin_huxley::CudaComposablePool::kMaxChannels];
    double mod_ch_E[hodgkin_huxley::CudaComposablePool::kMaxChannels];
    int has_mod_ch_E[hodgkin_huxley::CudaComposablePool::kMaxChannels];
    double mod_gate_shift[hodgkin_huxley::CudaComposablePool::kMaxGates];
    double mod_gate_scale[hodgkin_huxley::CudaComposablePool::kMaxGates];
    double mod_gate_tau[hodgkin_huxley::CudaComposablePool::kMaxGates];
    double mod_gate_expr[hodgkin_huxley::CudaComposablePool::kMaxGates];
    int has_mod_gate_expr[hodgkin_huxley::CudaComposablePool::kMaxGates];

    for (int g = 0; g < n_gates; ++g) {
        gate_vals[g] = gate_ptrs[g][i];
        mod_gate_shift[g] = 0.0;
        mod_gate_scale[g] = 1.0;
        mod_gate_tau[g] = 1.0;
        mod_gate_expr[g] = 0.0;
        has_mod_gate_expr[g] = 0;
    }
    for (int s = 0; s < n_substances; ++s) {
        x_vals[s] = substance_ptrs[s][i];
        e_nernst[s] = nernst_ptrs[s][i];
    }
    for (int c = 0; c < n_channels; ++c) {
        channel_currents[c] = 0.0;
        mod_ch_g[c] = 1.0;
        mod_ch_E[c] = 0.0;
        has_mod_ch_E[c] = 0;
    }

    double syn_g_scale = 1.0;

    for (int s = 0; s < n_substances; ++s) {
        const auto& intr = intr_descs[s];
        for (int m = 0; m < intr.mod_count; ++m) {
            const auto& mod = mod_descs[intr.mod_start + m];
            const int sidx = mod.substance_idx;
            const double xmod = (sidx >= 0 && sidx < n_substances) ? x_vals[sidx] : 0.0;
            switch (mod.target) {
                case 0: { // CHANNEL_G
                    if (mod.target_idx >= 0 && mod.target_idx < n_channels) {
                        const double val = (mod.vm_idx >= 0)
                            ? hodgkin_huxley::eval_vm_program(vm_programs[mod.vm_idx],
                                                              xmod, xmod, 0.0,
                                                              gate_vals, n_gates,
                                                              x_vals, n_substances)
                            : mod.shift_scale;
                        mod_ch_g[mod.target_idx] *= val;
                    }
                    break;
                }
                case 1: { // CHANNEL_EREV
                    if (mod.target_idx >= 0 && mod.target_idx < n_channels && mod.vm_idx >= 0) {
                        mod_ch_E[mod.target_idx] =
                            hodgkin_huxley::eval_vm_program(vm_programs[mod.vm_idx],
                                                            xmod, xmod, 0.0,
                                                            gate_vals, n_gates,
                                                            x_vals, n_substances);
                        has_mod_ch_E[mod.target_idx] = 1;
                    }
                    break;
                }
                case 2: { // GATE_INF_SHIFT
                    if (mod.target_idx >= 0 && mod.target_idx < n_gates) {
                        mod_gate_shift[mod.target_idx] += (mod.vm_idx >= 0)
                            ? hodgkin_huxley::eval_vm_program(vm_programs[mod.vm_idx],
                                                              xmod, xmod, 0.0,
                                                              gate_vals, n_gates,
                                                              x_vals, n_substances)
                            : (mod.shift_scale * xmod);
                    }
                    break;
                }
                case 3: { // GATE_INF_SCALE
                    if (mod.target_idx >= 0 && mod.target_idx < n_gates && mod.vm_idx >= 0) {
                        mod_gate_scale[mod.target_idx] *=
                            hodgkin_huxley::eval_vm_program(vm_programs[mod.vm_idx],
                                                            xmod, xmod, 0.0,
                                                            gate_vals, n_gates,
                                                            x_vals, n_substances);
                    }
                    break;
                }
                case 4: { // GATE_TAU_SCALE
                    if (mod.target_idx >= 0 && mod.target_idx < n_gates && mod.vm_idx >= 0) {
                        mod_gate_tau[mod.target_idx] *=
                            hodgkin_huxley::eval_vm_program(vm_programs[mod.vm_idx],
                                                            xmod, xmod, 0.0,
                                                            gate_vals, n_gates,
                                                            x_vals, n_substances);
                    }
                    break;
                }
                case 5: { // GATE_INF_EXPR
                    if (mod.target_idx >= 0 && mod.target_idx < n_gates && mod.vm_idx >= 0) {
                        mod_gate_expr[mod.target_idx] =
                            hodgkin_huxley::eval_vm_program(vm_programs[mod.vm_idx],
                                                            v, xmod, 0.0,
                                                            gate_vals, n_gates,
                                                            x_vals, n_substances);
                        has_mod_gate_expr[mod.target_idx] = 1;
                    }
                    break;
                }
                case 6: { // SYNAPSE_G
                    if (mod.vm_idx >= 0) {
                        syn_g_scale *= hodgkin_huxley::eval_vm_program(vm_programs[mod.vm_idx],
                                                                       xmod, xmod, 0.0,
                                                                       gate_vals, n_gates,
                                                                       x_vals, n_substances);
                    }
                    break;
                }
            }
        }
    }

    for (int g = 0; g < n_gates; ++g) {
        const auto& desc = gate_descs[g];
        const double dep_base = (desc.dependency == 1 && desc.intracellular_idx >= 0
                                 && desc.intracellular_idx < n_substances)
            ? x_vals[desc.intracellular_idx]
            : v;
        const double dep = dep_base + mod_gate_shift[g];

        switch (desc.update_form) {
            case 0: { // INF_TAU
                double x_inf = cuda_boltzmann_scalar(dep, desc.inf);
                x_inf *= mod_gate_scale[g];
                if (has_mod_gate_expr[g]) x_inf = mod_gate_expr[g];
                double tau_x = cuda_compute_tau_scalar(v, desc.tau) * mod_gate_tau[g];
                tau_x = fmax(tau_x, 1e-10);
                gate_vals[g] = x_inf + (gate_vals[g] - x_inf) * exp(-dt * desc.scale / tau_x);
                break;
            }
            case 1: { // ALPHA_BETA
                const double alpha = cuda_compute_rate_scalar(v, desc.alpha);
                const double beta = cuda_compute_rate_scalar(v, desc.beta);
                const double rate = fmax(alpha + beta, 1e-10);
                double x_inf = alpha / rate;
                x_inf *= mod_gate_scale[g];
                if (has_mod_gate_expr[g]) x_inf = mod_gate_expr[g];
                double tau_x = (1.0 / rate) * mod_gate_tau[g];
                tau_x = fmax(tau_x, 1e-10);
                gate_vals[g] = x_inf + (gate_vals[g] - x_inf) * exp(-dt / tau_x);
                break;
            }
            case 2: { // INSTANT
                double x_inf = cuda_boltzmann_scalar(dep, desc.inf) * mod_gate_scale[g];
                gate_vals[g] = has_mod_gate_expr[g] ? mod_gate_expr[g] : x_inf;
                break;
            }
            case 3: { // DERIVED
                if (desc.derived_source_gate >= 0 && desc.derived_source_gate < n_gates) {
                    gate_vals[g] = desc.derived_a *
                                   (desc.derived_b + desc.derived_c * gate_vals[desc.derived_source_gate]);
                }
                break;
            }
            case 4: { // CUSTOM_EXPR
                if (desc.dxdt_vm_idx >= 0) {
                    const double dxdt = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.dxdt_vm_idx],
                        dep, gate_vals[g], 0.0, gate_vals, n_gates, x_vals, n_substances);
                    gate_vals[g] += dt * desc.scale * dxdt;
                } else if (desc.inf_vm_idx >= 0 && desc.tau_vm_idx >= 0) {
                    double x_inf = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.inf_vm_idx],
                        dep, gate_vals[g], 0.0, gate_vals, n_gates, x_vals, n_substances);
                    x_inf *= mod_gate_scale[g];
                    if (has_mod_gate_expr[g]) x_inf = mod_gate_expr[g];
                    double tau_x = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.tau_vm_idx],
                        v, gate_vals[g], 0.0, gate_vals, n_gates, x_vals, n_substances);
                    tau_x = fmax(tau_x * mod_gate_tau[g], 1e-10);
                    gate_vals[g] = x_inf + (gate_vals[g] - x_inf) * exp(-dt * desc.scale / tau_x);
                } else if (desc.alpha_vm_idx >= 0 && desc.beta_vm_idx >= 0) {
                    const double alpha = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.alpha_vm_idx],
                        v, gate_vals[g], 0.0, gate_vals, n_gates, x_vals, n_substances);
                    const double beta = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.beta_vm_idx],
                        v, gate_vals[g], 0.0, gate_vals, n_gates, x_vals, n_substances);
                    const double rate = fmax(alpha + beta, 1e-10);
                    double x_inf = alpha / rate;
                    x_inf *= mod_gate_scale[g];
                    if (has_mod_gate_expr[g]) x_inf = mod_gate_expr[g];
                    double tau_x = (1.0 / rate) * mod_gate_tau[g];
                    tau_x = fmax(tau_x, 1e-10);
                    gate_vals[g] = x_inf + (gate_vals[g] - x_inf) * exp(-dt / tau_x);
                } else if (desc.inf_vm_idx >= 0) {
                    double x_inf = hodgkin_huxley::eval_vm_program(
                        vm_programs[desc.inf_vm_idx],
                        dep, gate_vals[g], 0.0, gate_vals, n_gates, x_vals, n_substances);
                    x_inf *= mod_gate_scale[g];
                    gate_vals[g] = has_mod_gate_expr[g] ? mod_gate_expr[g] : x_inf;
                }
                break;
            }
        }

        gate_vals[g] = clamp01(gate_vals[g]);
    }

    double I_total = 0.0;
    for (int c = 0; c < n_channels; ++c) {
        const auto& ch = channel_descs[c];
        double gate_prod = 1.0;
        if (ch.gate_product_vm_idx >= 0) {
            gate_prod = hodgkin_huxley::eval_vm_program(vm_programs[ch.gate_product_vm_idx],
                                                        v, 0.0, 0.0,
                                                        gate_vals, n_gates,
                                                        x_vals, n_substances);
        } else {
            for (int gi = 0; gi < ch.gate_ref_count; ++gi) {
                const auto& ref = channel_gate_refs[ch.gate_ref_start + gi];
                if (ref.gate_idx >= 0 && ref.gate_idx < n_gates) {
                    for (int p = 0; p < ref.power; ++p)
                        gate_prod *= gate_vals[ref.gate_idx];
                }
            }
        }

        const double E_rev = has_mod_ch_E[c]
            ? mod_ch_E[c]
            : ((ch.nernst_substance_idx >= 0 && ch.nernst_substance_idx < n_substances)
                   ? e_nernst[ch.nernst_substance_idx]
                   : ch.E_rev);

        double I_ci = 0.0;
        if (ch.is_ahp) {
            const double x_ahp = (ch.ahp_substance_idx >= 0 && ch.ahp_substance_idx < n_substances)
                ? x_vals[ch.ahp_substance_idx]
                : 0.0;
            const double ca_factor = x_ahp / fmax(x_ahp + ch.ahp_k1, 1e-10);
            I_ci = ch.g * mod_ch_g[c] * ca_factor * (v - E_rev);
        } else {
            I_ci = ch.g * mod_ch_g[c] * gate_prod * (v - E_rev);
        }
        channel_currents[c] = I_ci;
        I_total += I_ci;
    }

    v += dt * (-I_total + current) / C_m;

    for (int s = 0; s < n_substances; ++s) {
        const auto& intr = intr_descs[s];
        double dX = 0.0;
        if (intr.update_form == 0) { // DECAY
            dX = -intr.k_decay * x_vals[s];
        } else if (intr.update_form == 1 || intr.update_form == 2) { // DRIVEN_DECAY / NERNST
            double I_source = 0.0;
            for (int sc = 0; sc < intr.source_count; ++sc) {
                const int ch_idx = intr_source_channels[intr.source_start + sc];
                if (ch_idx >= 0 && ch_idx < n_channels)
                    I_source += channel_currents[ch_idx];
            }
            dX = intr.epsilon * (-I_source - intr.k_decay * x_vals[s]);
        } else if (intr.update_form == 3 && intr.ode_vm_idx >= 0) { // CUSTOM_EXPR
            double I_source = 0.0;
            for (int sc = 0; sc < intr.source_count; ++sc) {
                const int ch_idx = intr_source_channels[intr.source_start + sc];
                if (ch_idx >= 0 && ch_idx < n_channels)
                    I_source += channel_currents[ch_idx];
            }
            dX = hodgkin_huxley::eval_vm_program(vm_programs[intr.ode_vm_idx],
                                                 I_source, x_vals[s], 0.0,
                                                 gate_vals, n_gates,
                                                 x_vals, n_substances);
        }

        x_vals[s] = fmax(0.0, x_vals[s] + dt * dX);

        if (intr.nernst_enabled) {
            if (intr.nernst_vm_idx >= 0) {
                e_nernst[s] = hodgkin_huxley::eval_vm_program(vm_programs[intr.nernst_vm_idx],
                                                              x_vals[s], x_vals[s], 0.0,
                                                              gate_vals, n_gates,
                                                              x_vals, n_substances);
            } else {
                const double x_safe = fmax(x_vals[s], 1e-10);
                e_nernst[s] = (intr.nernst_R * intr.nernst_T)
                              / (intr.nernst_z * intr.nernst_F)
                              * log(intr.nernst_Ca_o / x_safe);
            }
        }
    }

    V[i] = v;
    synapse_g_scale[i] = syn_g_scale;
    for (int g = 0; g < n_gates; ++g)
        gate_ptrs[g][i] = gate_vals[g];
    for (int s = 0; s < n_substances; ++s) {
        substance_ptrs[s][i] = x_vals[s];
        nernst_ptrs[s][i] = e_nernst[s];
    }
}

} // namespace

namespace hodgkin_huxley {

CudaComposablePool::CudaComposablePool(
    int device_id, const NeuronModelSpec& model, size_t capacity, bool fast_math)
    : ComposablePool(model, capacity, fast_math), device_id_(device_id), capacity_(capacity)
{
    if (static_cast<int>(model_.gates.size()) > kMaxGates ||
        static_cast<int>(model_.channels.size()) > kMaxChannels ||
        static_cast<int>(model_.intracellular.size()) > kMaxSubstances) {
        throw std::runtime_error("CudaComposablePool model exceeds supported gate/channel/substance limits");
    }
    check_cuda(cudaSetDevice(device_id_), "CudaComposablePool cudaSetDevice");
}

CudaComposablePool::~CudaComposablePool() {
    free_device();
}

int CudaComposablePool::register_vm_program(const VmExpr& expr) {
    if (expr.empty()) return -1;
    vm_programs_.push_back(upload_vm_program(expr));
    return static_cast<int>(vm_programs_.size() - 1);
}

void CudaComposablePool::allocate_device() {
    if (device_ready_) return;
    check_cuda(cudaSetDevice(device_id_), "CudaComposablePool cudaSetDevice");

    auto alloc = [&](double*& ptr) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), capacity_ * sizeof(double)),
                   "CudaComposablePool cudaMalloc");
    };

    alloc(d_V_);
    alloc(d_I_ext_);
    alloc(d_synapse_g_scale_);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_net_idx_), capacity_ * sizeof(size_t)),
               "CudaComposablePool cudaMalloc net_idx");

    // Flat single-block allocations (stride = capacity_). One cudaMalloc each
    // instead of one per gate/substance — eliminates the pointer table.
    const size_t n_gates = model_.gates.size();
    const size_t n_subst = model_.intracellular.size();
    if (n_gates > 0) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_gate_state_),
                              n_gates * capacity_ * sizeof(double)),
                   "CudaComposablePool cudaMalloc gate_state");
    }
    if (n_subst > 0) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_substance_state_),
                              n_subst * capacity_ * sizeof(double)),
                   "CudaComposablePool cudaMalloc substance_state");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nernst_state_),
                              n_subst * capacity_ * sizeof(double)),
                   "CudaComposablePool cudaMalloc nernst_state");
    }

    device_ready_ = true;
}

void CudaComposablePool::free_device() {
    auto free_ptr = [](double*& ptr) {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
    };

    free_ptr(d_V_);
    free_ptr(d_I_ext_);
    free_ptr(d_synapse_g_scale_);
    if (d_net_idx_) cudaFree(d_net_idx_);
    d_net_idx_ = nullptr;

    free_ptr(d_gate_state_);
    free_ptr(d_substance_state_);
    free_ptr(d_nernst_state_);

    if (d_gate_descs_) cudaFree(d_gate_descs_);
    if (d_channel_descs_) cudaFree(d_channel_descs_);
    if (d_channel_gate_refs_) cudaFree(d_channel_gate_refs_);
    if (d_intr_descs_) cudaFree(d_intr_descs_);
    if (d_intr_source_channels_) cudaFree(d_intr_source_channels_);
    if (d_mod_descs_) cudaFree(d_mod_descs_);
    if (d_vm_programs_) cudaFree(d_vm_programs_);
    d_gate_descs_ = nullptr;
    d_channel_descs_ = nullptr;
    d_channel_gate_refs_ = nullptr;
    d_intr_descs_ = nullptr;
    d_intr_source_channels_ = nullptr;
    d_mod_descs_ = nullptr;
    d_vm_programs_ = nullptr;

    for (auto& prog : vm_programs_) free_vm_program(prog);
    vm_programs_.clear();

    device_ready_ = false;
    host_state_dirty_ = false;
}

void CudaComposablePool::upload_model_metadata() {
    std::vector<CudaGateDesc> gate_descs(model_.gates.size());
    for (size_t g = 0; g < model_.gates.size(); ++g) {
        const auto& src = model_.gates[g];
        auto& dst = gate_descs[g];
        dst.update_form = static_cast<int>(src.update_form);
        dst.dependency = static_cast<int>(src.dependency);
        dst.intracellular_idx = src.intracellular_idx;
        dst.scale = src.scale;
        dst.inf = src.inf;
        dst.tau = src.tau;
        dst.alpha = src.alpha;
        dst.beta = src.beta;
        dst.derived_source_gate = src.derived_source_gate;
        dst.derived_a = src.derived_a;
        dst.derived_b = src.derived_b;
        dst.derived_c = src.derived_c;
        dst.inf_vm_idx = register_vm_program(src.inf_vm);
        dst.tau_vm_idx = register_vm_program(src.tau_vm);
        dst.alpha_vm_idx = register_vm_program(src.alpha_vm);
        dst.beta_vm_idx = register_vm_program(src.beta_vm);
        dst.dxdt_vm_idx = register_vm_program(src.dxdt_vm);
    }

    std::vector<CudaChannelGateRef> gate_refs;
    std::vector<CudaChannelDesc> channel_descs(model_.channels.size());
    for (size_t c = 0; c < model_.channels.size(); ++c) {
        const auto& src = model_.channels[c];
        auto& dst = channel_descs[c];
        dst.g = src.g;
        dst.E_rev = src.E_rev;
        dst.nernst_substance_idx = src.nernst_substance_idx;
        dst.is_ahp = src.is_ahp ? 1 : 0;
        dst.ahp_k1 = src.ahp_k1;
        dst.ahp_substance_idx = src.ahp_substance_idx;
        dst.gate_ref_start = static_cast<int>(gate_refs.size());
        dst.gate_ref_count = static_cast<int>(src.gates.size());
        for (const auto& gp : src.gates)
            gate_refs.push_back({gp.first, gp.second});
        dst.gate_product_vm_idx = register_vm_program(src.gate_product_vm);
    }

    std::vector<int> intr_sources;
    std::vector<CudaIntracellularModDesc> mod_descs;
    std::vector<CudaIntracellularDesc> intr_descs(model_.intracellular.size());
    for (size_t s = 0; s < model_.intracellular.size(); ++s) {
        const auto& src = model_.intracellular[s];
        auto& dst = intr_descs[s];
        dst.update_form = static_cast<int>(src.update_form);
        dst.initial = src.initial;
        dst.epsilon = src.epsilon;
        dst.k_decay = src.k_decay;
        dst.source_start = static_cast<int>(intr_sources.size());
        dst.source_count = static_cast<int>(src.source_channels.size());
        intr_sources.insert(intr_sources.end(),
                            src.source_channels.begin(), src.source_channels.end());
        dst.nernst_enabled = src.nernst_enabled ? 1 : 0;
        dst.nernst_Ca_o = src.nernst_Ca_o;
        dst.nernst_z = src.nernst_z;
        dst.nernst_R = src.nernst_R;
        dst.nernst_F = src.nernst_F;
        dst.nernst_T = src.nernst_T;
        dst.nernst_vm_idx = register_vm_program(src.nernst_vm);
        dst.ode_vm_idx = register_vm_program(src.ode_vm);
        dst.mod_start = static_cast<int>(mod_descs.size());
        dst.mod_count = static_cast<int>(src.modulations.size());
        for (const auto& mod : src.modulations) {
            mod_descs.push_back({
                static_cast<int>(mod.target),
                mod.target_idx,
                mod.substance_idx,
                register_vm_program(mod.mod_vm),
                mod.shift_scale
            });
        }
    }

    if (!gate_descs.empty()) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_gate_descs_),
                              gate_descs.size() * sizeof(CudaGateDesc)),
                   "CudaComposablePool cudaMalloc gate_descs");
        check_cuda(cudaMemcpy(d_gate_descs_, gate_descs.data(),
                              gate_descs.size() * sizeof(CudaGateDesc),
                              cudaMemcpyHostToDevice),
                   "CudaComposablePool upload gate_descs");
    }
    if (!channel_descs.empty()) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_channel_descs_),
                              channel_descs.size() * sizeof(CudaChannelDesc)),
                   "CudaComposablePool cudaMalloc channel_descs");
        check_cuda(cudaMemcpy(d_channel_descs_, channel_descs.data(),
                              channel_descs.size() * sizeof(CudaChannelDesc),
                              cudaMemcpyHostToDevice),
                   "CudaComposablePool upload channel_descs");
    }
    if (!gate_refs.empty()) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_channel_gate_refs_),
                              gate_refs.size() * sizeof(CudaChannelGateRef)),
                   "CudaComposablePool cudaMalloc gate_refs");
        check_cuda(cudaMemcpy(d_channel_gate_refs_, gate_refs.data(),
                              gate_refs.size() * sizeof(CudaChannelGateRef),
                              cudaMemcpyHostToDevice),
                   "CudaComposablePool upload gate_refs");
    }
    if (!intr_descs.empty()) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_intr_descs_),
                              intr_descs.size() * sizeof(CudaIntracellularDesc)),
                   "CudaComposablePool cudaMalloc intr_descs");
        check_cuda(cudaMemcpy(d_intr_descs_, intr_descs.data(),
                              intr_descs.size() * sizeof(CudaIntracellularDesc),
                              cudaMemcpyHostToDevice),
                   "CudaComposablePool upload intr_descs");
    }
    if (!intr_sources.empty()) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_intr_source_channels_),
                              intr_sources.size() * sizeof(int)),
                   "CudaComposablePool cudaMalloc intr_sources");
        check_cuda(cudaMemcpy(d_intr_source_channels_, intr_sources.data(),
                              intr_sources.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "CudaComposablePool upload intr_sources");
    }
    if (!mod_descs.empty()) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_mod_descs_),
                              mod_descs.size() * sizeof(CudaIntracellularModDesc)),
                   "CudaComposablePool cudaMalloc mod_descs");
        check_cuda(cudaMemcpy(d_mod_descs_, mod_descs.data(),
                              mod_descs.size() * sizeof(CudaIntracellularModDesc),
                              cudaMemcpyHostToDevice),
                   "CudaComposablePool upload mod_descs");
    }
    if (!vm_programs_.empty()) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_vm_programs_),
                              vm_programs_.size() * sizeof(DeviceVmProgram)),
                   "CudaComposablePool cudaMalloc vm_programs");
        check_cuda(cudaMemcpy(d_vm_programs_, vm_programs_.data(),
                              vm_programs_.size() * sizeof(DeviceVmProgram),
                              cudaMemcpyHostToDevice),
                   "CudaComposablePool upload vm_programs");
    }
}

void CudaComposablePool::upload_state() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_V_, V_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaComposablePool upload V");
    for (size_t g = 0; g < gate_states_.size(); ++g) {
        check_cuda(cudaMemcpyAsync(d_gate_state_ + g * capacity_, gate_states_[g].data(),
                                   N_ * sizeof(double), cudaMemcpyHostToDevice, stream_),
                   "CudaComposablePool upload gates");
    }
    for (size_t s = 0; s < X_.size(); ++s) {
        check_cuda(cudaMemcpyAsync(d_substance_state_ + s * capacity_, X_[s].data(),
                                   N_ * sizeof(double), cudaMemcpyHostToDevice, stream_),
                   "CudaComposablePool upload substances");
        check_cuda(cudaMemcpyAsync(d_nernst_state_ + s * capacity_, E_nernst_[s].data(),
                                   N_ * sizeof(double), cudaMemcpyHostToDevice, stream_),
                   "CudaComposablePool upload nernst");
    }
    if (has_synapse_g_mods_ && synapse_g_scale_.size() == N_) {
        check_cuda(cudaMemcpyAsync(d_synapse_g_scale_, synapse_g_scale_.data(),
                                   N_ * sizeof(double), cudaMemcpyHostToDevice, stream_),
                   "CudaComposablePool upload synapse_g_scale");
    }
    host_state_dirty_ = false;
}

void CudaComposablePool::upload_currents() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_I_ext_, I_ext_.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "CudaComposablePool upload I_ext");
}

void CudaComposablePool::upload_index_map() {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(d_net_idx_, net_idx_.data(), N_ * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream_),
               "CudaComposablePool upload net_idx");
}

void CudaComposablePool::download_state() const {
    if (N_ == 0) return;
    check_cuda(cudaMemcpyAsync(const_cast<double*>(V_.data()), d_V_, N_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "CudaComposablePool download V");
    for (size_t g = 0; g < gate_states_.size(); ++g) {
        check_cuda(cudaMemcpyAsync(const_cast<double*>(gate_states_[g].data()),
                                   gate_device_ptrs_[g], N_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_),
                   "CudaComposablePool download gates");
    }
    for (size_t s = 0; s < X_.size(); ++s) {
        check_cuda(cudaMemcpyAsync(const_cast<double*>(X_[s].data()),
                                   substance_device_ptrs_[s], N_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_),
                   "CudaComposablePool download substances");
        check_cuda(cudaMemcpyAsync(const_cast<double*>(E_nernst_[s].data()),
                                   nernst_device_ptrs_[s], N_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_),
                   "CudaComposablePool download nernst");
    }
    if (has_synapse_g_mods_ && synapse_g_scale_.size() == N_) {
        check_cuda(cudaMemcpyAsync(const_cast<double*>(synapse_g_scale_.data()),
                                   d_synapse_g_scale_, N_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_),
                   "CudaComposablePool download synapse_g_scale");
    }
}

void CudaComposablePool::ensure_device_state() {
    allocate_device();
    upload_model_metadata();
    upload_state();
    upload_currents();
    upload_index_map();
    check_cuda(cudaStreamSynchronize(stream_), "CudaComposablePool initial sync");
}

void CudaComposablePool::ensure_host_state() const {
    if (!host_state_dirty_) return;
    const_cast<CudaComposablePool*>(this)->download_state();
    check_cuda(cudaStreamSynchronize(stream_), "CudaComposablePool host sync");
    host_state_dirty_ = false;
}

void CudaComposablePool::scatter_voltages(double* V_buf) const {
    ensure_host_state();
    ComposablePool::scatter_voltages(V_buf);
}

void CudaComposablePool::gather_currents(const double* I_buf) {
    ComposablePool::gather_currents(I_buf);
    if (!device_ready_) ensure_device_state();
    else upload_currents();
}

void CudaComposablePool::step(double dt) {
    if (N_ == 0 || dt == 0.0) return;
    if (!finalized_) finalize();
    if (!device_ready_) ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    composable_step_kernel<<<grid, block, 0, stream_>>>(
        d_V_, d_I_ext_, d_synapse_g_scale_,
        d_gate_ptrs_, d_substance_ptrs_, d_nernst_ptrs_,
        d_gate_descs_, d_channel_descs_, d_channel_gate_refs_,
        d_intr_descs_, d_intr_source_channels_, d_mod_descs_,
        d_vm_programs_,
        static_cast<int>(model_.gates.size()),
        static_cast<int>(model_.channels.size()),
        static_cast<int>(model_.intracellular.size()),
        model_.C_m,
        dt,
        static_cast<int>(N_));
    check_cuda(cudaGetLastError(), "CudaComposablePool composable_step_kernel");
    host_state_dirty_ = true;
}

void CudaComposablePool::sync_to_neurons(std::vector<std::unique_ptr<NeuronBase>>& neurons) const {
    ensure_host_state();
    ComposablePool::sync_to_neurons(neurons);
}

void CudaComposablePool::scatter_gate_states_into(double* buf, size_t max_gates,
                                                  size_t n_rec, size_t t_rec) const {
    ensure_host_state();
    ComposablePool::scatter_gate_states_into(buf, max_gates, n_rec, t_rec);
}

void CudaComposablePool::scatter_substance_into(size_t subst_idx, double* buf,
                                                size_t n_rec, size_t t_rec) const {
    ensure_host_state();
    ComposablePool::scatter_substance_into(subst_idx, buf, n_rec, t_rec);
}

void CudaComposablePool::scatter_synapse_g_scale(double* buf) const {
    ensure_host_state();
    ComposablePool::scatter_synapse_g_scale(buf);
}

void CudaComposablePool::scatter_voltages_device(double* d_V_buf) const {
    if (N_ == 0) return;
    if (!device_ready_) const_cast<CudaComposablePool*>(this)->ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    scatter_by_index_kernel<<<grid, block, 0, stream_>>>(d_net_idx_, d_V_, d_V_buf,
                                                          static_cast<int>(N_));
    check_cuda(cudaGetLastError(), "CudaComposablePool scatter voltages device");
}

void CudaComposablePool::gather_currents_device(const double* d_I_buf) {
    if (N_ == 0) return;
    if (!device_ready_) ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    gather_by_index_kernel<<<grid, block, 0, stream_>>>(d_net_idx_, d_I_buf, d_I_ext_,
                                                         static_cast<int>(N_));
    check_cuda(cudaGetLastError(), "CudaComposablePool gather currents device");
}

void CudaComposablePool::scatter_synapse_g_scale_device(double* d_buf) const {
    if (N_ == 0 || !has_synapse_g_mods_) return;
    if (!device_ready_) const_cast<CudaComposablePool*>(this)->ensure_device_state();
    const int block = 128;
    const int grid = static_cast<int>((N_ + block - 1) / block);
    scatter_by_index_kernel<<<grid, block, 0, stream_>>>(d_net_idx_, d_synapse_g_scale_, d_buf,
                                                          static_cast<int>(N_));
    check_cuda(cudaGetLastError(), "CudaComposablePool scatter synapse_g_scale device");
}

void CudaComposablePool::fill_coop_desc(CudaComposableDesc& out) const {
    if (!device_ready_) const_cast<CudaComposablePool*>(this)->ensure_device_state();

    int n_gate_refs = 0;
    for (const auto& ch : model_.channels)
        n_gate_refs += static_cast<int>(ch.gates.size());

    int n_mods = 0;
    for (const auto& s : model_.intracellular)
        n_mods += static_cast<int>(s.modulations.size());

    out.d_V                    = d_V_;
    out.d_I_ext                = d_I_ext_;
    out.d_synapse_g_scale      = d_synapse_g_scale_;
    out.d_gate_ptrs            = d_gate_ptrs_;
    out.d_substance_ptrs       = d_substance_ptrs_;
    out.d_nernst_ptrs          = d_nernst_ptrs_;
    out.d_gate_descs           = d_gate_descs_;
    out.d_channel_descs        = d_channel_descs_;
    out.d_channel_gate_refs    = d_channel_gate_refs_;
    out.d_intr_descs           = d_intr_descs_;
    out.d_intr_source_channels = d_intr_source_channels_;
    out.d_mod_descs            = d_mod_descs_;
    out.d_vm_programs          = d_vm_programs_;
    out.d_net_idx              = d_net_idx_;
    out.n                      = static_cast<int>(N_);
    out.n_gates                = static_cast<int>(model_.gates.size());
    out.n_channels             = static_cast<int>(model_.channels.size());
    out.n_intracellulars       = static_cast<int>(model_.intracellular.size());
    out.n_gate_refs            = n_gate_refs;
    out.n_mods                 = n_mods;
    out.C_m                    = model_.C_m;
    // Gate-major fast path is valid only for pure pattern-matched models:
    // no VM programs (custom expressions) and no modulations.
    out.fast_path              = (!needs_vm_programs() && n_mods == 0) ? 1 : 0;
}

bool CudaComposablePool::needs_vm_programs() const {
    for (const auto& g : model_.gates) {
        if (!g.inf_vm.empty() || !g.tau_vm.empty() ||
            !g.alpha_vm.empty() || !g.beta_vm.empty() || !g.dxdt_vm.empty())
            return true;
    }
    for (const auto& c : model_.channels) {
        if (!c.gate_product_vm.empty()) return true;
    }
    for (const auto& s : model_.intracellular) {
        if (!s.nernst_vm.empty() || !s.ode_vm.empty()) return true;
        for (const auto& m : s.modulations) {
            if (!m.mod_vm.empty()) return true;
        }
    }
    return false;
}

void CudaComposablePool::synchronize() {
    check_cuda(cudaStreamSynchronize(stream_), "CudaComposablePool synchronize");
}

void CudaComposablePool::migrate_to_device(int id) {
    if (id == device_id_) return;
    ensure_host_state();
    free_device();
    device_id_ = id;
    check_cuda(cudaSetDevice(device_id_), "CudaComposablePool migrate cudaSetDevice");
    if (N_ > 0) ensure_device_state();
}

} // namespace hodgkin_huxley
