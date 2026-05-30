#include "hodgkin_huxley/network.hpp"
#include "hodgkin_huxley/model/kinetics.hpp"
#ifdef HH_USE_CUDA
#  include <cuda_runtime_api.h>
#  include "hodgkin_huxley/cuda_sim_all.hpp"
#endif
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <limits>
#ifdef HH_USE_OPENMP
#  include <omp.h>
#endif

namespace hodgkin_huxley {

Network::~Network() {
#ifdef HH_USE_CUDA
    free_cuda_runtime();
    if (V_cache_pinned_) cudaFreeHost(V_cache_pinned_);
    if (I_syn_pinned_)   cudaFreeHost(I_syn_pinned_);
#endif
}

void Network::set_device(const Device& device) {
    if (pool_mgr_.on_cuda() &&
        (device.type != Device::Type::CUDA ||
         device.index != pool_mgr_.cuda_device_id())) {
#ifdef HH_USE_CUDA
        sync_cuda_synapses_to_host(true);
        free_cuda_runtime();
#endif
    }

    if (device.type == Device::Type::CUDA)
        pool_mgr_.assign_to_device(device.index);
    else
        pool_mgr_.assign_to_cpu();
    pools_dirty_ = true;
}

Device Network::get_device() const {
    return pool_mgr_.on_cuda()
        ? Device::cuda(pool_mgr_.cuda_device_id())
        : Device::cpu();
}

void Network::reallocate_pinned_buffers(size_t n) {
#ifdef HH_USE_CUDA
    if (V_cache_pinned_) cudaFreeHost(V_cache_pinned_);
    if (I_syn_pinned_)   cudaFreeHost(I_syn_pinned_);
    cudaMallocHost(reinterpret_cast<void**>(&V_cache_pinned_), n * sizeof(double));
    cudaMallocHost(reinterpret_cast<void**>(&I_syn_pinned_),   n * sizeof(double));
    use_pinned_memory_ = true;
    pinned_size_       = n;
#else
    (void)n;
#endif
}

#ifdef HH_USE_CUDA
bool Network::can_use_cuda_synapse_path() const {
    return pool_mgr_.on_cuda() && sa_.size() > 0 && !has_stdp_ && !has_stp_;
}

void Network::free_cuda_runtime() {
    if (sim_stream_) { cudaStreamDestroy(sim_stream_); sim_stream_ = nullptr; }
    if (d_V_out_)    { cudaFree(d_V_out_);    d_V_out_    = nullptr; d_V_out_cap_    = 0; }
    if (d_spike_buf_){ cudaFree(d_spike_buf_); d_spike_buf_= nullptr; d_spike_buf_cap_= 0; }
    if (d_V_cache_) cudaFree(d_V_cache_);
    if (d_I_syn_) cudaFree(d_I_syn_);
    if (d_synapse_g_scale_) cudaFree(d_synapse_g_scale_);
    d_V_cache_ = nullptr;
    d_I_syn_ = nullptr;
    d_synapse_g_scale_ = nullptr;
    cuda_buffer_size_ = 0;
    free_device_synapses(dev_syn_);
}

void Network::ensure_cuda_runtime_buffers(size_t n_neurons) {
    if (cuda_buffer_size_ == n_neurons && d_V_cache_ && d_I_syn_ && d_synapse_g_scale_)
        return;

    if (d_V_cache_) cudaFree(d_V_cache_);
    if (d_I_syn_) cudaFree(d_I_syn_);
    if (d_synapse_g_scale_) cudaFree(d_synapse_g_scale_);
    d_V_cache_ = nullptr;
    d_I_syn_ = nullptr;
    d_synapse_g_scale_ = nullptr;

    if (n_neurons == 0) {
        cuda_buffer_size_ = 0;
        return;
    }

    if (cudaSetDevice(pool_mgr_.cuda_device_id()) != cudaSuccess)
        throw std::runtime_error("Network::ensure_cuda_runtime_buffers: cudaSetDevice failed");

    if (cudaMalloc(reinterpret_cast<void**>(&d_V_cache_), n_neurons * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_I_syn_), n_neurons * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_synapse_g_scale_), n_neurons * sizeof(double)) != cudaSuccess) {
        throw std::runtime_error("Network::ensure_cuda_runtime_buffers: cudaMalloc failed");
    }
    cuda_buffer_size_ = n_neurons;
}

void Network::prepare_cuda_synapses(double dt) {
    if (!can_use_cuda_synapse_path()) return;

    const size_t n_neurons = neurons_.size();
    const size_t n_synapses = sa_.size();
    const size_t ring_size = event_slots_.empty() ? 1 : event_slots_.size();
    const size_t spec_count = synapse_specs_.size();

    ensure_cuda_runtime_buffers(n_neurons);

    const bool needs_realloc =
        dev_syn_.neuron_count != n_neurons ||
        dev_syn_.synapse_count != n_synapses ||
        dev_syn_.ring_size != ring_size ||
        dev_syn_.spec_count != spec_count ||
        dev_syn_.device_id != pool_mgr_.cuda_device_id();

    if (needs_realloc) {
        allocate_device_synapses(dev_syn_, n_neurons, n_synapses, ring_size,
                                 spec_count, pool_mgr_.cuda_device_id());
    }

    // Compute min delay across all synapses for kernel barrier optimization.
    // delay_steps_ was populated in build_injection_tables() before this call.
    uint32_t min_delay = std::numeric_limits<uint32_t>::max();
    for (size_t i = 0; i < n_synapses; ++i) {
        if (delay_steps_[i] < min_delay) min_delay = delay_steps_[i];
    }
    dev_syn_.min_delay_steps = (n_synapses == 0)
        ? std::numeric_limits<uint32_t>::max()  // no synapses → no race possible
        : min_delay;

    std::vector<DeviceSynapseSpecDesc> spec_descs(spec_count);
    std::vector<VmExpr> vm_exprs;

    auto register_vm = [&](const VmExpr& expr) -> int {
        if (expr.empty()) return -1;
        vm_exprs.push_back(expr);
        return static_cast<int>(vm_exprs.size()) - 1;
    };

    for (size_t i = 0; i < spec_count; ++i) {
        const auto& spec = synapse_specs_[i];
        auto& desc = spec_descs[i];
        desc.update_form = static_cast<int>(spec.update_form);
        desc.current_form = static_cast<int>(spec.current_form);
        desc.power = spec.power;
        desc.spec_g = spec.g;
        desc.delta_S = spec.delta_S;
        desc.delta_A = spec.delta_A;
        desc.tau_S = spec.tau_S;
        desc.tau_A = spec.tau_A;
        desc.norm_factor = spec.norm_factor;
        desc.tanh_amp = spec.tanh_amp;
        desc.tanh_vh = spec.tanh_vh;
        desc.tanh_k = spec.tanh_k;
        desc.tau_decay = spec.tau_decay;
        desc.s_inf = spec.s_inf;
        desc.tau = spec.tau;
        desc.alpha = spec.alpha;
        desc.beta = spec.beta;
        desc.mg_conc = spec.mg_conc;
        desc.mg_scale = spec.mg_scale;
        desc.mg_denom = spec.mg_denom;
        desc.dS_vm_idx = register_vm(spec.dS_dt_vm);
        desc.dA_vm_idx = register_vm(spec.dA_dt_vm);
        desc.current_vm_idx = register_vm(spec.current_vm);
    }

    std::vector<uint8_t> spike_ring(n_synapses * ring_size, 0u);
    for (size_t arrival_slot = 0; arrival_slot < event_slots_.size(); ++arrival_slot) {
        for (size_t syn_idx : event_slots_[arrival_slot]) {
            const size_t write_slot =
                (arrival_slot + ring_size - (delay_steps_[syn_idx] % ring_size)) % ring_size;
            spike_ring[syn_idx * ring_size + write_slot] = 1u;
        }
    }

    std::vector<uint32_t> spec_idx_u32 = sa_.spec_idx;

    upload_device_synapse_state(
        dev_syn_,
        sa_.weight,
        sa_.g,
        sa_.S,
        sa_.A,
        sa_.E_syn,
        pre_decoded_,
        post_decoded_,
        delay_steps_,
        spec_idx_u32,
        spec_descs,
        vm_exprs,
        V_prev_,
        spike_ring,
        dt,
        spike_threshold_);
}

void Network::sync_cuda_synapses_to_host(bool sync_pending_events) {
    if (!dev_syn_.d_g) return;

    std::vector<uint8_t> spike_ring;
    std::vector<double> v_prev;
    download_device_synapse_state(dev_syn_, sa_.g, sa_.S, sa_.A, v_prev,
                                  sync_pending_events ? &spike_ring : nullptr,
                                  nullptr);
    V_prev_ = std::move(v_prev);

    if (!sync_pending_events) return;

    event_slots_.assign(dev_syn_.ring_size, {});
    for (size_t syn_idx = 0; syn_idx < dev_syn_.synapse_count; ++syn_idx) {
        const size_t delay = delay_steps_[syn_idx];
        for (size_t write_slot = 0; write_slot < dev_syn_.ring_size; ++write_slot) {
            if (!spike_ring[syn_idx * dev_syn_.ring_size + write_slot]) continue;
            const size_t age = (current_step_ + dev_syn_.ring_size - 1 - write_slot) % dev_syn_.ring_size;
            if (age > delay) continue;
            const size_t future_offset = delay - age;
            const size_t arrival_slot = (current_step_ + future_offset) % dev_syn_.ring_size;
            event_slots_[arrival_slot].push_back(syn_idx);
        }
    }
}
#endif

static constexpr double kSynapseEpsilon = 1e-9;

// =============================================================================
// Constructors
// =============================================================================

Network::Network(size_t num_neurons) {
    neurons_.reserve(num_neurons);
    for (size_t i = 0; i < num_neurons; ++i) {
        neurons_.push_back(std::make_unique<HHNeuron>());
    }
}

Network::Network(size_t num_neurons, NeuronType type) {
    neurons_.reserve(num_neurons);
    for (size_t i = 0; i < num_neurons; ++i) {
        add_neuron(type);
    }
}

// =============================================================================
// Add neurons (unchanged)
// =============================================================================

size_t Network::add_neuron() {
    neurons_.push_back(std::make_unique<HHNeuron>());
    return neurons_.size() - 1;
}

size_t Network::add_neuron(const HHNeuron::Parameters& params) {
    neurons_.push_back(std::make_unique<HHNeuron>(params));
    return neurons_.size() - 1;
}

size_t Network::add_neuron(NeuronType type) {
    switch (type) {
        case NeuronType::HH:
            return add_hh_neuron();
        case NeuronType::IZHIKEVICH_RS:
            return add_izhikevich_neuron(IzhikevichNeuron::Type::REGULAR_SPIKING);
        case NeuronType::IZHIKEVICH_FS:
            return add_izhikevich_neuron(IzhikevichNeuron::Type::FAST_SPIKING);
        case NeuronType::IZHIKEVICH_IB:
            return add_izhikevich_neuron(IzhikevichNeuron::Type::INTRINSICALLY_BURSTING);
        case NeuronType::IZHIKEVICH_CH:
            return add_izhikevich_neuron(IzhikevichNeuron::Type::CHATTERING);
        case NeuronType::IZHIKEVICH_LTS:
            return add_izhikevich_neuron(IzhikevichNeuron::Type::LOW_THRESHOLD_SPIKING);
        case NeuronType::IZHIKEVICH_CUSTOM:
            return add_izhikevich_neuron(IzhikevichNeuron::Type::CUSTOM);
        case NeuronType::COMPOSABLE:
            throw std::runtime_error("COMPOSABLE type requires NeuronModelSpec overload");
        default:
            return add_hh_neuron();
    }
}

size_t Network::add_neuron(const IzhikevichNeuron::Parameters& params) {
    neurons_.push_back(std::make_unique<IzhikevichNeuron>(params));
    return neurons_.size() - 1;
}

size_t Network::add_neuron(const NeuronModelSpec& spec) {
    spec.validate();
    neurons_.push_back(std::make_unique<ComposableNeuron>(spec));
    return neurons_.size() - 1;
}

size_t Network::add_hh_neuron() {
    neurons_.push_back(std::make_unique<HHNeuron>());
    return neurons_.size() - 1;
}

size_t Network::add_hh_neuron(const HHNeuron::Parameters& params) {
    neurons_.push_back(std::make_unique<HHNeuron>(params));
    return neurons_.size() - 1;
}

size_t Network::add_izhikevich_neuron(IzhikevichNeuron::Type type) {
    neurons_.push_back(std::make_unique<IzhikevichNeuron>(type));
    return neurons_.size() - 1;
}

size_t Network::add_izhikevich_neuron(const IzhikevichNeuron::Parameters& params) {
    neurons_.push_back(std::make_unique<IzhikevichNeuron>(params));
    return neurons_.size() - 1;
}

// =============================================================================
// Neuron accessors (unchanged)
// =============================================================================

const HHNeuron& Network::hh_neuron(size_t idx) const {
    if (idx >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }
    auto* hh = dynamic_cast<const HHNeuron*>(neurons_[idx].get());
    if (!hh) {
        throw std::runtime_error("Neuron at index " + std::to_string(idx) + " is not an HH neuron");
    }
    return *hh;
}

HHNeuron& Network::hh_neuron(size_t idx) {
    if (idx >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }
    auto* hh = dynamic_cast<HHNeuron*>(neurons_[idx].get());
    if (!hh) {
        throw std::runtime_error("Neuron at index " + std::to_string(idx) + " is not an HH neuron");
    }
    return *hh;
}

const IzhikevichNeuron& Network::iz_neuron(size_t idx) const {
    if (idx >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }
    auto* iz = dynamic_cast<const IzhikevichNeuron*>(neurons_[idx].get());
    if (!iz) {
        throw std::runtime_error("Neuron at index " + std::to_string(idx) + " is not an Izhikevich neuron");
    }
    return *iz;
}

IzhikevichNeuron& Network::iz_neuron(size_t idx) {
    if (idx >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }
    auto* iz = dynamic_cast<IzhikevichNeuron*>(neurons_[idx].get());
    if (!iz) {
        throw std::runtime_error("Neuron at index " + std::to_string(idx) + " is not an Izhikevich neuron");
    }
    return *iz;
}

// =============================================================================
// Synapse accessor — lazily builds the SynapseBase view vector (task27.5)
// =============================================================================

void Network::ensure_synapses_built() const {
    if (!synapses_dirty_ && synapses_.size() == sa_.size()) return;
    const size_t S = sa_.size();
    synapses_.resize(S);
    for (size_t i = 0; i < S; ++i)
        synapses_[i] = SynapseBase(i, this);
    synapses_dirty_ = false;
}

const SynapseBase& Network::synapse(size_t idx) const {
    ensure_synapses_built();
    if (idx >= synapses_.size()) throw std::out_of_range("Synapse index out of range");
    return synapses_[idx];
}

// =============================================================================
// Unified add_synapse — primary method, all forms
// =============================================================================

size_t Network::add_synapse(size_t pre, size_t post, double weight,
                             const SynapseSpec& spec, double delay) {
    if (pre >= neurons_.size() || post >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }

    // Dedup spec by name; populate spec cache for new entries
    size_t sidx = synapse_specs_.size();
    for (size_t i = 0; i < synapse_specs_.size(); ++i) {
        if (synapse_specs_[i].name == spec.name) { sidx = i; break; }
    }
    if (sidx == synapse_specs_.size()) {
        synapse_specs_.push_back(spec);
        SynapseSpecCache c;
        c.delta_S   = spec.delta_S;
        c.delta_A   = spec.delta_A;
        c.tau_S     = spec.tau_S;
        c.tau_A     = spec.tau_A;
        c.inv_tau_A = spec.tau_A > 0.0 ? 1.0 / spec.tau_A : 0.0;
        c.norm      = spec.norm_factor;
        // decay fields left 0 — filled by rebuild_spec_caches() at simulate time
        synapse_spec_caches_.push_back(c);
    }

    // Restore pre/post arrays if they were freed after a prior build
    if (pre_post_cleared_) reconstruct_pre_post();

    // Common SoA fields — narrow types: task27.2 (pre/post/delay), task27.4 (E_syn/spec_idx)
    sa_.pre.push_back(static_cast<uint32_t>(pre));
    sa_.post.push_back(static_cast<uint32_t>(post));
    sa_.weight.push_back(weight);
    sa_.E_syn.push_back(static_cast<float>(spec.E_syn));
    sa_.g.push_back(0.0);
    sa_.delay.push_back(static_cast<float>(delay));

    // Unified state (spec-derived constants now live in synapse_spec_caches_)
    sa_.S.push_back(spec.S_init);
    sa_.A.push_back(spec.A_init);
    sa_.spec_idx.push_back(static_cast<uint32_t>(sidx));

    // Plasticity defaults (NONE — no state allocated)
    sa_.plast_type.push_back(PlasticityType::NONE);
    sa_.plast_state_idx.push_back(-1);
    sa_.plast_spec_idx_arr.push_back(static_cast<uint32_t>(0));
    sa_.is_active.push_back(false);

    // synapses_ is built lazily on first synapse(idx) call (task27.5)
    synapses_dirty_ = true;

    sa_.cached_dt = -1.0;
    soa_sorted_ = false;
    groups_built_ = false;
    injection_tables_built_ = false;

    return sa_.size() - 1;
}

// =============================================================================
// add_synapse overload with PlasticitySpec
// =============================================================================

size_t Network::add_or_find_plasticity_spec(const PlasticitySpec& ps) {
    for (size_t i = 0; i < plasticity_specs_.size(); ++i) {
        const auto& e = plasticity_specs_[i];
        if (e.type != ps.type) continue;
        if (ps.type == PlasticityType::STDP) {
            const auto& a = e.stdp; const auto& b = ps.stdp;
            if (a.A_plus == b.A_plus && a.A_minus == b.A_minus &&
                a.tau_plus == b.tau_plus && a.tau_minus == b.tau_minus &&
                a.w_min == b.w_min && a.w_max == b.w_max &&
                a.modulator_pop_start == b.modulator_pop_start &&
                a.modulator_substance_idx == b.modulator_substance_idx)
                return i;
        } else if (ps.type == PlasticityType::STP) {
            const auto& a = e.stp; const auto& b = ps.stp;
            if (a.U == b.U && a.tau_u == b.tau_u && a.tau_x == b.tau_x)
                return i;
        }
    }
    plasticity_specs_.push_back(ps);
    return plasticity_specs_.size() - 1;
}

size_t Network::add_synapse(size_t pre, size_t post, double weight,
                             const SynapseSpec& spec, double delay,
                             const PlasticitySpec& plast) {
    size_t syn_i = add_synapse(pre, post, weight, spec, delay);

    if (plast.type == PlasticityType::NONE) return syn_i;

    size_t ps_idx = add_or_find_plasticity_spec(plast);
    sa_.plast_type[syn_i] = plast.type;
    sa_.plast_spec_idx_arr[syn_i] = static_cast<uint32_t>(ps_idx);

    if (plast.type == PlasticityType::STDP) {
        int32_t state_i = (int32_t)sa_.plast_x_pre.size();
        sa_.plast_state_idx[syn_i] = state_i;
        sa_.plast_x_pre.push_back(0.0);
        sa_.plast_x_post.push_back(0.0);
    } else if (plast.type == PlasticityType::STP) {
        int32_t state_i = (int32_t)sa_.stp_u.size();
        sa_.plast_state_idx[syn_i] = state_i;
        sa_.stp_u.push_back(plast.stp.U);
        sa_.stp_x.push_back(1.0);
    }

    return syn_i;
}

// =============================================================================
// Backward-compatible convenience wrappers — delegate to add_synapse()
// =============================================================================

void Network::add_synapse(size_t pre_idx, size_t post_idx, double weight,
                           double E_syn, double tau, double delay) {
    add_synapse(pre_idx, post_idx, weight,
                SynapseSpec::exponential(tau, 1.0, E_syn), delay);
    // weight is passed separately; spec.g=1.0 so effective conductance = weight * S
    // Correct the weight already pushed so the product matches legacy behaviour:
    // legacy: g[i] += weight on spike; g[i] *= exp(-dt/tau)
    // new: g = spec.g * weight * S; spec.g=1 so g = weight * S — same.
    // (No adjustment needed — exponential preset sets delta_S=1, tau_S=tau, g=1)
    // Overwrite E_syn in SoA (spec already set it, but name-dedup may have reused
    // a spec with different E_syn; store per-synapse E_syn in SoA directly)
    sa_.E_syn.back() = E_syn;
}

void Network::add_alpha_synapse(size_t pre_idx, size_t post_idx, double weight,
                                double E_syn, double tau, double delay) {
    add_synapse(pre_idx, post_idx, weight,
                SynapseSpec::alpha_function(tau, 1.0, E_syn), delay);
    sa_.E_syn.back() = E_syn;
}

void Network::add_double_exp_synapse(size_t pre_idx, size_t post_idx, double weight,
                                     double E_syn,
                                     double tau_rise, double tau_decay,
                                     double delay) {
    add_synapse(pre_idx, post_idx, weight,
                SynapseSpec::double_exponential(tau_rise, tau_decay, 1.0, E_syn), delay);
    sa_.E_syn.back() = E_syn;
}

// Receptor-type convenience methods
void Network::add_ampa_synapse(size_t pre_idx, size_t post_idx, double weight,
                               double delay) {
    add_synapse(pre_idx, post_idx, weight, SynapseSpec::ampa(), delay);
}

void Network::add_nmda_synapse(size_t pre_idx, size_t post_idx, double weight,
                               double delay) {
    add_synapse(pre_idx, post_idx, weight, SynapseSpec::nmda(), delay);
}

void Network::add_gaba_a_synapse(size_t pre_idx, size_t post_idx, double weight,
                                 double delay) {
    add_synapse(pre_idx, post_idx, weight, SynapseSpec::gaba_a(), delay);
}

void Network::add_receptor_synapse(size_t pre_idx, size_t post_idx, double weight,
                                   ReceptorType receptor, double delay) {
    switch (receptor) {
        case ReceptorType::AMPA:   add_ampa_synapse(pre_idx, post_idx, weight, delay);  break;
        case ReceptorType::NMDA:   add_nmda_synapse(pre_idx, post_idx, weight, delay);  break;
        case ReceptorType::GABA_A: add_gaba_a_synapse(pre_idx, post_idx, weight, delay); break;
    }
}

// Kinetic synapse (now delegates to unified add_synapse)
size_t Network::add_kinetic_synapse(size_t pre, size_t post, double weight,
                                    const SynapseSpec& spec, double delay) {
    return add_synapse(pre, post, weight, spec, delay);
}

// =============================================================================
// Kinetic state accessors (now read from unified SoA)
// =============================================================================

double Network::get_kin_S(size_t synapse_idx) const {
    if (synapse_idx >= sa_.size()) throw std::out_of_range("Synapse index out of range");
    return sa_.S[synapse_idx];
}

double Network::get_kin_g(size_t synapse_idx) const {
    if (synapse_idx >= sa_.size()) throw std::out_of_range("Synapse index out of range");
    return sa_.g[synapse_idx];
}

// =============================================================================
// Utility methods
// =============================================================================

std::vector<double> Network::get_potentials() const {
    std::vector<double> potentials;
    potentials.reserve(neurons_.size());
    for (const auto& neuron : neurons_) {
        potentials.push_back(neuron->membrane_potential());
    }
    return potentials;
}

void Network::ensure_buffers() {
    size_t n = neurons_.size();
    if (I_syn_buffer_.size() != n) {
        I_syn_buffer_.resize(n, 0.0);
        V_cache_.resize(n, 0.0);
        synapse_g_scale_.assign(n, 1.0);
    }
}

void Network::cache_voltages() {
    for (size_t i = 0; i < neurons_.size(); ++i) {
        V_cache_[i] = neurons_[i]->membrane_potential();
    }
}

// =============================================================================
// Thread control
// =============================================================================

void Network::set_num_threads(int n) {
    num_threads_ = n;
    pool_mgr_.set_num_threads(n);
#ifdef HH_USE_OPENMP
    if (n > 0) omp_set_num_threads(n);
#endif
}

// =============================================================================
// Reset
// =============================================================================

void Network::reset() {
    for (auto& neuron : neurons_) {
        neuron->reset();
    }

    const size_t N = sa_.size();

    // Reset unified mutable state
    std::fill(sa_.g.begin(), sa_.g.end(), 0.0);

    for (size_t i = 0; i < N; ++i) {
        const auto& spec = synapse_specs_[sa_.spec_idx[i]];
        sa_.S[i] = spec.S_init;
        sa_.A[i] = spec.A_init;
    }

    // Clear in-flight spike events and reset step counter
    for (auto& slot : event_slots_) slot.clear();
    current_step_ = 0;

    // Reinitialise per-neuron previous voltage
    const size_t N_neurons = neurons_.size();
    V_prev_.resize(N_neurons);
    for (size_t n = 0; n < N_neurons; ++n)
        V_prev_[n] = neurons_[n]->membrane_potential();

    sa_.cached_dt = -1.0;

    soa_dirty_ = false;
    pools_dirty_ = true;
}

// =============================================================================
// Spec cache rebuild — O(n_spec_types), called once when dt changes (task27.1)
// =============================================================================

void Network::rebuild_spec_caches(double dt) {
    if (dt == sa_.cached_dt) return;

    using UF = SynapseSpec::UpdateForm;
    synapse_spec_caches_.resize(synapse_specs_.size());
    for (size_t i = 0; i < synapse_specs_.size(); ++i) {
        const auto& spec = synapse_specs_[i];
        auto& c = synapse_spec_caches_[i];
        c.delta_S   = spec.delta_S;
        c.delta_A   = spec.delta_A;
        c.tau_S     = spec.tau_S;
        c.tau_A     = spec.tau_A;
        c.inv_tau_A = spec.tau_A > 0.0 ? 1.0 / spec.tau_A : 0.0;
        c.norm      = spec.norm_factor;
        const auto uf = spec.update_form;
        c.decay_S = (uf == UF::EXP_DECAY || uf == UF::DOUBLE_EXP)
                    ? std::exp(-dt / spec.tau_S) : 0.0;
        c.decay_A = (uf == UF::DOUBLE_EXP)
                    ? std::exp(-dt / spec.tau_A) : 0.0;
    }
    sa_.cached_dt = dt;
}

// =============================================================================
// Synaptic current computation (SoA — contiguous, cache-friendly)
// =============================================================================

void Network::compute_synaptic_currents() {
    std::fill(I_syn_buffer_.begin(), I_syn_buffer_.end(), 0.0);

    const double*   g    = sa_.g.data();
    const float*    E_syn = sa_.E_syn.data();  // float; promotes to double in arithmetic
    const uint32_t* post  = post_decoded_.empty() ? sa_.post.data() : post_decoded_.data();
    const double*   V    = V_cache_.data();
    double*         I_syn = I_syn_buffer_.data();

    if (!synapse_g_scale_.empty() && pool_mgr_.has_synapse_g_mods()) {
        const double* gscale = synapse_g_scale_.data();
        for (size_t i : syn_groups_.active_g)
            I_syn[post[i]] += g[i] * gscale[post[i]] * (E_syn[i] - V[post[i]]);
    } else {
        for (size_t i : syn_groups_.active_g)
            I_syn[post[i]] += g[i] * (E_syn[i] - V[post[i]]);
    }
}

// =============================================================================
// Step
// =============================================================================

void Network::step(double dt, const std::vector<double>& I_ext) {
    if (I_ext.size() != neurons_.size()) {
        throw std::invalid_argument("I_ext size must match number of neurons");
    }

    ensure_buffers();
    sort_synapses_by_pre();
    if (!groups_built_) build_synapse_groups();

    cache_voltages();
    compute_synaptic_currents();

    for (size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i]->step(dt, I_ext[i] + I_syn_buffer_[i]);
    }

    cache_voltages();
    update_synapses_grouped(dt);

    soa_dirty_ = true;
}

// =============================================================================
// Reconstruct sa_.pre / sa_.post from compressed data (task27.2)
// Called by add_synapse() when new synapses are added after a prior build.
// =============================================================================

void Network::reconstruct_pre_post() {
    const size_t S = post_decoded_.size();
    sa_.pre.resize(S);
    sa_.post.resize(S);
    for (size_t n = 0; n < neuron_connectivity_.size(); ++n) {
        const auto& conn = neuron_connectivity_[n];
        for (size_t j = 0; j < conn.syn_count; ++j) {
            sa_.pre[conn.syn_start + j]  = static_cast<uint32_t>(n);
            sa_.post[conn.syn_start + j] = post_decoded_[conn.syn_start + j];
        }
    }
    pre_post_cleared_ = false;
}

// =============================================================================
// Sort SoA arrays by presynaptic index for cache-friendly access
// =============================================================================

void Network::sort_synapses_by_pre() {
    if (soa_sorted_) return;

    const size_t S = sa_.size();
    if (S <= 1) { soa_sorted_ = true; return; }

    std::vector<size_t> perm(S);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        if (sa_.pre[a] != sa_.pre[b]) return sa_.pre[a] < sa_.pre[b];
        return sa_.post[a] < sa_.post[b];
    });

    auto permute = [&](auto& vec) {
        using T = typename std::decay<decltype(vec)>::type;
        T tmp(vec.size());
        for (size_t i = 0; i < S; ++i) tmp[i] = std::move(vec[perm[i]]);
        vec = std::move(tmp);
    };

    // Common fields
    permute(sa_.pre);
    permute(sa_.post);
    permute(sa_.weight);
    permute(sa_.E_syn);
    permute(sa_.g);
    permute(sa_.delay);

    // Unified state
    permute(sa_.S);
    permute(sa_.A);
    permute(sa_.spec_idx);

    // is_active: vector<bool> uses proxy refs — permute manually
    {
        std::vector<bool> tmp(S);
        for (size_t i = 0; i < S; ++i) tmp[i] = sa_.is_active[perm[i]];
        sa_.is_active = std::move(tmp);
    }

    // synapses_ indices no longer valid after permutation — mark dirty (task27.5)
    synapses_dirty_ = true;

    // Active lists reference old indices — clear; will be rebuilt by build_synapse_groups
    syn_groups_.active_exp_decay.clear();
    syn_groups_.active_alpha_func.clear();
    syn_groups_.active_double_exp.clear();
    syn_groups_.active_g.clear();

    sa_.cached_dt = -1.0;
    soa_sorted_ = true;
}

// =============================================================================
// Build UpdateForm-separated synapse index lists
// =============================================================================

void Network::build_synapse_groups() {
    syn_groups_.exp_decay.clear();
    syn_groups_.alpha_func.clear();
    syn_groups_.double_exp.clear();
    syn_groups_.voltage_gated.clear();
    syn_groups_.stdp.clear();
    syn_groups_.stp.clear();
    syn_groups_.active_exp_decay.clear();
    syn_groups_.active_alpha_func.clear();
    syn_groups_.active_double_exp.clear();
    syn_groups_.active_g.clear();

    using UF = SynapseSpec::UpdateForm;
    const size_t S = sa_.size();
    sa_.is_active.assign(S, false);
    for (size_t i = 0; i < S; ++i) {
        auto form = synapse_specs_[sa_.spec_idx[i]].update_form;
        if      (form == UF::EXP_DECAY)  syn_groups_.exp_decay.push_back(i);
        else if (form == UF::ALPHA_FUNC) syn_groups_.alpha_func.push_back(i);
        else if (form == UF::DOUBLE_EXP) syn_groups_.double_exp.push_back(i);
        else                              syn_groups_.voltage_gated.push_back(i);

        if (sa_.plast_type[i] == PlasticityType::STDP) syn_groups_.stdp.push_back(i);
        if (sa_.plast_type[i] == PlasticityType::STP)  syn_groups_.stp.push_back(i);

        // Re-activate any synapse already carrying conductance (e.g. after re-sort)
        if (form == UF::EXP_DECAY && sa_.S[i] >= kSynapseEpsilon) {
            sa_.is_active[i] = true;
            syn_groups_.active_exp_decay.push_back(i);
        } else if (form == UF::ALPHA_FUNC &&
                   (sa_.S[i] >= kSynapseEpsilon || sa_.A[i] >= kSynapseEpsilon)) {
            sa_.is_active[i] = true;
            syn_groups_.active_alpha_func.push_back(i);
        } else if (form == UF::DOUBLE_EXP &&
                   (sa_.S[i] >= kSynapseEpsilon || sa_.A[i] >= kSynapseEpsilon)) {
            sa_.is_active[i] = true;
            syn_groups_.active_double_exp.push_back(i);
        }
    }

    spike_detected_.resize(S);

    has_stdp_ = !syn_groups_.stdp.empty();
    has_stp_  = !syn_groups_.stp.empty();

    if (has_stdp_) {
        size_t N = neurons_.size();
        V_all_prev_.assign(N, spike_threshold_ - 10.0);
        post_spiked_.assign(N, 0);
    }

    groups_built_ = true;
}

// =============================================================================
// Forward-injection spike delivery tables (task26)
// =============================================================================

void Network::build_injection_tables(double dt) {
    const size_t N = neurons_.size();

    // On subsequent simulate() calls the tables are unchanged; only reset V_prev_.
    if (injection_tables_built_) {
        V_prev_.resize(N);
        for (size_t n = 0; n < N; ++n)
            V_prev_[n] = neurons_[n]->membrane_potential();
        return;
    }

    const size_t S = sa_.size();

    delay_steps_.resize(S);
    size_t max_delay_steps = 0;
    for (size_t i = 0; i < S; ++i) {
        delay_steps_[i] = static_cast<uint32_t>(
            std::round(static_cast<double>(sa_.delay[i]) / dt));
        if (delay_steps_[i] > max_delay_steps) max_delay_steps = delay_steps_[i];
    }

    post_from_.assign(N, {});
    for (size_t i = 0; i < S; ++i)
        post_from_[sa_.pre[i]].push_back({static_cast<uint32_t>(i), delay_steps_[i]});

    // +1 so the slot for step t and the slot for step t+max_delay don't collide
    event_slots_.assign(max_delay_steps + 1, {});
    current_step_ = 0;

    V_prev_.resize(N);
    for (size_t n = 0; n < N; ++n)
        V_prev_[n] = neurons_[n]->membrane_potential();

    build_neuron_connectivity();
    injection_tables_built_ = true;
}

// =============================================================================
// Build compressed per-neuron connectivity (task27.2)
// Called at the end of build_injection_tables() once sa_.pre is sorted.
// Populates neuron_connectivity_, post_decoded_, pre_decoded_;
// then clears sa_.pre / sa_.post to reclaim ~8 bytes/synapse.
// =============================================================================

void Network::build_neuron_connectivity() {
    const size_t N = neurons_.size();
    const size_t S = sa_.size();

    neuron_connectivity_.assign(N, NeuronConnectivity{});
    post_decoded_.resize(S);
    pre_decoded_.resize(S);

    size_t k = 0;
    for (size_t n = 0; n < N; ++n) {
        auto& conn = neuron_connectivity_[n];
        conn.syn_start = k;
        while (k < S && sa_.pre[k] == n) ++k;
        conn.syn_count = k - conn.syn_start;

        for (size_t j = 0; j < conn.syn_count; ++j) {
            const size_t idx = conn.syn_start + j;
            pre_decoded_[idx] = static_cast<uint32_t>(n);
            post_decoded_[idx] = sa_.post[idx];
        }

        if (conn.syn_count == 0) continue;

        // Delta-encode post indices (sorted ascending after sort_synapses_by_pre)
        conn.post_deltas.resize(conn.syn_count);
        uint32_t prev = 0;
        for (size_t j = 0; j < conn.syn_count; ++j) {
            uint32_t cur   = sa_.post[conn.syn_start + j];
            uint32_t delta = cur - prev;
            if (delta < 255u) {
                conn.post_deltas[j] = static_cast<uint8_t>(delta);
            } else {
                conn.post_deltas[j] = 255u;
                conn.post_overflow.push_back(static_cast<uint32_t>(j));
                conn.post_overflow.push_back(cur);
            }
            prev = cur;
        }

        // Uniform weight detection
        double w0 = sa_.weight[conn.syn_start];
        conn.uniform_weight = true;
        for (size_t j = 1; j < conn.syn_count; ++j) {
            if (sa_.weight[conn.syn_start + j] != w0) {
                conn.uniform_weight = false;
                break;
            }
        }
        conn.weight_value = w0;
    }

    // Free sa_.pre and sa_.post — connectivity is now in neuron_connectivity_ + decoded arrays
    sa_.pre.clear();  sa_.pre.shrink_to_fit();
    sa_.post.clear(); sa_.post.shrink_to_fit();
    pre_post_cleared_ = true;
}

// =============================================================================
// Unified synapse update — four tight sub-loops, no branch on type within each
// =============================================================================

void Network::update_synapses_grouped(double dt) {
    rebuild_spec_caches(dt);

    const size_t S = sa_.size();
    const double spike_threshold = spike_threshold_;

    // Phase 0: post-synaptic spike detection (STDP only, zero cost otherwise)
    if (has_stdp_) {
        const size_t N = neurons_.size();
        for (size_t j = 0; j < N; ++j) {
            double Vj = V_cache_[j];
            post_spiked_[j] = (Vj > spike_threshold && V_all_prev_[j] <= spike_threshold) ? 1 : 0;
            V_all_prev_[j] = Vj;
        }
    }

    // Phase 1: forward injection + event dispatch
    // Lazy-build tables when called via the step() public API (no prior simulate call).
    // Skip if dt=0 — spike timing is undefined with zero timestep.
    if (event_slots_.empty() && dt > 0.0)
        build_injection_tables(dt);

    std::fill(spike_detected_.begin(), spike_detected_.end(), 0);
    if (!event_slots_.empty()) {
        const size_t N = neurons_.size();
        const size_t step = current_step_++;
        // Detect spikes at neuron level; push arrivals into the correct future slot.
        // Injection before dispatch ensures delay=0 fires in the same step.
        for (size_t n = 0; n < N; ++n) {
            double V = V_cache_[n];
            bool spiked = (V > spike_threshold) && (V_prev_[n] <= spike_threshold);
            V_prev_[n] = V;
            if (!spiked) continue;
            for (const SynapseRef& ref : post_from_[n]) {
                size_t slot = (step + ref.delay_steps) % event_slots_.size();
                event_slots_[slot].push_back(ref.syn_idx);
            }
        }
        // Populate spike_detected_ from events due this step, then clear the slot.
        const size_t cur_slot = step % event_slots_.size();
        for (size_t k : event_slots_[cur_slot])
            spike_detected_[k] = 1;
        event_slots_[cur_slot].clear();
    }

    // Phase 2a: EXP_DECAY — spike additive jump, exact multiplicative decay
    // Activate newly spiked synapses
    for (size_t k : syn_groups_.exp_decay) {
        if (spike_detected_[k] && !sa_.is_active[k]) {
            sa_.is_active[k] = true;
            syn_groups_.active_exp_decay.push_back(k);
        }
    }
    // Iterate active only; swap-and-pop on decay
    {
        size_t pos = 0;
        while (pos < syn_groups_.active_exp_decay.size()) {
            const size_t k = syn_groups_.active_exp_decay[pos];
            const auto& cache = synapse_spec_caches_[sa_.spec_idx[k]];
            if (spike_detected_[k]) sa_.S[k] += cache.delta_S;
            sa_.S[k] *= cache.decay_S;
            if (sa_.S[k] < 0.0) sa_.S[k] = 0.0;
            const auto& spec = synapse_specs_[sa_.spec_idx[k]];
            sa_.g[k] = spec.g * sa_.weight[k] * sa_.S[k];
            if (sa_.S[k] < kSynapseEpsilon) {
                sa_.g[k] = 0.0;
                sa_.is_active[k] = false;
                syn_groups_.active_exp_decay[pos] = syn_groups_.active_exp_decay.back();
                syn_groups_.active_exp_decay.pop_back();
            } else { ++pos; }
        }
    }

    // Phase 2b: ALPHA_FUNC — 2-variable Euler, spike on A
    for (size_t k : syn_groups_.alpha_func) {
        if (spike_detected_[k] && !sa_.is_active[k]) {
            sa_.is_active[k] = true;
            syn_groups_.active_alpha_func.push_back(k);
        }
    }
    {
        size_t pos = 0;
        while (pos < syn_groups_.active_alpha_func.size()) {
            const size_t k = syn_groups_.active_alpha_func[pos];
            const auto& cache = synapse_spec_caches_[sa_.spec_idx[k]];
            if (spike_detected_[k]) sa_.A[k] += cache.delta_A;
            double dS  = (sa_.A[k] - sa_.S[k]) * cache.inv_tau_A;
            double dA  = -sa_.A[k] * cache.inv_tau_A;
            sa_.S[k] += dt * dS;
            sa_.A[k] += dt * dA;
            if (sa_.S[k] < 0.0) sa_.S[k] = 0.0;
            const auto& spec = synapse_specs_[sa_.spec_idx[k]];
            sa_.g[k] = spec.g * sa_.weight[k] * sa_.S[k];
            if (sa_.S[k] < kSynapseEpsilon && sa_.A[k] < kSynapseEpsilon) {
                sa_.g[k] = 0.0;
                sa_.is_active[k] = false;
                syn_groups_.active_alpha_func[pos] = syn_groups_.active_alpha_func.back();
                syn_groups_.active_alpha_func.pop_back();
            } else { ++pos; }
        }
    }

    // Phase 2c: DOUBLE_EXP — two independent exact decays, spike on both
    for (size_t k : syn_groups_.double_exp) {
        if (spike_detected_[k] && !sa_.is_active[k]) {
            sa_.is_active[k] = true;
            syn_groups_.active_double_exp.push_back(k);
        }
    }
    {
        size_t pos = 0;
        while (pos < syn_groups_.active_double_exp.size()) {
            const size_t k = syn_groups_.active_double_exp[pos];
            const auto& cache = synapse_spec_caches_[sa_.spec_idx[k]];
            if (spike_detected_[k]) {
                sa_.S[k] += cache.delta_S;
                sa_.A[k] += cache.delta_A;
            }
            sa_.S[k] *= cache.decay_S;
            sa_.A[k] *= cache.decay_A;
            double g_eff = cache.norm * (sa_.S[k] - sa_.A[k]);
            if (g_eff < 0.0) g_eff = 0.0;
            const auto& spec = synapse_specs_[sa_.spec_idx[k]];
            sa_.g[k] = spec.g * sa_.weight[k] * g_eff;
            if (sa_.S[k] < kSynapseEpsilon && sa_.A[k] < kSynapseEpsilon) {
                sa_.g[k] = 0.0;
                sa_.is_active[k] = false;
                syn_groups_.active_double_exp[pos] = syn_groups_.active_double_exp.back();
                syn_groups_.active_double_exp.pop_back();
            } else { ++pos; }
        }
    }

    // Phase 2d: voltage-gated — TANH_GATE, BOLTZMANN_GATE, ALPHA_BETA, CUSTOM_EXPR
    using UF = SynapseSpec::UpdateForm;
    using CF = SynapseSpec::CurrentForm;

    for (size_t k : syn_groups_.voltage_gated) {
        double Vpre  = V_cache_[pre_decoded_[k]];
        double Vpost = V_cache_[post_decoded_[k]];
        const auto& spec = synapse_specs_[sa_.spec_idx[k]];
        double S = sa_.S[k];

        if (spec.update_form == UF::TANH_GATE) {
            double rate_open = spec.tanh_amp
                               * (1.0 + std::tanh((Vpre - spec.tanh_vh) / spec.tanh_k));
            double rate = rate_open + 1.0 / spec.tau_decay;
            double S_inf = rate_open / rate;
            S = S_inf + (S - S_inf) * std::exp(-dt * rate);

        } else if (spec.update_form == UF::BOLTZMANN_GATE) {
            double S_inf = boltzmann_scalar(Vpre, spec.s_inf);
            double tau_s = compute_tau_scalar(Vpre, spec.tau);
            if (tau_s < 1e-10) tau_s = 1e-10;
            S = S_inf + (S - S_inf) * std::exp(-dt / tau_s);

        } else if (spec.update_form == UF::ALPHA_BETA) {
            double a = compute_rate_scalar(Vpre, spec.alpha);
            double b = compute_rate_scalar(Vpre, spec.beta);
            double rate = a + b;
            double S_inf = (rate > 1e-10) ? a / rate : S;
            S = S_inf + (S - S_inf) * std::exp(-dt * rate);

        } else if (spec.update_form == UF::CUSTOM_EXPR) {
            if (!spec.dS_dt_vm.empty()) {
                if (!spec.dA_dt_vm.empty()) {
                    // 2-variable ODE: both S and A are live
                    double A_k = sa_.A[k];
                    double dS  = vm_eval_scalar_3arg(spec.dS_dt_vm, Vpre, S, A_k);
                    double dA  = vm_eval_scalar_3arg(spec.dA_dt_vm, Vpre, S, A_k);
                    S = std::max(0.0, std::min(1.0, S + dt * dS));
                    sa_.A[k] = A_k + dt * dA;
                } else {
                    // 1-variable ODE
                    double dS = vm_eval_scalar_2arg(spec.dS_dt_vm, Vpre, S);
                    S = std::max(0.0, std::min(1.0, S + dt * dS));
                }
            }
        }

        sa_.S[k] = S;

        // Current computation
        if (spec.current_form == CF::CUSTOM_EXPR && !spec.current_vm.empty()) {
            sa_.g[k] = vm_eval_scalar_3arg(spec.current_vm, Vpost, S, sa_.A[k])
                       * sa_.weight[k];
        } else {
            double gS = spec.g * sa_.weight[k];
            for (int p = 0; p < spec.power; ++p) gS *= S;
            if (spec.current_form == CF::MG_BLOCK) {
                double mg = 1.0 + spec.mg_conc
                                 * std::exp(-spec.mg_scale * Vpost) / spec.mg_denom;
                gS /= mg;
            }
            sa_.g[k] = gS;
        }
    }

    // Rebuild active_g: spike-driven actives + voltage-gated with g > epsilon
    syn_groups_.active_g.clear();
    for (size_t k : syn_groups_.active_exp_decay)
        syn_groups_.active_g.push_back(k);
    for (size_t k : syn_groups_.active_alpha_func)
        syn_groups_.active_g.push_back(k);
    for (size_t k : syn_groups_.active_double_exp)
        syn_groups_.active_g.push_back(k);
    for (size_t k : syn_groups_.voltage_gated)
        if (sa_.g[k] > kSynapseEpsilon)
            syn_groups_.active_g.push_back(k);

    // Phase 3: STDP weight updates
    if (has_stdp_) apply_stdp(dt);

    // Phase 4: STP conductance scaling
    if (has_stp_) apply_stp(dt);
}

// =============================================================================
// Simulate — batched pools + type-separated synapse loops
// =============================================================================

std::vector<std::vector<double>> Network::simulate(
    double duration,
    double dt,
    const std::vector<std::vector<double>>& I_ext
) {
    const size_t num_steps = static_cast<size_t>(duration / dt);
    const size_t n_neurons = neurons_.size();

    std::vector<double> V_flat(n_neurons * num_steps, 0.0);

    simulate_into_buffers(
        duration, dt, I_ext,
        V_flat.data(),
        nullptr, 0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        1,
        num_steps,
        0.0
    );

    std::vector<std::vector<double>> traces(n_neurons);
    for (size_t i = 0; i < n_neurons; ++i)
        traces[i].assign(V_flat.data() + i * num_steps,
                         V_flat.data() + (i + 1) * num_steps);
    return traces;
}

// =============================================================================
// max_gate_count / get_synapse_pre/post_indices
// =============================================================================

size_t Network::max_gate_count() const {
    size_t max_gates = 0;
    for (const auto& n : neurons_) {
        if (auto* cn = dynamic_cast<ComposableNeuron*>(n.get())) {
            max_gates = std::max(max_gates, cn->gate_states().size());
        }
    }
    return max_gates;
}

std::vector<size_t> Network::get_synapse_pre_indices() const {
    if (!pre_decoded_.empty())
        return std::vector<size_t>(pre_decoded_.begin(), pre_decoded_.end());
    return std::vector<size_t>(sa_.pre.begin(), sa_.pre.end());
}

std::vector<size_t> Network::get_synapse_post_indices() const {
    if (!post_decoded_.empty())
        return std::vector<size_t>(post_decoded_.begin(), post_decoded_.end());
    return std::vector<size_t>(sa_.post.begin(), sa_.post.end());
}

// =============================================================================
// simulate_into_buffers — batched pools + recording into caller-allocated bufs
// =============================================================================

void Network::simulate_into_buffers(
    double duration, double dt,
    const std::vector<std::vector<double>>& I_ext,
    double* V_buf,
    double* gate_buf,   size_t max_gates,
    double* calcium_buf,
    double* u_buf,
    double* g_syn_buf,
    double* I_syn_buf,
    double* spike_event_buf,
    size_t  interval,
    size_t  n_rec,
    double  spike_threshold
) {
    spike_threshold_ = spike_threshold;

    size_t num_steps = static_cast<size_t>(duration / dt);
    size_t n_neurons = neurons_.size();

    if (I_ext.size() != n_neurons) {
        throw std::invalid_argument("I_ext outer size must match number of neurons");
    }
    for (const auto& curr : I_ext) {
        if (curr.size() < num_steps) {
            throw std::invalid_argument("I_ext vectors too short for simulation duration");
        }
    }

    sort_synapses_by_pre();
    build_synapse_groups();
    build_injection_tables(dt);

    if (pools_dirty_) {
        pool_mgr_.build_from_neurons(neurons_, fast_math_);
        pools_dirty_ = false;
    }

    ensure_buffers();
    const size_t S = sa_.size();

#ifdef HH_USE_CUDA
    const bool use_cuda_synapses = can_use_cuda_synapse_path();
    if (use_cuda_synapses && (!use_pinned_memory_ || pinned_size_ != n_neurons))
        reallocate_pinned_buffers(n_neurons);
    if (use_cuda_synapses) {
        prepare_cuda_synapses(dt);
        if (pool_mgr_.has_synapse_g_mods()) {
            fill_device_buffer(d_synapse_g_scale_, 1.0, n_neurons);
            pool_mgr_.scatter_synapse_g_scale_device(d_synapse_g_scale_);
            pool_mgr_.synchronize_cuda();
        }
    }
    double* V_host = use_pinned_memory_ ? V_cache_pinned_ : V_cache_.data();
    double* I_host = use_pinned_memory_ ? I_syn_pinned_   : I_syn_buffer_.data();
    std::vector<uint8_t> spike_arrived_host;
    if (use_cuda_synapses && spike_event_buf)
        spike_arrived_host.resize(S);
#else
    const bool use_cuda_synapses = false;
    double* V_host = V_cache_.data();
    double* I_host = I_syn_buffer_.data();
#endif

    std::vector<double> syn_spike_accum(n_neurons, 0.0);

    for (size_t t = 0; t < num_steps; ++t) {
        if (use_cuda_synapses) {
#ifdef HH_USE_CUDA
            pool_mgr_.scatter_all_voltages_device(d_V_cache_);
            pool_mgr_.synchronize_cuda();
#endif
        } else {
            pool_mgr_.scatter_all_voltages(V_cache_.data());
        }

        if (t % interval == 0) {
            size_t tr = t / interval;

            if (V_buf) {
                if (use_cuda_synapses &&
                    !(gate_buf || calcium_buf || u_buf)) {
#ifdef HH_USE_CUDA
                    if (cudaMemcpy(V_host, d_V_cache_,
                                   n_neurons * sizeof(double),
                                   cudaMemcpyDeviceToHost) != cudaSuccess) {
                        throw std::runtime_error("simulate_into_buffers: cudaMemcpy V_cache D2H failed");
                    }
#endif
                } else {
                    pool_mgr_.scatter_all_voltages(V_host);
                }

                for (size_t i = 0; i < n_neurons; ++i)
                    V_buf[i * n_rec + tr] = V_host[i];
            }

            if (gate_buf && max_gates > 0)
                pool_mgr_.scatter_gates(gate_buf, max_gates, n_rec, tr);

            if (calcium_buf)
                pool_mgr_.scatter_calcium(calcium_buf, n_rec, tr);

            if (u_buf)
                pool_mgr_.scatter_recoveries(u_buf, n_rec, tr);

            if (g_syn_buf) {
                if (use_cuda_synapses) {
#ifdef HH_USE_CUDA
                    if (cudaMemcpy(sa_.g.data(), dev_syn_.d_g,
                                   S * sizeof(double),
                                   cudaMemcpyDeviceToHost) != cudaSuccess) {
                        throw std::runtime_error("simulate_into_buffers: cudaMemcpy synapse g D2H failed");
                    }
#endif
                }
                const double* g = sa_.g.data();
                for (size_t i = 0; i < S; ++i)
                    g_syn_buf[i * n_rec + tr] = g[i];
            }

            if (spike_event_buf) {
                for (size_t i = 0; i < n_neurons; ++i) {
                    spike_event_buf[i * n_rec + tr] = syn_spike_accum[i];
                    syn_spike_accum[i] = 0.0;
                }
            }
        }

        if (use_cuda_synapses) {
#ifdef HH_USE_CUDA
            for (size_t i = 0; i < n_neurons; ++i)
                I_host[i] = I_ext[i][t];

            if (cudaMemcpy(d_I_syn_, I_host, n_neurons * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("simulate_into_buffers: cudaMemcpy I_ext H2D failed");
            }

            accumulate_cuda_synaptic_currents(
                dev_syn_, d_V_cache_, d_I_syn_,
                pool_mgr_.has_synapse_g_mods() ? d_synapse_g_scale_ : nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("simulate_into_buffers: CUDA synaptic current kernel failed");

            if (I_syn_buf && t % interval == 0) {
                size_t tr = t / interval;
                if (cudaMemcpy(I_host, d_I_syn_, n_neurons * sizeof(double),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("simulate_into_buffers: cudaMemcpy I_syn D2H failed");
                }
                for (size_t i = 0; i < n_neurons; ++i)
                    I_syn_buf[i * n_rec + tr] = I_host[i] - I_ext[i][t];
            }

            pool_mgr_.gather_all_currents_device(d_I_syn_);
            pool_mgr_.synchronize_cuda();
            pool_mgr_.step_all(dt);

            if (pool_mgr_.has_synapse_g_mods()) {
                fill_device_buffer(d_synapse_g_scale_, 1.0, n_neurons);
                pool_mgr_.scatter_synapse_g_scale_device(d_synapse_g_scale_);
                pool_mgr_.synchronize_cuda();
            }

            pool_mgr_.scatter_all_voltages_device(d_V_cache_);
            pool_mgr_.synchronize_cuda();

            const size_t step = current_step_++;
            update_cuda_synapses(dev_syn_, d_V_cache_, step);
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("simulate_into_buffers: CUDA synapse update kernel failed");

            if (spike_event_buf) {
                if (cudaMemcpy(spike_arrived_host.data(), dev_syn_.d_spike_arrived,
                               S * sizeof(uint8_t),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("simulate_into_buffers: cudaMemcpy spike_arrived D2H failed");
                }
                for (size_t j = 0; j < S; ++j)
                    if (spike_arrived_host[j]) syn_spike_accum[post_decoded_[j]] += 1.0;
            }
#endif
        } else {
            for (size_t i = 0; i < n_neurons; ++i)
                I_syn_buffer_[i] = I_ext[i][t];

            const double*   g          = sa_.g.data();
            const float*    E_syn_data = sa_.E_syn.data();  // float; promotes in arithmetic
            const uint32_t* post       = post_decoded_.data();
            const double*   V          = V_cache_.data();
            double*         I_buf      = I_syn_buffer_.data();
            if (pool_mgr_.has_synapse_g_mods()) {
                const double* gscale = synapse_g_scale_.data();
                for (size_t i = 0; i < S; ++i)
                    I_buf[post[i]] += g[i] * gscale[post[i]] * (E_syn_data[i] - V[post[i]]);
            } else {
                for (size_t i = 0; i < S; ++i)
                    I_buf[post[i]] += g[i] * (E_syn_data[i] - V[post[i]]);
            }

            if (I_syn_buf && t % interval == 0) {
                size_t tr = t / interval;
                for (size_t i = 0; i < n_neurons; ++i)
                    I_syn_buf[i * n_rec + tr] = I_syn_buffer_[i] - I_ext[i][t];
            }

            pool_mgr_.gather_all_currents(I_syn_buffer_.data());
            pool_mgr_.step_all(dt);
            if (pool_mgr_.has_synapse_g_mods()) {
                std::fill(synapse_g_scale_.begin(), synapse_g_scale_.end(), 1.0);
                pool_mgr_.scatter_synapse_g_scale(synapse_g_scale_.data());
            }
            pool_mgr_.scatter_all_voltages(V_cache_.data());

            update_synapses_grouped(dt);

            if (spike_event_buf) {
                for (size_t j = 0; j < S; ++j)
                    if (spike_detected_[j]) syn_spike_accum[post_decoded_[j]] += 1.0;
            }
        }
    }

#ifdef HH_USE_CUDA
    if (use_cuda_synapses)
        sync_cuda_synapses_to_host(true);
#endif
    pool_mgr_.sync_all_to_neurons(neurons_);
    if (pool_mgr_.has_synapse_g_mods()) {
        std::fill(synapse_g_scale_.begin(), synapse_g_scale_.end(), 1.0);
        pool_mgr_.scatter_synapse_g_scale(synapse_g_scale_.data());
    }
}

// =============================================================================
// simulate_with_descriptors — compact StimPlan instead of dense I_ext matrix
// =============================================================================

void Network::simulate_with_descriptors(
    double duration, double dt,
    const StimPlan& stim,
    double* V_buf,
    double* gate_buf,       size_t max_gates,
    double* calcium_buf,
    double* u_buf,
    double* g_syn_buf,
    double* I_syn_buf,
    double* spike_event_buf,
    uint8_t* spike_buf_uint8,
    size_t  interval,
    size_t  n_rec,
    double  spike_threshold
) {
    spike_threshold_ = spike_threshold;

    size_t num_steps = static_cast<size_t>(duration / dt);
    size_t n_neurons = neurons_.size();

    if (stim.I_const.size() != n_neurons) {
        throw std::invalid_argument(
            "StimPlan::I_const size must match number of neurons");
    }

    sort_synapses_by_pre();
    build_synapse_groups();
    build_injection_tables(dt);

    if (pools_dirty_) {
        pool_mgr_.build_from_neurons(neurons_, fast_math_);
        pools_dirty_ = false;
    }

    ensure_buffers();

    const size_t S = sa_.size();

#ifdef HH_USE_CUDA
    // Cooperative kernel path: entire simulation on-device, no per-step CPU round-trips
    if (pool_mgr_.on_cuda() && !has_stdp_ && !has_stp_
        && pool_mgr_.has_coop_capable_cuda_pools()
        && !gate_buf && !calcium_buf && !u_buf && !g_syn_buf && !I_syn_buf && !spike_event_buf) {

        if (!use_pinned_memory_ || pinned_size_ != n_neurons)
            reallocate_pinned_buffers(n_neurons);
        ensure_cuda_runtime_buffers(n_neurons);
        prepare_cuda_synapses(dt);

        if (!sim_stream_) {
            if (cudaStreamCreate(&sim_stream_) != cudaSuccess)
                throw std::runtime_error("simulate_with_descriptors: cudaStreamCreate failed");
        }

        // Allocate device recording buffers
        const size_t v_needed = V_buf ? n_neurons * n_rec : 0;
        if (v_needed > d_V_out_cap_) {
            if (d_V_out_) cudaFree(d_V_out_);
            if (v_needed > 0 &&
                cudaMalloc(reinterpret_cast<void**>(&d_V_out_),
                           v_needed * sizeof(double)) != cudaSuccess)
                throw std::runtime_error("simulate_with_descriptors: cudaMalloc d_V_out_ failed");
            d_V_out_cap_ = v_needed;
        }
        const size_t s_needed = spike_buf_uint8 ? n_neurons * n_rec : 0;
        if (s_needed > d_spike_buf_cap_) {
            if (d_spike_buf_) cudaFree(d_spike_buf_);
            if (s_needed > 0 &&
                cudaMalloc(reinterpret_cast<void**>(&d_spike_buf_),
                           s_needed * sizeof(uint8_t)) != cudaSuccess)
                throw std::runtime_error("simulate_with_descriptors: cudaMalloc d_spike_buf_ failed");
            d_spike_buf_cap_ = s_needed;
        }
        if (spike_buf_uint8 && d_spike_buf_)
            cudaMemset(d_spike_buf_, 0, s_needed * sizeof(uint8_t));

        // Collect pool descriptors (ensures device state is ready)
        std::vector<CudaHHDesc>         hh_descs;
        std::vector<CudaIzDesc>         iz_descs;
        std::vector<CudaComposableDesc> comp_descs;
        pool_mgr_.collect_hh_descs(hh_descs);
        pool_mgr_.collect_iz_descs(iz_descs);
        pool_mgr_.collect_comp_descs(comp_descs);

        CudaStimRaw   stim_raw = upload_stim_raw(stim, n_neurons, pool_mgr_.cuda_device_id());
        DeviceSynapseRaw syn_raw = make_syn_raw(dev_syn_);

        simulate_all_steps(
            hh_descs.data(),  static_cast<int>(hh_descs.size()),
            iz_descs.data(),  static_cast<int>(iz_descs.size()),
            comp_descs.data(), static_cast<int>(comp_descs.size()),
            syn_raw, stim_raw,
            d_V_cache_, d_I_syn_,
            V_buf           ? d_V_out_     : nullptr,
            spike_buf_uint8 ? d_spike_buf_ : nullptr,
            n_neurons, num_steps, dt,
            current_step_, interval, n_rec,
            sim_stream_);

        current_step_ += num_steps;

        if (V_buf && d_V_out_)
            cudaMemcpy(V_buf, d_V_out_, v_needed * sizeof(double), cudaMemcpyDeviceToHost);
        if (spike_buf_uint8 && d_spike_buf_)
            cudaMemcpy(spike_buf_uint8, d_spike_buf_, s_needed * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        sync_cuda_synapses_to_host(true);
        free_stim_raw(stim_raw);

        pool_mgr_.sync_all_to_neurons(neurons_);
        return;
    }
#endif

    double* V_ptr = V_cache_.data();
    double* I_ptr = I_syn_buffer_.data();
#ifdef HH_USE_CUDA
    if (pool_mgr_.on_cuda()) {
        if (!use_pinned_memory_ || pinned_size_ != n_neurons)
            reallocate_pinned_buffers(n_neurons);
        V_ptr = V_cache_pinned_;
        I_ptr = I_syn_pinned_;
    }
#endif

    std::vector<double> syn_spike_accum(n_neurons, 0.0);
    std::vector<double> I_stim_cache(n_neurons, 0.0);

    for (size_t t = 0; t < num_steps; ++t) {
        pool_mgr_.scatter_all_voltages(V_ptr);

        if (t % interval == 0) {
            size_t tr = t / interval;
            if (V_buf) {
                for (size_t i = 0; i < n_neurons; ++i)
                    V_buf[i * n_rec + tr] = V_ptr[i];
            }
            if (gate_buf && max_gates > 0)
                pool_mgr_.scatter_gates(gate_buf, max_gates, n_rec, tr);
            if (calcium_buf)
                pool_mgr_.scatter_calcium(calcium_buf, n_rec, tr);
            if (u_buf)
                pool_mgr_.scatter_recoveries(u_buf, n_rec, tr);
            if (g_syn_buf) {
                const double* g = sa_.g.data();
                for (size_t i = 0; i < S; ++i)
                    g_syn_buf[i * n_rec + tr] = g[i];
            }
            if (spike_event_buf) {
                for (size_t i = 0; i < n_neurons; ++i) {
                    spike_event_buf[i * n_rec + tr] = syn_spike_accum[i];
                    syn_spike_accum[i] = 0.0;
                }
            }
        }

        for (size_t i = 0; i < n_neurons; ++i)
            I_ptr[i] = I_stim_cache[i] = stim.I_const[i];
        for (const auto& p : stim.pulses) {
            if (t >= p.onset_step && t < p.end_step)
                for (size_t i = p.neuron_start; i < p.neuron_end; ++i) {
                    I_ptr[i] += p.amplitude;
                    I_stim_cache[i] += p.amplitude;
                }
        }
        for (const auto& d : stim.dbs) {
            if (d.isi_steps == 0) continue;
            size_t phase = t % d.isi_steps;
            if (phase < d.pw_steps)
                for (size_t i = d.neuron_start; i < d.neuron_end; ++i) {
                    I_ptr[i] += d.amplitude;
                    I_stim_cache[i] += d.amplitude;
                }
        }

        const double*   g          = sa_.g.data();
        const float*    E_syn_data = sa_.E_syn.data();
        const uint32_t* post       = post_decoded_.data();
        const double*   V          = V_ptr;
        double*         I_buf      = I_ptr;
        if (pool_mgr_.has_synapse_g_mods()) {
            const double* gscale = synapse_g_scale_.data();
            for (size_t i : syn_groups_.active_g)
                I_buf[post[i]] += g[i] * gscale[post[i]] * (E_syn_data[i] - V[post[i]]);
        } else {
            for (size_t i : syn_groups_.active_g)
                I_buf[post[i]] += g[i] * (E_syn_data[i] - V[post[i]]);
        }

        if (I_syn_buf && t % interval == 0) {
            size_t tr = t / interval;
            for (size_t i = 0; i < n_neurons; ++i)
                I_syn_buf[i * n_rec + tr] = I_ptr[i] - I_stim_cache[i];
        }

        pool_mgr_.gather_all_currents(I_ptr);
        pool_mgr_.step_all(dt);
        if (pool_mgr_.has_synapse_g_mods()) {
            std::fill(synapse_g_scale_.begin(), synapse_g_scale_.end(), 1.0);
            pool_mgr_.scatter_synapse_g_scale(synapse_g_scale_.data());
        }
        pool_mgr_.scatter_all_voltages(V_ptr);
#ifdef HH_USE_CUDA
        if (pool_mgr_.on_cuda())
            std::copy(V_ptr, V_ptr + n_neurons, V_cache_.data());
#endif
        update_synapses_grouped(dt);

        if (spike_event_buf) {
            for (size_t j = 0; j < S; ++j)
                if (spike_detected_[j]) syn_spike_accum[post_decoded_[j]] += 1.0;
        }
    }

    pool_mgr_.sync_all_to_neurons(neurons_);
    if (pool_mgr_.has_synapse_g_mods()) {
        std::fill(synapse_g_scale_.begin(), synapse_g_scale_.end(), 1.0);
        pool_mgr_.scatter_synapse_g_scale(synapse_g_scale_.data());
    }
}

// =============================================================================
// Phase 2 delay-decomposition parallel simulation
//
// Each thread group owns a disjoint set of neurons and their pre-synapses.
// Cross-group communication happens only through the existing per-synapse delay
// ring buffers (sa_.spike_buf) which are already partitioned by pre-neuron.
//
// Synchronisation: std::atomic<size_t> step_done[gid] — incremented (with
// release ordering) after each step. Group B at step t waits for each source
// group A to reach step_done[A] >= t (acquire) before reading g[k] for
// inter-group A→B synapses (those values were written during A's step t-1).
//
// Note: STDP / STP weight updates are skipped in this path (they require a
// network-wide barrier each step). Support can be added in a follow-on once
// a per-step barrier primitive is introduced (see task16.md §16.2 for the
// SpikeTransport abstraction that will generalise this to CUDA P2P in task17).
// =============================================================================

void Network::simulate_with_descriptors_parallel(
    double duration, double dt,
    const StimPlan& stim,
    const std::map<int, std::vector<size_t>>& group_neurons,
    const std::map<int, std::vector<std::string>>& group_pools,
    double* V_buf,
    double* gate_buf,        size_t max_gates,
    double* calcium_buf,
    double* u_buf,
    double* g_syn_buf,
    double* I_syn_buf,
    double* spike_event_buf,
    size_t  interval,
    size_t  n_rec,
    double  spike_threshold
) {
    spike_threshold_ = spike_threshold;
    const size_t num_steps = static_cast<size_t>(duration / dt);
    const size_t n_neurons = neurons_.size();

    if (stim.I_const.size() != n_neurons)
        throw std::invalid_argument("StimPlan::I_const size must match number of neurons");

    // --- Pre-simulate preparation (same as serial path) ---
    sort_synapses_by_pre();
    build_synapse_groups();
    build_injection_tables(dt);
    if (pools_dirty_) {
        pool_mgr_.build_from_neurons(neurons_, fast_math_);
        pools_dirty_ = false;
    }
    ensure_buffers();
    rebuild_spec_caches(dt);   // once on main thread — dt is constant for this run

    // --- Build GroupDef structs ---
    // Map neuron global index → group id
    std::unordered_map<size_t, int> neuron_gid;
    for (const auto& kv : group_neurons)
        for (size_t ni : kv.second)
            neuron_gid[ni] = kv.first;

    const size_t n_groups = group_neurons.size();
    std::vector<GroupDef> groups(n_groups);

    for (const auto& kv : group_neurons) {
        int gid = kv.first;
        GroupDef& g = groups[gid];
        g.id = gid;
        g.neuron_indices = kv.second;
        auto pit = group_pools.find(gid);
        if (pit != group_pools.end()) g.pool_names = pit->second;
    }

    // Partition synapse type lists by pre-neuron group
    // (uses pre_decoded_ — sa_.pre was cleared by build_neuron_connectivity)
    auto assign_pre = [&](const std::vector<size_t>& src, std::vector<size_t> GroupDef::* member) {
        for (size_t k : src) {
            auto it = neuron_gid.find(static_cast<size_t>(pre_decoded_[k]));
            if (it != neuron_gid.end())
                (groups[it->second].*member).push_back(k);
        }
    };
    assign_pre(syn_groups_.exp_decay,      &GroupDef::pre_exp_decay);
    assign_pre(syn_groups_.alpha_func,     &GroupDef::pre_alpha_func);
    assign_pre(syn_groups_.double_exp,     &GroupDef::pre_double_exp);
    assign_pre(syn_groups_.voltage_gated,  &GroupDef::pre_voltage_gated);

    // Build pre_all (union of all pre-synapse type lists) for spike detection Phase 1
    for (size_t k = 0; k < sa_.size(); ++k) {
        auto it = neuron_gid.find(static_cast<size_t>(pre_decoded_[k]));
        if (it != neuron_gid.end())
            groups[it->second].pre_all.push_back(k);
    }

    // Build post_syn (synapse indices where post[k] in this group) — I_syn accumulation
    for (size_t k = 0; k < sa_.size(); ++k) {
        auto it = neuron_gid.find(static_cast<size_t>(post_decoded_[k]));
        if (it != neuron_gid.end())
            groups[it->second].post_syn.push_back(k);
    }

    // Build src_group_ids for each group
    for (auto& g : groups) {
        std::unordered_set<int> srcs;
        for (size_t k : g.post_syn) {
            auto it = neuron_gid.find(static_cast<size_t>(pre_decoded_[k]));
            if (it != neuron_gid.end() && it->second != g.id)
                srcs.insert(it->second);
        }
        g.src_group_ids.assign(srcs.begin(), srcs.end());
    }

    // Build consumer_group_ids (groups that read this group's g[k] values)
    // Inverse of src_group_ids: if B has A in src_group_ids, then A has B in consumer_group_ids
    for (int b = 0; b < (int)n_groups; ++b)
        for (int src : groups[b].src_group_ids)
            groups[src].consumer_group_ids.push_back(b);

    // Build intra_syn: synapses where both pre and post are in the same group
    for (size_t k = 0; k < sa_.size(); ++k) {
        auto pre_it  = neuron_gid.find(static_cast<size_t>(pre_decoded_[k]));
        auto post_it = neuron_gid.find(static_cast<size_t>(post_decoded_[k]));
        if (pre_it != neuron_gid.end() && post_it != neuron_gid.end()
                && pre_it->second == post_it->second)
            groups[pre_it->second].intra_syn.push_back(k);
    }

    // Fill hh_local_indices / iz_local_indices for each group
    for (auto& g : groups)
        pool_mgr_.fill_group_hh_iz_indices(g.neuron_indices,
                                            g.hh_local_indices,
                                            g.iz_local_indices);

    // Build per-group forward-injection structures (task26)
    for (auto& grp : groups) {
        // Collect unique pre-neuron indices for this group's pre-synapses
        std::unordered_set<size_t> pre_neuron_set;
        for (size_t k : grp.pre_all)
            pre_neuron_set.insert(static_cast<size_t>(pre_decoded_[k]));
        grp.pre_neurons.assign(pre_neuron_set.begin(), pre_neuron_set.end());

        // Size event buffer to cover max delay among this group's synapses
        size_t max_delay = 0;
        for (size_t n : grp.pre_neurons)
            for (const auto& ref : post_from_[n])
                if (ref.delay_steps > max_delay) max_delay = ref.delay_steps;
        grp.event_slots.assign(max_delay + 1, {});
    }

    // --- Initial V_cache scatter (t=0 needs pre-existing pool state) ---
    pool_mgr_.scatter_all_voltages(V_cache_.data());

    const bool has_gmod = pool_mgr_.has_synapse_g_mods();

    // Two-counter synchronization to prevent write-after-read race on g[k]:
    //   step_done[A]: incremented after A's Phase U2 (full step t complete).
    //     B waits for step_done[A] >= t before reading g[k] in Phase A of step t.
    //   read_done[B]: incremented after B's Phase A (done reading g[k]).
    //     A waits for read_done[B] >= t+1 before writing g[k] in Phase U2 of step t.
    std::vector<std::atomic<size_t>> step_done(n_groups);
    for (auto& s : step_done) s.store(0, std::memory_order_relaxed);
    std::vector<std::atomic<size_t>> read_done(n_groups);
    for (auto& s : read_done) s.store(0, std::memory_order_relaxed);

    // Local aliases for hot-loop access (avoids repeated member dereference)
    double* const V_cache      = V_cache_.data();
    double* const I_buf        = I_syn_buffer_.data();
    double* const gscale       = synapse_g_scale_.data();
    const double* const I_const = stim.I_const.data();
    const uint32_t* const pre_arr  = pre_decoded_.data();
    const uint32_t* const post_arr = post_decoded_.data();
    const float* const  E_arr    = sa_.E_syn.data();  // float; promotes in arithmetic
    double* const g_arr          = sa_.g.data();

    // --- Launch one std::thread per group ---
    std::vector<std::thread> threads;
    threads.reserve(n_groups);

    for (int gid = 0; gid < static_cast<int>(n_groups); ++gid) {
        threads.emplace_back([&, gid]() {
            GroupDef& grp = groups[gid];

            // Thread-local per-step stimulus cache (for I_syn = I_total - I_stim)
            std::vector<double> grp_I_stim_cache(n_neurons, 0.0);
            // Thread-local spike-event accumulator (intra-group only)
            std::vector<double> grp_spike_accum(n_neurons, 0.0);

            for (size_t t = 0; t < num_steps; ++t) {

                // Phase W: wait for source groups to complete step t-1
                // (ensures g[k] for inter-group synapses is from step t-1)
                for (int src : grp.src_group_ids) {
                    while (step_done[src].load(std::memory_order_acquire) < t)
                        std::this_thread::yield();
                }

                // Phase R: record (from previous step's scatter / accumulation)
                if (t % interval == 0) {
                    size_t tr = t / interval;

                    if (V_buf)
                        for (size_t i : grp.neuron_indices)
                            V_buf[i * n_rec + tr] = V_cache[i];

                    if (gate_buf && max_gates > 0)
                        pool_mgr_.scatter_gates_for_names(
                            grp.pool_names, gate_buf, max_gates, n_rec, tr);

                    if (calcium_buf)
                        pool_mgr_.scatter_calcium_for_names(
                            grp.pool_names, calcium_buf, n_rec, tr);

                    if (u_buf)
                        pool_mgr_.scatter_recoveries_for_iz(
                            grp.iz_local_indices, u_buf, n_rec, tr);

                    if (g_syn_buf)
                        for (size_t k : grp.post_syn)
                            g_syn_buf[k * n_rec + tr] = g_arr[k];

                    if (spike_event_buf) {
                        for (size_t i : grp.neuron_indices) {
                            spike_event_buf[i * n_rec + tr] = grp_spike_accum[i];
                            grp_spike_accum[i] = 0.0;
                        }
                    }
                }

                // Phase I: seed I_syn from compact descriptors for own neurons
                for (size_t i : grp.neuron_indices)
                    I_buf[i] = grp_I_stim_cache[i] = I_const[i];

                for (const auto& p : stim.pulses) {
                    if (t >= p.onset_step && t < p.end_step) {
                        for (size_t i : grp.neuron_indices)
                            if (i >= p.neuron_start && i < p.neuron_end) {
                                I_buf[i] += p.amplitude;
                                grp_I_stim_cache[i] += p.amplitude;
                            }
                    }
                }
                for (const auto& d : stim.dbs) {
                    if (d.isi_steps == 0) continue;
                    if (t % d.isi_steps < d.pw_steps) {
                        for (size_t i : grp.neuron_indices)
                            if (i >= d.neuron_start && i < d.neuron_end) {
                                I_buf[i] += d.amplitude;
                                grp_I_stim_cache[i] += d.amplitude;
                            }
                    }
                }

                // Phase A: I_syn accumulation — post_syn: post[k] in this group
                // g[k] for inter-group synapses is from src group's step t-1 (wait above).
                // Check g > 0.0: spike-driven types set g=0.0 on deactivation; this matches
                // the serial path's active_g which includes all synapses with S >= epsilon.
                if (has_gmod) {
                    for (size_t k : grp.post_syn)
                        if (g_arr[k] > 0.0)
                            I_buf[post_arr[k]] += g_arr[k] * gscale[post_arr[k]]
                                                * (E_arr[k] - V_cache[post_arr[k]]);
                } else {
                    for (size_t k : grp.post_syn)
                        if (g_arr[k] > 0.0)
                            I_buf[post_arr[k]] += g_arr[k] * (E_arr[k] - V_cache[post_arr[k]]);
                }
                // Signal: done reading inter-group g[k] values for step t
                read_done[gid].fetch_add(1, std::memory_order_release);

                // Record I_syn = synaptic current (total minus external stimulus)
                if (I_syn_buf && t % interval == 0) {
                    size_t tr = t / interval;
                    for (size_t i : grp.neuron_indices)
                        I_syn_buf[i * n_rec + tr] = I_buf[i] - grp_I_stim_cache[i];
                }

                // Phase G: gather currents for all pools in this group
                pool_mgr_.gather_currents_for_names(grp.pool_names, I_buf);
                pool_mgr_.gather_currents_for_hh(grp.hh_local_indices, I_buf);
                pool_mgr_.gather_currents_for_iz(grp.iz_local_indices, I_buf);

                // Phase S: step all pools in this group
                pool_mgr_.step_for_names(grp.pool_names, dt);
                pool_mgr_.step_for_hh(grp.hh_local_indices, dt);
                pool_mgr_.step_for_iz(grp.iz_local_indices, dt);

                if (has_gmod) {
                    for (size_t i : grp.neuron_indices) gscale[i] = 1.0;
                    pool_mgr_.scatter_synapse_g_scale_for_names(grp.pool_names, gscale);
                }

                // Phase V: scatter voltages for all pools in this group → V_cache
                pool_mgr_.scatter_voltages_for_names(grp.pool_names, V_cache);
                pool_mgr_.scatter_voltages_for_hh(grp.hh_local_indices, V_cache);
                pool_mgr_.scatter_voltages_for_iz(grp.iz_local_indices, V_cache);

                // Phase U1: forward injection + event dispatch (task26)
                // Clear spike_detected_ for own pre-synapses
                for (size_t k : grp.pre_all) spike_detected_[k] = 0;
                // Detect spikes at neuron level; inject into future slots
                for (size_t n : grp.pre_neurons) {
                    double V = V_cache[n];
                    bool spiked = (V > spike_threshold_) && (V_prev_[n] <= spike_threshold_);
                    V_prev_[n] = V;
                    if (!spiked) continue;
                    for (const SynapseRef& ref : post_from_[n]) {
                        size_t slot = (t + ref.delay_steps) % grp.event_slots.size();
                        grp.event_slots[slot].push_back(ref.syn_idx);
                    }
                }
                // Dispatch events due this step
                const size_t cur_slot = t % grp.event_slots.size();
                for (size_t k : grp.event_slots[cur_slot])
                    spike_detected_[k] = 1;
                grp.event_slots[cur_slot].clear();

                // Spike-event accumulation: count intra-group spike arrivals
                // (inter-group spikes omitted — spike_detected_ from other groups
                //  is not safe to read without additional synchronization)
                if (spike_event_buf) {
                    for (size_t k : grp.intra_syn)
                        if (spike_detected_[k]) grp_spike_accum[post_arr[k]] += 1.0;
                }

                // Phase W2: wait until all consumer groups have finished reading g[k]
                // (read_done[consumer] >= t+1) before overwriting g[k] in Phase U2
                for (int cid : grp.consumer_group_ids) {
                    while (read_done[cid].load(std::memory_order_acquire) < t + 1)
                        std::this_thread::yield();
                }

                // Phase U2a: EXP_DECAY
                for (size_t k : grp.pre_exp_decay) {
                    if (spike_detected_[k] && !sa_.is_active[k]) {
                        sa_.is_active[k] = true;
                        grp.active_pre_exp_decay.push_back(k);
                    }
                }
                {
                    size_t pos = 0;
                    while (pos < grp.active_pre_exp_decay.size()) {
                        const size_t k = grp.active_pre_exp_decay[pos];
                        const auto& cache = synapse_spec_caches_[sa_.spec_idx[k]];
                        if (spike_detected_[k]) sa_.S[k] += cache.delta_S;
                        sa_.S[k] *= cache.decay_S;
                        if (sa_.S[k] < 0.0) sa_.S[k] = 0.0;
                        const auto& spec = synapse_specs_[sa_.spec_idx[k]];
                        sa_.g[k] = spec.g * sa_.weight[k] * sa_.S[k];
                        if (sa_.S[k] < kSynapseEpsilon) {
                            sa_.g[k] = 0.0;
                            sa_.is_active[k] = false;
                            grp.active_pre_exp_decay[pos] = grp.active_pre_exp_decay.back();
                            grp.active_pre_exp_decay.pop_back();
                        } else { ++pos; }
                    }
                }

                // Phase U2b: ALPHA_FUNC
                for (size_t k : grp.pre_alpha_func) {
                    if (spike_detected_[k] && !sa_.is_active[k]) {
                        sa_.is_active[k] = true;
                        grp.active_pre_alpha_func.push_back(k);
                    }
                }
                {
                    size_t pos = 0;
                    while (pos < grp.active_pre_alpha_func.size()) {
                        const size_t k = grp.active_pre_alpha_func[pos];
                        const auto& cache = synapse_spec_caches_[sa_.spec_idx[k]];
                        if (spike_detected_[k]) sa_.A[k] += cache.delta_A;
                        double dS  = (sa_.A[k] - sa_.S[k]) * cache.inv_tau_A;
                        double dA  = -sa_.A[k] * cache.inv_tau_A;
                        sa_.S[k] += dt * dS;
                        sa_.A[k] += dt * dA;
                        if (sa_.S[k] < 0.0) sa_.S[k] = 0.0;
                        const auto& spec = synapse_specs_[sa_.spec_idx[k]];
                        sa_.g[k] = spec.g * sa_.weight[k] * sa_.S[k];
                        if (sa_.S[k] < kSynapseEpsilon && sa_.A[k] < kSynapseEpsilon) {
                            sa_.g[k] = 0.0;
                            sa_.is_active[k] = false;
                            grp.active_pre_alpha_func[pos] = grp.active_pre_alpha_func.back();
                            grp.active_pre_alpha_func.pop_back();
                        } else { ++pos; }
                    }
                }

                // Phase U2c: DOUBLE_EXP
                for (size_t k : grp.pre_double_exp) {
                    if (spike_detected_[k] && !sa_.is_active[k]) {
                        sa_.is_active[k] = true;
                        grp.active_pre_double_exp.push_back(k);
                    }
                }
                {
                    size_t pos = 0;
                    while (pos < grp.active_pre_double_exp.size()) {
                        const size_t k = grp.active_pre_double_exp[pos];
                        const auto& cache = synapse_spec_caches_[sa_.spec_idx[k]];
                        if (spike_detected_[k]) {
                            sa_.S[k] += cache.delta_S;
                            sa_.A[k] += cache.delta_A;
                        }
                        sa_.S[k] *= cache.decay_S;
                        sa_.A[k] *= cache.decay_A;
                        double g_eff = cache.norm * (sa_.S[k] - sa_.A[k]);
                        if (g_eff < 0.0) g_eff = 0.0;
                        const auto& spec = synapse_specs_[sa_.spec_idx[k]];
                        sa_.g[k] = spec.g * sa_.weight[k] * g_eff;
                        if (sa_.S[k] < kSynapseEpsilon && sa_.A[k] < kSynapseEpsilon) {
                            sa_.g[k] = 0.0;
                            sa_.is_active[k] = false;
                            grp.active_pre_double_exp[pos] = grp.active_pre_double_exp.back();
                            grp.active_pre_double_exp.pop_back();
                        } else { ++pos; }
                    }
                }

                // Phase U2d: VOLTAGE_GATED (TANH_GATE, BOLTZMANN_GATE, ALPHA_BETA, CUSTOM_EXPR)
                using UF = SynapseSpec::UpdateForm;
                using CF = SynapseSpec::CurrentForm;
                for (size_t k : grp.pre_voltage_gated) {
                    double Vpre  = V_cache[pre_arr[k]];
                    double Vpost = V_cache[post_arr[k]];
                    const auto& spec = synapse_specs_[sa_.spec_idx[k]];
                    double S = sa_.S[k];

                    if (spec.update_form == UF::TANH_GATE) {
                        double rate_open = spec.tanh_amp
                                           * (1.0 + std::tanh((Vpre - spec.tanh_vh) / spec.tanh_k));
                        double rate = rate_open + 1.0 / spec.tau_decay;
                        double S_inf = rate_open / rate;
                        S = S_inf + (S - S_inf) * std::exp(-dt * rate);
                    } else if (spec.update_form == UF::BOLTZMANN_GATE) {
                        double S_inf = boltzmann_scalar(Vpre, spec.s_inf);
                        double tau_s = compute_tau_scalar(Vpre, spec.tau);
                        if (tau_s < 1e-10) tau_s = 1e-10;
                        S = S_inf + (S - S_inf) * std::exp(-dt / tau_s);
                    } else if (spec.update_form == UF::ALPHA_BETA) {
                        double a = compute_rate_scalar(Vpre, spec.alpha);
                        double b = compute_rate_scalar(Vpre, spec.beta);
                        double rate = a + b;
                        double S_inf = (rate > 1e-10) ? a / rate : S;
                        S = S_inf + (S - S_inf) * std::exp(-dt * rate);
                    } else if (spec.update_form == UF::CUSTOM_EXPR) {
                        if (!spec.dS_dt_vm.empty()) {
                            if (!spec.dA_dt_vm.empty()) {
                                double A_k = sa_.A[k];
                                double dS = vm_eval_scalar_3arg(spec.dS_dt_vm, Vpre, S, A_k);
                                double dA = vm_eval_scalar_3arg(spec.dA_dt_vm, Vpre, S, A_k);
                                S = std::max(0.0, std::min(1.0, S + dt * dS));
                                sa_.A[k] = A_k + dt * dA;
                            } else {
                                double dS = vm_eval_scalar_2arg(spec.dS_dt_vm, Vpre, S);
                                S = std::max(0.0, std::min(1.0, S + dt * dS));
                            }
                        }
                    }
                    sa_.S[k] = S;

                    if (spec.current_form == CF::CUSTOM_EXPR && !spec.current_vm.empty()) {
                        sa_.g[k] = vm_eval_scalar_3arg(spec.current_vm, Vpost, S, sa_.A[k])
                                   * sa_.weight[k];
                    } else {
                        double gS = spec.g * sa_.weight[k];
                        for (int p = 0; p < spec.power; ++p) gS *= S;
                        if (spec.current_form == CF::MG_BLOCK) {
                            double mg = 1.0 + spec.mg_conc
                                             * std::exp(-spec.mg_scale * Vpost) / spec.mg_denom;
                            gS /= mg;
                        }
                        sa_.g[k] = gS;
                    }
                }

                // Phase D: signal this step complete
                step_done[gid].fetch_add(1, std::memory_order_release);
            }
        }); // end thread lambda
    }

    for (auto& thr : threads) thr.join();

    pool_mgr_.sync_all_to_neurons(neurons_);
}

// =============================================================================
// Plasticity update methods
// =============================================================================

void Network::apply_stdp(double dt) {
    for (size_t k : syn_groups_.stdp) {
        int32_t si = sa_.plast_state_idx[k];
        const auto& spec = plasticity_specs_[sa_.plast_spec_idx_arr[k]].stdp;

        // Trace decay
        sa_.plast_x_pre[si]  *= std::exp(-dt / spec.tau_plus);
        sa_.plast_x_post[si] *= std::exp(-dt / spec.tau_minus);

        double& w = sa_.weight[k];
        double mod_scale = 1.0;
        if (spec.modulator_pop_start >= 0 && spec.modulator_substance_idx >= 0) {
            size_t post_n = post_decoded_[k];
            mod_scale = spec.modulator_scale
                        * pool_mgr_.get_substance(post_n,
                                                   (size_t)spec.modulator_substance_idx);
        }

        // Pre-spike: LTP
        if (spike_detected_[k]) {
            w += mod_scale * spec.A_plus * sa_.plast_x_post[si];
            sa_.plast_x_pre[si] += 1.0;
        }
        // Post-spike: LTD
        if (post_spiked_[post_decoded_[k]]) {
            w -= mod_scale * spec.A_minus * sa_.plast_x_pre[si];
            sa_.plast_x_post[si] += 1.0;
        }

        w = std::max(spec.w_min, std::min(spec.w_max, w));
    }
}

void Network::apply_stp(double dt) {
    for (size_t k : syn_groups_.stp) {
        int32_t si = sa_.plast_state_idx[k];
        const auto& spec = plasticity_specs_[sa_.plast_spec_idx_arr[k]].stp;

        double& u = sa_.stp_u[si];
        double& x = sa_.stp_x[si];

        // Recovery toward baseline every step
        u += dt * (spec.U - u) / spec.tau_u;
        x += dt * (1.0 - x)   / spec.tau_x;

        if (spike_detected_[k]) {
            // Scale S persistently so the depressed conductance decays from
            // the correct amplitude (u*x) rather than reverting next step.
            // Phase 2a already ran: S was jumped by delta_S then decayed once.
            // We back-correct: remove the unscaled jump and add the scaled one.
            // Correction: S -= delta_S * decay_S * (1 - u*x)
            double ux = u * x;
            const auto& syn_cache = synapse_spec_caches_[sa_.spec_idx[k]];
            sa_.S[k] -= syn_cache.delta_S * syn_cache.decay_S * (1.0 - ux);
            if (sa_.S[k] < 0.0) sa_.S[k] = 0.0;
            // Recompute g from corrected S
            const auto& syn_spec = synapse_specs_[sa_.spec_idx[k]];
            sa_.g[k] = syn_spec.g * sa_.weight[k] * sa_.S[k];
            // Update facilitation then depression (standard TM order)
            u += spec.U * (1.0 - u);
            x -= u * x;
        }
    }
}

std::vector<double> Network::get_synapse_weights() const {
    return sa_.weight;
}

} // namespace hodgkin_huxley
