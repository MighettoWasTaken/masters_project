#include "hodgkin_huxley/network.hpp"
#include "hodgkin_huxley/model/kinetics.hpp"
#ifdef HH_USE_CUDA
#  include <cuda_runtime_api.h>
#endif
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#ifdef HH_USE_OPENMP
#  include <omp.h>
#endif

namespace hodgkin_huxley {

Network::~Network() {
#ifdef HH_USE_CUDA
    if (V_cache_pinned_) cudaFreeHost(V_cache_pinned_);
    if (I_syn_pinned_)   cudaFreeHost(I_syn_pinned_);
#endif
}

void Network::set_device(const Device& device) {
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
// Synapse accessor — SynapseBase is always a valid view (no lazy sync needed)
// =============================================================================

const SynapseBase& Network::synapse(size_t idx) const {
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

    // Dedup spec by name
    size_t sidx = synapse_specs_.size();
    for (size_t i = 0; i < synapse_specs_.size(); ++i) {
        if (synapse_specs_[i].name == spec.name) { sidx = i; break; }
    }
    if (sidx == synapse_specs_.size()) synapse_specs_.push_back(spec);

    // Common SoA fields
    sa_.pre.push_back(pre);
    sa_.post.push_back(post);
    sa_.weight.push_back(weight);
    sa_.E_syn.push_back(spec.E_syn);
    sa_.g.push_back(0.0);
    sa_.V_pre_prev.push_back(neurons_[pre]->membrane_potential());
    sa_.delay.push_back(delay);
    sa_.spike_buf.emplace_back();
    sa_.buf_head.push_back(0);
    sa_.delay_init.push_back(false);

    // Unified state
    sa_.S.push_back(spec.S_init);
    sa_.A.push_back(spec.A_init);
    sa_.delta_S.push_back(spec.delta_S);
    sa_.delta_A.push_back(spec.delta_A);
    sa_.tau_S.push_back(spec.tau_S);
    sa_.tau_A.push_back(spec.tau_A);
    sa_.inv_tau_A.push_back(spec.tau_A > 0.0 ? 1.0 / spec.tau_A : 0.0);
    sa_.norm.push_back(spec.norm_factor);
    sa_.decay_S.push_back(0.0);
    sa_.decay_A.push_back(0.0);
    sa_.spec_idx.push_back(sidx);

    // Plasticity defaults (NONE — no state allocated)
    sa_.plast_type.push_back(PlasticityType::NONE);
    sa_.plast_state_idx.push_back(-1);
    sa_.plast_spec_idx_arr.push_back(0);
    sa_.is_active.push_back(false);

    // Push lightweight view (always valid — no heap per synapse)
    synapses_.emplace_back(sa_.size() - 1, this);

    sa_.cached_dt = -1.0;
    soa_sorted_ = false;
    groups_built_ = false;

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
    sa_.plast_spec_idx_arr[syn_i] = ps_idx;

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
        sa_.V_pre_prev[i] = neurons_[sa_.pre[i]]->membrane_potential();
        if (sa_.delay_init[i]) {
            std::fill(sa_.spike_buf[i].begin(), sa_.spike_buf[i].end(), false);
            sa_.buf_head[i] = 0;
        }
    }
    sa_.cached_dt = -1.0;

    soa_dirty_ = false;
    pools_dirty_ = true;
}

// =============================================================================
// Decay factor caching (called once when dt changes)
// =============================================================================

void Network::update_decay_factors(double dt) {
    if (dt == sa_.cached_dt) return;

    using UF = SynapseSpec::UpdateForm;
    const size_t N = sa_.size();
    for (size_t i = 0; i < N; ++i) {
        const auto& spec = synapse_specs_[sa_.spec_idx[i]];
        if (spec.update_form == UF::EXP_DECAY) {
            sa_.decay_S[i] = std::exp(-dt / sa_.tau_S[i]);
        } else if (spec.update_form == UF::DOUBLE_EXP) {
            sa_.decay_S[i] = std::exp(-dt / sa_.tau_S[i]);
            sa_.decay_A[i] = std::exp(-dt / sa_.tau_A[i]);
        }
        // ALPHA_FUNC uses Euler — no precomputed decay needed
        // Voltage-gated forms compute their own exact integration per step
    }
    sa_.cached_dt = dt;
}

// =============================================================================
// Synaptic current computation (SoA — contiguous, cache-friendly)
// =============================================================================

void Network::compute_synaptic_currents() {
    std::fill(I_syn_buffer_.begin(), I_syn_buffer_.end(), 0.0);

    const double* g = sa_.g.data();
    const double* E_syn = sa_.E_syn.data();
    const size_t* post = sa_.post.data();
    const double* V = V_cache_.data();
    double* I_syn = I_syn_buffer_.data();

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
    permute(sa_.V_pre_prev);
    permute(sa_.delay);
    permute(sa_.spike_buf);
    permute(sa_.buf_head);
    permute(sa_.delay_init);

    // Unified state
    permute(sa_.S);
    permute(sa_.A);
    permute(sa_.delta_S);
    permute(sa_.delta_A);
    permute(sa_.tau_S);
    permute(sa_.tau_A);
    permute(sa_.inv_tau_A);
    permute(sa_.norm);
    permute(sa_.decay_S);
    permute(sa_.decay_A);
    permute(sa_.spec_idx);

    // is_active: vector<bool> uses proxy refs — permute manually
    {
        std::vector<bool> tmp(S);
        for (size_t i = 0; i < S; ++i) tmp[i] = sa_.is_active[perm[i]];
        sa_.is_active = std::move(tmp);
    }

    // Reorder SynapseBase views and re-bind indices
    std::vector<SynapseBase> reordered(S);
    for (size_t i = 0; i < S; ++i) reordered[i] = SynapseBase(i, this);
    synapses_ = std::move(reordered);

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
// Unified synapse update — four tight sub-loops, no branch on type within each
// =============================================================================

void Network::update_synapses_grouped(double dt) {
    update_decay_factors(dt);

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

    // Phase 1: spike detection + delay processing for ALL synapses
    for (size_t i = 0; i < S; ++i) {
        double V_pre = V_cache_[sa_.pre[i]];
        bool spiked = (V_pre > spike_threshold) && (sa_.V_pre_prev[i] <= spike_threshold);
        sa_.V_pre_prev[i] = V_pre;

        if (sa_.delay[i] > 0.0) {
            if (!sa_.delay_init[i]) {
                size_t steps = static_cast<size_t>(std::round(sa_.delay[i] / dt));
                if (steps > 0) {
                    sa_.spike_buf[i].assign(steps, false);
                    sa_.buf_head[i] = 0;
                    sa_.delay_init[i] = true;
                }
            }
            if (sa_.delay_init[i]) {
                bool delayed = sa_.spike_buf[i][sa_.buf_head[i]];
                sa_.spike_buf[i][sa_.buf_head[i]] = spiked;
                sa_.buf_head[i] = (sa_.buf_head[i] + 1) % sa_.spike_buf[i].size();
                spiked = delayed;
            }
        }

        spike_detected_[i] = spiked ? 1 : 0;
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
            if (spike_detected_[k]) sa_.S[k] += sa_.delta_S[k];
            sa_.S[k] *= sa_.decay_S[k];
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
            if (spike_detected_[k]) sa_.A[k] += sa_.delta_A[k];
            double inv = sa_.inv_tau_A[k];
            double dS  = (sa_.A[k] - sa_.S[k]) * inv;
            double dA  = -sa_.A[k] * inv;
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
            if (spike_detected_[k]) {
                sa_.S[k] += sa_.delta_S[k];
                sa_.A[k] += sa_.delta_A[k];
            }
            sa_.S[k] *= sa_.decay_S[k];
            sa_.A[k] *= sa_.decay_A[k];
            double g_eff = sa_.norm[k] * (sa_.S[k] - sa_.A[k]);
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
        double Vpre  = V_cache_[sa_.pre[k]];
        double Vpost = V_cache_[sa_.post[k]];
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
    return sa_.pre;
}

std::vector<size_t> Network::get_synapse_post_indices() const {
    return sa_.post;
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

    if (pools_dirty_) {
        pool_mgr_.build_from_neurons(neurons_, fast_math_);
        pools_dirty_ = false;
    }

    ensure_buffers();

    const size_t S = sa_.size();
    std::vector<double> syn_spike_accum(n_neurons, 0.0);

    for (size_t t = 0; t < num_steps; ++t) {
        pool_mgr_.scatter_all_voltages(V_cache_.data());

        if (t % interval == 0) {
            size_t tr = t / interval;

            if (V_buf) {
                for (size_t i = 0; i < n_neurons; ++i)
                    V_buf[i * n_rec + tr] = V_cache_[i];
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
            I_syn_buffer_[i] = I_ext[i][t];

        const double* g = sa_.g.data();
        const double* E_syn_data = sa_.E_syn.data();
        const size_t* post = sa_.post.data();
        const double* V = V_cache_.data();
        double* I_buf = I_syn_buffer_.data();
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
                if (spike_detected_[j]) syn_spike_accum[sa_.post[j]] += 1.0;
        }
    }

    pool_mgr_.sync_all_to_neurons(neurons_);
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

    if (pools_dirty_) {
        pool_mgr_.build_from_neurons(neurons_, fast_math_);
        pools_dirty_ = false;
    }

    ensure_buffers();

#ifdef HH_USE_CUDA
    if (pool_mgr_.on_cuda() && (!use_pinned_memory_ || pinned_size_ != n_neurons))
        reallocate_pinned_buffers(n_neurons);
#endif
    double* V_ptr = use_pinned_memory_ ? V_cache_pinned_ : V_cache_.data();
    double* I_ptr = use_pinned_memory_ ? I_syn_pinned_   : I_syn_buffer_.data();

    const size_t S = sa_.size();
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

        // Seed from compact descriptors
        for (size_t i = 0; i < n_neurons; ++i)
            I_ptr[i] = I_stim_cache[i] = stim.I_const[i];

        for (const auto& p : stim.pulses) {
            if (t >= p.onset_step && t < p.end_step) {
                for (size_t i = p.neuron_start; i < p.neuron_end; ++i) {
                    I_ptr[i]          += p.amplitude;
                    I_stim_cache[i]   += p.amplitude;
                }
            }
        }

        for (const auto& d : stim.dbs) {
            if (d.isi_steps == 0) continue;
            size_t phase = t % d.isi_steps;
            if (phase < d.pw_steps) {
                for (size_t i = d.neuron_start; i < d.neuron_end; ++i) {
                    I_ptr[i]         += d.amplitude;
                    I_stim_cache[i]  += d.amplitude;
                }
            }
        }

        const double* g = sa_.g.data();
        const double* E_syn_data = sa_.E_syn.data();
        const size_t* post = sa_.post.data();
        const double* V = V_ptr;
        double* I_buf = I_ptr;
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

        update_synapses_grouped(dt);

        if (spike_event_buf) {
            for (size_t j = 0; j < S; ++j)
                if (spike_detected_[j]) syn_spike_accum[sa_.post[j]] += 1.0;
        }
    }

    pool_mgr_.sync_all_to_neurons(neurons_);
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
    if (pools_dirty_) {
        pool_mgr_.build_from_neurons(neurons_, fast_math_);
        pools_dirty_ = false;
    }
    ensure_buffers();
    update_decay_factors(dt);   // once on main thread — dt is constant for this run

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
    auto assign_pre = [&](const std::vector<size_t>& src, std::vector<size_t> GroupDef::* member) {
        for (size_t k : src) {
            auto it = neuron_gid.find(sa_.pre[k]);
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
        auto it = neuron_gid.find(sa_.pre[k]);
        if (it != neuron_gid.end())
            groups[it->second].pre_all.push_back(k);
    }

    // Build post_syn (synapse indices where post[k] in this group) — I_syn accumulation
    for (size_t k = 0; k < sa_.size(); ++k) {
        auto it = neuron_gid.find(sa_.post[k]);
        if (it != neuron_gid.end())
            groups[it->second].post_syn.push_back(k);
    }

    // Build src_group_ids for each group
    for (auto& g : groups) {
        std::unordered_set<int> srcs;
        for (size_t k : g.post_syn) {
            auto it = neuron_gid.find(sa_.pre[k]);
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
        auto pre_it  = neuron_gid.find(sa_.pre[k]);
        auto post_it = neuron_gid.find(sa_.post[k]);
        if (pre_it != neuron_gid.end() && post_it != neuron_gid.end()
                && pre_it->second == post_it->second)
            groups[pre_it->second].intra_syn.push_back(k);
    }

    // Fill hh_local_indices / iz_local_indices for each group
    for (auto& g : groups)
        pool_mgr_.fill_group_hh_iz_indices(g.neuron_indices,
                                            g.hh_local_indices,
                                            g.iz_local_indices);

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
    const size_t* const pre_arr  = sa_.pre.data();
    const size_t* const post_arr = sa_.post.data();
    const double* const E_arr    = sa_.E_syn.data();
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

                // Phase U1: spike detection for own pre-synapses
                for (size_t k : grp.pre_all) {
                    double Vpre = V_cache[pre_arr[k]];
                    bool spiked = (Vpre > spike_threshold_) && (sa_.V_pre_prev[k] <= spike_threshold_);
                    sa_.V_pre_prev[k] = Vpre;

                    if (sa_.delay[k] > 0.0) {
                        if (!sa_.delay_init[k]) {
                            size_t steps = static_cast<size_t>(std::round(sa_.delay[k] / dt));
                            if (steps > 0) {
                                sa_.spike_buf[k].assign(steps, false);
                                sa_.buf_head[k] = 0;
                                sa_.delay_init[k] = true;
                            }
                        }
                        if (sa_.delay_init[k]) {
                            bool delayed = sa_.spike_buf[k][sa_.buf_head[k]];
                            sa_.spike_buf[k][sa_.buf_head[k]] = spiked;
                            sa_.buf_head[k] = (sa_.buf_head[k] + 1) % sa_.spike_buf[k].size();
                            spiked = delayed;
                        }
                    }
                    spike_detected_[k] = spiked ? 1 : 0;
                }

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
                        if (spike_detected_[k]) sa_.S[k] += sa_.delta_S[k];
                        sa_.S[k] *= sa_.decay_S[k];
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
                        if (spike_detected_[k]) sa_.A[k] += sa_.delta_A[k];
                        double inv = sa_.inv_tau_A[k];
                        double dS  = (sa_.A[k] - sa_.S[k]) * inv;
                        double dA  = -sa_.A[k] * inv;
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
                        if (spike_detected_[k]) {
                            sa_.S[k] += sa_.delta_S[k];
                            sa_.A[k] += sa_.delta_A[k];
                        }
                        sa_.S[k] *= sa_.decay_S[k];
                        sa_.A[k] *= sa_.decay_A[k];
                        double g_eff = sa_.norm[k] * (sa_.S[k] - sa_.A[k]);
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
            size_t post_n = sa_.post[k];
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
        if (post_spiked_[sa_.post[k]]) {
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
            sa_.S[k] -= sa_.delta_S[k] * sa_.decay_S[k] * (1.0 - ux);
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
