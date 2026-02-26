#include "hodgkin_huxley/network.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>

namespace hodgkin_huxley {

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
// Synapse accessor with lazy sync
// =============================================================================

const SynapseBase& Network::synapse(size_t idx) const {
    if (idx >= synapses_.size()) throw std::out_of_range("Synapse index out of range");
    if (!synapses_[idx]) throw std::runtime_error("Synapse at index " + std::to_string(idx) + " is a kinetic synapse (no API object)");
    if (soa_dirty_) sync_soa_to_objects();
    return *synapses_[idx];
}

// =============================================================================
// Add synapses — populate both API objects and SoA arrays
// =============================================================================

void Network::add_synapse(size_t pre_idx, size_t post_idx, double weight,
                          double E_syn, double tau, double delay) {
    if (pre_idx >= neurons_.size() || post_idx >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }

    // API object
    synapses_.push_back(
        std::make_unique<ExponentialSynapse>(pre_idx, post_idx, weight, E_syn, tau, delay));

    // SoA common
    sa_.pre.push_back(pre_idx);
    sa_.post.push_back(post_idx);
    sa_.weight.push_back(weight);
    sa_.E_syn.push_back(E_syn);
    sa_.g.push_back(0.0);
    sa_.type.push_back(SynType::SYN_EXP);
    sa_.V_pre_prev.push_back(neurons_[pre_idx]->membrane_potential());
    sa_.delay.push_back(delay);
    sa_.spike_buf.emplace_back();
    sa_.buf_head.push_back(0);
    sa_.delay_init.push_back(false);

    // Type-specific defaults
    sa_.push_type_defaults();

    // Overwrite exp-specific
    sa_.exp_tau.back() = tau;

    // Invalidate caches
    sa_.cached_dt = -1.0;
    soa_sorted_ = false;
}

void Network::add_alpha_synapse(size_t pre_idx, size_t post_idx, double weight,
                                double E_syn, double tau, double delay) {
    if (pre_idx >= neurons_.size() || post_idx >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }

    synapses_.push_back(
        std::make_unique<AlphaSynapse>(pre_idx, post_idx, weight, E_syn, tau, delay));

    sa_.pre.push_back(pre_idx);
    sa_.post.push_back(post_idx);
    sa_.weight.push_back(weight);
    sa_.E_syn.push_back(E_syn);
    sa_.g.push_back(0.0);
    sa_.type.push_back(SynType::SYN_ALPHA);
    sa_.V_pre_prev.push_back(neurons_[pre_idx]->membrane_potential());
    sa_.delay.push_back(delay);
    sa_.spike_buf.emplace_back();
    sa_.buf_head.push_back(0);
    sa_.delay_init.push_back(false);

    sa_.push_type_defaults();

    // Overwrite alpha-specific
    sa_.alpha_inv_tau.back() = 1.0 / tau;

    sa_.cached_dt = -1.0;
    soa_sorted_ = false;
}

void Network::add_double_exp_synapse(size_t pre_idx, size_t post_idx, double weight,
                                     double E_syn,
                                     double tau_rise, double tau_decay, double delay) {
    if (pre_idx >= neurons_.size() || post_idx >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }

    // API object (validates tau_rise < tau_decay)
    synapses_.push_back(
        std::make_unique<DoubleExponentialSynapse>(
            pre_idx, post_idx, weight, E_syn, tau_rise, tau_decay, delay));

    sa_.pre.push_back(pre_idx);
    sa_.post.push_back(post_idx);
    sa_.weight.push_back(weight);
    sa_.E_syn.push_back(E_syn);
    sa_.g.push_back(0.0);
    sa_.type.push_back(SynType::SYN_DEXP);
    sa_.V_pre_prev.push_back(neurons_[pre_idx]->membrane_potential());
    sa_.delay.push_back(delay);
    sa_.spike_buf.emplace_back();
    sa_.buf_head.push_back(0);
    sa_.delay_init.push_back(false);

    sa_.push_type_defaults();

    // Overwrite dexp-specific
    sa_.dexp_tau_rise.back() = tau_rise;
    sa_.dexp_tau_decay.back() = tau_decay;

    // Normalization factor so peak conductance = weight
    double t_peak = (tau_decay * tau_rise) / (tau_decay - tau_rise)
                    * std::log(tau_decay / tau_rise);
    double peak_val = std::exp(-t_peak / tau_decay) - std::exp(-t_peak / tau_rise);
    sa_.dexp_norm.back() = 1.0 / peak_val;

    sa_.cached_dt = -1.0;
    soa_sorted_ = false;
}

// =============================================================================
// Receptor-type convenience methods
// =============================================================================

void Network::add_ampa_synapse(size_t pre_idx, size_t post_idx, double weight,
                               double delay) {
    add_double_exp_synapse(pre_idx, post_idx, weight,
                           0.0, 0.5, 2.5, delay);
}

void Network::add_nmda_synapse(size_t pre_idx, size_t post_idx, double weight,
                               double delay) {
    add_double_exp_synapse(pre_idx, post_idx, weight,
                           0.0, 2.0, 67.0, delay);
}

void Network::add_gaba_a_synapse(size_t pre_idx, size_t post_idx, double weight,
                                 double delay) {
    add_double_exp_synapse(pre_idx, post_idx, weight,
                           -80.0, 0.4, 7.7, delay);
}

void Network::add_receptor_synapse(size_t pre_idx, size_t post_idx, double weight,
                                   ReceptorType receptor, double delay) {
    switch (receptor) {
        case ReceptorType::AMPA:
            add_ampa_synapse(pre_idx, post_idx, weight, delay);
            break;
        case ReceptorType::NMDA:
            add_nmda_synapse(pre_idx, post_idx, weight, delay);
            break;
        case ReceptorType::GABA_A:
            add_gaba_a_synapse(pre_idx, post_idx, weight, delay);
            break;
    }
}

// =============================================================================
// Kinetic synapse
// =============================================================================

size_t Network::add_kinetic_synapse(size_t pre, size_t post, double weight,
                                    const KineticSynapseSpec& spec, double delay) {
    if (pre >= neurons_.size() || post >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range");
    }

    // Dedup spec by name
    size_t spec_idx = kinetic_specs_.size();
    for (size_t i = 0; i < kinetic_specs_.size(); ++i) {
        if (kinetic_specs_[i].name == spec.name) { spec_idx = i; break; }
    }
    if (spec_idx == kinetic_specs_.size()) kinetic_specs_.push_back(spec);

    // No API synapse object for kinetic type — use nullptr slot to keep indices aligned
    synapses_.push_back(nullptr);

    // Common SoA fields
    sa_.pre.push_back(pre);
    sa_.post.push_back(post);
    sa_.weight.push_back(weight);
    sa_.E_syn.push_back(spec.E_syn);
    sa_.g.push_back(0.0);
    sa_.type.push_back(SynType::SYN_KINETIC);
    sa_.V_pre_prev.push_back(neurons_[pre]->membrane_potential());
    sa_.delay.push_back(0.0);   // delay not supported for kinetic
    sa_.spike_buf.emplace_back();
    sa_.buf_head.push_back(0);
    sa_.delay_init.push_back(false);

    // Type-specific defaults (fills exp/alpha/dexp fields with 0)
    sa_.push_type_defaults();

    // Overwrite kinetic-specific (push_type_defaults already pushed 0/0)
    sa_.kin_S.back() = spec.S_init;
    sa_.kin_spec_idx.back() = spec_idx;

    sa_.cached_dt = -1.0;
    soa_sorted_ = false;

    return sa_.size() - 1;
}

// =============================================================================
// Kinetic state accessors
// =============================================================================

double Network::get_kin_S(size_t synapse_idx) const {
    if (synapse_idx >= sa_.size()) throw std::out_of_range("Synapse index out of range");
    return sa_.kin_S[synapse_idx];
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
    }
}

void Network::cache_voltages() {
    for (size_t i = 0; i < neurons_.size(); ++i) {
        V_cache_[i] = neurons_[i]->membrane_potential();
    }
}

void Network::sync_soa_to_objects() const {
    for (size_t i = 0; i < synapses_.size(); ++i) {
        if (synapses_[i]) synapses_[i]->set_conductance(sa_.g[i]);
    }
    soa_dirty_ = false;
}

// =============================================================================
// Reset
// =============================================================================

void Network::reset() {
    for (auto& neuron : neurons_) {
        neuron->reset();
    }

    const size_t S = sa_.size();

    // Reset SoA mutable state
    std::fill(sa_.g.begin(), sa_.g.end(), 0.0);
    std::fill(sa_.alpha_x.begin(), sa_.alpha_x.end(), 0.0);
    std::fill(sa_.dexp_g_rise.begin(), sa_.dexp_g_rise.end(), 0.0);
    std::fill(sa_.dexp_g_decay.begin(), sa_.dexp_g_decay.end(), 0.0);

    for (size_t i = 0; i < S; ++i) {
        sa_.V_pre_prev[i] = neurons_[sa_.pre[i]]->membrane_potential();
        if (sa_.delay_init[i]) {
            std::fill(sa_.spike_buf[i].begin(), sa_.spike_buf[i].end(), false);
            sa_.buf_head[i] = 0;
        }
    }
    sa_.cached_dt = -1.0;

    // Reset kinetic gating variables
    for (size_t i = 0; i < S; ++i) {
        if (sa_.type[i] == SynType::SYN_KINETIC) {
            sa_.kin_S[i] = kinetic_specs_[sa_.kin_spec_idx[i]].S_init;
            sa_.g[i] = 0.0;
        }
    }

    // Reset API objects (skip nullptr kinetic slots)
    for (auto& syn : synapses_) {
        if (syn) {
            syn->reset();
            syn->reset_delay_buffer();
        }
    }

    soa_dirty_ = false;
}

// =============================================================================
// Decay factor caching (called once when dt changes)
// =============================================================================

void Network::update_decay_factors(double dt) {
    if (dt == sa_.cached_dt) return;

    const size_t S = sa_.size();
    for (size_t i = 0; i < S; ++i) {
        switch (sa_.type[i]) {
            case SynType::SYN_EXP:
                sa_.exp_decay[i] = std::exp(-dt / sa_.exp_tau[i]);
                break;
            case SynType::SYN_DEXP:
                sa_.dexp_rise_decay[i] = std::exp(-dt / sa_.dexp_tau_rise[i]);
                sa_.dexp_fall_decay[i] = std::exp(-dt / sa_.dexp_tau_decay[i]);
                break;
            default:
                break;
        }
    }
    sa_.cached_dt = dt;
}

// =============================================================================
// Synaptic current computation (SoA — contiguous, cache-friendly)
// =============================================================================

void Network::compute_synaptic_currents() {
    std::fill(I_syn_buffer_.begin(), I_syn_buffer_.end(), 0.0);

    const size_t S = sa_.size();
    const double* g = sa_.g.data();
    const double* E_syn = sa_.E_syn.data();
    const size_t* post = sa_.post.data();
    const double* V = V_cache_.data();
    double* I_syn = I_syn_buffer_.data();

    for (size_t i = 0; i < S; ++i) {
        I_syn[post[i]] += g[i] * (E_syn[i] - V[post[i]]);
    }
}

// =============================================================================
// Synapse state update (SoA — spike detection + type-specific kinetics)
// =============================================================================

void Network::update_synapses(double dt) {
    update_decay_factors(dt);

    const size_t S = sa_.size();
    const double spike_threshold = 0.0;
    static constexpr double E_CONST = 2.718281828459045;

    for (size_t i = 0; i < S; ++i) {
        // Spike detection using cached voltages
        double V_pre = V_cache_[sa_.pre[i]];
        bool spiked = (V_pre > spike_threshold) && (sa_.V_pre_prev[i] <= spike_threshold);
        sa_.V_pre_prev[i] = V_pre;

        // Delay processing
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

        // Type-specific conductance update
        switch (sa_.type[i]) {
            case SynType::SYN_EXP:
                if (spiked) sa_.g[i] += sa_.weight[i];
                sa_.g[i] *= sa_.exp_decay[i];
                break;

            case SynType::SYN_ALPHA: {
                if (spiked) sa_.alpha_x[i] += sa_.weight[i] * E_CONST;
                double inv_tau = sa_.alpha_inv_tau[i];
                double dx = -sa_.alpha_x[i] * inv_tau;
                double dg = (sa_.alpha_x[i] - sa_.g[i]) * inv_tau;
                sa_.alpha_x[i] += dt * dx;
                sa_.g[i] += dt * dg;
                if (sa_.g[i] < 0.0) sa_.g[i] = 0.0;
                break;
            }

            case SynType::SYN_DEXP:
                if (spiked) {
                    sa_.dexp_g_rise[i] += 1.0;
                    sa_.dexp_g_decay[i] += 1.0;
                }
                sa_.dexp_g_rise[i] *= sa_.dexp_rise_decay[i];
                sa_.dexp_g_decay[i] *= sa_.dexp_fall_decay[i];
                sa_.g[i] = sa_.weight[i] * sa_.dexp_norm[i]
                         * (sa_.dexp_g_decay[i] - sa_.dexp_g_rise[i]);
                if (sa_.g[i] < 0.0) sa_.g[i] = 0.0;
                break;

            case SynType::SYN_KINETIC:
                // Kinetic synapses are updated in update_synapses_grouped() phase 3
                break;
        }
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

    // Cache voltages for synaptic current computation
    cache_voltages();
    compute_synaptic_currents();

    // Step each neuron
    for (size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i]->step(dt, I_ext[i] + I_syn_buffer_[i]);
    }

    // Re-cache voltages for spike detection
    cache_voltages();
    update_synapses(dt);

    soa_dirty_ = true;
}

// =============================================================================
// Sort SoA arrays by presynaptic index for cache-friendly access
// =============================================================================

void Network::sort_synapses_by_pre() {
    if (soa_sorted_) return;

    const size_t S = sa_.size();
    if (S <= 1) { soa_sorted_ = true; return; }

    // Build sort permutation: sort by (pre, post) for locality
    std::vector<size_t> perm(S);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        if (sa_.pre[a] != sa_.pre[b]) return sa_.pre[a] < sa_.pre[b];
        return sa_.post[a] < sa_.post[b];
    });

    // Apply permutation to all parallel arrays using a temporary
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
    permute(sa_.type);
    permute(sa_.V_pre_prev);
    permute(sa_.delay);
    permute(sa_.spike_buf);
    permute(sa_.buf_head);
    permute(sa_.delay_init);

    // Exponential
    permute(sa_.exp_tau);
    permute(sa_.exp_decay);

    // Alpha
    permute(sa_.alpha_x);
    permute(sa_.alpha_inv_tau);

    // Double-exponential
    permute(sa_.dexp_g_rise);
    permute(sa_.dexp_g_decay);
    permute(sa_.dexp_tau_rise);
    permute(sa_.dexp_tau_decay);
    permute(sa_.dexp_rise_decay);
    permute(sa_.dexp_fall_decay);
    permute(sa_.dexp_norm);

    // Kinetic
    permute(sa_.kin_S);
    permute(sa_.kin_spec_idx);

    // Reorder API synapse objects to match
    std::vector<std::unique_ptr<SynapseBase>> reordered(S);
    for (size_t i = 0; i < S; ++i) reordered[i] = std::move(synapses_[perm[i]]);
    synapses_ = std::move(reordered);

    // Invalidate cached decay factors (order changed)
    sa_.cached_dt = -1.0;
    soa_sorted_ = true;
}

// =============================================================================
// Build type-separated synapse index lists
// =============================================================================

void Network::build_synapse_groups() {
    syn_groups_.exp.clear();
    syn_groups_.alpha.clear();
    syn_groups_.dexp.clear();
    syn_groups_.kinetic.clear();

    const size_t S = sa_.size();
    for (size_t i = 0; i < S; ++i) {
        switch (sa_.type[i]) {
            case SynType::SYN_EXP:     syn_groups_.exp.push_back(i);     break;
            case SynType::SYN_ALPHA:   syn_groups_.alpha.push_back(i);   break;
            case SynType::SYN_DEXP:    syn_groups_.dexp.push_back(i);    break;
            case SynType::SYN_KINETIC: syn_groups_.kinetic.push_back(i); break;
        }
    }

    spike_detected_.resize(S);
}

// =============================================================================
// Type-separated synapse update (no switch in inner loops)
// =============================================================================

void Network::update_synapses_grouped(double dt) {
    update_decay_factors(dt);

    const size_t S = sa_.size();
    const double spike_threshold = 0.0;
    static constexpr double E_CONST = 2.718281828459045;

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

    // Phase 2: type-separated kinetics — tight loops, no branch on type

    for (size_t j = 0, je = syn_groups_.exp.size(); j < je; ++j) {
        size_t i = syn_groups_.exp[j];
        if (spike_detected_[i]) sa_.g[i] += sa_.weight[i];
        sa_.g[i] *= sa_.exp_decay[i];
    }

    for (size_t j = 0, je = syn_groups_.alpha.size(); j < je; ++j) {
        size_t i = syn_groups_.alpha[j];
        if (spike_detected_[i]) sa_.alpha_x[i] += sa_.weight[i] * E_CONST;
        double inv_tau = sa_.alpha_inv_tau[i];
        double dx = -sa_.alpha_x[i] * inv_tau;
        double dg = (sa_.alpha_x[i] - sa_.g[i]) * inv_tau;
        sa_.alpha_x[i] += dt * dx;
        sa_.g[i] += dt * dg;
        if (sa_.g[i] < 0.0) sa_.g[i] = 0.0;
    }

    for (size_t j = 0, je = syn_groups_.dexp.size(); j < je; ++j) {
        size_t i = syn_groups_.dexp[j];
        if (spike_detected_[i]) {
            sa_.dexp_g_rise[i] += 1.0;
            sa_.dexp_g_decay[i] += 1.0;
        }
        sa_.dexp_g_rise[i] *= sa_.dexp_rise_decay[i];
        sa_.dexp_g_decay[i] *= sa_.dexp_fall_decay[i];
        sa_.g[i] = sa_.weight[i] * sa_.dexp_norm[i]
                 * (sa_.dexp_g_decay[i] - sa_.dexp_g_rise[i]);
        if (sa_.g[i] < 0.0) sa_.g[i] = 0.0;
    }

    // Phase 3: kinetic synapses — continuous V_pre-dependent gating
    for (size_t k : syn_groups_.kinetic) {
        double Vpre  = V_cache_[sa_.pre[k]];
        double Vpost = V_cache_[sa_.post[k]];
        const auto& spec = kinetic_specs_[sa_.kin_spec_idx[k]];
        double S = sa_.kin_S[k];

        // Exact exponential integration
        using UF = KineticSynapseSpec::UpdateForm;
        if (spec.update_form == UF::TANH_GATE) {
            double rate_open = spec.tanh_amp
                               * (1.0 + std::tanh((Vpre - spec.tanh_vh) / spec.tanh_k));
            double rate = rate_open + 1.0 / spec.tau_decay;
            double S_inf = rate_open / rate;
            S = S_inf + (S - S_inf) * std::exp(-dt * rate);
        } else if (spec.update_form == UF::ALPHA_BETA) {
            double a = compute_rate_scalar(Vpre, spec.alpha);
            double b = compute_rate_scalar(Vpre, spec.beta);
            double rate = a + b;
            double S_inf = (rate > 1e-10) ? a / rate : S;
            S = S_inf + (S - S_inf) * std::exp(-dt * rate);
        } else { // BOLTZMANN_GATE
            double S_inf = boltzmann_scalar(Vpre, spec.s_inf);
            double tau_s = compute_tau_scalar(Vpre, spec.tau);
            if (tau_s < 1e-10) tau_s = 1e-10;
            S = S_inf + (S - S_inf) * std::exp(-dt / tau_s);
        }
        sa_.kin_S[k] = S;

        // Compute effective conductance (written to sa_.g[k] for current summation)
        double gS = spec.g * sa_.weight[k];
        for (int p = 0; p < spec.power; ++p) gS *= S;

        using CF = KineticSynapseSpec::CurrentForm;
        if (spec.current_form == CF::MG_BLOCK) {
            double mg = 1.0 + spec.mg_conc
                             * std::exp(-spec.mg_scale * Vpost) / spec.mg_denom;
            gS /= mg;
        }
        sa_.g[k] = gS;
    }
}

// =============================================================================
// Simulate — batched pools + type-separated synapse loops
// =============================================================================

std::vector<std::vector<double>> Network::simulate(
    double duration,
    double dt,
    const std::vector<std::vector<double>>& I_ext
) {
    size_t num_steps = static_cast<size_t>(duration / dt);
    size_t n_neurons = neurons_.size();

    // Validate input
    if (I_ext.size() != n_neurons) {
        throw std::invalid_argument("I_ext outer size must match number of neurons");
    }
    for (const auto& curr : I_ext) {
        if (curr.size() < num_steps) {
            throw std::invalid_argument("I_ext vectors too short for simulation duration");
        }
    }

    // Pre-allocate output
    std::vector<std::vector<double>> traces(n_neurons);
    for (auto& trace : traces) {
        trace.resize(num_steps);
    }

    // Sort synapses by pre-index for cache-friendly access
    sort_synapses_by_pre();
    build_synapse_groups();

    // =========================================================================
    // Build batched neuron pools — classify via dynamic_cast once
    // =========================================================================
    size_t n_hh = 0, n_iz = 0;
    std::map<std::string, size_t> composable_counts;
    for (size_t i = 0; i < n_neurons; ++i) {
        if (dynamic_cast<HHNeuron*>(neurons_[i].get())) ++n_hh;
        else if (dynamic_cast<IzhikevichNeuron*>(neurons_[i].get())) ++n_iz;
        else if (auto* cn = dynamic_cast<ComposableNeuron*>(neurons_[i].get())) {
            composable_counts[cn->model_spec().name]++;
        }
    }

    HHPool hh_pool(n_hh, fast_math_);
    IzPool iz_pool(n_iz);

    // Create one ComposablePool per distinct model name
    std::map<std::string, ComposablePool> comp_pools;
    for (const auto& kv : composable_counts) {
        // Find first neuron with this model to get the spec
        for (size_t i = 0; i < n_neurons; ++i) {
            auto* cn = dynamic_cast<ComposableNeuron*>(neurons_[i].get());
            if (cn && cn->model_spec().name == kv.first) {
                comp_pools.emplace(kv.first,
                    ComposablePool(cn->model_spec(), kv.second, fast_math_));
                break;
            }
        }
    }

    for (size_t i = 0; i < n_neurons; ++i) {
        if (auto* hh = dynamic_cast<HHNeuron*>(neurons_[i].get())) {
            hh_pool.add(i, hh->parameters(), hh->state());
        } else if (auto* iz = dynamic_cast<IzhikevichNeuron*>(neurons_[i].get())) {
            iz_pool.add(i, iz->parameters(), iz->state());
        } else if (auto* cn = dynamic_cast<ComposableNeuron*>(neurons_[i].get())) {
            auto it = comp_pools.find(cn->model_spec().name);
            if (it != comp_pools.end()) {
                it->second.add(i, cn->membrane_potential(),
                               cn->gate_states(), cn->calcium());
            }
        }
    }

    // Pre-allocate working buffers
    ensure_buffers();

    // =========================================================================
    // Hot loop — pools do batched SIMD math, no virtual dispatch
    // =========================================================================
    for (size_t t = 0; t < num_steps; ++t) {
        // Read voltages from pools into V_cache_
        hh_pool.scatter_voltages(V_cache_.data());
        iz_pool.scatter_voltages(V_cache_.data());
        for (auto& kv : comp_pools) kv.second.scatter_voltages(V_cache_.data());

        // Record traces + gather external currents into I_syn_buffer_
        // (reuse I_syn_buffer_ as combined I buffer — compute_synaptic_currents
        //  will ADD into it after we seed it with I_ext)
        for (size_t i = 0; i < n_neurons; ++i) {
            traces[i][t] = V_cache_[i];
            I_syn_buffer_[i] = I_ext[i][t];
        }

        // Compute synaptic currents — ADDS g*(E-V) into I_syn_buffer_
        const size_t S = sa_.size();
        const double* g = sa_.g.data();
        const double* E_syn = sa_.E_syn.data();
        const size_t* post = sa_.post.data();
        const double* V = V_cache_.data();
        double* I_buf = I_syn_buffer_.data();
        for (size_t i = 0; i < S; ++i) {
            I_buf[post[i]] += g[i] * (E_syn[i] - V[post[i]]);
        }

        // Feed combined currents to pools
        hh_pool.gather_currents(I_syn_buffer_.data());
        iz_pool.gather_currents(I_syn_buffer_.data());
        for (auto& kv : comp_pools) kv.second.gather_currents(I_syn_buffer_.data());

        // Step pools (batched SIMD — the main performance win)
        hh_pool.step_rk4(dt);
        iz_pool.step_euler(dt);
        for (auto& kv : comp_pools) kv.second.step(dt);

        // Re-read voltages for spike detection
        hh_pool.scatter_voltages(V_cache_.data());
        iz_pool.scatter_voltages(V_cache_.data());
        for (auto& kv : comp_pools) kv.second.scatter_voltages(V_cache_.data());

        // Type-separated synapse update (no switch in inner loops)
        update_synapses_grouped(dt);
    }

    // =========================================================================
    // Sync pool state back to API objects once at the end
    // =========================================================================
    hh_pool.sync_to_neurons(neurons_);
    iz_pool.sync_to_neurons(neurons_);
    for (const auto& kv : comp_pools) kv.second.sync_to_neurons(neurons_);
    sync_soa_to_objects();

    return traces;
}

} // namespace hodgkin_huxley
