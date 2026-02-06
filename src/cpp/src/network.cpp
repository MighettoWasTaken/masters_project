#include "hodgkin_huxley/network.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

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
        default:
            return add_hh_neuron();
    }
}

size_t Network::add_neuron(const IzhikevichNeuron::Parameters& params) {
    neurons_.push_back(std::make_unique<IzhikevichNeuron>(params));
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
    sa_.type.push_back(SYN_EXP);
    sa_.V_pre_prev.push_back(neurons_[pre_idx]->membrane_potential());
    sa_.delay.push_back(delay);
    sa_.spike_buf.emplace_back();
    sa_.buf_head.push_back(0);
    sa_.delay_init.push_back(false);

    // Type-specific defaults
    sa_.push_type_defaults();

    // Overwrite exp-specific
    sa_.exp_tau.back() = tau;

    // Invalidate cached decay factors
    sa_.cached_dt = -1.0;
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
    sa_.type.push_back(SYN_ALPHA);
    sa_.V_pre_prev.push_back(neurons_[pre_idx]->membrane_potential());
    sa_.delay.push_back(delay);
    sa_.spike_buf.emplace_back();
    sa_.buf_head.push_back(0);
    sa_.delay_init.push_back(false);

    sa_.push_type_defaults();

    // Overwrite alpha-specific
    sa_.alpha_inv_tau.back() = 1.0 / tau;

    sa_.cached_dt = -1.0;
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
    sa_.type.push_back(SYN_DEXP);
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
        synapses_[i]->set_conductance(sa_.g[i]);
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

    // Reset API objects
    for (auto& syn : synapses_) {
        syn->reset();
        syn->reset_delay_buffer();
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
            case SYN_EXP:
                sa_.exp_decay[i] = std::exp(-dt / sa_.exp_tau[i]);
                break;
            case SYN_DEXP:
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
            case SYN_EXP:
                if (spiked) sa_.g[i] += sa_.weight[i];
                sa_.g[i] *= sa_.exp_decay[i];
                break;

            case SYN_ALPHA: {
                if (spiked) sa_.alpha_x[i] += sa_.weight[i] * E_CONST;
                double inv_tau = sa_.alpha_inv_tau[i];
                double dx = -sa_.alpha_x[i] * inv_tau;
                double dg = (sa_.alpha_x[i] - sa_.g[i]) * inv_tau;
                sa_.alpha_x[i] += dt * dx;
                sa_.g[i] += dt * dg;
                if (sa_.g[i] < 0.0) sa_.g[i] = 0.0;
                break;
            }

            case SYN_DEXP:
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
// Simulate (inlined step logic — avoids per-step validation and sync)
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

    // Pre-allocate working buffers
    ensure_buffers();
    std::vector<double> I_step(n_neurons);

    // Run simulation
    for (size_t t = 0; t < num_steps; ++t) {
        // Cache voltages and record traces in one pass
        cache_voltages();
        for (size_t i = 0; i < n_neurons; ++i) {
            traces[i][t] = V_cache_[i];
            I_step[i] = I_ext[i][t];
        }

        // Compute synaptic currents from cached voltages
        compute_synaptic_currents();

        // Step each neuron
        for (size_t i = 0; i < n_neurons; ++i) {
            neurons_[i]->step(dt, I_step[i] + I_syn_buffer_[i]);
        }

        // Re-cache voltages for spike detection
        cache_voltages();
        update_synapses(dt);
    }

    // Sync conductances back to API objects once at the end
    sync_soa_to_objects();

    return traces;
}

} // namespace hodgkin_huxley
