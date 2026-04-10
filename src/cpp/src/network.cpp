#include "hodgkin_huxley/network.hpp"
#include "hodgkin_huxley/model/kinetics.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

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

    // Push lightweight view (always valid — no heap per synapse)
    synapses_.emplace_back(sa_.size() - 1, this);

    sa_.cached_dt = -1.0;
    soa_sorted_ = false;
    groups_built_ = false;

    return sa_.size() - 1;
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
    }
}

void Network::cache_voltages() {
    for (size_t i = 0; i < neurons_.size(); ++i) {
        V_cache_[i] = neurons_[i]->membrane_potential();
    }
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

    // Reorder SynapseBase views and re-bind indices
    std::vector<SynapseBase> reordered(S);
    for (size_t i = 0; i < S; ++i) reordered[i] = SynapseBase(i, this);
    synapses_ = std::move(reordered);

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

    using UF = SynapseSpec::UpdateForm;
    const size_t S = sa_.size();
    for (size_t i = 0; i < S; ++i) {
        auto form = synapse_specs_[sa_.spec_idx[i]].update_form;
        if      (form == UF::EXP_DECAY)  syn_groups_.exp_decay.push_back(i);
        else if (form == UF::ALPHA_FUNC) syn_groups_.alpha_func.push_back(i);
        else if (form == UF::DOUBLE_EXP) syn_groups_.double_exp.push_back(i);
        else                              syn_groups_.voltage_gated.push_back(i);
    }

    spike_detected_.resize(S);
    groups_built_ = true;
}

// =============================================================================
// Unified synapse update — four tight sub-loops, no branch on type within each
// =============================================================================

void Network::update_synapses_grouped(double dt) {
    update_decay_factors(dt);

    const size_t S = sa_.size();
    const double spike_threshold = spike_threshold_;

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
    for (size_t k : syn_groups_.exp_decay) {
        if (spike_detected_[k]) sa_.S[k] += sa_.delta_S[k];
        sa_.S[k] *= sa_.decay_S[k];
        if (sa_.S[k] < 0.0) sa_.S[k] = 0.0;
        const auto& spec = synapse_specs_[sa_.spec_idx[k]];
        sa_.g[k] = spec.g * sa_.weight[k] * sa_.S[k];
    }

    // Phase 2b: ALPHA_FUNC — 2-variable Euler, spike on A
    for (size_t k : syn_groups_.alpha_func) {
        if (spike_detected_[k]) sa_.A[k] += sa_.delta_A[k];
        double inv = sa_.inv_tau_A[k];
        double dS  = (sa_.A[k] - sa_.S[k]) * inv;
        double dA  = -sa_.A[k] * inv;
        sa_.S[k] += dt * dS;
        sa_.A[k] += dt * dA;
        if (sa_.S[k] < 0.0) sa_.S[k] = 0.0;
        const auto& spec = synapse_specs_[sa_.spec_idx[k]];
        sa_.g[k] = spec.g * sa_.weight[k] * sa_.S[k];
    }

    // Phase 2c: DOUBLE_EXP — two independent exact decays, spike on both
    for (size_t k : syn_groups_.double_exp) {
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
        for (size_t i = 0; i < S; ++i)
            I_buf[post[i]] += g[i] * (E_syn_data[i] - V[post[i]]);

        if (I_syn_buf && t % interval == 0) {
            size_t tr = t / interval;
            for (size_t i = 0; i < n_neurons; ++i)
                I_syn_buf[i * n_rec + tr] = I_syn_buffer_[i] - I_ext[i][t];
        }

        pool_mgr_.gather_all_currents(I_syn_buffer_.data());
        pool_mgr_.step_all(dt);
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

    const size_t S = sa_.size();
    std::vector<double> syn_spike_accum(n_neurons, 0.0);
    std::vector<double> I_stim_cache(n_neurons, 0.0);

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

        // Seed from compact descriptors
        for (size_t i = 0; i < n_neurons; ++i)
            I_syn_buffer_[i] = I_stim_cache[i] = stim.I_const[i];

        for (const auto& p : stim.pulses) {
            if (t >= p.onset_step && t < p.end_step) {
                for (size_t i = p.neuron_start; i < p.neuron_end; ++i) {
                    I_syn_buffer_[i] += p.amplitude;
                    I_stim_cache[i]  += p.amplitude;
                }
            }
        }

        for (const auto& d : stim.dbs) {
            if (d.isi_steps == 0) continue;
            size_t phase = t % d.isi_steps;
            if (phase < d.pw_steps) {
                for (size_t i = d.neuron_start; i < d.neuron_end; ++i) {
                    I_syn_buffer_[i] += d.amplitude;
                    I_stim_cache[i]  += d.amplitude;
                }
            }
        }

        const double* g = sa_.g.data();
        const double* E_syn_data = sa_.E_syn.data();
        const size_t* post = sa_.post.data();
        const double* V = V_cache_.data();
        double* I_buf = I_syn_buffer_.data();
        for (size_t i = 0; i < S; ++i)
            I_buf[post[i]] += g[i] * (E_syn_data[i] - V[post[i]]);

        if (I_syn_buf && t % interval == 0) {
            size_t tr = t / interval;
            for (size_t i = 0; i < n_neurons; ++i)
                I_syn_buf[i * n_rec + tr] = I_syn_buffer_[i] - I_stim_cache[i];
        }

        pool_mgr_.gather_all_currents(I_syn_buffer_.data());
        pool_mgr_.step_all(dt);
        pool_mgr_.scatter_all_voltages(V_cache_.data());

        update_synapses_grouped(dt);

        if (spike_event_buf) {
            for (size_t j = 0; j < S; ++j)
                if (spike_detected_[j]) syn_spike_accum[sa_.post[j]] += 1.0;
        }
    }

    pool_mgr_.sync_all_to_neurons(neurons_);
}

} // namespace hodgkin_huxley
