#include "hodgkin_huxley/composable_pool.hpp"
#include "hodgkin_huxley/composable_neuron.hpp"
#include "hodgkin_huxley/model/kinetics.hpp"
#include <cstring>

namespace hodgkin_huxley {

ComposablePool::ComposablePool(const NeuronModelSpec& model, size_t capacity, bool fast_math)
    : model_(model), N_(0), fast_math_(fast_math)
{
    net_idx_.reserve(capacity);
    gate_states_.resize(model_.gates.size());
    // Arrays will be conservativeResize'd in add()
}

void ComposablePool::add(size_t network_idx, double V_init,
                          const std::vector<double>& gate_inits, double Ca_init) {
    size_t i = N_++;
    net_idx_.push_back(network_idx);

    if (i == 0) contiguous_ = true;
    else if (contiguous_ && network_idx != net_idx_[i - 1] + 1) contiguous_ = false;

    // Grow arrays
    V_.conservativeResize(N_);
    Ca_.conservativeResize(N_);
    E_Ca_.conservativeResize(N_);
    I_ext_.conservativeResize(N_);

    V_(i) = V_init;
    Ca_(i) = Ca_init;

    if (model_.calcium.enabled && model_.calcium.use_nernst && Ca_init > 0.0) {
        E_Ca_(i) = (model_.calcium.R * model_.calcium.T)
                   / (model_.calcium.z * model_.calcium.F)
                   * std::log(model_.calcium.Ca_o / Ca_init);
    } else {
        E_Ca_(i) = 120.0;
    }

    for (size_t g = 0; g < gate_states_.size(); ++g) {
        gate_states_[g].conservativeResize(N_);
        if (g < gate_inits.size()) {
            gate_states_[g](i) = gate_inits[g];
        } else {
            gate_states_[g](i) = 0.0;
        }
    }

    finalized_ = false;
}

void ComposablePool::finalize() {
    if (finalized_) return;
    I_total_.resize(N_);
    tmp_.resize(N_);
    tmp2_.resize(N_);
    tmp_exp_r_.resize(N_);
    finalized_ = true;
}

void ComposablePool::scatter_voltages(double* V_buf) const {
    if (contiguous_ && N_ > 0) {
        std::memcpy(V_buf + net_idx_[0], V_.data(), N_ * sizeof(double));
    } else {
        for (size_t i = 0; i < N_; ++i)
            V_buf[net_idx_[i]] = V_(i);
    }
}

void ComposablePool::gather_currents(const double* I_buf) {
    if (contiguous_ && N_ > 0) {
        std::memcpy(I_ext_.data(), I_buf + net_idx_[0], N_ * sizeof(double));
    } else {
        for (size_t i = 0; i < N_; ++i)
            I_ext_(i) = I_buf[net_idx_[i]];
    }
}

// Delegate to shared implementations in model/kinetics.hpp
void ComposablePool::fast_exp(const Eigen::ArrayXd& src, Eigen::ArrayXd& dst) {
    hodgkin_huxley::fast_exp(src, dst, tmp_exp_r_);
}

void ComposablePool::step(double dt) {
    if (N_ == 0 || dt == 0.0) return;
    finalize();

    const size_t ng = model_.gates.size();
    const size_t nc = model_.channels.size();

    // =========================================================================
    // 1. Update gates
    // =========================================================================
    for (size_t gi = 0; gi < ng; ++gi) {
        const auto& gs = model_.gates[gi];
        Eigen::ArrayXd& X = gate_states_[gi];

        switch (gs.update_form) {
            case GateSpec::UpdateForm::INF_TAU: {
                // Materialize dependency to avoid ternary type mismatch
                Eigen::ArrayXd dep = (gs.dependency == GateSpec::Dependency::CALCIUM)
                    ? Eigen::ArrayXd(Ca_) : Eigen::ArrayXd(V_);
                Eigen::ArrayXd x_inf = hodgkin_huxley::boltzmann_vec(dep, gs.inf);
                Eigen::ArrayXd tau_x = hodgkin_huxley::compute_tau_vec(V_, gs.tau, tmp_);
                tau_x = tau_x.max(1e-10);
                X = x_inf + (X - x_inf) * (-dt * gs.scale / tau_x).exp();
                break;
            }

            case GateSpec::UpdateForm::ALPHA_BETA: {
                Eigen::ArrayXd alpha = hodgkin_huxley::compute_rate_vec(V_, gs.alpha, tmp_);
                Eigen::ArrayXd beta  = hodgkin_huxley::compute_rate_vec(V_, gs.beta,  tmp2_);
                Eigen::ArrayXd rate  = (alpha + beta).max(1e-10);
                Eigen::ArrayXd x_inf = alpha / rate;
                X = x_inf + (X - x_inf) * (-dt * rate).exp();
                break;
            }

            case GateSpec::UpdateForm::INSTANT: {
                Eigen::ArrayXd dep = (gs.dependency == GateSpec::Dependency::CALCIUM)
                    ? Eigen::ArrayXd(Ca_) : Eigen::ArrayXd(V_);
                X = hodgkin_huxley::boltzmann_vec(dep, gs.inf);
                break;
            }

            case GateSpec::UpdateForm::DERIVED: {
                int src = gs.derived_source_gate;
                if (src >= 0 && src < static_cast<int>(ng)) {
                    X = gs.derived_a * (gs.derived_b + gs.derived_c * gate_states_[src]);
                }
                break;
            }
        }

        // Clamp to [0, 1]
        X = X.max(0.0).min(1.0);
    }

    // =========================================================================
    // 2. Compute total ionic current
    // =========================================================================
    I_total_.setZero();

    for (size_t ci = 0; ci < nc; ++ci) {
        const auto& ch = model_.channels[ci];

        // Gate product
        Eigen::ArrayXd gate_prod = Eigen::ArrayXd::Ones(N_);
        for (const auto& gp : ch.gates) {
            int idx = gp.first;
            int power = gp.second;
            if (idx >= 0 && idx < static_cast<int>(ng)) {
                for (int p = 0; p < power; ++p) {
                    gate_prod *= gate_states_[idx];
                }
            }
        }

        // Reversal potential
        Eigen::ArrayXd E_rev = ch.use_calcium_nernst
            ? Eigen::ArrayXd(E_Ca_)
            : Eigen::ArrayXd::Constant(N_, ch.E_rev);

        if (ch.is_ahp) {
            Eigen::ArrayXd ca_factor = Ca_ / (Ca_ + ch.ahp_k1).max(1e-10);
            I_total_ += ch.g * ca_factor * (V_ - E_rev);
        } else {
            I_total_ += ch.g * gate_prod * (V_ - E_rev);
        }
    }

    // =========================================================================
    // 3. Voltage update: dV/dt = (-I_ion + I_ext) / C_m
    // =========================================================================
    V_ += dt * (-I_total_ + I_ext_) / model_.C_m;

    // =========================================================================
    // 4. Calcium dynamics
    // =========================================================================
    if (model_.calcium.enabled) {
        Eigen::ArrayXd I_Ca = Eigen::ArrayXd::Zero(N_);
        for (int ch_idx : model_.calcium.source_channels) {
            if (ch_idx < 0 || ch_idx >= static_cast<int>(nc)) continue;
            const auto& ch = model_.channels[ch_idx];

            Eigen::ArrayXd gp = Eigen::ArrayXd::Ones(N_);
            for (const auto& gate_pair : ch.gates) {
                int idx = gate_pair.first;
                int power = gate_pair.second;
                if (idx >= 0 && idx < static_cast<int>(ng)) {
                    for (int p = 0; p < power; ++p) {
                        gp *= gate_states_[idx];
                    }
                }
            }

            Eigen::ArrayXd E_rev = ch.use_calcium_nernst
                ? Eigen::ArrayXd(E_Ca_)
                : Eigen::ArrayXd::Constant(N_, ch.E_rev);

            I_Ca += ch.g * gp * (V_ - E_rev);
        }

        Ca_ += dt * (model_.calcium.epsilon * (-I_Ca - model_.calcium.K_Ca * Ca_));
        Ca_ = Ca_.max(0.0);

        if (model_.calcium.use_nernst) {
            Eigen::ArrayXd ca_safe = Ca_.max(1e-10);
            E_Ca_ = (model_.calcium.R * model_.calcium.T)
                    / (model_.calcium.z * model_.calcium.F)
                    * (model_.calcium.Ca_o / ca_safe).log();
        }
    }
}

void ComposablePool::scatter_gate_states_into(double* buf, size_t max_gates,
                                               size_t n_rec, size_t t_rec) const {
    const size_t ng = model_.gates.size();
    for (size_t i = 0; i < N_; ++i) {
        size_t net_i = net_idx_[i];
        for (size_t g = 0; g < ng; ++g) {
            buf[net_i * max_gates * n_rec + g * n_rec + t_rec] = gate_states_[g](i);
        }
    }
}

void ComposablePool::scatter_calcium_into(double* buf, size_t n_rec, size_t t_rec) const {
    for (size_t i = 0; i < N_; ++i) {
        buf[net_idx_[i] * n_rec + t_rec] = Ca_(i);
    }
}

void ComposablePool::sync_to_neurons(
    std::vector<std::unique_ptr<NeuronBase>>& neurons) const
{
    const size_t ng = model_.gates.size();

    for (size_t i = 0; i < N_; ++i) {
        auto* cn = dynamic_cast<ComposableNeuron*>(neurons[net_idx_[i]].get());
        if (!cn) continue;

        cn->set_membrane_potential(V_(i));

        std::vector<double> gs(ng);
        for (size_t g = 0; g < ng; ++g) {
            gs[g] = gate_states_[g](i);
        }
        cn->set_gate_states(gs);
        cn->set_calcium(Ca_(i));
        cn->set_E_Ca(E_Ca_(i));
    }
}

} // namespace hodgkin_huxley
