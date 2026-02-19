#include "hodgkin_huxley/composable_pool.hpp"
#include "hodgkin_huxley/composable_neuron.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

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

// Fast exp: range reduction by 32 + degree-7 Taylor + 5 squarings
void ComposablePool::fast_exp(const Eigen::ArrayXd& src, Eigen::ArrayXd& dst) {
    tmp_exp_r_ = src * (1.0 / 32.0);

    dst = tmp_exp_r_ * (1.0 / 5040.0) + (1.0 / 720.0);
    dst = dst * tmp_exp_r_ + (1.0 / 120.0);
    dst = dst * tmp_exp_r_ + (1.0 / 24.0);
    dst = dst * tmp_exp_r_ + (1.0 / 6.0);
    dst = dst * tmp_exp_r_ + 0.5;
    dst = dst * tmp_exp_r_ + 1.0;
    dst = dst * tmp_exp_r_ + 1.0;

    dst *= dst; dst *= dst; dst *= dst; dst *= dst; dst *= dst;
}

// Vectorized Boltzmann: 1 / (1 + exp(-(x - v_half) / k))
Eigen::ArrayXd ComposablePool::boltzmann_vec(const Eigen::ArrayXd& x, const BoltzmannParams& p) {
    Eigen::ArrayXd arg = -(x - p.v_half) / p.k;
    arg = arg.max(-500.0).min(500.0);
    return 1.0 / (1.0 + arg.exp());
}

Eigen::ArrayXd ComposablePool::compute_tau_vec(const Eigen::ArrayXd& V,
                                                 const TauParams& tau,
                                                 Eigen::ArrayXd& tmp) {
    const Eigen::Index N = V.size();
    switch (tau.form) {
        case TauParams::Form::CONSTANT:
            return Eigen::ArrayXd::Constant(N, tau.params[0]);

        case TauParams::Form::BOLTZMANN: {
            double base = tau.params[0], amp = tau.params[1];
            double vh = tau.params[2], k = tau.params[3];
            tmp = (-(V - vh) / k).max(-500.0).min(500.0);
            return base + amp / (1.0 + tmp.exp());
        }

        case TauParams::Form::DOUBLE_EXP_SUM: {
            double base = tau.params[0], amp = tau.params[1];
            double v1 = tau.params[2], s1 = tau.params[3];
            double v2 = tau.params[5], s2 = tau.params[6];
            Eigen::ArrayXd e1 = ((V + v1) / s1).exp();
            Eigen::ArrayXd e2 = (-(V + v2) / s2).exp();
            return base + amp / (e1 + e2).max(1e-10);
        }

        case TauParams::Form::OFFSET_DOUBLE_EXP: {
            double base = tau.params[0], a1 = tau.params[1];
            double v1 = tau.params[2], s1 = tau.params[3];
            double a2 = tau.params[4], v2 = tau.params[5], s2 = tau.params[6];
            tmp = (V + v1) / s1;
            Eigen::ArrayXd t1 = a1 * (-tmp * tmp).exp();
            tmp = (V + v2) / s2;
            return base + t1 + a2 * (-tmp * tmp).exp();
        }

        case TauParams::Form::SCALED_EXP: {
            double scale = tau.params[0], vh = tau.params[1], k = tau.params[2];
            tmp = ((V - vh) / (2.0 * k)).max(-500.0).min(500.0);
            return scale / tmp.cosh().max(1e-10);
        }

        case TauParams::Form::COMPOUND_AB: {
            double aA = tau.params[0], aB = tau.params[1], aC = tau.params[2];
            double bA = tau.params[3], bB = tau.params[4], bC = tau.params[5];
            Eigen::ArrayXd alpha = aA * ((V + aB) / aC).exp();
            Eigen::ArrayXd beta = bA * ((V + bB) / bC).exp();
            return 1.0 / (alpha + beta).max(1e-10);
        }
    }
    return Eigen::ArrayXd::Constant(N, 1.0);
}

Eigen::ArrayXd ComposablePool::compute_rate_vec(const Eigen::ArrayXd& V,
                                                  const RateFuncParams& rate,
                                                  Eigen::ArrayXd& tmp) {
    const Eigen::Index N = V.size();
    switch (rate.form) {
        case RateFuncParams::Form::LINEAR_OVER_EXP: {
            Eigen::ArrayXd x = V + rate.B;
            Eigen::ArrayXd xc = x / rate.C;
            Eigen::ArrayXd e = xc.exp();
            Eigen::ArrayXd result = rate.A * x / (e - 1.0);
            return (xc.abs() < 1e-6).select(
                Eigen::ArrayXd::Constant(N, rate.A * rate.C), result);
        }

        case RateFuncParams::Form::EXP_DECAY: {
            tmp = ((V + rate.B) / rate.C).max(-500.0).min(500.0);
            return rate.A * tmp.exp();
        }

        case RateFuncParams::Form::LINEAR_OVER_EXPM1: {
            Eigen::ArrayXd x = V + rate.B;
            Eigen::ArrayXd xc = x / rate.C;
            Eigen::ArrayXd e = (-xc).exp();
            Eigen::ArrayXd result = rate.A * x / (1.0 - e);
            return (xc.abs() < 1e-6).select(
                Eigen::ArrayXd::Constant(N, rate.A * rate.C), result);
        }

        case RateFuncParams::Form::SIGMOID: {
            tmp = ((V + rate.B) / rate.C).max(-500.0).min(500.0);
            return rate.A / (1.0 + tmp.exp());
        }
    }
    return Eigen::ArrayXd::Zero(N);
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
                Eigen::ArrayXd x_inf = boltzmann_vec(dep, gs.inf);
                Eigen::ArrayXd tau_x = compute_tau_vec(V_, gs.tau, tmp_);
                tau_x = tau_x.max(1e-10);
                X = x_inf + (X - x_inf) * (-dt * gs.scale / tau_x).exp();
                break;
            }

            case GateSpec::UpdateForm::ALPHA_BETA: {
                Eigen::ArrayXd alpha = compute_rate_vec(V_, gs.alpha, tmp_);
                Eigen::ArrayXd beta = compute_rate_vec(V_, gs.beta, tmp2_);
                X += dt * (alpha * (1.0 - X) - beta * X);
                break;
            }

            case GateSpec::UpdateForm::INSTANT: {
                Eigen::ArrayXd dep = (gs.dependency == GateSpec::Dependency::CALCIUM)
                    ? Eigen::ArrayXd(Ca_) : Eigen::ArrayXd(V_);
                X = boltzmann_vec(dep, gs.inf);
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
