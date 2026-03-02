#include "hodgkin_huxley/composable_neuron.hpp"
#include <cmath>
#include <algorithm>

namespace hodgkin_huxley {

ComposableNeuron::ComposableNeuron(const NeuronModelSpec& spec)
    : spec_(spec),
      V_(spec.V_init),
      gate_states_(spec.gates.size()),
      Ca_(spec.calcium.Ca_init),
      E_Ca_(120.0)
{
    spec_.validate();
    for (size_t i = 0; i < spec_.gates.size(); ++i) {
        gate_states_[i] = spec_.gates[i].initial_value;
    }
    if (spec_.calcium.enabled && spec_.calcium.use_nernst && Ca_ > 0.0) {
        double ratio = spec_.calcium.Ca_o / Ca_;
        E_Ca_ = (spec_.calcium.R * spec_.calcium.T)
                / (spec_.calcium.z * spec_.calcium.F)
                * std::log(ratio);
    }
}

double ComposableNeuron::membrane_potential() const { return V_; }
void ComposableNeuron::set_membrane_potential(double V) { V_ = V; }
std::string ComposableNeuron::type_name() const { return "Composable:" + spec_.name; }
const NeuronModelSpec& ComposableNeuron::model_spec() const { return spec_; }
const std::vector<double>& ComposableNeuron::gate_states() const { return gate_states_; }
double ComposableNeuron::calcium() const { return Ca_; }

void ComposableNeuron::set_gate_states(const std::vector<double>& states) {
    for (size_t i = 0; i < states.size() && i < gate_states_.size(); ++i) {
        gate_states_[i] = states[i];
    }
}

void ComposableNeuron::reset_gates_to_steady_state() {
    const size_t ng = spec_.gates.size();

    // First pass: compute steady-state for INF_TAU, ALPHA_BETA, INSTANT gates
    for (size_t i = 0; i < ng; ++i) {
        const auto& gs = spec_.gates[i];
        switch (gs.update_form) {
            case GateSpec::UpdateForm::INF_TAU: {
                double dep = (gs.dependency == GateSpec::Dependency::CALCIUM) ? Ca_ : V_;
                gate_states_[i] = boltzmann(dep, gs.inf);
                gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
                break;
            }
            case GateSpec::UpdateForm::ALPHA_BETA: {
                double alpha = compute_rate(V_, gs.alpha);
                double beta  = compute_rate(V_, gs.beta);
                double rate  = alpha + beta;
                if (rate > 1e-10) gate_states_[i] = alpha / rate;
                gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
                break;
            }
            case GateSpec::UpdateForm::INSTANT: {
                double dep = (gs.dependency == GateSpec::Dependency::CALCIUM) ? Ca_ : V_;
                gate_states_[i] = boltzmann(dep, gs.inf);
                gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
                break;
            }
            case GateSpec::UpdateForm::DERIVED:
                break;  // handled in second pass
        }
    }

    // Second pass: DERIVED gates (use the just-computed steady states of their sources)
    for (size_t i = 0; i < ng; ++i) {
        const auto& gs = spec_.gates[i];
        if (gs.update_form == GateSpec::UpdateForm::DERIVED) {
            int src = gs.derived_source_gate;
            if (src >= 0 && src < static_cast<int>(ng)) {
                gate_states_[i] = gs.derived_a * (gs.derived_b + gs.derived_c * gate_states_[src]);
                gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
            }
        }
    }
}
void ComposableNeuron::set_calcium(double ca) { Ca_ = ca; }
void ComposableNeuron::set_E_Ca(double e_ca) { E_Ca_ = e_ca; }

void ComposableNeuron::reset() {
    V_ = spec_.V_init;
    for (size_t i = 0; i < spec_.gates.size(); ++i) {
        gate_states_[i] = spec_.gates[i].initial_value;
    }
    Ca_ = spec_.calcium.Ca_init;
    if (spec_.calcium.enabled && spec_.calcium.use_nernst && Ca_ > 0.0) {
        double ratio = spec_.calcium.Ca_o / Ca_;
        E_Ca_ = (spec_.calcium.R * spec_.calcium.T)
                / (spec_.calcium.z * spec_.calcium.F)
                * std::log(ratio);
    }
}

// Boltzmann sigmoid: 1 / (1 + exp(-(x - v_half) / k))
double ComposableNeuron::boltzmann(double x, const BoltzmannParams& p) {
    double arg = -(x - p.v_half) / p.k;
    // Clamp to prevent overflow
    if (arg > 500.0) return 0.0;
    if (arg < -500.0) return 1.0;
    return 1.0 / (1.0 + std::exp(arg));
}

double ComposableNeuron::compute_tau(double V, const TauParams& tau) {
    switch (tau.form) {
        case TauParams::Form::CONSTANT:
            return tau.params[0];

        case TauParams::Form::BOLTZMANN: {
            // tau = base + amp / (1 + exp(-(V - v_half) / k))
            double base = tau.params[0];
            double amp = tau.params[1];
            double vh = tau.params[2];
            double k = tau.params[3];
            double arg = -(V - vh) / k;
            arg = std::max(-500.0, std::min(500.0, arg));
            return base + amp / (1.0 + std::exp(arg));
        }

        case TauParams::Form::DOUBLE_EXP_SUM: {
            // tau = base + amp / (exp((V + v1) / s1) + exp(-(V + v2) / s2))
            double base = tau.params[0];
            double amp = tau.params[1];
            double v1 = tau.params[2];
            double s1 = tau.params[3];
            double v2 = tau.params[5];
            double s2 = tau.params[6];
            double e1 = std::exp((V + v1) / s1);
            double e2 = std::exp(-(V + v2) / s2);
            double denom = e1 + e2;
            if (denom < 1e-10) denom = 1e-10;
            return base + amp / denom;
        }

        case TauParams::Form::OFFSET_DOUBLE_EXP: {
            // tau = base + a1*exp(-((V+v1)/s1)^2) + a2*exp(-((V+v2)/s2)^2)
            double base = tau.params[0];
            double a1 = tau.params[1];
            double v1 = tau.params[2];
            double s1 = tau.params[3];
            double a2 = tau.params[4];
            double v2 = tau.params[5];
            double s2 = tau.params[6];
            double x1 = (V + v1) / s1;
            double x2 = (V + v2) / s2;
            return base + a1 * std::exp(-x1 * x1) + a2 * std::exp(-x2 * x2);
        }

        case TauParams::Form::SCALED_EXP: {
            // tau = scale / cosh((V - v_half) / (2 * k))
            double scale = tau.params[0];
            double vh = tau.params[1];
            double k = tau.params[2];
            double arg = (V - vh) / (2.0 * k);
            arg = std::max(-500.0, std::min(500.0, arg));
            double ch = std::cosh(arg);
            if (ch < 1e-10) ch = 1e-10;
            return scale / ch;
        }

        case TauParams::Form::COMPOUND_AB: {
            // alpha and beta as rate functions, tau = 1/(alpha+beta)
            double aA = tau.params[0], aB = tau.params[1], aC = tau.params[2];
            double bA = tau.params[3], bB = tau.params[4], bC = tau.params[5];
            // Simple exp-based rates
            double alpha = aA * std::exp((V + aB) / aC);
            double beta = bA * std::exp((V + bB) / bC);
            double sum = alpha + beta;
            if (sum < 1e-10) sum = 1e-10;
            return 1.0 / sum;
        }
    }
    return 1.0;
}

double ComposableNeuron::compute_rate(double V, const RateFuncParams& rate) {
    switch (rate.form) {
        case RateFuncParams::Form::LINEAR_OVER_EXP: {
            // A*(V+B) / (exp((V+B)/C) - 1)
            double x = V + rate.B;
            double xc = x / rate.C;
            // Handle singularity using L'Hopital: limit -> A*C
            if (std::abs(xc) < 1e-6) {
                return rate.A * rate.C * (1.0 + xc * 0.5);
            }
            double e = std::exp(xc);
            return rate.A * x / (e - 1.0);
        }

        case RateFuncParams::Form::EXP_DECAY: {
            // A * exp((V+B)/C)
            double arg = (V + rate.B) / rate.C;
            arg = std::max(-500.0, std::min(500.0, arg));
            return rate.A * std::exp(arg);
        }

        case RateFuncParams::Form::LINEAR_OVER_EXPM1: {
            // A*(V+B) / (1 - exp(-(V+B)/C))
            double x = V + rate.B;
            double xc = x / rate.C;
            // Handle singularity: limit -> A*C
            if (std::abs(xc) < 1e-6) {
                return rate.A * rate.C * (1.0 + xc * 0.5);
            }
            double e = std::exp(-xc);
            return rate.A * x / (1.0 - e);
        }

        case RateFuncParams::Form::SIGMOID: {
            // A / (1 + exp((V+B)/C))
            double arg = (V + rate.B) / rate.C;
            arg = std::max(-500.0, std::min(500.0, arg));
            return rate.A / (1.0 + std::exp(arg));
        }
    }
    return 0.0;
}

void ComposableNeuron::update_gates(double dt) {
    const size_t ng = spec_.gates.size();

    for (size_t i = 0; i < ng; ++i) {
        const auto& gs = spec_.gates[i];

        switch (gs.update_form) {
            case GateSpec::UpdateForm::INF_TAU: {
                double dep_var = (gs.dependency == GateSpec::Dependency::CALCIUM) ? Ca_ : V_;
                double x_inf = boltzmann(dep_var, gs.inf);
                double tau_x = compute_tau(V_, gs.tau);
                if (tau_x < 1e-10) tau_x = 1e-10;
                gate_states_[i] = x_inf + (gate_states_[i] - x_inf) * std::exp(-dt * gs.scale / tau_x);
                break;
            }

            case GateSpec::UpdateForm::ALPHA_BETA: {
                double alpha = compute_rate(V_, gs.alpha);
                double beta  = compute_rate(V_, gs.beta);
                double rate  = alpha + beta;
                double x_inf = (rate > 1e-10) ? alpha / rate : gate_states_[i];
                double tau_x = (rate > 1e-10) ? 1.0 / rate  : 1e10;
                gate_states_[i] = x_inf + (gate_states_[i] - x_inf) * std::exp(-dt / tau_x);
                break;
            }

            case GateSpec::UpdateForm::INSTANT: {
                double dep_var = (gs.dependency == GateSpec::Dependency::CALCIUM) ? Ca_ : V_;
                gate_states_[i] = boltzmann(dep_var, gs.inf);
                break;
            }

            case GateSpec::UpdateForm::DERIVED: {
                int src = gs.derived_source_gate;
                if (src >= 0 && src < static_cast<int>(ng)) {
                    gate_states_[i] = gs.derived_a * (gs.derived_b + gs.derived_c * gate_states_[src]);
                }
                break;
            }
        }

        // Clamp gate values to [0, 1]
        gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
    }
}

double ComposableNeuron::compute_channel_current() const {
    double I_total = 0.0;

    for (const auto& ch : spec_.channels) {
        double gate_product = 1.0;
        for (const auto& gp : ch.gates) {
            int idx = gp.first;
            int power = gp.second;
            if (idx >= 0 && idx < static_cast<int>(gate_states_.size())) {
                double val = gate_states_[idx];
                for (int p = 0; p < power; ++p) {
                    gate_product *= val;
                }
            }
        }

        double E = ch.E_rev;
        if (ch.use_calcium_nernst) {
            E = E_Ca_;
        }

        if (ch.is_ahp) {
            // I_AHP = g * Ca/(Ca + k1) * (V - E)
            double ca_factor = (Ca_ + ch.ahp_k1 > 0.0) ? Ca_ / (Ca_ + ch.ahp_k1) : 0.0;
            I_total += ch.g * ca_factor * (V_ - E);
        } else {
            I_total += ch.g * gate_product * (V_ - E);
        }
    }

    return I_total;
}

void ComposableNeuron::update_calcium(double dt) {
    if (!spec_.calcium.enabled) return;

    // Compute Ca influx from source channels
    double I_Ca = 0.0;
    for (int ch_idx : spec_.calcium.source_channels) {
        if (ch_idx < 0 || ch_idx >= static_cast<int>(spec_.channels.size())) continue;
        const auto& ch = spec_.channels[ch_idx];
        double gate_product = 1.0;
        for (const auto& gp : ch.gates) {
            int idx = gp.first;
            int power = gp.second;
            if (idx >= 0 && idx < static_cast<int>(gate_states_.size())) {
                double val = gate_states_[idx];
                for (int p = 0; p < power; ++p) {
                    gate_product *= val;
                }
            }
        }
        double E = ch.use_calcium_nernst ? E_Ca_ : ch.E_rev;
        I_Ca += ch.g * gate_product * (V_ - E);
    }

    // dCa/dt = epsilon * (-I_Ca - K_Ca * Ca)
    // Note: I_Ca is outward for V > E_Ca, so -I_Ca is inward (positive influx)
    Ca_ += dt * (spec_.calcium.epsilon * (-I_Ca - spec_.calcium.K_Ca * Ca_));
    Ca_ = std::max(0.0, Ca_);

    // Update Nernst E_Ca
    if (spec_.calcium.use_nernst) {
        double ca_safe = std::max(Ca_, 1e-10);
        double ratio = spec_.calcium.Ca_o / ca_safe;
        E_Ca_ = (spec_.calcium.R * spec_.calcium.T)
                / (spec_.calcium.z * spec_.calcium.F)
                * std::log(ratio);
    }
}

void ComposableNeuron::step(double dt, double I_ext) {
    if (dt == 0.0) return;

    // 1. Update gates (INF_TAU / ALPHA_BETA / INSTANT / DERIVED all use V_old)
    update_gates(dt);

    // 2. Compute total ionic current (with gates at V_old)
    double I_ion = compute_channel_current();

    // 3. dV/dt = (-I_ion + I_ext) / C_m
    V_ += dt * (-I_ion + I_ext) / spec_.C_m;

    // 4. Update calcium
    update_calcium(dt);

    // 5. Re-sync algebraic gates to the new voltage so callers always see
    //    a consistent (V, gate) state.  INF_TAU / ALPHA_BETA gates already
    //    hold their best estimate and are left unchanged here.
    const size_t ng = spec_.gates.size();
    for (size_t i = 0; i < ng; ++i) {
        const auto& gs = spec_.gates[i];
        if (gs.update_form == GateSpec::UpdateForm::INSTANT) {
            double dep = (gs.dependency == GateSpec::Dependency::CALCIUM) ? Ca_ : V_;
            gate_states_[i] = boltzmann(dep, gs.inf);
        } else if (gs.update_form == GateSpec::UpdateForm::DERIVED) {
            int src = gs.derived_source_gate;
            if (src >= 0 && src < static_cast<int>(ng)) {
                double val = gs.derived_a * (gs.derived_b + gs.derived_c * gate_states_[src]);
                gate_states_[i] = std::max(0.0, std::min(1.0, val));
            }
        }
    }
}

} // namespace hodgkin_huxley
