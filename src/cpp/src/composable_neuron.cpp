#include "hodgkin_huxley/composable_neuron.hpp"
#include "hodgkin_huxley/model/kinetics.hpp"
#include <cmath>
#include <algorithm>

namespace hodgkin_huxley {

// ---------------------------------------------------------------------------
// Helpers to initialize substance state from spec
// ---------------------------------------------------------------------------
static void init_substances(const NeuronModelSpec& spec,
                             std::vector<double>& X,
                             std::vector<double>& E_nernst) {
    const size_t ns = spec.intracellular.size();
    X.resize(ns);
    E_nernst.resize(ns, 120.0);
    for (size_t si = 0; si < ns; ++si) {
        const auto& ic = spec.intracellular[si];
        X[si] = ic.initial;
        if (ic.nernst_enabled && ic.initial > 0.0) {
            E_nernst[si] = (ic.nernst_R * ic.nernst_T)
                           / (ic.nernst_z * ic.nernst_F)
                           * std::log(ic.nernst_Ca_o / ic.initial);
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor / set_model
// ---------------------------------------------------------------------------

ComposableNeuron::ComposableNeuron(const NeuronModelSpec& spec)
    : spec_(spec),
      V_(spec.V_init),
      gate_states_(spec.gates.size())
{
    spec_.validate();
    for (size_t i = 0; i < spec_.gates.size(); ++i)
        gate_states_[i] = spec_.gates[i].initial_value;
    init_substances(spec_, X_, E_nernst_);
}

void ComposableNeuron::set_model(const NeuronModelSpec& spec) {
    // Resize gate state vector if gate count changed
    gate_states_.resize(spec.gates.size(), 0.0);
    // Resize substance vectors, preserving existing values where possible
    size_t ns_old = X_.size();
    size_t ns_new = spec.intracellular.size();
    X_.resize(ns_new, 0.0);
    E_nernst_.resize(ns_new, 120.0);
    for (size_t si = ns_old; si < ns_new; ++si) {
        const auto& ic = spec.intracellular[si];
        X_[si] = ic.initial;
        if (ic.nernst_enabled && ic.initial > 0.0) {
            E_nernst_[si] = (ic.nernst_R * ic.nernst_T)
                            / (ic.nernst_z * ic.nernst_F)
                            * std::log(ic.nernst_Ca_o / ic.initial);
        }
    }
    spec_ = spec;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double ComposableNeuron::membrane_potential() const { return V_; }
void ComposableNeuron::set_membrane_potential(double V) { V_ = V; }
std::string ComposableNeuron::type_name() const { return "Composable:" + spec_.name; }
const NeuronModelSpec& ComposableNeuron::model_spec() const { return spec_; }
const std::vector<double>& ComposableNeuron::gate_states() const { return gate_states_; }

void ComposableNeuron::set_gate_states(const std::vector<double>& states) {
    for (size_t i = 0; i < states.size() && i < gate_states_.size(); ++i)
        gate_states_[i] = states[i];
}

void ComposableNeuron::set_substances(const std::vector<double>& xs,
                                      const std::vector<double>& es) {
    for (size_t i = 0; i < xs.size() && i < X_.size(); ++i)
        X_[i] = xs[i];
    for (size_t i = 0; i < es.size() && i < E_nernst_.size(); ++i)
        E_nernst_[i] = es[i];
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void ComposableNeuron::reset() {
    V_ = spec_.V_init;
    for (size_t i = 0; i < spec_.gates.size(); ++i)
        gate_states_[i] = spec_.gates[i].initial_value;
    init_substances(spec_, X_, E_nernst_);
}

// ---------------------------------------------------------------------------
// reset_gates_to_steady_state
// ---------------------------------------------------------------------------

void ComposableNeuron::reset_gates_to_steady_state() {
    const size_t ng = spec_.gates.size();

    auto get_dep = [&](const GateSpec& gs) -> double {
        if (gs.dependency == GateSpec::Dependency::INTRACELLULAR) {
            int iidx = gs.intracellular_idx;
            if (iidx >= 0 && iidx < static_cast<int>(X_.size()))
                return X_[iidx];
            return 0.0;
        }
        return V_;
    };

    for (size_t i = 0; i < ng; ++i) {
        const auto& gs = spec_.gates[i];
        switch (gs.update_form) {
            case GateSpec::UpdateForm::INF_TAU: {
                double dep = get_dep(gs);
                gate_states_[i] = boltzmann_scalar(dep, gs.inf);
                gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
                break;
            }
            case GateSpec::UpdateForm::ALPHA_BETA: {
                double alpha = compute_rate_scalar(V_, gs.alpha);
                double beta  = compute_rate_scalar(V_, gs.beta);
                double rate  = alpha + beta;
                if (rate > 1e-10) gate_states_[i] = alpha / rate;
                gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
                break;
            }
            case GateSpec::UpdateForm::INSTANT: {
                double dep = get_dep(gs);
                gate_states_[i] = boltzmann_scalar(dep, gs.inf);
                gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
                break;
            }
            case GateSpec::UpdateForm::DERIVED:
                break;  // handled in second pass
            case GateSpec::UpdateForm::CUSTOM_EXPR: {
                double dep = get_dep(gs);
                if (!gs.inf_vm.empty()) {
                    double v = hodgkin_huxley::vm_eval_scalar(gs.inf_vm, dep);
                    gate_states_[i] = std::max(0.0, std::min(1.0, v));
                } else if (!gs.alpha_vm.empty() && !gs.beta_vm.empty()) {
                    double a = hodgkin_huxley::vm_eval_scalar(gs.alpha_vm, V_);
                    double b = hodgkin_huxley::vm_eval_scalar(gs.beta_vm, V_);
                    if (a+b > 1e-10) gate_states_[i] = std::max(0.0, std::min(1.0, a/(a+b)));
                }
                break;
            }
        }
    }

    // Second pass: DERIVED gates
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

// ---------------------------------------------------------------------------
// update_gates
// ---------------------------------------------------------------------------

void ComposableNeuron::update_gates(double dt) {
    const size_t ng = spec_.gates.size();

    auto get_dep = [&](const GateSpec& gs) -> double {
        if (gs.dependency == GateSpec::Dependency::INTRACELLULAR) {
            int iidx = gs.intracellular_idx;
            if (iidx >= 0 && iidx < static_cast<int>(X_.size()))
                return X_[iidx];
            return 0.0;
        }
        return V_;
    };

    for (size_t i = 0; i < ng; ++i) {
        const auto& gs = spec_.gates[i];

        switch (gs.update_form) {
            case GateSpec::UpdateForm::INF_TAU: {
                double dep_var = get_dep(gs);
                double x_inf = boltzmann_scalar(dep_var, gs.inf);
                double tau_x = compute_tau_scalar(V_, gs.tau);
                if (tau_x < 1e-10) tau_x = 1e-10;
                gate_states_[i] = x_inf + (gate_states_[i] - x_inf)
                                   * std::exp(-dt * gs.scale / tau_x);
                break;
            }

            case GateSpec::UpdateForm::ALPHA_BETA: {
                double alpha = compute_rate_scalar(V_, gs.alpha);
                double beta  = compute_rate_scalar(V_, gs.beta);
                double rate  = alpha + beta;
                double x_inf = (rate > 1e-10) ? alpha / rate : gate_states_[i];
                double tau_x = (rate > 1e-10) ? 1.0 / rate  : 1e10;
                gate_states_[i] = x_inf + (gate_states_[i] - x_inf) * std::exp(-dt / tau_x);
                break;
            }

            case GateSpec::UpdateForm::INSTANT: {
                double dep_var = get_dep(gs);
                gate_states_[i] = boltzmann_scalar(dep_var, gs.inf);
                break;
            }

            case GateSpec::UpdateForm::DERIVED: {
                int src = gs.derived_source_gate;
                if (src >= 0 && src < static_cast<int>(ng))
                    gate_states_[i] = gs.derived_a * (gs.derived_b + gs.derived_c * gate_states_[src]);
                break;
            }

            case GateSpec::UpdateForm::CUSTOM_EXPR: {
                double dep = get_dep(gs);
                if (!gs.dxdt_vm.empty()) {
                    double dxdt = hodgkin_huxley::vm_eval_scalar_2arg(gs.dxdt_vm, dep, gate_states_[i]);
                    gate_states_[i] += dt * gs.scale * dxdt;
                } else if (!gs.inf_vm.empty() && !gs.tau_vm.empty()) {
                    double x_inf = hodgkin_huxley::vm_eval_scalar(gs.inf_vm, dep);
                    double tau_x = std::max(1e-10, hodgkin_huxley::vm_eval_scalar(gs.tau_vm, V_));
                    gate_states_[i] = x_inf + (gate_states_[i]-x_inf)*std::exp(-dt*gs.scale/tau_x);
                } else if (!gs.alpha_vm.empty() && !gs.beta_vm.empty()) {
                    double a = hodgkin_huxley::vm_eval_scalar(gs.alpha_vm, V_);
                    double b = hodgkin_huxley::vm_eval_scalar(gs.beta_vm, V_);
                    double rate = std::max(1e-10, a+b);
                    gate_states_[i] = a/rate + (gate_states_[i]-a/rate)*std::exp(-dt*rate);
                } else if (!gs.inf_vm.empty()) {
                    double x_inf = hodgkin_huxley::vm_eval_scalar(gs.inf_vm, dep);
                    gate_states_[i] = std::max(0.0, std::min(1.0, x_inf));
                }
                break;
            }
        }

        gate_states_[i] = std::max(0.0, std::min(1.0, gate_states_[i]));
    }
}

// ---------------------------------------------------------------------------
// compute_single_channel_current  (for use by update_intracellular)
// ---------------------------------------------------------------------------

double ComposableNeuron::compute_single_channel_current(int channel_idx) const {
    const size_t ng = spec_.gates.size();
    const size_t nc = spec_.channels.size();
    if (channel_idx < 0 || channel_idx >= static_cast<int>(nc)) return 0.0;

    const auto& ch = spec_.channels[channel_idx];
    double gate_product;
    if (!ch.gate_product_vm.empty()) {
        gate_product = hodgkin_huxley::vm_eval_gate_product_scalar(
            ch.gate_product_vm, V_, gate_states_);
    } else {
        gate_product = 1.0;
        for (const auto& gp : ch.gates) {
            int idx = gp.first, power = gp.second;
            if (idx >= 0 && idx < static_cast<int>(gate_states_.size())) {
                double val = gate_states_[idx];
                for (int p = 0; p < power; ++p) gate_product *= val;
            }
        }
    }
    double E;
    if (ch.nernst_substance_idx >= 0 && ch.nernst_substance_idx < static_cast<int>(E_nernst_.size()))
        E = E_nernst_[ch.nernst_substance_idx];
    else
        E = ch.E_rev;
    return ch.g * gate_product * (V_ - E);
}

// ---------------------------------------------------------------------------
// compute_channel_current
// ---------------------------------------------------------------------------

double ComposableNeuron::compute_channel_current() const {
    double I_total = 0.0;
    const size_t nc = spec_.channels.size();

    for (size_t ci = 0; ci < nc; ++ci) {
        const auto& ch = spec_.channels[ci];
        double gate_product;
        if (!ch.gate_product_vm.empty()) {
            gate_product = hodgkin_huxley::vm_eval_gate_product_scalar(
                ch.gate_product_vm, V_, gate_states_);
        } else {
            gate_product = 1.0;
            for (const auto& gp : ch.gates) {
                int idx = gp.first, power = gp.second;
                if (idx >= 0 && idx < static_cast<int>(gate_states_.size())) {
                    double val = gate_states_[idx];
                    for (int p = 0; p < power; ++p) gate_product *= val;
                }
            }
        }

        double E;
        if (ch.nernst_substance_idx >= 0 && ch.nernst_substance_idx < static_cast<int>(E_nernst_.size()))
            E = E_nernst_[ch.nernst_substance_idx];
        else
            E = ch.E_rev;

        if (ch.is_ahp) {
            int aidx = ch.ahp_substance_idx;
            double X_ahp = (aidx >= 0 && aidx < static_cast<int>(X_.size())) ? X_[aidx] : 0.0;
            double ca_factor = (X_ahp + ch.ahp_k1 > 0.0) ? X_ahp / (X_ahp + ch.ahp_k1) : 0.0;
            I_total += ch.g * ca_factor * (V_ - E);
        } else {
            I_total += ch.g * gate_product * (V_ - E);
        }
    }

    return I_total;
}

// ---------------------------------------------------------------------------
// update_intracellular  (replaces update_calcium)
// ---------------------------------------------------------------------------

void ComposableNeuron::update_intracellular(double dt) {
    const size_t ns = spec_.intracellular.size();
    if (ns == 0) return;

    for (size_t si = 0; si < ns; ++si) {
        const auto& ic = spec_.intracellular[si];
        using UF = IntracellularSpec::UpdateForm;

        double dX = 0.0;
        switch (ic.update_form) {
            case UF::DECAY:
                dX = -ic.k_decay * X_[si];
                break;

            case UF::DRIVEN_DECAY:
            case UF::DRIVEN_DECAY_NERNST: {
                double I_src = 0.0;
                for (int ch_idx : ic.source_channels)
                    I_src += compute_single_channel_current(ch_idx);
                dX = ic.epsilon * (-I_src - ic.k_decay * X_[si]);
                break;
            }

            case UF::CUSTOM_EXPR: {
                if (!ic.ode_vm.empty()) {
                    double I_src = 0.0;
                    for (int ch_idx : ic.source_channels)
                        I_src += compute_single_channel_current(ch_idx);
                    // Scalar VM evaluation — PUSH_DEP=I_src, PUSH_S=X_[si]
                    // PUSH_X(n) not supported in scalar path for cross-substance refs
                    dX = hodgkin_huxley::vm_eval_scalar_2arg(ic.ode_vm, I_src, X_[si]);
                }
                break;
            }
        }

        X_[si] += dt * dX;
        X_[si] = std::max(0.0, X_[si]);

        if (ic.nernst_enabled) {
            if (!ic.nernst_vm.empty()) {
                E_nernst_[si] = hodgkin_huxley::vm_eval_scalar(ic.nernst_vm, X_[si]);
            } else {
                double x_safe = std::max(X_[si], 1e-10);
                E_nernst_[si] = (ic.nernst_R * ic.nernst_T)
                                / (ic.nernst_z * ic.nernst_F)
                                * std::log(ic.nernst_Ca_o / x_safe);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// step
// ---------------------------------------------------------------------------

void ComposableNeuron::step(double dt, double I_ext) {
    if (dt == 0.0) return;

    update_gates(dt);
    double I_ion = compute_channel_current();
    V_ += dt * (-I_ion + I_ext) / spec_.C_m;
    update_intracellular(dt);

    // Re-sync INSTANT and DERIVED gates to the new voltage
    const size_t ng = spec_.gates.size();
    auto get_dep = [&](const GateSpec& gs) -> double {
        if (gs.dependency == GateSpec::Dependency::INTRACELLULAR) {
            int iidx = gs.intracellular_idx;
            if (iidx >= 0 && iidx < static_cast<int>(X_.size()))
                return X_[iidx];
            return 0.0;
        }
        return V_;
    };

    for (size_t i = 0; i < ng; ++i) {
        const auto& gs = spec_.gates[i];
        if (gs.update_form == GateSpec::UpdateForm::INSTANT) {
            gate_states_[i] = boltzmann_scalar(get_dep(gs), gs.inf);
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
