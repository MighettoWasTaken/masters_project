#include "hodgkin_huxley/model/synapse_spec.hpp"
#include <cmath>
#include <stdexcept>

namespace hodgkin_huxley {

// =============================================================================
// Spike-driven presets (EXP_DECAY / ALPHA_FUNC / DOUBLE_EXP)
// =============================================================================

SynapseSpec SynapseSpec::exponential(double tau_S, double g, double E_syn) {
    SynapseSpec s;
    s.name         = "exponential";
    s.update_form  = UpdateForm::EXP_DECAY;
    s.current_form = CurrentForm::LINEAR;
    s.g            = g;
    s.E_syn        = E_syn;
    s.power        = 1;
    s.S_init       = 0.0;
    s.A_init       = 0.0;
    s.delta_S      = 1.0;   // S += 1 on spike; g = spec.g * weight * S
    s.delta_A      = 0.0;
    s.tau_S        = tau_S;
    s.tau_A        = 1.0;   // unused for EXP_DECAY
    s.norm_factor  = 1.0;
    return s;
}

SynapseSpec SynapseSpec::alpha_function(double tau, double g, double E_syn) {
    // Peak conductance = spec.g * weight when delta_A = e and tau_A = tau.
    // Derivation: alpha ODE peak S(tau) = delta_A / e, so spec.g * weight * delta_A/e = weight
    // requires spec.g * delta_A = e  →  delta_A = e / spec.g.
    // With g=1: delta_A = e.
    static constexpr double E_CONST = 2.718281828459045;

    SynapseSpec s;
    s.name         = "alpha_function";
    s.update_form  = UpdateForm::ALPHA_FUNC;
    s.current_form = CurrentForm::LINEAR;
    s.g            = g;
    s.E_syn        = E_syn;
    s.power        = 1;
    s.S_init       = 0.0;
    s.A_init       = 0.0;
    s.delta_S      = 0.0;
    s.delta_A      = E_CONST / g;  // ensures g_peak = weight (normalized)
    s.tau_S        = tau;           // S time constant (primary var, lower in 2-var ODE)
    s.tau_A        = tau;           // A time constant (auxiliary var, upper in 2-var ODE)
    s.norm_factor  = 1.0;
    return s;
}

SynapseSpec SynapseSpec::double_exponential(double tau_rise, double tau_decay,
                                             double g, double E_syn) {
    if (tau_rise <= 0.0 || tau_decay <= 0.0)
        throw std::invalid_argument("double_exponential: tau_rise and tau_decay must be positive");
    if (tau_rise >= tau_decay)
        throw std::invalid_argument("double_exponential: tau_rise must be strictly less than tau_decay");

    // S = slow component (tau_S = tau_decay), A = fast component (tau_A = tau_rise)
    // g_eff = norm * (S - A);  peak conductance = spec.g * weight * 1.0 = spec.g * weight
    double t_peak = (tau_decay * tau_rise) / (tau_decay - tau_rise)
                    * std::log(tau_decay / tau_rise);
    double peak_val = std::exp(-t_peak / tau_decay) - std::exp(-t_peak / tau_rise);

    SynapseSpec s;
    s.name         = "double_exponential";
    s.update_form  = UpdateForm::DOUBLE_EXP;
    s.current_form = CurrentForm::LINEAR;
    s.g            = g;
    s.E_syn        = E_syn;
    s.power        = 1;
    s.S_init       = 0.0;
    s.A_init       = 0.0;
    s.delta_S      = 1.0;   // S (slow) jumps by 1 on spike
    s.delta_A      = 1.0;   // A (fast) jumps by 1 on spike
    s.tau_S        = tau_decay;
    s.tau_A        = tau_rise;
    s.norm_factor  = (peak_val > 1e-15) ? 1.0 / peak_val : 1.0;
    return s;
}

// =============================================================================
// Voltage-gated kinetic presets
// =============================================================================

SynapseSpec SynapseSpec::gaba_kinetic() {
    // Kumaravelu 2016: tanh open-rate model
    SynapseSpec s;
    s.name         = "GABA_kinetic";
    s.update_form  = UpdateForm::TANH_GATE;
    s.current_form = CurrentForm::LINEAR;
    s.tanh_amp     = 2.0;
    s.tanh_vh      = 0.0;
    s.tanh_k       = 4.0;
    s.tau_decay    = 13.0;
    s.g            = 0.1;
    s.E_syn        = -80.0;
    s.power        = 1;
    s.S_init       = 0.0;
    return s;
}

SynapseSpec SynapseSpec::nmda_kinetic() {
    // NMDA with Mg2+ block
    SynapseSpec s;
    s.name             = "NMDA_kinetic";
    s.update_form      = UpdateForm::BOLTZMANN_GATE;
    s.current_form     = CurrentForm::MG_BLOCK;
    s.s_inf.v_half     = -20.0;
    s.s_inf.k          = 16.0;
    s.tau.form         = TauParams::Form::CONSTANT;
    s.tau.params[0]    = 80.0;
    s.g                = 0.1;
    s.E_syn            = 0.0;
    s.power            = 1;
    s.mg_conc          = 1.0;
    s.mg_scale         = 0.062;
    s.mg_denom         = 3.57;
    s.S_init           = 0.0;
    return s;
}

SynapseSpec SynapseSpec::gaba_b() {
    // Slow GABA-B with cooperative gating (power=4)
    SynapseSpec s;
    s.name          = "GABA_B";
    s.update_form   = UpdateForm::BOLTZMANN_GATE;
    s.current_form  = CurrentForm::LINEAR;
    s.s_inf.v_half  = -60.0;
    s.s_inf.k       = -8.0;    // inverted sigmoid (decreases with V)
    s.tau.form      = TauParams::Form::CONSTANT;
    s.tau.params[0] = 200.0;
    s.g             = 0.1;
    s.E_syn         = -95.0;
    s.power         = 4;       // cooperative gating
    s.S_init        = 0.0;
    return s;
}

// =============================================================================
// Receptor presets (double-exponential: biologically accurate kinetics)
// =============================================================================

SynapseSpec SynapseSpec::ampa() {
    auto s = double_exponential(0.5, 2.5, 1.0, 0.0);
    s.name = "AMPA";
    return s;
}

SynapseSpec SynapseSpec::nmda() {
    auto s = double_exponential(2.0, 67.0, 1.0, 0.0);
    s.name = "NMDA";
    return s;
}

SynapseSpec SynapseSpec::gaba_a() {
    auto s = double_exponential(0.4, 7.7, 1.0, -80.0);
    s.name = "GABA_A";
    return s;
}

} // namespace hodgkin_huxley
