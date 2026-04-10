#pragma once

#include "hodgkin_huxley/model/gate_spec.hpp"
#include <cstdint>
#include <string>

namespace hodgkin_huxley {

// =============================================================================
// SynapseSpec — unified synapse model specification (mirrors GateSpec pattern)
//
// A single flat struct with an UpdateForm discriminant covers all synapse types:
//   EXP_DECAY     — single-variable, spike-jump + exponential decay
//   ALPHA_FUNC    — two-variable Euler (alpha function)
//   DOUBLE_EXP    — two-variable exact double-exponential
//   TANH_GATE     — voltage-gated, exact exponential integration
//   BOLTZMANN_GATE— voltage-gated, exact exponential integration
//   ALPHA_BETA    — voltage-gated, alpha/beta rate functions
//   CUSTOM_EXPR   — arbitrary VM bytecode; optional second ODE (dA_dt_vm)
//
// All update logic lives in Network::update_synapses_grouped(); this struct
// is pure data.  Unused fields for a given UpdateForm are ignored at runtime.
// =============================================================================

struct SynapseSpec {
    std::string name;

    // -------------------------------------------------------------------------
    // Discriminant
    // -------------------------------------------------------------------------
    enum class UpdateForm {
        EXP_DECAY,        // S += delta_S on spike; S *= exp(-dt/tau_S) each step
        ALPHA_FUNC,       // dS/dt=(A-S)/tau_A, dA/dt=-A/tau_A; A+=delta_A on spike (Euler)
        DOUBLE_EXP,       // S*=exp(-dt/tau_S), A*=exp(-dt/tau_A); g = norm*(S-A)
        TANH_GATE,        // voltage-gated: tanh open-rate, exact exp integration
        BOLTZMANN_GATE,   // voltage-gated: Boltzmann S_inf, exact exp integration
        ALPHA_BETA,       // voltage-gated: alpha/beta rate functions, exact exp integration
        CUSTOM_EXPR,      // VM bytecode dS_dt_vm (+ optional dA_dt_vm for 2-variable forms)
    };
    UpdateForm update_form = UpdateForm::TANH_GATE;

    enum class CurrentForm { LINEAR, MG_BLOCK, CUSTOM_EXPR };
    CurrentForm current_form = CurrentForm::LINEAR;

    // -------------------------------------------------------------------------
    // Shared conductance / reversal potential
    // -------------------------------------------------------------------------
    double g     = 0.1;
    double E_syn = 0.0;
    int    power = 1;       // S^power for LINEAR current (voltage-gated forms)

    // -------------------------------------------------------------------------
    // Initial state
    // -------------------------------------------------------------------------
    double S_init = 0.0;
    double A_init = 0.0;

    // -------------------------------------------------------------------------
    // EXP_DECAY / ALPHA_FUNC / DOUBLE_EXP / CUSTOM_EXPR: spike-jump increments
    // -------------------------------------------------------------------------
    double delta_S = 0.0;   // added to S on spike detection
    double delta_A = 0.0;   // added to A on spike detection

    // -------------------------------------------------------------------------
    // EXP_DECAY / DOUBLE_EXP: time constants and normalization
    // -------------------------------------------------------------------------
    double tau_S       = 10.0;  // S decay time constant (ms)
    double tau_A       = 1.0;   // A decay / rise time constant (ms)
    double norm_factor = 1.0;   // DOUBLE_EXP: 1/peak_conductance (normalizes peak to g*weight)

    // -------------------------------------------------------------------------
    // TANH_GATE: dS/dt = tanh_amp*(1+tanh((V-tanh_vh)/tanh_k))*(1-S) - S/tau_decay
    // -------------------------------------------------------------------------
    double tanh_amp = 2.0, tanh_vh = 0.0, tanh_k = 4.0, tau_decay = 13.0;

    // -------------------------------------------------------------------------
    // BOLTZMANN_GATE: dS/dt = (S_inf(V) - S) / tau(V)
    // -------------------------------------------------------------------------
    BoltzmannParams s_inf;
    TauParams       tau;

    // -------------------------------------------------------------------------
    // ALPHA_BETA: dS/dt = alpha(V)*(1-S) - beta(V)*S
    // -------------------------------------------------------------------------
    RateFuncParams alpha, beta;

    // -------------------------------------------------------------------------
    // MG_BLOCK (NMDA): voltage-dependent Mg2+ unblock
    // -------------------------------------------------------------------------
    double mg_conc  = 1.0;
    double mg_scale = 0.062;
    double mg_denom = 3.57;

    // -------------------------------------------------------------------------
    // CUSTOM_EXPR: VM bytecode
    //   dS_dt_vm  : PUSH_DEP = V_pre, PUSH_S = S, PUSH_A = A
    //   dA_dt_vm  : same mapping (2-variable novel forms; empty = 1-variable)
    //   current_vm: PUSH_DEP = V_post, PUSH_S = S, PUSH_A = A
    // -------------------------------------------------------------------------
    VmExpr dS_dt_vm;
    VmExpr dA_dt_vm;
    VmExpr current_vm;

    // -------------------------------------------------------------------------
    // Static preset factories
    // -------------------------------------------------------------------------

    // Spike-driven (EXP_DECAY / ALPHA_FUNC / DOUBLE_EXP)
    static SynapseSpec exponential(double tau_S, double g, double E_syn);
    static SynapseSpec alpha_function(double tau, double g, double E_syn);
    static SynapseSpec double_exponential(double tau_rise, double tau_decay,
                                          double g, double E_syn);

    // Voltage-gated kinetic presets
    static SynapseSpec gaba_kinetic();   // Kumaravelu 2016
    static SynapseSpec nmda_kinetic();   // NMDA + Mg2+ block
    static SynapseSpec gaba_b();         // slow GABA-B

    // Receptor presets (convenience wrappers)
    static SynapseSpec ampa();
    static SynapseSpec nmda();
    static SynapseSpec gaba_a();
};

} // namespace hodgkin_huxley
