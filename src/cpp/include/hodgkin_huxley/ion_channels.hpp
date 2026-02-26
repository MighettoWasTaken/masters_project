#pragma once

#include <string>
#include <vector>
#include <utility>
#include <cmath>

namespace hodgkin_huxley {

struct BoltzmannParams {
    double v_half = 0.0;
    double k = 1.0;
};

struct TauParams {
    enum class Form {
        CONSTANT,          // params[0] = tau_const
        BOLTZMANN,         // params[0] = tau_base, params[1] = tau_amp, params[2] = v_half, params[3] = k
        DOUBLE_EXP_SUM,    // params[0] = base, params[1] = a1, params[2] = v1, params[3] = s1, params[4] = a2, params[5] = v2, params[6] = s2
        OFFSET_DOUBLE_EXP, // params[0] = base, params[1] = a1, params[2] = v1, params[3] = s1, params[4] = a2, params[5] = v2, params[6] = s2
        SCALED_EXP,        // params[0] = scale, params[1] = v_half, params[2] = k (tau = scale / cosh((V-v_half)/(2*k)))
        COMPOUND_AB        // params[0..2] = alpha A,B,C; params[3..5] = beta A,B,C; tau = 1/(alpha+beta)
    };
    Form form = Form::CONSTANT;
    double params[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

struct RateFuncParams {
    enum class Form {
        LINEAR_OVER_EXP,   // alpha = A*(V+B) / (exp((V+B)/C) - 1)   [HH standard]
        EXP_DECAY,         // alpha = A * exp((V+B)/C)
        LINEAR_OVER_EXPM1, // alpha = A*(V+B) / (1 - exp(-(V+B)/C))  [equivalent rewrite]
        SIGMOID            // alpha = A / (1 + exp((V+B)/C))
    };
    Form form = Form::LINEAR_OVER_EXP;
    double A = 0.0, B = 0.0, C = 1.0;
};

struct GateSpec {
    enum class UpdateForm { INF_TAU, ALPHA_BETA, INSTANT, DERIVED };
    enum class Dependency { VOLTAGE, CALCIUM };

    std::string name;
    UpdateForm update_form = UpdateForm::INF_TAU;
    Dependency dependency = Dependency::VOLTAGE;
    double scale = 1.0;
    double initial_value = 0.0;

    // For INF_TAU and INSTANT forms
    BoltzmannParams inf;
    TauParams tau;

    // For ALPHA_BETA form
    RateFuncParams alpha;
    RateFuncParams beta;

    // For DERIVED form: X = a * (b + c * source_gate)
    int derived_source_gate = -1;
    double derived_a = 1.0;
    double derived_b = 0.0;
    double derived_c = 1.0;
};

struct ChannelSpec {
    std::string name;
    double g = 0.0;
    double E_rev = 0.0;
    bool use_calcium_nernst = false;
    std::vector<std::pair<int, int>> gates;  // (gate_index, power)
    bool is_ahp = false;
    double ahp_k1 = 0.0;
};

struct CalciumSpec {
    bool enabled = false;
    bool use_nernst = false;
    double epsilon = 1e-4;
    double K_Ca = 15.0;
    double Ca_init = 0.1;
    double Ca_o = 2000.0;  // extracellular Ca (uM)
    double z = 2.0;
    double F = 96485.0;    // Faraday constant
    double R = 8314.0;     // gas constant (mJ/(mol*K))
    double T = 298.0;      // temperature (K)
    std::vector<int> source_channels;  // channel indices contributing to Ca influx
};

struct NeuronModelSpec {
    std::string name;
    double C_m = 1.0;
    double V_init = -65.0;
    std::vector<GateSpec> gates;
    std::vector<ChannelSpec> channels;
    CalciumSpec calcium;

    // Preset factories
    static NeuronModelSpec thalamic();
    static NeuronModelSpec stn();
    static NeuronModelSpec gpe();
    static NeuronModelSpec gpi();
    static NeuronModelSpec striatum(double pd = 0.0);
};

// =============================================================================
// Free math helpers — shared between composable_neuron.cpp and network.cpp
// =============================================================================

inline double boltzmann_scalar(double x, const BoltzmannParams& p) {
    double arg = -(x - p.v_half) / p.k;
    if (arg > 500.0) return 0.0;
    if (arg < -500.0) return 1.0;
    return 1.0 / (1.0 + std::exp(arg));
}

inline double compute_tau_scalar(double V, const TauParams& p) {
    switch (p.form) {
        case TauParams::Form::CONSTANT:
            return p.params[0];
        case TauParams::Form::BOLTZMANN: {
            double arg = -(V - p.params[2]) / p.params[3];
            arg = std::max(-500.0, std::min(500.0, arg));
            return p.params[0] + p.params[1] / (1.0 + std::exp(arg));
        }
        case TauParams::Form::DOUBLE_EXP_SUM: {
            double e1 = std::exp((V + p.params[2]) / p.params[3]);
            double e2 = std::exp(-(V + p.params[5]) / p.params[6]);
            double denom = e1 + e2;
            if (denom < 1e-10) denom = 1e-10;
            return p.params[0] + p.params[1] / denom;
        }
        case TauParams::Form::OFFSET_DOUBLE_EXP: {
            double x1 = (V + p.params[2]) / p.params[3];
            double x2 = (V + p.params[5]) / p.params[6];
            return p.params[0] + p.params[1] * std::exp(-x1 * x1)
                               + p.params[4] * std::exp(-x2 * x2);
        }
        case TauParams::Form::SCALED_EXP: {
            double arg = (V - p.params[1]) / (2.0 * p.params[2]);
            arg = std::max(-500.0, std::min(500.0, arg));
            double ch = std::cosh(arg);
            if (ch < 1e-10) ch = 1e-10;
            return p.params[0] / ch;
        }
        case TauParams::Form::COMPOUND_AB: {
            double alpha = p.params[0] * std::exp((V + p.params[1]) / p.params[2]);
            double beta  = p.params[3] * std::exp((V + p.params[4]) / p.params[5]);
            double sum = alpha + beta;
            if (sum < 1e-10) sum = 1e-10;
            return 1.0 / sum;
        }
    }
    return 1.0;
}

inline double compute_rate_scalar(double V, const RateFuncParams& p) {
    switch (p.form) {
        case RateFuncParams::Form::LINEAR_OVER_EXP: {
            double x = V + p.B;
            double xc = x / p.C;
            if (std::abs(xc) < 1e-6) return p.A * p.C * (1.0 + xc * 0.5);
            return p.A * x / (std::exp(xc) - 1.0);
        }
        case RateFuncParams::Form::EXP_DECAY: {
            double arg = (V + p.B) / p.C;
            arg = std::max(-500.0, std::min(500.0, arg));
            return p.A * std::exp(arg);
        }
        case RateFuncParams::Form::LINEAR_OVER_EXPM1: {
            double x = V + p.B;
            double xc = x / p.C;
            if (std::abs(xc) < 1e-6) return p.A * p.C * (1.0 + xc * 0.5);
            return p.A * x / (1.0 - std::exp(-xc));
        }
        case RateFuncParams::Form::SIGMOID: {
            double arg = (V + p.B) / p.C;
            arg = std::max(-500.0, std::min(500.0, arg));
            return p.A / (1.0 + std::exp(arg));
        }
    }
    return 0.0;
}

// =============================================================================
// KineticSynapseSpec — composable kinetic synapse
// =============================================================================

struct KineticSynapseSpec {
    std::string name;

    enum class UpdateForm { ALPHA_BETA, TANH_GATE, BOLTZMANN_GATE };
    UpdateForm update_form = UpdateForm::TANH_GATE;

    // ALPHA_BETA: dS/dt = alpha(V)*(1-S) - beta(V)*S
    RateFuncParams alpha, beta;

    // TANH_GATE: dS/dt = tanh_amp*(1+tanh((V-tanh_vh)/tanh_k))*(1-S) - S/tau_decay
    double tanh_amp = 2.0, tanh_vh = 0.0, tanh_k = 4.0, tau_decay = 13.0;

    // BOLTZMANN_GATE: dS/dt = (S_inf(V) - S) / tau(V)
    BoltzmannParams s_inf;
    TauParams tau;

    enum class CurrentForm { LINEAR, MG_BLOCK };
    CurrentForm current_form = CurrentForm::LINEAR;

    double g = 0.1, E_syn = -80.0;
    int power = 1;

    // MG_BLOCK (NMDA)
    double mg_conc = 1.0, mg_scale = 0.062, mg_denom = 3.57;

    double S_init = 0.0;

    static KineticSynapseSpec gaba_kinetic();  // Kumaravelu 2016
    static KineticSynapseSpec nmda_kinetic();  // NMDA + Mg block
    static KineticSynapseSpec gaba_b();        // slow GABA-B
};

} // namespace hodgkin_huxley
