#include "hodgkin_huxley/neuron.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace hodgkin_huxley {

HHNeuron::HHNeuron() : params_(), state_() {
    method_ = IntegrationMethod::RK4;
    reset();
}

HHNeuron::HHNeuron(const Parameters& params) : params_(params), state_() {
    method_ = IntegrationMethod::RK4;
    reset();
}

HHNeuron::HHNeuron(const Parameters& params, IntegrationMethod method)
    : params_(params), state_() {
    method_ = method;
    reset();
}

void HHNeuron::reset() {
    state_.V = -65.0;
    state_.m = m_inf(state_.V);
    state_.h = h_inf(state_.V);
    state_.n = n_inf(state_.V);
}

// Rate functions for the gating variables
// These follow the original Hodgkin-Huxley formulation
// With safeguards against numerical overflow for extreme voltages

namespace {
    // Safe exponential that prevents overflow
    inline double safe_exp(double x) {
        constexpr double MAX_EXP_ARG = 700.0;  // exp(700) ~ 1e304, near double max
        if (x > MAX_EXP_ARG) return std::exp(MAX_EXP_ARG);
        if (x < -MAX_EXP_ARG) return 0.0;
        return std::exp(x);
    }
}

double HHNeuron::alpha_m(double V) {
    double dV = V + 40.0;
    if (std::abs(dV) < 1e-7) {
        return 1.0;  // L'Hopital's rule limit
    }
    double exp_term = safe_exp(-dV / 10.0);
    if (exp_term == 1.0) return 0.1 * 10.0;  // Limit as dV -> 0 from numerical issues
    return 0.1 * dV / (1.0 - exp_term);
}

double HHNeuron::beta_m(double V) {
    return 4.0 * safe_exp(-(V + 65.0) / 18.0);
}

double HHNeuron::alpha_h(double V) {
    return 0.07 * safe_exp(-(V + 65.0) / 20.0);
}

double HHNeuron::beta_h(double V) {
    return 1.0 / (1.0 + safe_exp(-(V + 35.0) / 10.0));
}

double HHNeuron::alpha_n(double V) {
    double dV = V + 55.0;
    if (std::abs(dV) < 1e-7) {
        return 0.1;  // L'Hopital's rule limit
    }
    double exp_term = safe_exp(-dV / 10.0);
    if (exp_term == 1.0) return 0.01 * 10.0;  // Limit as dV -> 0 from numerical issues
    return 0.01 * dV / (1.0 - exp_term);
}

double HHNeuron::beta_n(double V) {
    return 0.125 * safe_exp(-(V + 65.0) / 80.0);
}

// Steady-state activation functions
double HHNeuron::m_inf(double V) {
    double am = alpha_m(V);
    return am / (am + beta_m(V));
}

double HHNeuron::h_inf(double V) {
    double ah = alpha_h(V);
    return ah / (ah + beta_h(V));
}

double HHNeuron::n_inf(double V) {
    double an = alpha_n(V);
    return an / (an + beta_n(V));
}

void HHNeuron::compute_derivatives(double I_ext, double& dV, double& dm, double& dh, double& dn) const {
    const double V = state_.V;
    const double m = state_.m;
    const double h = state_.h;
    const double n = state_.n;

    // Ionic currents — use (n*n)*(n*n) to help compiler reuse n^2
    double n2 = n * n;
    double I_Na = params_.g_Na * m * m * m * h * (V - params_.E_Na);
    double I_K = params_.g_K * n2 * n2 * (V - params_.E_K);
    double I_L = params_.g_L * (V - params_.E_L);

    dV = (I_ext - I_Na - I_K - I_L) / params_.C_m;

    // Inline all rate functions to avoid function call overhead and enable
    // shared subexpression elimination. safe_exp removed: for V in [-100, +100]
    // all exp arguments are well within double range.
    double V_plus_65 = V + 65.0;

    // alpha_m
    double dV_m = V + 40.0;
    double am;
    if (std::abs(dV_m) < 1e-7) {
        am = 1.0;
    } else {
        am = 0.1 * dV_m / (1.0 - std::exp(-dV_m * 0.1));
    }
    // beta_m
    double bm = 4.0 * std::exp(-V_plus_65 / 18.0);

    // alpha_h
    double ah = 0.07 * std::exp(-V_plus_65 * 0.05);
    // beta_h
    double bh = 1.0 / (1.0 + std::exp(-(V + 35.0) * 0.1));

    // alpha_n
    double dV_n = V + 55.0;
    double an;
    if (std::abs(dV_n) < 1e-7) {
        an = 0.1;
    } else {
        an = 0.01 * dV_n / (1.0 - std::exp(-dV_n * 0.1));
    }
    // beta_n
    double bn = 0.125 * std::exp(-V_plus_65 * 0.0125);

    dm = am * (1.0 - m) - bm * m;
    dh = ah * (1.0 - h) - bh * h;
    dn = an * (1.0 - n) - bn * n;
}

void HHNeuron::euler_step(double dt, double I_ext) {
    double dV, dm, dh, dn;
    compute_derivatives(I_ext, dV, dm, dh, dn);

    state_.V += dt * dV;
    state_.m += dt * dm;
    state_.h += dt * dh;
    state_.n += dt * dn;

    // Clamp gating variables to [0, 1]
    state_.m = std::max(0.0, std::min(1.0, state_.m));
    state_.h = std::max(0.0, std::min(1.0, state_.h));
    state_.n = std::max(0.0, std::min(1.0, state_.n));
}

void HHNeuron::rk4_step(double dt, double I_ext) {
    double dV1, dm1, dh1, dn1;
    double dV2, dm2, dh2, dn2;
    double dV3, dm3, dh3, dn3;
    double dV4, dm4, dh4, dn4;

    // Pre-compute dt fractions
    const double dt_half = dt * 0.5;
    const double dt_sixth = dt / 6.0;

    // Helper to clamp gating variables to [0, 1]
    auto clamp_gates = [this]() {
        state_.m = std::max(0.0, std::min(1.0, state_.m));
        state_.h = std::max(0.0, std::min(1.0, state_.h));
        state_.n = std::max(0.0, std::min(1.0, state_.n));
    };

    State orig = state_;

    // k1
    compute_derivatives(I_ext, dV1, dm1, dh1, dn1);

    // k2
    state_.V = orig.V + dt_half * dV1;
    state_.m = orig.m + dt_half * dm1;
    state_.h = orig.h + dt_half * dh1;
    state_.n = orig.n + dt_half * dn1;
    clamp_gates();
    compute_derivatives(I_ext, dV2, dm2, dh2, dn2);

    // k3
    state_.V = orig.V + dt_half * dV2;
    state_.m = orig.m + dt_half * dm2;
    state_.h = orig.h + dt_half * dh2;
    state_.n = orig.n + dt_half * dn2;
    clamp_gates();
    compute_derivatives(I_ext, dV3, dm3, dh3, dn3);

    // k4
    state_.V = orig.V + dt * dV3;
    state_.m = orig.m + dt * dm3;
    state_.h = orig.h + dt * dh3;
    state_.n = orig.n + dt * dn3;
    clamp_gates();
    compute_derivatives(I_ext, dV4, dm4, dh4, dn4);

    // Combine
    state_.V = orig.V + dt_sixth * (dV1 + 2.0 * dV2 + 2.0 * dV3 + dV4);
    state_.m = orig.m + dt_sixth * (dm1 + 2.0 * dm2 + 2.0 * dm3 + dm4);
    state_.h = orig.h + dt_sixth * (dh1 + 2.0 * dh2 + 2.0 * dh3 + dh4);
    state_.n = orig.n + dt_sixth * (dn1 + 2.0 * dn2 + 2.0 * dn3 + dn4);

    // Final clamp
    clamp_gates();
}

void HHNeuron::step(double dt, double I_ext) {
    // Warn about potentially unstable timesteps
    static bool warned_euler = false;
    static bool warned_rk4 = false;

    if (method_ == IntegrationMethod::EULER && dt > 0.01 && !warned_euler) {
        std::cerr << "Warning: dt=" << dt << "ms may be unstable for Euler integration. "
                  << "Consider dt <= 0.01ms or use RK4.\n";
        warned_euler = true;
    } else if (method_ != IntegrationMethod::EULER && dt > 0.05 && !warned_rk4) {
        std::cerr << "Warning: dt=" << dt << "ms may be unstable for RK4 integration. "
                  << "Consider dt <= 0.05ms.\n";
        warned_rk4 = true;
    }

    switch (method_) {
        case IntegrationMethod::EULER:
            euler_step(dt, I_ext);
            break;
        case IntegrationMethod::RK4:
        case IntegrationMethod::RK45_ADAPTIVE:
            rk4_step(dt, I_ext);
            break;
    }
}

// simulate() methods are inherited from NeuronBase

} // namespace hodgkin_huxley
