#pragma once

#include "neuron_base.hpp"
#include <vector>
#include <cmath>

namespace hodgkin_huxley {

/**
 * @brief Hodgkin-Huxley neuron model
 *
 * Implements the classic Hodgkin-Huxley model with Na+, K+, and leak channels.
 * All units are in standard SI: mV, ms, uA/cm^2, mS/cm^2, uF/cm^2
 */
class HHNeuron : public NeuronBase {
public:
    // Default HH parameters (squid giant axon at 6.3°C)
    struct Parameters {
        double C_m = 1.0;       // Membrane capacitance (uF/cm^2)
        double g_Na = 120.0;    // Sodium conductance (mS/cm^2)
        double g_K = 36.0;      // Potassium conductance (mS/cm^2)
        double g_L = 0.3;       // Leak conductance (mS/cm^2)
        double E_Na = 50.0;     // Sodium reversal potential (mV)
        double E_K = -77.0;     // Potassium reversal potential (mV)
        double E_L = -54.387;   // Leak reversal potential (mV)
    };

    // State variables
    struct State {
        double V = -65.0;   // Membrane potential (mV)
        double m = 0.05;    // Na+ activation gate
        double h = 0.6;     // Na+ inactivation gate
        double n = 0.32;    // K+ activation gate
    };

    HHNeuron();
    explicit HHNeuron(const Parameters& params);
    HHNeuron(const Parameters& params, IntegrationMethod method);

    // =========================================================================
    // NeuronBase interface implementation
    // =========================================================================

    [[nodiscard]] double membrane_potential() const override { return state_.V; }
    void set_membrane_potential(double V) override { state_.V = V; }
    void reset() override;
    void step(double dt, double I_ext) override;
    [[nodiscard]] std::string type_name() const override { return "HH"; }

    [[nodiscard]] IntegrationMethod integration_method() const override { return method_; }
    void set_integration_method(IntegrationMethod method) override { method_ = method; }

    // =========================================================================
    // HH-specific interface
    // =========================================================================

    [[nodiscard]] const State& state() const { return state_; }
    [[nodiscard]] const Parameters& parameters() const { return params_; }

    void set_state(const State& state) { state_ = state; }
    void set_parameters(const Parameters& params) { params_ = params; }

    // Compute derivatives for the state variables
    void compute_derivatives(double I_ext, double& dV, double& dm, double& dh, double& dn) const;

private:
    Parameters params_;
    State state_;

    // Integration step implementations
    void euler_step(double dt, double I_ext);
    void rk4_step(double dt, double I_ext);

    // Rate functions (alpha and beta for each gate)
    [[nodiscard]] static double alpha_m(double V);
    [[nodiscard]] static double beta_m(double V);
    [[nodiscard]] static double alpha_h(double V);
    [[nodiscard]] static double beta_h(double V);
    [[nodiscard]] static double alpha_n(double V);
    [[nodiscard]] static double beta_n(double V);

    // Steady-state values and time constants
    [[nodiscard]] static double m_inf(double V);
    [[nodiscard]] static double h_inf(double V);
    [[nodiscard]] static double n_inf(double V);
};

} // namespace hodgkin_huxley
