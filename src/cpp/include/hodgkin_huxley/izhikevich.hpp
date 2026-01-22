#pragma once

#include "neuron_base.hpp"
#include <vector>

namespace hodgkin_huxley {

/**
 * @brief Izhikevich neuron model
 *
 * A computationally efficient neuron model that can reproduce many
 * biologically realistic spiking patterns with only 2 state variables.
 *
 * Equations:
 *   dv/dt = 0.04*v^2 + 5*v + 140 - u + I
 *   du/dt = a*(b*v - u)
 *
 *   if v >= 30 mV:
 *       v = c
 *       u = u + d
 *
 * Reference: Izhikevich (2003) "Simple Model of Spiking Neurons"
 */
class IzhikevichNeuron : public NeuronBase {
public:
    /**
     * @brief Model parameters
     *
     * Different parameter sets produce different spiking patterns:
     * - Regular Spiking (RS): a=0.02, b=0.2, c=-65, d=8
     * - Fast Spiking (FS): a=0.1, b=0.2, c=-65, d=2
     * - Intrinsically Bursting (IB): a=0.02, b=0.2, c=-55, d=4
     * - Chattering (CH): a=0.02, b=0.2, c=-50, d=2
     * - Low-threshold Spiking (LTS): a=0.02, b=0.25, c=-65, d=2
     */
    struct Parameters {
        double a = 0.02;    // Time scale of recovery variable u
        double b = 0.2;     // Sensitivity of u to subthreshold v
        double c = -65.0;   // After-spike reset value of v (mV)
        double d = 8.0;     // After-spike reset increment of u
    };

    /**
     * @brief State variables
     */
    struct State {
        double v = -65.0;   // Membrane potential (mV)
        double u = -13.0;   // Recovery variable (initialized to b*v)
    };

    /**
     * @brief Preset parameter configurations
     */
    enum class Type {
        REGULAR_SPIKING,        // Cortical excitatory (RS)
        FAST_SPIKING,           // Cortical inhibitory (FS)
        INTRINSICALLY_BURSTING, // Cortical excitatory (IB)
        CHATTERING,             // Cortical excitatory (CH)
        LOW_THRESHOLD_SPIKING,  // Cortical inhibitory (LTS)
        CUSTOM                  // User-defined parameters
    };

    // Constructors
    IzhikevichNeuron();
    explicit IzhikevichNeuron(Type type);
    explicit IzhikevichNeuron(const Parameters& params);

    // =========================================================================
    // NeuronBase interface implementation
    // =========================================================================

    [[nodiscard]] double membrane_potential() const override { return state_.v; }
    void set_membrane_potential(double V) override { state_.v = V; }
    void reset() override;
    void step(double dt, double I_ext) override;
    [[nodiscard]] std::string type_name() const override { return "Izhikevich"; }

    // Izhikevich typically uses Euler due to discontinuous spike reset
    [[nodiscard]] IntegrationMethod integration_method() const override { return IntegrationMethod::EULER; }
    void set_integration_method(IntegrationMethod method) override;

    // =========================================================================
    // Izhikevich-specific interface
    // =========================================================================

    [[nodiscard]] const State& state() const { return state_; }
    [[nodiscard]] const Parameters& parameters() const { return params_; }
    [[nodiscard]] double recovery_variable() const { return state_.u; }

    void set_state(const State& state) { state_ = state; }
    void set_parameters(const Parameters& params) { params_ = params; }
    void set_recovery_variable(double u) { state_.u = u; }

    /**
     * @brief Check if neuron just spiked in the last step
     * @return true if spike occurred
     */
    [[nodiscard]] bool spiked() const { return spiked_; }

    /**
     * @brief Get parameters for a preset neuron type
     */
    static Parameters get_preset(Type type);

private:
    Parameters params_;
    State state_;
    bool spiked_ = false;  // Track if spike occurred in last step

    static constexpr double SPIKE_THRESHOLD = 30.0;  // mV
};

} // namespace hodgkin_huxley
