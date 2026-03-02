#pragma once

#include "neuron_base.hpp"
#include "ion_channels.hpp"
#include <vector>
#include <string>

namespace hodgkin_huxley {

class ComposableNeuron : public NeuronBase {
public:
    explicit ComposableNeuron(const NeuronModelSpec& spec);

    double membrane_potential() const override;
    void set_membrane_potential(double V) override;
    void reset() override;
    void step(double dt, double I_ext) override;
    std::string type_name() const override;

    const NeuronModelSpec& model_spec() const;
    const std::vector<double>& gate_states() const;
    double calcium() const;

    // Setters for pool sync
    void set_gate_states(const std::vector<double>& states);
    void set_calcium(double ca);
    void set_E_Ca(double e_ca);

    // Reset all gate variables to their steady-state values at the current V/Ca
    void reset_gates_to_steady_state();

private:
    NeuronModelSpec spec_;
    double V_;
    std::vector<double> gate_states_;
    double Ca_;
    double E_Ca_;

    // Helpers
    static double boltzmann(double x, const BoltzmannParams& p);
    static double compute_tau(double V, const TauParams& tau);
    static double compute_rate(double V, const RateFuncParams& rate);
    void update_gates(double dt);
    double compute_channel_current() const;
    void update_calcium(double dt);
};

} // namespace hodgkin_huxley
