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
    const std::string& pool_key() const;
    const std::vector<double>& gate_states() const;

    // Substance accessors (backward compat: calcium = X_[0])
    double calcium() const { return X_.empty() ? 0.0 : X_[0]; }
    const std::vector<double>& substances() const { return X_; }

    // Setters for pool sync
    void set_gate_states(const std::vector<double>& states);
    void set_calcium(double ca)   { if (!X_.empty())        X_[0] = ca; }
    void set_E_Ca(double e_ca)    { if (!E_nernst_.empty()) E_nernst_[0] = e_ca; }
    void set_substances(const std::vector<double>& xs, const std::vector<double>& es);

    // Update spec (called by RegionalNetwork::update_population_spec)
    void set_model(const NeuronModelSpec& spec);

    // Reset all gate variables to their steady-state values at the current V/X
    void reset_gates_to_steady_state();

private:
    NeuronModelSpec spec_;
    std::string pool_key_;
    double V_;
    std::vector<double> gate_states_;
    std::vector<double> X_;        // substance concentrations (X_[0] = calcium by convention)
    std::vector<double> E_nernst_; // Nernst reversals, one per substance

    // Helpers
    void update_gates(double dt);
    double compute_channel_current() const;
    void update_intracellular(double dt);

    // Compute channel current for one channel (needed by update_intracellular)
    double compute_single_channel_current(int channel_idx) const;
};

} // namespace hodgkin_huxley
