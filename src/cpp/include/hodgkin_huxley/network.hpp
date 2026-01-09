#pragma once

#include "neuron.hpp"
#include <vector>
#include <memory>

namespace hodgkin_huxley {

/**
 * @brief Synapse model for connecting neurons
 */
struct Synapse {
    size_t pre_idx;         // Pre-synaptic neuron index
    size_t post_idx;        // Post-synaptic neuron index
    double weight;          // Synaptic weight (conductance)
    double E_syn;           // Synaptic reversal potential (mV)
    double tau;             // Synaptic time constant (ms)

    // Synapse state
    double g = 0.0;         // Current synaptic conductance
    double V_pre_prev = -65.0;  // Previous presynaptic voltage (for spike detection)
};

/**
 * @brief Network of Hodgkin-Huxley neurons
 *
 * Allows simulation of interconnected neurons with synaptic connections.
 */
class Network {
public:
    Network() = default;
    explicit Network(size_t num_neurons);

    // Add neurons
    size_t add_neuron();
    size_t add_neuron(const HHNeuron::Parameters& params);

    // Add synaptic connection
    void add_synapse(size_t pre_idx, size_t post_idx, double weight,
                     double E_syn = 0.0, double tau = 2.0);

    // Getters
    [[nodiscard]] size_t num_neurons() const { return neurons_.size(); }
    [[nodiscard]] size_t num_synapses() const { return synapses_.size(); }
    [[nodiscard]] const HHNeuron& neuron(size_t idx) const { return neurons_[idx]; }
    [[nodiscard]] HHNeuron& neuron(size_t idx) { return neurons_[idx]; }

    // Get all membrane potentials
    [[nodiscard]] std::vector<double> get_potentials() const;

    // Reset all neurons
    void reset();

    // Step the entire network
    void step(double dt, const std::vector<double>& I_ext);

    // Simulate network, returns matrix of voltage traces (neurons x time)
    std::vector<std::vector<double>> simulate(
        double duration,
        double dt,
        const std::vector<std::vector<double>>& I_ext
    );

private:
    std::vector<HHNeuron> neurons_;
    std::vector<Synapse> synapses_;

    // Compute synaptic currents for each neuron
    [[nodiscard]] std::vector<double> compute_synaptic_currents() const;

    // Update synapse states
    void update_synapses(double dt);
};

} // namespace hodgkin_huxley
