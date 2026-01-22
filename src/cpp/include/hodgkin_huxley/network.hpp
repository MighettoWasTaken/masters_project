#pragma once

#include "neuron_base.hpp"
#include "neuron.hpp"
#include "izhikevich.hpp"
#include <vector>
#include <memory>
#include <string>

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
 * @brief Network of neurons with polymorphic neuron support
 *
 * Allows simulation of interconnected neurons (HH, Izhikevich, or mixed)
 * with synaptic connections.
 */
class Network {
public:
    /**
     * @brief Enum for neuron types when adding neurons
     */
    enum class NeuronType {
        HH,
        IZHIKEVICH_RS,
        IZHIKEVICH_FS,
        IZHIKEVICH_IB,
        IZHIKEVICH_CH,
        IZHIKEVICH_LTS,
        IZHIKEVICH_CUSTOM
    };

    Network() = default;

    /**
     * @brief Create network with N HH neurons (backward compatible)
     */
    explicit Network(size_t num_neurons);

    /**
     * @brief Create network with N neurons of specified type
     */
    Network(size_t num_neurons, NeuronType type);

    // Add neurons
    size_t add_neuron();  // Add default HH neuron
    size_t add_neuron(const HHNeuron::Parameters& params);
    size_t add_neuron(NeuronType type);
    size_t add_neuron(const IzhikevichNeuron::Parameters& params);

    /**
     * @brief Add a neuron with explicit type specification
     */
    size_t add_hh_neuron();
    size_t add_hh_neuron(const HHNeuron::Parameters& params);
    size_t add_izhikevich_neuron(IzhikevichNeuron::Type type = IzhikevichNeuron::Type::REGULAR_SPIKING);
    size_t add_izhikevich_neuron(const IzhikevichNeuron::Parameters& params);

    // Add synaptic connection
    void add_synapse(size_t pre_idx, size_t post_idx, double weight,
                     double E_syn = 0.0, double tau = 2.0);

    // Getters
    [[nodiscard]] size_t num_neurons() const { return neurons_.size(); }
    [[nodiscard]] size_t num_synapses() const { return synapses_.size(); }

    /**
     * @brief Get neuron by index (polymorphic access)
     */
    [[nodiscard]] const NeuronBase& neuron(size_t idx) const { return *neurons_[idx]; }
    [[nodiscard]] NeuronBase& neuron(size_t idx) { return *neurons_[idx]; }

    /**
     * @brief Get neuron as HH (throws if wrong type)
     */
    [[nodiscard]] const HHNeuron& hh_neuron(size_t idx) const;
    [[nodiscard]] HHNeuron& hh_neuron(size_t idx);

    /**
     * @brief Get neuron as Izhikevich (throws if wrong type)
     */
    [[nodiscard]] const IzhikevichNeuron& iz_neuron(size_t idx) const;
    [[nodiscard]] IzhikevichNeuron& iz_neuron(size_t idx);

    /**
     * @brief Get neuron type name
     */
    [[nodiscard]] std::string neuron_type(size_t idx) const { return neurons_[idx]->type_name(); }

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
    std::vector<std::unique_ptr<NeuronBase>> neurons_;
    std::vector<Synapse> synapses_;

    // Compute synaptic currents for each neuron
    [[nodiscard]] std::vector<double> compute_synaptic_currents() const;

    // Update synapse states
    void update_synapses(double dt);
};

} // namespace hodgkin_huxley
