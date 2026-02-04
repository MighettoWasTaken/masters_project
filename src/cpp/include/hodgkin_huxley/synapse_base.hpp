#pragma once

#include <string>
#include <cstddef>

namespace hodgkin_huxley {

/**
 * @brief Abstract base class for synapse models
 *
 * Provides a common interface for different synapse kinetics
 * (Exponential, Alpha, Double-Exponential). Spike detection is handled
 * by Network; each subclass implements its own conductance update logic.
 */
class SynapseBase {
public:
    virtual ~SynapseBase() = default;

    /**
     * @brief Update conductance after a timestep
     * @param dt Time step in milliseconds
     * @param spiked Whether the presynaptic neuron spiked this step
     */
    virtual void update(double dt, bool spiked) = 0;

    /**
     * @brief Reset synapse state to initial conditions
     */
    virtual void reset() = 0;

    /**
     * @brief Get synapse type name
     */
    [[nodiscard]] virtual std::string type_name() const = 0;

    [[nodiscard]] double conductance() const { return g_; }
    [[nodiscard]] double reversal_potential() const { return E_syn_; }
    [[nodiscard]] double weight() const { return weight_; }
    [[nodiscard]] size_t pre_idx() const { return pre_idx_; }
    [[nodiscard]] size_t post_idx() const { return post_idx_; }

protected:
    size_t pre_idx_ = 0;
    size_t post_idx_ = 0;
    double weight_ = 0.0;
    double E_syn_ = 0.0;
    double g_ = 0.0;
};

} // namespace hodgkin_huxley
