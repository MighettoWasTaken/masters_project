#pragma once

#include <string>
#include <cstddef>
#include <vector>
#include <cmath>

namespace hodgkin_huxley {

/**
 * @brief Abstract base class for synapse models
 *
 * Provides a common interface for different synapse kinetics
 * (Exponential, Alpha, Double-Exponential). Spike detection is handled
 * by Network; each subclass implements its own conductance update logic.
 *
 * Supports axonal conduction delays via an internal circular buffer.
 * When delay > 0, spikes are buffered and delivered after the specified
 * delay period. The buffer is lazily initialized on first use.
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

    /**
     * @brief Process a spike through the delay buffer
     *
     * If delay is 0, returns the spike immediately (pass-through).
     * Otherwise, pushes the spike into a circular buffer and returns
     * the oldest buffered value (the spike from delay_steps ago).
     *
     * @param spiked Whether a spike was detected this step
     * @param dt Time step in milliseconds (used for lazy buffer init)
     * @return The delayed spike value
     */
    bool process_spike(bool spiked, double dt) {
        if (delay_ == 0.0) return spiked;

        if (!delay_initialized_) {
            delay_steps_ = static_cast<size_t>(std::round(delay_ / dt));
            if (delay_steps_ == 0) return spiked;
            spike_buffer_.assign(delay_steps_, false);
            buffer_head_ = 0;
            delay_initialized_ = true;
        }

        // Read oldest entry, overwrite with current, advance head
        bool delayed = spike_buffer_[buffer_head_];
        spike_buffer_[buffer_head_] = spiked;
        buffer_head_ = (buffer_head_ + 1) % spike_buffer_.size();
        return delayed;
    }

    /**
     * @brief Reset the delay buffer to all-false
     */
    void reset_delay_buffer() {
        if (delay_initialized_) {
            std::fill(spike_buffer_.begin(), spike_buffer_.end(), false);
            buffer_head_ = 0;
        }
    }

    [[nodiscard]] double conductance() const { return g_; }
    [[nodiscard]] double reversal_potential() const { return E_syn_; }
    [[nodiscard]] double weight() const { return weight_; }
    [[nodiscard]] size_t pre_idx() const { return pre_idx_; }
    [[nodiscard]] size_t post_idx() const { return post_idx_; }
    [[nodiscard]] double delay() const { return delay_; }
    void set_conductance(double g) { g_ = g; }

protected:
    size_t pre_idx_ = 0;
    size_t post_idx_ = 0;
    double weight_ = 0.0;
    double E_syn_ = 0.0;
    double g_ = 0.0;
    double delay_ = 0.0;

private:
    std::vector<bool> spike_buffer_;
    size_t buffer_head_ = 0;
    size_t delay_steps_ = 0;
    bool delay_initialized_ = false;
};

} // namespace hodgkin_huxley
