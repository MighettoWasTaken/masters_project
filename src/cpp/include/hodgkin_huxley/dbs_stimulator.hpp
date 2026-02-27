#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
#include <cstddef>

namespace hodgkin_huxley {

/**
 * @brief Deep Brain Stimulation (DBS) pulse-train current generator.
 *
 * Generates periodic rectangular current pulses:
 *
 *   I(t) = amplitude   if  t mod (1000/frequency) < pulse_width
 *           0           otherwise
 *
 * Matches the benchmark creatdbs() implementation exactly:
 * pulses begin at t=0, spaced by round(ISI/dt) steps.
 */
class DBSStimulator {
public:
    struct Parameters {
        double frequency   = 130.0;  ///< Stimulation frequency in Hz (0 = off)
        double amplitude   = 0.0;    ///< Pulse amplitude in uA/cm^2
        double pulse_width = 0.06;   ///< Pulse width in ms
    };

    DBSStimulator() = default;
    explicit DBSStimulator(const Parameters& params);

    /**
     * @brief Generate full current trace for given duration and time step.
     *
     * Returns a vector of length ceil((duration + dt) / dt), matching the
     * np.arange(0, tmax+dt, dt) convention from the benchmark.
     *
     * @param duration  Total duration in ms.
     * @param dt        Time step in ms.
     */
    std::vector<double> generate(double duration, double dt) const;

    /**
     * @brief Compute current at a specific simulation step index.
     *
     * @param step_index  Zero-based step index.
     * @param dt          Time step in ms.
     */
    double current_at(size_t step_index, double dt) const;

    void set_parameters(const Parameters& params);
    const Parameters& parameters() const { return params_; }

private:
    Parameters params_;

    static void validate(const Parameters& p);
};

} // namespace hodgkin_huxley
