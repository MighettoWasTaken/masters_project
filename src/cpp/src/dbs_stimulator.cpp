#include "hodgkin_huxley/dbs_stimulator.hpp"

namespace hodgkin_huxley {

// =============================================================================
// Validation
// =============================================================================

void DBSStimulator::validate(const Parameters& p) {
    if (p.frequency < 0.0) {
        throw std::invalid_argument(
            "DBSStimulator: frequency must be >= 0 Hz (0 = off), got " +
            std::to_string(p.frequency));
    }
    if (p.pulse_width <= 0.0) {
        throw std::invalid_argument(
            "DBSStimulator: pulse_width must be > 0 ms, got " +
            std::to_string(p.pulse_width));
    }
    if (p.frequency > 0.0) {
        double isi = 1000.0 / p.frequency;
        if (p.pulse_width >= isi) {
            throw std::invalid_argument(
                "DBSStimulator: pulse_width (" + std::to_string(p.pulse_width) +
                " ms) must be less than the inter-stimulus interval "
                "(1000 / frequency = " + std::to_string(isi) + " ms)");
        }
    }
}

// =============================================================================
// Construction / parameter update
// =============================================================================

DBSStimulator::DBSStimulator(const Parameters& params) {
    set_parameters(params);
}

void DBSStimulator::set_parameters(const Parameters& params) {
    validate(params);
    params_ = params;
}

// =============================================================================
// generate()  — full trace, matching benchmark creatdbs()
// =============================================================================

std::vector<double> DBSStimulator::generate(double duration, double dt) const {
    // Match benchmark: t = np.arange(0, tmax+dt, dt) → length ≈ tmax/dt + 1
    size_t n = static_cast<size_t>((duration + dt) / dt);
    std::vector<double> trace(n, 0.0);

    if (params_.frequency <= 0.0 || n == 0) return trace;

    size_t pulse_steps = static_cast<size_t>(params_.pulse_width / dt);
    if (pulse_steps == 0) pulse_steps = 1;

    double isi_ms  = 1000.0 / params_.frequency;
    size_t step_isi = static_cast<size_t>(std::round(isi_ms / dt));
    if (step_isi == 0) step_isi = 1;

    // Benchmark loop: i starts at 0, increments by round(ISI/dt)
    size_t i = 0;
    while (i < n - 1) {
        size_t plen = pulse_steps;
        if (i + plen > n) plen = n - i;
        for (size_t k = 0; k < plen; ++k) {
            trace[i + k] = params_.amplitude;
        }
        i += step_isi;
    }
    return trace;
}

// =============================================================================
// current_at()  — single step query using modular arithmetic
// =============================================================================

double DBSStimulator::current_at(size_t step_index, double dt) const {
    if (params_.frequency <= 0.0) return 0.0;

    size_t pulse_steps = static_cast<size_t>(params_.pulse_width / dt);
    if (pulse_steps == 0) pulse_steps = 1;

    double isi_ms  = 1000.0 / params_.frequency;
    size_t step_isi = static_cast<size_t>(std::round(isi_ms / dt));
    if (step_isi == 0) step_isi = 1;

    size_t phase = step_index % step_isi;
    return (phase < pulse_steps) ? params_.amplitude : 0.0;
}

} // namespace hodgkin_huxley
