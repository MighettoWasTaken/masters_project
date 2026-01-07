#include "hodgkin_huxley/solver.hpp"
#include <cmath>
#include <algorithm>

namespace hodgkin_huxley {

Solver::Solver() : options_() {}

Solver::Solver(const Options& options) : options_(options) {}

void Solver::euler_step(
    std::vector<double>& state,
    const std::function<std::vector<double>(const std::vector<double>&)>& derivatives,
    double dt
) {
    auto k = derivatives(state);
    for (size_t i = 0; i < state.size(); ++i) {
        state[i] += dt * k[i];
    }
}

void Solver::rk4_step(
    std::vector<double>& state,
    const std::function<std::vector<double>(const std::vector<double>&)>& derivatives,
    double dt
) {
    const size_t n = state.size();
    std::vector<double> temp(n);

    // k1
    auto k1 = derivatives(state);

    // k2
    for (size_t i = 0; i < n; ++i) {
        temp[i] = state[i] + 0.5 * dt * k1[i];
    }
    auto k2 = derivatives(temp);

    // k3
    for (size_t i = 0; i < n; ++i) {
        temp[i] = state[i] + 0.5 * dt * k2[i];
    }
    auto k3 = derivatives(temp);

    // k4
    for (size_t i = 0; i < n; ++i) {
        temp[i] = state[i] + dt * k3[i];
    }
    auto k4 = derivatives(temp);

    // Combine
    for (size_t i = 0; i < n; ++i) {
        state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

void Solver::step(
    std::vector<double>& state,
    const std::function<std::vector<double>(const std::vector<double>&)>& derivatives,
    double dt
) const {
    switch (options_.method) {
        case IntegrationMethod::EULER:
            euler_step(state, derivatives, dt);
            break;
        case IntegrationMethod::RK4:
            rk4_step(state, derivatives, dt);
            break;
        case IntegrationMethod::RK45_ADAPTIVE:
            // TODO: Implement adaptive RK45 (Dormand-Prince)
            // For now, fall back to RK4
            rk4_step(state, derivatives, dt);
            break;
    }
}

} // namespace hodgkin_huxley
