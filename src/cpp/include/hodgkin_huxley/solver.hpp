#pragma once

#include <vector>
#include <functional>

namespace hodgkin_huxley {

/**
 * @brief Numerical integration methods for ODE solving
 */
enum class IntegrationMethod {
    EULER,          // Forward Euler (1st order, fast but less accurate)
    RK4,            // Classic Runge-Kutta 4th order (good balance)
    RK45_ADAPTIVE   // Adaptive step size RK45 (Dormand-Prince)
};

/**
 * @brief ODE solver for Hodgkin-Huxley type equations
 *
 * Provides various integration methods with configurable accuracy.
 */
class Solver {
public:
    struct Options {
        IntegrationMethod method;
        double dt;
        double abs_tol;
        double rel_tol;
        double min_dt;
        double max_dt;

        Options()
            : method(IntegrationMethod::RK4)
            , dt(0.01)
            , abs_tol(1e-6)
            , rel_tol(1e-3)
            , min_dt(1e-6)
            , max_dt(1.0)
        {}
    };

    Solver();
    explicit Solver(const Options& options);

    const Options& options() const { return options_; }
    void set_options(const Options& options) { options_ = options; }

    /**
     * @brief Integrate a system of ODEs
     *
     * @param state Current state vector (modified in place)
     * @param derivatives Function that computes derivatives given state
     * @param dt Time step
     */
    void step(
        std::vector<double>& state,
        const std::function<std::vector<double>(const std::vector<double>&)>& derivatives,
        double dt
    ) const;

    /**
     * @brief Perform a single RK4 step
     */
    static void rk4_step(
        std::vector<double>& state,
        const std::function<std::vector<double>(const std::vector<double>&)>& derivatives,
        double dt
    );

    /**
     * @brief Perform a single Euler step
     */
    static void euler_step(
        std::vector<double>& state,
        const std::function<std::vector<double>(const std::vector<double>&)>& derivatives,
        double dt
    );

private:
    Options options_;
};

} // namespace hodgkin_huxley
