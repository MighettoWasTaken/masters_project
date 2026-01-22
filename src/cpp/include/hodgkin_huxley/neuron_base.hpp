#pragma once

#include <vector>
#include <string>

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
 * @brief Abstract base class for all neuron models
 *
 * Provides a common interface for different neuron implementations
 * (Hodgkin-Huxley, Izhikevich, etc.) enabling polymorphic networks
 * and consistent simulation interfaces.
 *
 * To implement a new neuron model:
 * 1. Inherit from NeuronBase
 * 2. Implement all pure virtual functions
 * 3. Optionally override simulate() if custom behavior needed
 *
 * For Python-level custom neurons (future):
 * - Use pybind11 trampoline class pattern
 * - Override step() with Python callback
 * - Performance note: Python callbacks will be slower than C++
 */
class NeuronBase {
public:
    virtual ~NeuronBase() = default;

    // =========================================================================
    // Core Interface (must be implemented by subclasses)
    // =========================================================================

    /**
     * @brief Get the current membrane potential
     * @return Membrane potential in mV
     */
    [[nodiscard]] virtual double membrane_potential() const = 0;

    /**
     * @brief Set the membrane potential directly
     * @param V New membrane potential in mV
     */
    virtual void set_membrane_potential(double V) = 0;

    /**
     * @brief Reset neuron to resting state
     */
    virtual void reset() = 0;

    /**
     * @brief Advance simulation by one timestep
     * @param dt Time step in milliseconds
     * @param I_ext External current in uA/cm^2
     */
    virtual void step(double dt, double I_ext) = 0;

    /**
     * @brief Get a string identifier for this neuron type
     * @return Type name (e.g., "HH", "Izhikevich")
     */
    [[nodiscard]] virtual std::string type_name() const = 0;

    // =========================================================================
    // Integration Method (optional override)
    // =========================================================================

    /**
     * @brief Get the current integration method
     * @return Integration method enum value
     */
    [[nodiscard]] virtual IntegrationMethod integration_method() const {
        return method_;
    }

    /**
     * @brief Set the integration method
     * @param method New integration method
     * @note Not all neuron types support all methods
     */
    virtual void set_integration_method(IntegrationMethod method) {
        method_ = method;
    }

    // =========================================================================
    // Simulation (default implementation using step())
    // =========================================================================

    /**
     * @brief Run simulation with constant current
     * @param duration Simulation duration in ms
     * @param dt Time step in ms
     * @param I_ext Constant external current in uA/cm^2
     * @return Vector of membrane potentials at each timestep
     */
    virtual std::vector<double> simulate(double duration, double dt, double I_ext);

    /**
     * @brief Run simulation with time-varying current
     * @param duration Simulation duration in ms
     * @param dt Time step in ms
     * @param I_ext Vector of external currents (one per timestep)
     * @return Vector of membrane potentials at each timestep
     */
    virtual std::vector<double> simulate(double duration, double dt,
                                         const std::vector<double>& I_ext);

protected:
    IntegrationMethod method_ = IntegrationMethod::RK4;
};

} // namespace hodgkin_huxley
