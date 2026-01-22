#include "hodgkin_huxley/izhikevich.hpp"
#include <iostream>
#include <stdexcept>

namespace hodgkin_huxley {

IzhikevichNeuron::IzhikevichNeuron() : params_(), state_() {
    method_ = IntegrationMethod::EULER;
    reset();
}

IzhikevichNeuron::IzhikevichNeuron(Type type) : params_(get_preset(type)), state_() {
    method_ = IntegrationMethod::EULER;
    reset();
}

IzhikevichNeuron::IzhikevichNeuron(const Parameters& params) : params_(params), state_() {
    method_ = IntegrationMethod::EULER;
    reset();
}

IzhikevichNeuron::Parameters IzhikevichNeuron::get_preset(Type type) {
    Parameters p;
    switch (type) {
        case Type::REGULAR_SPIKING:
            p.a = 0.02; p.b = 0.2; p.c = -65.0; p.d = 8.0;
            break;
        case Type::FAST_SPIKING:
            p.a = 0.1; p.b = 0.2; p.c = -65.0; p.d = 2.0;
            break;
        case Type::INTRINSICALLY_BURSTING:
            p.a = 0.02; p.b = 0.2; p.c = -55.0; p.d = 4.0;
            break;
        case Type::CHATTERING:
            p.a = 0.02; p.b = 0.2; p.c = -50.0; p.d = 2.0;
            break;
        case Type::LOW_THRESHOLD_SPIKING:
            p.a = 0.02; p.b = 0.25; p.c = -65.0; p.d = 2.0;
            break;
        case Type::CUSTOM:
        default:
            // Default to regular spiking
            p.a = 0.02; p.b = 0.2; p.c = -65.0; p.d = 8.0;
            break;
    }
    return p;
}

void IzhikevichNeuron::reset() {
    state_.v = params_.c;  // Reset to after-spike value
    state_.u = params_.b * state_.v;  // u at steady state for given v
    spiked_ = false;
}

void IzhikevichNeuron::set_integration_method(IntegrationMethod method) {
    // Warn if trying to use RK4 - it doesn't help much with discontinuous reset
    if (method != IntegrationMethod::EULER) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "Warning: Izhikevich model uses Euler integration due to "
                      << "discontinuous spike reset. RK4 setting ignored.\n";
            warned = true;
        }
    }
    // Always use Euler for Izhikevich
    method_ = IntegrationMethod::EULER;
}

void IzhikevichNeuron::step(double dt, double I_ext) {
    spiked_ = false;

    // Check for spike first (from previous integration step)
    if (state_.v >= SPIKE_THRESHOLD) {
        state_.v = params_.c;
        state_.u += params_.d;
        spiked_ = true;
    }

    // Izhikevich equations:
    // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    // du/dt = a*(b*v - u)
    //
    // Note: The original model uses dt=1ms with specific scaling.
    // For smaller dt, we integrate properly.

    double v = state_.v;
    double u = state_.u;

    // Euler integration
    // Using the standard form which assumes time in ms
    double dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I_ext);
    double du = params_.a * (params_.b * v - u);

    state_.v += dt * dv;
    state_.u += dt * du;

    // Bound v to prevent numerical explosion (optional safety)
    if (state_.v > 100.0) {
        state_.v = 100.0;
    }
}

} // namespace hodgkin_huxley
