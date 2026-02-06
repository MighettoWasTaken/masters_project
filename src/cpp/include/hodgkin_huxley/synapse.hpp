#pragma once

#include "synapse_base.hpp"

namespace hodgkin_huxley {

/**
 * @brief Exponential decay synapse: g decays as exp(-t/tau) after spike
 */
class ExponentialSynapse : public SynapseBase {
public:
    ExponentialSynapse(size_t pre_idx, size_t post_idx, double weight,
                       double E_syn = 0.0, double tau = 2.0,
                       double delay = 0.0);

    void update(double dt, bool spiked) override;
    void reset() override;
    [[nodiscard]] std::string type_name() const override { return "Exponential"; }

    [[nodiscard]] double tau() const { return tau_; }

private:
    double tau_;
    double decay_factor_ = 0.0;   // cached exp(-dt/tau)
    double cached_dt_ = -1.0;     // dt used to compute cache
};

/**
 * @brief Alpha-function synapse: g(t) = g_peak * (t/tau) * exp(1 - t/tau)
 *
 * Implemented as an ODE system to avoid tracking individual spike times:
 *   dx/dt = -x/tau
 *   dg/dt = (x - g)/tau
 * On spike: x += weight * e
 * Peaks at g = weight when t = tau after a spike.
 */
class AlphaSynapse : public SynapseBase {
public:
    AlphaSynapse(size_t pre_idx, size_t post_idx, double weight,
                 double E_syn = 0.0, double tau = 2.0,
                 double delay = 0.0);

    void update(double dt, bool spiked) override;
    void reset() override;
    [[nodiscard]] std::string type_name() const override { return "Alpha"; }

    [[nodiscard]] double tau() const { return tau_; }

private:
    static constexpr double E_CONST = 2.718281828459045;  // exp(1)
    double tau_;
    double inv_tau_ = 0.0;  // cached 1.0/tau
    double x_ = 0.0;
};

/**
 * @brief Double-exponential synapse: g(t) = g_peak * f * (exp(-t/tau_d) - exp(-t/tau_r))
 *
 * Separate rise and decay time constants. Used for AMPA/NMDA with distinct
 * kinetics. Normalized so peak conductance equals weight.
 * Requires tau_rise < tau_decay.
 */
class DoubleExponentialSynapse : public SynapseBase {
public:
    DoubleExponentialSynapse(size_t pre_idx, size_t post_idx, double weight,
                             double E_syn = 0.0,
                             double tau_rise = 0.4, double tau_decay = 2.5,
                             double delay = 0.0);

    void update(double dt, bool spiked) override;
    void reset() override;
    [[nodiscard]] std::string type_name() const override { return "DoubleExponential"; }

    [[nodiscard]] double tau_rise() const { return tau_rise_; }
    [[nodiscard]] double tau_decay() const { return tau_decay_; }

private:
    double tau_rise_;
    double tau_decay_;
    double norm_factor_;
    double g_rise_ = 0.0;
    double g_decay_ = 0.0;
    double rise_decay_ = 0.0;   // cached exp(-dt/tau_rise)
    double fall_decay_ = 0.0;   // cached exp(-dt/tau_decay)
    double cached_dt_ = -1.0;   // dt used to compute cache
};

} // namespace hodgkin_huxley
