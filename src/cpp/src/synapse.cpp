#include "hodgkin_huxley/synapse.hpp"
#include <cmath>
#include <stdexcept>

namespace hodgkin_huxley {

// =============================================================================
// ExponentialSynapse
// =============================================================================

ExponentialSynapse::ExponentialSynapse(size_t pre_idx, size_t post_idx,
                                       double weight, double E_syn, double tau)
    : tau_(tau)
{
    pre_idx_ = pre_idx;
    post_idx_ = post_idx;
    weight_ = weight;
    E_syn_ = E_syn;
    g_ = 0.0;
}

void ExponentialSynapse::update(double dt, bool spiked) {
    if (spiked) {
        g_ += weight_;
    }
    g_ *= std::exp(-dt / tau_);
}

void ExponentialSynapse::reset() {
    g_ = 0.0;
}

// =============================================================================
// AlphaSynapse
// =============================================================================

AlphaSynapse::AlphaSynapse(size_t pre_idx, size_t post_idx,
                           double weight, double E_syn, double tau)
    : tau_(tau), x_(0.0)
{
    pre_idx_ = pre_idx;
    post_idx_ = post_idx;
    weight_ = weight;
    E_syn_ = E_syn;
    g_ = 0.0;
}

void AlphaSynapse::update(double dt, bool spiked) {
    if (spiked) {
        x_ += weight_ * std::exp(1.0);
    }

    double dx = -x_ / tau_;
    double dg = (x_ - g_) / tau_;

    x_ += dt * dx;
    g_ += dt * dg;

    if (g_ < 0.0) g_ = 0.0;
}

void AlphaSynapse::reset() {
    g_ = 0.0;
    x_ = 0.0;
}

// =============================================================================
// DoubleExponentialSynapse
// =============================================================================

DoubleExponentialSynapse::DoubleExponentialSynapse(
    size_t pre_idx, size_t post_idx,
    double weight, double E_syn,
    double tau_rise, double tau_decay)
    : tau_rise_(tau_rise), tau_decay_(tau_decay),
      g_rise_(0.0), g_decay_(0.0)
{
    if (tau_rise >= tau_decay) {
        throw std::invalid_argument(
            "DoubleExponentialSynapse: tau_rise must be less than tau_decay");
    }

    pre_idx_ = pre_idx;
    post_idx_ = post_idx;
    weight_ = weight;
    E_syn_ = E_syn;
    g_ = 0.0;

    // Normalization so peak conductance = weight
    double t_peak = (tau_decay_ * tau_rise_) / (tau_decay_ - tau_rise_)
                    * std::log(tau_decay_ / tau_rise_);
    double peak_val = std::exp(-t_peak / tau_decay_) - std::exp(-t_peak / tau_rise_);
    norm_factor_ = 1.0 / peak_val;
}

void DoubleExponentialSynapse::update(double dt, bool spiked) {
    if (spiked) {
        g_rise_ += 1.0;
        g_decay_ += 1.0;
    }

    g_rise_ *= std::exp(-dt / tau_rise_);
    g_decay_ *= std::exp(-dt / tau_decay_);

    g_ = weight_ * norm_factor_ * (g_decay_ - g_rise_);

    if (g_ < 0.0) g_ = 0.0;
}

void DoubleExponentialSynapse::reset() {
    g_ = 0.0;
    g_rise_ = 0.0;
    g_decay_ = 0.0;
}

} // namespace hodgkin_huxley
