#include "hodgkin_huxley/iz_pool.hpp"
#include <cstring>

namespace hodgkin_huxley {

// =============================================================================
// Construction — pre-allocate all buffers
// =============================================================================

IzPool::IzPool(size_t capacity) : N_(0) {
    net_idx_.reserve(capacity);
    v_.resize(capacity);  u_.resize(capacity);
    a_.resize(capacity);  b_.resize(capacity);
    c_.resize(capacity);  d_.resize(capacity);
    I_ext_.resize(capacity);
    dv_.resize(capacity); du_.resize(capacity);
}

void IzPool::add(size_t network_idx, const IzhikevichNeuron::Parameters& params,
                 const IzhikevichNeuron::State& state) {
    size_t i = N_++;
    net_idx_.push_back(network_idx);

    // Track whether indices are contiguous (enables memcpy fast path)
    if (i == 0) contiguous_ = true;
    else if (contiguous_ && network_idx != net_idx_[i - 1] + 1) contiguous_ = false;

    v_(i) = state.v;  u_(i) = state.u;
    a_(i) = params.a;  b_(i) = params.b;
    c_(i) = params.c;  d_(i) = params.d;
}

// =============================================================================
// Scatter / gather
// =============================================================================

void IzPool::scatter_voltages(double* V_buf) const {
    if (contiguous_ && N_ > 0) {
        std::memcpy(V_buf + net_idx_[0], v_.data(), N_ * sizeof(double));
    } else {
        for (size_t i = 0; i < N_; ++i)
            V_buf[net_idx_[i]] = v_(i);
    }
}

void IzPool::scatter_recoveries(double* u_buf, size_t n_rec, size_t tr) const {
    for (size_t i = 0; i < N_; ++i)
        u_buf[net_idx_[i] * n_rec + tr] = u_(i);
}

void IzPool::gather_currents(const double* I_buf) {
    if (contiguous_ && N_ > 0) {
        std::memcpy(I_ext_.data(), I_buf + net_idx_[0], N_ * sizeof(double));
    } else {
        for (size_t i = 0; i < N_; ++i)
            I_ext_(i) = I_buf[net_idx_[i]];
    }
}

// =============================================================================
// Batched Euler step with branchless spike reset — zero allocation
// =============================================================================

void IzPool::step_euler(double dt) {
    if (N_ == 0) return;

    // Branchless spike detection via mask.
    // .eval() materializes the boolean array so modifying v_ below
    // does not invalidate the mask (Eigen lazy-expression gotcha).
    auto spiked = (v_ >= SPIKE_THRESHOLD).eval();

    // Reset spiked neurons
    v_ = spiked.select(c_, v_);
    u_ = spiked.select(u_ + d_, u_);

    // Izhikevich ODE — write to pre-allocated dv_/du_ buffers
    dv_ = 0.04 * v_ * v_ + 5.0 * v_ + 140.0 - u_ + I_ext_;
    du_ = a_ * (b_ * v_ - u_);

    v_ += dt * dv_;
    u_ += dt * du_;

    // Safety clamp
    v_ = v_.min(100.0);
}

// =============================================================================
// Per-group subset ops for Phase 2 parallelism (scalar loop over local indices)
// =============================================================================

void IzPool::gather_currents_subset(const std::vector<size_t>& local_indices,
                                     const double* I_buf) {
    for (size_t li : local_indices)
        I_ext_(li) = I_buf[net_idx_[li]];
}

void IzPool::step_subset(const std::vector<size_t>& local_indices, double dt) {
    for (size_t li : local_indices) {
        double v = v_(li), u = u_(li);
        if (v >= SPIKE_THRESHOLD) {
            v  = c_(li);
            u += d_(li);
        }
        double dv = 0.04*v*v + 5.0*v + 140.0 - u + I_ext_(li);
        double du = a_(li) * (b_(li)*v - u);
        v = std::min(v + dt * dv, 100.0);
        u = u + dt * du;
        v_(li) = v;
        u_(li) = u;
    }
}

void IzPool::scatter_voltages_subset(const std::vector<size_t>& local_indices,
                                      double* V_buf) const {
    for (size_t li : local_indices)
        V_buf[net_idx_[li]] = v_(li);
}

void IzPool::scatter_recoveries_subset(const std::vector<size_t>& local_indices,
                                        double* u_buf, size_t n_rec, size_t tr) const {
    for (size_t li : local_indices)
        u_buf[net_idx_[li] * n_rec + tr] = u_(li);
}

// =============================================================================
// Sync back to polymorphic API objects
// =============================================================================

void IzPool::sync_to_neurons(
    std::vector<std::unique_ptr<NeuronBase>>& neurons) const
{
    for (size_t i = 0; i < N_; ++i) {
        IzhikevichNeuron* iz =
            static_cast<IzhikevichNeuron*>(neurons[net_idx_[i]].get());
        IzhikevichNeuron::State s;
        s.v = v_(i);  s.u = u_(i);
        iz->set_state(s);
    }
}

} // namespace hodgkin_huxley
