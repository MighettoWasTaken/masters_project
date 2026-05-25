#include "hodgkin_huxley/hh_pool.hpp"
#include "hodgkin_huxley/model/kinetics.hpp"
#include <cstring>

namespace hodgkin_huxley {

// =============================================================================
// Construction — pre-allocate every buffer so the hot loop does 0 mallocs
// =============================================================================

HHPool::HHPool(size_t capacity, bool fast_math) : N_(0), fast_math_(fast_math) {
    net_idx_.reserve(capacity);

    // State
    V_.resize(capacity);  m_.resize(capacity);
    h_.resize(capacity);  n_.resize(capacity);

    // Parameters
    C_m_.resize(capacity);  g_Na_.resize(capacity); g_K_.resize(capacity);
    g_L_.resize(capacity);  E_Na_.resize(capacity); E_K_.resize(capacity);
    E_L_.resize(capacity);

    // Current input
    I_ext_.resize(capacity);

    // RK4 working buffers
    kV_.resize(capacity);   km_.resize(capacity);
    kh_.resize(capacity);   kn_.resize(capacity);
    accV_.resize(capacity); accm_.resize(capacity);
    acch_.resize(capacity); accn_.resize(capacity);
    V0_.resize(capacity);   m0_.resize(capacity);
    h0_.resize(capacity);   n0_.resize(capacity);

    // Derivative intermediates
    tmp_n2_.resize(capacity);   tmp_Vp65_.resize(capacity);
    tmp_dVm_.resize(capacity);  tmp_dVn_.resize(capacity);

    // Rate function buffers
    tmp_am_.resize(capacity);  tmp_bm_.resize(capacity);
    tmp_ah_.resize(capacity);  tmp_bh_.resize(capacity);
    tmp_an_.resize(capacity);  tmp_bn_.resize(capacity);

    // Fast exp working buffer
    tmp_exp_r_.resize(capacity);
}

HHPool::~HHPool() = default;

void HHPool::add(size_t network_idx, const HHNeuron::Parameters& params,
                 const HHNeuron::State& state) {
    size_t i = N_++;
    net_idx_.push_back(network_idx);

    // Track whether indices are contiguous (enables memcpy fast path)
    if (i == 0) contiguous_ = true;
    else if (contiguous_ && network_idx != net_idx_[i - 1] + 1) contiguous_ = false;

    V_(i) = state.V;  m_(i) = state.m;
    h_(i) = state.h;  n_(i) = state.n;

    C_m_(i)  = params.C_m;  g_Na_(i) = params.g_Na;
    g_K_(i)  = params.g_K;  g_L_(i)  = params.g_L;
    E_Na_(i) = params.E_Na; E_K_(i)  = params.E_K;
    E_L_(i)  = params.E_L;
}

// =============================================================================
// Scatter / gather
// =============================================================================

void HHPool::scatter_voltages(double* V_buf) const {
    if (contiguous_ && N_ > 0) {
        std::memcpy(V_buf + net_idx_[0], V_.data(), N_ * sizeof(double));
    } else {
        for (size_t i = 0; i < N_; ++i)
            V_buf[net_idx_[i]] = V_(i);
    }
}

void HHPool::gather_currents(const double* I_buf) {
    if (contiguous_ && N_ > 0) {
        std::memcpy(I_ext_.data(), I_buf + net_idx_[0], N_ * sizeof(double));
    } else {
        for (size_t i = 0; i < N_; ++i)
            I_ext_(i) = I_buf[net_idx_[i]];
    }
}

// Delegate to the shared free function in model/kinetics.hpp
void HHPool::fast_exp(const Eigen::ArrayXd& src, Eigen::ArrayXd& dst) {
    hodgkin_huxley::fast_exp(src, dst, tmp_exp_r_);
}

// =============================================================================
// Vectorized HH derivative computation — zero allocation
//
// Multi-use intermediates (n², V+65, V+40, V+55) write to pre-allocated
// tmp_ buffers.  Rate functions are materialized into tmp_am_ .. tmp_bn_.
// When fast_math_ is true, uses polynomial fast_exp (~8 digits);
// otherwise uses Eigen's built-in exp (full precision).
// =============================================================================

void HHPool::compute_derivatives(
    const Eigen::ArrayXd& V, const Eigen::ArrayXd& m,
    const Eigen::ArrayXd& h, const Eigen::ArrayXd& n,
    const Eigen::ArrayXd& I_ext,
    Eigen::ArrayXd& dV, Eigen::ArrayXd& dm,
    Eigen::ArrayXd& dh, Eigen::ArrayXd& dn)
{
    // Materialise intermediates used >1 time into pre-allocated buffers
    tmp_n2_   = n * n;
    tmp_Vp65_ = V + 65.0;
    tmp_dVm_  = V + 40.0;
    tmp_dVn_  = V + 55.0;

    // Ionic currents — fused into a single pass writing to dV
    dV = (I_ext
          - g_Na_ * m.cube() * h * (V - E_Na_)
          - g_K_ * tmp_n2_ * tmp_n2_ * (V - E_K_)
          - g_L_ * (V - E_L_)) / C_m_;

    // fast_exp wins at large N (amortized over many elements) but has too
    // much loop overhead for tiny arrays.  Threshold ~64 = 8 AVX2 registers.
    if (fast_math_ && N_ > 64) {
        // --- Fast polynomial exp (~8 significant digits) ---

        // beta_m: 4 * exp(-(V+65)/18)
        tmp_bm_ = -tmp_Vp65_ / 18.0;
        fast_exp(tmp_bm_, tmp_bm_);
        tmp_bm_ *= 4.0;

        // alpha_h: 0.07 * exp(-0.05*(V+65))
        tmp_ah_ = -tmp_Vp65_ * 0.05;
        fast_exp(tmp_ah_, tmp_ah_);
        tmp_ah_ *= 0.07;

        // beta_h: 1 / (1 + exp(-0.1*(V+35)))
        tmp_bh_ = -(V + 35.0) * 0.1;
        fast_exp(tmp_bh_, tmp_bh_);
        tmp_bh_ = 1.0 / (1.0 + tmp_bh_);

        // beta_n: 0.125 * exp(-0.0125*(V+65))
        tmp_bn_ = -tmp_Vp65_ * 0.0125;
        fast_exp(tmp_bn_, tmp_bn_);
        tmp_bn_ *= 0.125;

        // alpha_m: 0.1*(V+40) / (1 - exp(-0.1*(V+40))), singularity at V=-40
        tmp_am_ = -tmp_dVm_ * 0.1;
        fast_exp(tmp_am_, tmp_am_);
        tmp_am_ = (tmp_dVm_.abs() < 1e-7).select(
            Eigen::ArrayXd::Ones(N_),
            0.1 * tmp_dVm_ / (1.0 - tmp_am_));

        // alpha_n: 0.01*(V+55) / (1 - exp(-0.1*(V+55))), singularity at V=-55
        tmp_an_ = -tmp_dVn_ * 0.1;
        fast_exp(tmp_an_, tmp_an_);
        tmp_an_ = (tmp_dVn_.abs() < 1e-7).select(
            Eigen::ArrayXd::Constant(N_, 0.1),
            0.01 * tmp_dVn_ / (1.0 - tmp_an_));
    } else {
        // --- Full-precision Eigen exp ---

        tmp_am_ = (tmp_dVm_.abs() < 1e-7).select(
            Eigen::ArrayXd::Ones(N_),
            0.1 * tmp_dVm_ / (1.0 - (-tmp_dVm_ * 0.1).exp()));

        tmp_bm_ = 4.0 * (-tmp_Vp65_ / 18.0).exp();
        tmp_ah_ = 0.07 * (-tmp_Vp65_ * 0.05).exp();
        tmp_bh_ = 1.0 / (1.0 + (-(V + 35.0) * 0.1).exp());

        tmp_an_ = (tmp_dVn_.abs() < 1e-7).select(
            Eigen::ArrayXd::Constant(N_, 0.1),
            0.01 * tmp_dVn_ / (1.0 - (-tmp_dVn_ * 0.1).exp()));

        tmp_bn_ = 0.125 * (-tmp_Vp65_ * 0.0125).exp();
    }

    dm = tmp_am_ * (1.0 - m) - tmp_bm_ * m;
    dh = tmp_ah_ * (1.0 - h) - tmp_bh_ * h;
    dn = tmp_an_ * (1.0 - n) - tmp_bn_ * n;
}

// =============================================================================
// Batched RK4 — accumulator pattern, zero allocation
//
// Instead of storing all four k-vectors (16 arrays), we use a running
// accumulator: acc = k1, then += 2*k2, += 2*k3, += k4.
// Only needs: 4 k + 4 acc + 4 saved = 12 pre-allocated arrays.
// =============================================================================

void HHPool::step_rk4(double dt) {
    if (N_ == 0) return;

    const double dt_half  = dt * 0.5;
    const double dt_sixth = dt / 6.0;

    // Save original state (writes to pre-allocated buffers, no alloc)
    V0_ = V_;  m0_ = m_;  h0_ = h_;  n0_ = n_;

    // --- k1 ---
    compute_derivatives(V_, m_, h_, n_, I_ext_, kV_, km_, kh_, kn_);
    accV_ = kV_;  accm_ = km_;  acch_ = kh_;  accn_ = kn_;

    // --- k2 (midpoint from k1) ---
    V_ = V0_ + dt_half * kV_;
    m_ = (m0_ + dt_half * km_).max(0.0).min(1.0);
    h_ = (h0_ + dt_half * kh_).max(0.0).min(1.0);
    n_ = (n0_ + dt_half * kn_).max(0.0).min(1.0);
    compute_derivatives(V_, m_, h_, n_, I_ext_, kV_, km_, kh_, kn_);
    accV_ += 2.0 * kV_;  accm_ += 2.0 * km_;
    acch_ += 2.0 * kh_;  accn_ += 2.0 * kn_;

    // --- k3 (midpoint from k2) ---
    V_ = V0_ + dt_half * kV_;
    m_ = (m0_ + dt_half * km_).max(0.0).min(1.0);
    h_ = (h0_ + dt_half * kh_).max(0.0).min(1.0);
    n_ = (n0_ + dt_half * kn_).max(0.0).min(1.0);
    compute_derivatives(V_, m_, h_, n_, I_ext_, kV_, km_, kh_, kn_);
    accV_ += 2.0 * kV_;  accm_ += 2.0 * km_;
    acch_ += 2.0 * kh_;  accn_ += 2.0 * kn_;

    // --- k4 (endpoint from k3) ---
    V_ = V0_ + dt * kV_;
    m_ = (m0_ + dt * km_).max(0.0).min(1.0);
    h_ = (h0_ + dt * kh_).max(0.0).min(1.0);
    n_ = (n0_ + dt * kn_).max(0.0).min(1.0);
    compute_derivatives(V_, m_, h_, n_, I_ext_, kV_, km_, kh_, kn_);
    accV_ += kV_;  accm_ += km_;  acch_ += kh_;  accn_ += kn_;

    // --- Combine ---
    V_ = V0_ + dt_sixth * accV_;
    m_ = (m0_ + dt_sixth * accm_).max(0.0).min(1.0);
    h_ = (h0_ + dt_sixth * acch_).max(0.0).min(1.0);
    n_ = (n0_ + dt_sixth * accn_).max(0.0).min(1.0);
}

// =============================================================================
// Per-group subset ops for Phase 2 parallelism (scalar loop over local indices)
// =============================================================================

void HHPool::gather_currents_subset(const std::vector<size_t>& local_indices,
                                     const double* I_buf) {
    for (size_t li : local_indices)
        I_ext_(li) = I_buf[net_idx_[li]];
}

void HHPool::scatter_voltages_subset(const std::vector<size_t>& local_indices,
                                      double* V_buf) const {
    for (size_t li : local_indices)
        V_buf[net_idx_[li]] = V_(li);
}

namespace {
// Scalar HH rate functions and derivatives — mirrors the vectorized logic.
inline void hh_scalar_derivs(
    double V, double m, double h, double n,
    double Cm, double gNa, double gK, double gL,
    double ENa, double EK, double EL, double I,
    double& dV, double& dm, double& dh, double& dn)
{
    double Vp40 = V + 40.0;
    double am = (std::abs(Vp40) < 1e-7)
                  ? 1.0
                  : 0.1 * Vp40 / (1.0 - std::exp(-0.1 * Vp40));
    double bm = 4.0 * std::exp(-(V + 65.0) / 18.0);
    double ah = 0.07 * std::exp(-0.05 * (V + 65.0));
    double bh = 1.0 / (1.0 + std::exp(-0.1 * (V + 35.0)));
    double Vp55 = V + 55.0;
    double an = (std::abs(Vp55) < 1e-7)
                  ? 0.1
                  : 0.01 * Vp55 / (1.0 - std::exp(-0.1 * Vp55));
    double bn = 0.125 * std::exp(-0.0125 * (V + 65.0));

    dV = (I - gNa*m*m*m*h*(V-ENa) - gK*n*n*n*n*(V-EK) - gL*(V-EL)) / Cm;
    dm = am*(1.0-m) - bm*m;
    dh = ah*(1.0-h) - bh*h;
    dn = an*(1.0-n) - bn*n;
}
inline double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }
} // namespace

void HHPool::step_subset(const std::vector<size_t>& local_indices, double dt) {
    const double dt2 = dt * 0.5;
    const double dt6 = dt / 6.0;

    for (size_t li : local_indices) {
        const double Cm  = C_m_(li),  gNa = g_Na_(li), gK = g_K_(li),  gL = g_L_(li);
        const double ENa = E_Na_(li), EK  = E_K_(li),  EL = E_L_(li),  I  = I_ext_(li);

        double V0 = V_(li), m0 = m_(li), h0 = h_(li), n0 = n_(li);
        double k1V, k1m, k1h, k1n;
        hh_scalar_derivs(V0, m0, h0, n0, Cm, gNa, gK, gL, ENa, EK, EL, I,
                         k1V, k1m, k1h, k1n);

        double k2V, k2m, k2h, k2n;
        hh_scalar_derivs(V0+dt2*k1V, clamp01(m0+dt2*k1m), clamp01(h0+dt2*k1h), clamp01(n0+dt2*k1n),
                         Cm, gNa, gK, gL, ENa, EK, EL, I,
                         k2V, k2m, k2h, k2n);

        double k3V, k3m, k3h, k3n;
        hh_scalar_derivs(V0+dt2*k2V, clamp01(m0+dt2*k2m), clamp01(h0+dt2*k2h), clamp01(n0+dt2*k2n),
                         Cm, gNa, gK, gL, ENa, EK, EL, I,
                         k3V, k3m, k3h, k3n);

        double k4V, k4m, k4h, k4n;
        hh_scalar_derivs(V0+dt*k3V, clamp01(m0+dt*k3m), clamp01(h0+dt*k3h), clamp01(n0+dt*k3n),
                         Cm, gNa, gK, gL, ENa, EK, EL, I,
                         k4V, k4m, k4h, k4n);

        V_(li) = V0 + dt6*(k1V + 2*k2V + 2*k3V + k4V);
        m_(li) = clamp01(m0 + dt6*(k1m + 2*k2m + 2*k3m + k4m));
        h_(li) = clamp01(h0 + dt6*(k1h + 2*k2h + 2*k3h + k4h));
        n_(li) = clamp01(n0 + dt6*(k1n + 2*k2n + 2*k3n + k4n));
    }
}

// =============================================================================
// Sync back to polymorphic API objects
// =============================================================================

void HHPool::sync_to_neurons(
    std::vector<std::unique_ptr<NeuronBase>>& neurons) const
{
    for (size_t i = 0; i < N_; ++i) {
        HHNeuron* hh = static_cast<HHNeuron*>(neurons[net_idx_[i]].get());
        HHNeuron::State s;
        s.V = V_(i);  s.m = m_(i);
        s.h = h_(i);  s.n = n_(i);
        hh->set_state(s);
    }
}

} // namespace hodgkin_huxley
