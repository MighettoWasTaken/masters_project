#pragma once

// =============================================================================
// model/kinetics.hpp — single authoritative kinetic math implementation
//
// Provides scalar and vectorized (Eigen) versions of:
//   boltzmann    — sigmoid steady-state
//   compute_tau  — voltage-dependent time constant (6 forms)
//   compute_rate — HH-style rate function (4 forms)
//   fast_exp     — polynomial approximation for SIMD hot loops
//
// All free functions; no class dependencies.  Scalar functions are identical
// to the old ion_channels.hpp free functions.  Vectorized functions replace
// the former static methods on ComposablePool.
//
// Insertion point for task13 (SymPy codegen): swap this header with a
// generated kinetics_sympy.hpp that overloads the same function names.
// =============================================================================

#include "hodgkin_huxley/model/gate_spec.hpp"
#include <Eigen/Core>
#include <cmath>
#include <algorithm>

namespace hodgkin_huxley {

// =============================================================================
// Scalar implementations
// =============================================================================

inline double boltzmann_scalar(double x, const BoltzmannParams& p) {
    double arg = -(x - p.v_half) / p.k;
    if (arg > 500.0) return 0.0;
    if (arg < -500.0) return 1.0;
    return 1.0 / (1.0 + std::exp(arg));
}

inline double compute_tau_scalar(double V, const TauParams& p) {
    switch (p.form) {
        case TauParams::Form::CONSTANT:
            return p.params[0];
        case TauParams::Form::BOLTZMANN: {
            double arg = -(V - p.params[2]) / p.params[3];
            arg = std::max(-500.0, std::min(500.0, arg));
            return p.params[0] + p.params[1] / (1.0 + std::exp(arg));
        }
        case TauParams::Form::DOUBLE_EXP_SUM: {
            double e1 = std::exp((V + p.params[2]) / p.params[3]);
            double e2 = std::exp(-(V + p.params[5]) / p.params[6]);
            double denom = e1 + e2;
            if (denom < 1e-10) denom = 1e-10;
            return p.params[0] + p.params[1] / denom;
        }
        case TauParams::Form::OFFSET_DOUBLE_EXP: {
            double x1 = (V + p.params[2]) / p.params[3];
            double x2 = (V + p.params[5]) / p.params[6];
            return p.params[0] + p.params[1] * std::exp(-x1 * x1)
                               + p.params[4] * std::exp(-x2 * x2);
        }
        case TauParams::Form::SCALED_EXP: {
            double arg = (V - p.params[1]) / (2.0 * p.params[2]);
            arg = std::max(-500.0, std::min(500.0, arg));
            double ch = std::cosh(arg);
            if (ch < 1e-10) ch = 1e-10;
            return p.params[0] / ch;
        }
        case TauParams::Form::COMPOUND_AB: {
            double alpha = p.params[0] * std::exp((V + p.params[1]) / p.params[2]);
            double beta  = p.params[3] * std::exp((V + p.params[4]) / p.params[5]);
            double sum = alpha + beta;
            if (sum < 1e-10) sum = 1e-10;
            return 1.0 / sum;
        }
    }
    return 1.0;
}

inline double compute_rate_scalar(double V, const RateFuncParams& p) {
    switch (p.form) {
        case RateFuncParams::Form::LINEAR_OVER_EXP: {
            double x = V + p.B;
            double xc = x / p.C;
            if (std::abs(xc) < 1e-6) return p.A * p.C * (1.0 + xc * 0.5);
            return p.A * x / (std::exp(xc) - 1.0);
        }
        case RateFuncParams::Form::EXP_DECAY: {
            double arg = (V + p.B) / p.C;
            arg = std::max(-500.0, std::min(500.0, arg));
            return p.A * std::exp(arg);
        }
        case RateFuncParams::Form::LINEAR_OVER_EXPM1: {
            double x = V + p.B;
            double xc = x / p.C;
            if (std::abs(xc) < 1e-6) return p.A * p.C * (1.0 + xc * 0.5);
            return p.A * x / (1.0 - std::exp(-xc));
        }
        case RateFuncParams::Form::SIGMOID: {
            double arg = (V + p.B) / p.C;
            arg = std::max(-500.0, std::min(500.0, arg));
            return p.A / (1.0 + std::exp(arg));
        }
    }
    return 0.0;
}

// =============================================================================
// fast_exp — range reduction by 32 + degree-7 Taylor + 5 squarings
//
// exp(x) = exp(x/32)^32.  Degree-7 Taylor on [-0.31, 0.09] gives ~8
// significant digits after 5 squarings.  Entirely Eigen-vectorized.
// Safe to call with src == dst.  tmp_r must be pre-allocated to src.size().
// =============================================================================

inline void fast_exp(const Eigen::ArrayXd& src, Eigen::ArrayXd& dst, Eigen::ArrayXd& tmp_r) {
    tmp_r = src * (1.0 / 32.0);

    // Degree-7 Taylor in Horner form: 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720 + r⁷/5040
    dst = tmp_r * (1.0 / 5040.0) + (1.0 / 720.0);
    dst = dst * tmp_r + (1.0 / 120.0);
    dst = dst * tmp_r + (1.0 / 24.0);
    dst = dst * tmp_r + (1.0 / 6.0);
    dst = dst * tmp_r + 0.5;
    dst = dst * tmp_r + 1.0;
    dst = dst * tmp_r + 1.0;

    // 5 squarings: exp(x/32)^32 = exp(x)
    dst *= dst; dst *= dst; dst *= dst; dst *= dst; dst *= dst;
}

// =============================================================================
// Vectorized implementations — operate on Eigen arrays (SIMD)
// =============================================================================

inline Eigen::ArrayXd boltzmann_vec(const Eigen::ArrayXd& x, const BoltzmannParams& p) {
    Eigen::ArrayXd arg = -(x - p.v_half) / p.k;
    arg = arg.max(-500.0).min(500.0);
    return 1.0 / (1.0 + arg.exp());
}

inline Eigen::ArrayXd compute_tau_vec(const Eigen::ArrayXd& V, const TauParams& tau,
                                       Eigen::ArrayXd& tmp) {
    const Eigen::Index N = V.size();
    switch (tau.form) {
        case TauParams::Form::CONSTANT:
            return Eigen::ArrayXd::Constant(N, tau.params[0]);

        case TauParams::Form::BOLTZMANN: {
            double base = tau.params[0], amp = tau.params[1];
            double vh = tau.params[2], k = tau.params[3];
            tmp = (-(V - vh) / k).max(-500.0).min(500.0);
            return base + amp / (1.0 + tmp.exp());
        }

        case TauParams::Form::DOUBLE_EXP_SUM: {
            double base = tau.params[0], amp = tau.params[1];
            double v1 = tau.params[2], s1 = tau.params[3];
            double v2 = tau.params[5], s2 = tau.params[6];
            Eigen::ArrayXd e1 = ((V + v1) / s1).exp();
            Eigen::ArrayXd e2 = (-(V + v2) / s2).exp();
            return base + amp / (e1 + e2).max(1e-10);
        }

        case TauParams::Form::OFFSET_DOUBLE_EXP: {
            double base = tau.params[0], a1 = tau.params[1];
            double v1 = tau.params[2], s1 = tau.params[3];
            double a2 = tau.params[4], v2 = tau.params[5], s2 = tau.params[6];
            tmp = (V + v1) / s1;
            Eigen::ArrayXd t1 = a1 * (-tmp * tmp).exp();
            tmp = (V + v2) / s2;
            return base + t1 + a2 * (-tmp * tmp).exp();
        }

        case TauParams::Form::SCALED_EXP: {
            double scale = tau.params[0], vh = tau.params[1], k = tau.params[2];
            tmp = ((V - vh) / (2.0 * k)).max(-500.0).min(500.0);
            return scale / tmp.cosh().max(1e-10);
        }

        case TauParams::Form::COMPOUND_AB: {
            double aA = tau.params[0], aB = tau.params[1], aC = tau.params[2];
            double bA = tau.params[3], bB = tau.params[4], bC = tau.params[5];
            Eigen::ArrayXd alpha = aA * ((V + aB) / aC).exp();
            Eigen::ArrayXd beta  = bA * ((V + bB) / bC).exp();
            return 1.0 / (alpha + beta).max(1e-10);
        }
    }
    return Eigen::ArrayXd::Constant(N, 1.0);
}

inline Eigen::ArrayXd compute_rate_vec(const Eigen::ArrayXd& V, const RateFuncParams& rate,
                                        Eigen::ArrayXd& tmp) {
    const Eigen::Index N = V.size();
    switch (rate.form) {
        case RateFuncParams::Form::LINEAR_OVER_EXP: {
            Eigen::ArrayXd x = V + rate.B;
            Eigen::ArrayXd xc = x / rate.C;
            Eigen::ArrayXd e = xc.exp();
            Eigen::ArrayXd result = rate.A * x / (e - 1.0);
            return (xc.abs() < 1e-6).select(
                Eigen::ArrayXd::Constant(N, rate.A * rate.C), result);
        }

        case RateFuncParams::Form::EXP_DECAY: {
            tmp = ((V + rate.B) / rate.C).max(-500.0).min(500.0);
            return rate.A * tmp.exp();
        }

        case RateFuncParams::Form::LINEAR_OVER_EXPM1: {
            Eigen::ArrayXd x = V + rate.B;
            Eigen::ArrayXd xc = x / rate.C;
            Eigen::ArrayXd e = (-xc).exp();
            Eigen::ArrayXd result = rate.A * x / (1.0 - e);
            return (xc.abs() < 1e-6).select(
                Eigen::ArrayXd::Constant(N, rate.A * rate.C), result);
        }

        case RateFuncParams::Form::SIGMOID: {
            tmp = ((V + rate.B) / rate.C).max(-500.0).min(500.0);
            return rate.A / (1.0 + tmp.exp());
        }
    }
    return Eigen::ArrayXd::Zero(N);
}

} // namespace hodgkin_huxley
