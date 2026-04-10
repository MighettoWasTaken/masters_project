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

// =============================================================================
// Bytecode VM evaluators — interpret VmExpr programs compiled from SymPy trees
// =============================================================================

// tmp_fast_exp: when non-null, EXP opcodes use fast_exp() instead of Eigen's .exp().
// Pass the pool's pre-allocated tmp_exp_r_ buffer; leave null for scalar / low-N cases.
inline Eigen::ArrayXd vm_eval_vec(const VmExpr& prog, const Eigen::ArrayXd& dep,
                                   Eigen::ArrayXd* tmp_fast_exp = nullptr) {
    const Eigen::Index N = dep.size();
    std::vector<Eigen::ArrayXd> stk;
    stk.reserve(8);
    for (const auto& ins : prog.instructions) {
        switch (ins.op) {
            case VmOp::PUSH_DEP:   stk.push_back(dep); break;
            case VmOp::PUSH_CONST: stk.push_back(Eigen::ArrayXd::Constant(N, prog.constants[ins.operand])); break;
            case VmOp::ADD: { auto b=std::move(stk.back()); stk.pop_back(); stk.back()+=b; break; }
            case VmOp::MUL: { auto b=std::move(stk.back()); stk.pop_back(); stk.back()*=b; break; }
            case VmOp::NEG:      stk.back() = -stk.back(); break;
            case VmOp::RCP:      stk.back() = 1.0/stk.back(); break;
            case VmOp::POW_INT:  stk.back() = stk.back().pow(ins.operand); break;
            case VmOp::POW_HALF: stk.back() = stk.back().sqrt(); break;
            case VmOp::POW_GEN:  { auto e=std::move(stk.back()); stk.pop_back(); stk.back()=stk.back().pow(e); break; }
            case VmOp::EXP:
                if (tmp_fast_exp) { fast_exp(stk.back(), stk.back(), *tmp_fast_exp); }
                else              { stk.back() = stk.back().exp(); }
                break;
            case VmOp::LOG:      stk.back() = stk.back().log();  break;
            case VmOp::TANH:     stk.back() = stk.back().tanh(); break;
            case VmOp::SIN:      stk.back() = stk.back().sin();  break;
            case VmOp::COS:      stk.back() = stk.back().cos();  break;
            case VmOp::SQRT:     stk.back() = stk.back().sqrt(); break;
            case VmOp::ABS:      stk.back() = stk.back().abs();  break;
        }
    }
    return stk.empty() ? Eigen::ArrayXd::Zero(N) : stk.back();
}

inline double vm_eval_scalar(const VmExpr& prog, double dep) {
    std::vector<double> stk;
    stk.reserve(8);
    for (const auto& ins : prog.instructions) {
        switch (ins.op) {
            case VmOp::PUSH_DEP:   stk.push_back(dep); break;
            case VmOp::PUSH_CONST: stk.push_back(prog.constants[ins.operand]); break;
            case VmOp::ADD: { double b=stk.back(); stk.pop_back(); stk.back()+=b; break; }
            case VmOp::MUL: { double b=stk.back(); stk.pop_back(); stk.back()*=b; break; }
            case VmOp::NEG:      stk.back()=-stk.back(); break;
            case VmOp::RCP:      stk.back()=1.0/stk.back(); break;
            case VmOp::POW_INT:  stk.back()=std::pow(stk.back(),ins.operand); break;
            case VmOp::POW_HALF: stk.back()=std::sqrt(stk.back()); break;
            case VmOp::POW_GEN:  { double e=stk.back(); stk.pop_back(); stk.back()=std::pow(stk.back(),e); break; }
            case VmOp::EXP:      stk.back()=std::exp(stk.back());  break;
            case VmOp::LOG:      stk.back()=std::log(stk.back());  break;
            case VmOp::TANH:     stk.back()=std::tanh(stk.back()); break;
            case VmOp::SIN:      stk.back()=std::sin(stk.back());  break;
            case VmOp::COS:      stk.back()=std::cos(stk.back());  break;
            case VmOp::SQRT:     stk.back()=std::sqrt(stk.back()); break;
            case VmOp::ABS:      stk.back()=std::abs(stk.back());  break;
        }
    }
    return stk.empty() ? 0.0 : stk.back();
}

// =============================================================================
// 2-argument VM evaluator — vectorized version of vm_eval_scalar_2arg.
// dep  = V (or Ca) array; x_arr = current gate state array (pushed by PUSH_S).
// Used for arbitrary gate ODEs: dx/dt = F(x, V).
// =============================================================================

inline Eigen::ArrayXd vm_eval_vec_2arg(const VmExpr& prog,
                                        const Eigen::ArrayXd& dep,
                                        const Eigen::ArrayXd& x_arr,
                                        Eigen::ArrayXd* tmp_fast_exp = nullptr) {
    const Eigen::Index N = dep.size();
    std::vector<Eigen::ArrayXd> stk;
    stk.reserve(8);
    for (const auto& ins : prog.instructions) {
        switch (ins.op) {
            case VmOp::PUSH_DEP:   stk.push_back(dep);   break;
            case VmOp::PUSH_CONST: stk.push_back(Eigen::ArrayXd::Constant(N, prog.constants[ins.operand])); break;
            case VmOp::PUSH_S:     stk.push_back(x_arr); break;
            case VmOp::ADD: { auto b=std::move(stk.back()); stk.pop_back(); stk.back()+=b; break; }
            case VmOp::MUL: { auto b=std::move(stk.back()); stk.pop_back(); stk.back()*=b; break; }
            case VmOp::NEG:      stk.back() = -stk.back(); break;
            case VmOp::RCP:      stk.back() = 1.0/stk.back(); break;
            case VmOp::POW_INT:  stk.back() = stk.back().pow(ins.operand); break;
            case VmOp::POW_HALF: stk.back() = stk.back().sqrt(); break;
            case VmOp::POW_GEN:  { auto e=std::move(stk.back()); stk.pop_back(); stk.back()=stk.back().pow(e); break; }
            case VmOp::EXP:
                if (tmp_fast_exp) { fast_exp(stk.back(), stk.back(), *tmp_fast_exp); }
                else              { stk.back() = stk.back().exp(); }
                break;
            case VmOp::LOG:      stk.back() = stk.back().log();  break;
            case VmOp::TANH:     stk.back() = stk.back().tanh(); break;
            case VmOp::SIN:      stk.back() = stk.back().sin();  break;
            case VmOp::COS:      stk.back() = stk.back().cos();  break;
            case VmOp::SQRT:     stk.back() = stk.back().sqrt(); break;
            case VmOp::ABS:      stk.back() = stk.back().abs();  break;
            default: break;  // PUSH_GATE not applicable here
        }
    }
    return stk.empty() ? Eigen::ArrayXd::Zero(N) : stk.back();
}

// =============================================================================
// Gate-product VM evaluators — like vm_eval_* but handle PUSH_GATE opcode
// which pushes a gate state array (pool path) or scalar (single-neuron path).
// =============================================================================

inline Eigen::ArrayXd vm_eval_gate_product_vec(
    const VmExpr& prog,
    const Eigen::ArrayXd& V,
    const std::vector<Eigen::ArrayXd>& gate_states,
    Eigen::ArrayXd* tmp_fast_exp = nullptr)
{
    const Eigen::Index N = V.size();
    std::vector<Eigen::ArrayXd> stk;
    stk.reserve(8);
    for (const auto& ins : prog.instructions) {
        switch (ins.op) {
            case VmOp::PUSH_DEP:   stk.push_back(V); break;
            case VmOp::PUSH_CONST: stk.push_back(Eigen::ArrayXd::Constant(N, prog.constants[ins.operand])); break;
            case VmOp::PUSH_GATE:  stk.push_back(gate_states[ins.operand]); break;
            case VmOp::ADD: { auto b=std::move(stk.back()); stk.pop_back(); stk.back()+=b; break; }
            case VmOp::MUL: { auto b=std::move(stk.back()); stk.pop_back(); stk.back()*=b; break; }
            case VmOp::NEG:      stk.back() = -stk.back(); break;
            case VmOp::RCP:      stk.back() = 1.0/stk.back(); break;
            case VmOp::POW_INT:  stk.back() = stk.back().pow(ins.operand); break;
            case VmOp::POW_HALF: stk.back() = stk.back().sqrt(); break;
            case VmOp::POW_GEN:  { auto e=std::move(stk.back()); stk.pop_back(); stk.back()=stk.back().pow(e); break; }
            case VmOp::EXP:
                if (tmp_fast_exp) { fast_exp(stk.back(), stk.back(), *tmp_fast_exp); }
                else              { stk.back() = stk.back().exp(); }
                break;
            case VmOp::LOG:      stk.back() = stk.back().log();  break;
            case VmOp::TANH:     stk.back() = stk.back().tanh(); break;
            case VmOp::SIN:      stk.back() = stk.back().sin();  break;
            case VmOp::COS:      stk.back() = stk.back().cos();  break;
            case VmOp::SQRT:     stk.back() = stk.back().sqrt(); break;
            case VmOp::ABS:      stk.back() = stk.back().abs();  break;
        }
    }
    return stk.empty() ? Eigen::ArrayXd::Ones(N) : stk.back();
}

inline double vm_eval_gate_product_scalar(
    const VmExpr& prog,
    double V,
    const std::vector<double>& gate_states)
{
    std::vector<double> stk;
    stk.reserve(8);
    for (const auto& ins : prog.instructions) {
        switch (ins.op) {
            case VmOp::PUSH_DEP:   stk.push_back(V); break;
            case VmOp::PUSH_CONST: stk.push_back(prog.constants[ins.operand]); break;
            case VmOp::PUSH_GATE:  stk.push_back(gate_states[ins.operand]); break;
            case VmOp::ADD: { double b=stk.back(); stk.pop_back(); stk.back()+=b; break; }
            case VmOp::MUL: { double b=stk.back(); stk.pop_back(); stk.back()*=b; break; }
            case VmOp::NEG:      stk.back()=-stk.back(); break;
            case VmOp::RCP:      stk.back()=1.0/stk.back(); break;
            case VmOp::POW_INT:  stk.back()=std::pow(stk.back(),ins.operand); break;
            case VmOp::POW_HALF: stk.back()=std::sqrt(stk.back()); break;
            case VmOp::POW_GEN:  { double e=stk.back(); stk.pop_back(); stk.back()=std::pow(stk.back(),e); break; }
            case VmOp::EXP:      stk.back()=std::exp(stk.back());  break;
            case VmOp::LOG:      stk.back()=std::log(stk.back());  break;
            case VmOp::TANH:     stk.back()=std::tanh(stk.back()); break;
            case VmOp::SIN:      stk.back()=std::sin(stk.back());  break;
            case VmOp::COS:      stk.back()=std::cos(stk.back());  break;
            case VmOp::SQRT:     stk.back()=std::sqrt(stk.back()); break;
            case VmOp::ABS:      stk.back()=std::abs(stk.back());  break;
        }
    }
    return stk.empty() ? 1.0 : stk.back();
}

// =============================================================================
// Kinetic-synapse VM evaluator — like vm_eval_scalar but handles PUSH_S.
// dep  = V_pre  (for dS_dt expressions)  or  V_post  (for current expressions)
// S    = current value of the synapse gating variable
// =============================================================================

inline double vm_eval_scalar_2arg(const VmExpr& prog, double dep, double S) {
    std::vector<double> stk;
    stk.reserve(8);
    for (const auto& ins : prog.instructions) {
        switch (ins.op) {
            case VmOp::PUSH_DEP:   stk.push_back(dep); break;
            case VmOp::PUSH_CONST: stk.push_back(prog.constants[ins.operand]); break;
            case VmOp::PUSH_S:     stk.push_back(S);   break;
            case VmOp::ADD: { double b=stk.back(); stk.pop_back(); stk.back()+=b; break; }
            case VmOp::MUL: { double b=stk.back(); stk.pop_back(); stk.back()*=b; break; }
            case VmOp::NEG:      stk.back()=-stk.back(); break;
            case VmOp::RCP:      stk.back()=1.0/stk.back(); break;
            case VmOp::POW_INT:  stk.back()=std::pow(stk.back(),ins.operand); break;
            case VmOp::POW_HALF: stk.back()=std::sqrt(stk.back()); break;
            case VmOp::POW_GEN:  { double e=stk.back(); stk.pop_back(); stk.back()=std::pow(stk.back(),e); break; }
            case VmOp::EXP:      stk.back()=std::exp(stk.back());  break;
            case VmOp::LOG:      stk.back()=std::log(stk.back());  break;
            case VmOp::TANH:     stk.back()=std::tanh(stk.back()); break;
            case VmOp::SIN:      stk.back()=std::sin(stk.back());  break;
            case VmOp::COS:      stk.back()=std::cos(stk.back());  break;
            case VmOp::SQRT:     stk.back()=std::sqrt(stk.back()); break;
            case VmOp::ABS:      stk.back()=std::abs(stk.back());  break;
            default: break;  // PUSH_GATE not applicable here
        }
    }
    return stk.empty() ? 0.0 : stk.back();
}

// =============================================================================
// 3-argument VM evaluator — like vm_eval_scalar_2arg but also handles PUSH_A.
// dep  = V_pre (or V_post for current); S = primary gating var; A = auxiliary.
// Used for 2-variable novel synapse ODEs and current expressions.
// =============================================================================

inline double vm_eval_scalar_3arg(const VmExpr& prog, double dep, double S, double A) {
    std::vector<double> stk;
    stk.reserve(8);
    for (const auto& ins : prog.instructions) {
        switch (ins.op) {
            case VmOp::PUSH_DEP:   stk.push_back(dep); break;
            case VmOp::PUSH_CONST: stk.push_back(prog.constants[ins.operand]); break;
            case VmOp::PUSH_S:     stk.push_back(S);   break;
            case VmOp::PUSH_A:     stk.push_back(A);   break;
            case VmOp::ADD: { double b=stk.back(); stk.pop_back(); stk.back()+=b; break; }
            case VmOp::MUL: { double b=stk.back(); stk.pop_back(); stk.back()*=b; break; }
            case VmOp::NEG:      stk.back()=-stk.back(); break;
            case VmOp::RCP:      stk.back()=1.0/stk.back(); break;
            case VmOp::POW_INT:  stk.back()=std::pow(stk.back(),ins.operand); break;
            case VmOp::POW_HALF: stk.back()=std::sqrt(stk.back()); break;
            case VmOp::POW_GEN:  { double e=stk.back(); stk.pop_back(); stk.back()=std::pow(stk.back(),e); break; }
            case VmOp::EXP:      stk.back()=std::exp(stk.back());  break;
            case VmOp::LOG:      stk.back()=std::log(stk.back());  break;
            case VmOp::TANH:     stk.back()=std::tanh(stk.back()); break;
            case VmOp::SIN:      stk.back()=std::sin(stk.back());  break;
            case VmOp::COS:      stk.back()=std::cos(stk.back());  break;
            case VmOp::SQRT:     stk.back()=std::sqrt(stk.back()); break;
            case VmOp::ABS:      stk.back()=std::abs(stk.back());  break;
            default: break;  // PUSH_GATE not applicable here
        }
    }
    return stk.empty() ? 0.0 : stk.back();
}

} // namespace hodgkin_huxley
