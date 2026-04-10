#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hodgkin_huxley {

struct BoltzmannParams {
    double v_half = 0.0;
    double k = 1.0;
};

struct TauParams {
    enum class Form {
        CONSTANT,          // params[0] = tau_const
        BOLTZMANN,         // params[0] = tau_base, params[1] = tau_amp, params[2] = v_half, params[3] = k
        DOUBLE_EXP_SUM,    // params[0] = base, params[1] = a1, params[2] = v1, params[3] = s1, params[4] = a2, params[5] = v2, params[6] = s2
        OFFSET_DOUBLE_EXP, // params[0] = base, params[1] = a1, params[2] = v1, params[3] = s1, params[4] = a2, params[5] = v2, params[6] = s2
        SCALED_EXP,        // params[0] = scale, params[1] = v_half, params[2] = k (tau = scale / cosh((V-v_half)/(2*k)))
        COMPOUND_AB        // params[0..2] = alpha A,B,C; params[3..5] = beta A,B,C; tau = 1/(alpha+beta)
    };
    Form form = Form::CONSTANT;
    double params[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

struct RateFuncParams {
    enum class Form {
        LINEAR_OVER_EXP,   // alpha = A*(V+B) / (exp((V+B)/C) - 1)   [HH standard]
        EXP_DECAY,         // alpha = A * exp((V+B)/C)
        LINEAR_OVER_EXPM1, // alpha = A*(V+B) / (1 - exp(-(V+B)/C))  [equivalent rewrite]
        SIGMOID            // alpha = A / (1 + exp((V+B)/C))
    };
    Form form = Form::LINEAR_OVER_EXP;
    double A = 0.0, B = 0.0, C = 1.0;
};

// =============================================================================
// Bytecode VM types — defined before GateSpec so GateSpec can embed them
// =============================================================================

enum class VmOp : uint8_t {
    PUSH_DEP=0, PUSH_CONST=1,          // stack inputs
    ADD=2, MUL=3, NEG=4, RCP=5,        // arithmetic (RCP = 1/x)
    POW_INT=6, POW_HALF=7, POW_GEN=8,  // powers
    EXP=9, LOG=10, TANH=11,            // transcendental
    SIN=12, COS=13, SQRT=14, ABS=15,   // more functions
    PUSH_GATE=16,                           // push gate_states_[operand] onto stack
    PUSH_S=17,                              // push kinetic synapse gating variable S
    PUSH_A=18,                              // push auxiliary synapse state variable A
};

struct VmInstruction {
    VmOp    op      = VmOp::PUSH_CONST;
    int32_t operand = 0;  // const index (PUSH_CONST) or integer exponent (POW_INT)
};

struct VmExpr {
    std::vector<VmInstruction> instructions;
    std::vector<double>        constants;
    bool empty() const { return instructions.empty(); }
    void add_instruction(VmOp op, int32_t operand = 0) {
        instructions.push_back({op, operand});
    }
    int32_t add_constant(double val) {
        constants.push_back(val);
        return static_cast<int32_t>(constants.size() - 1);
    }
};

struct GateSpec {
    enum class UpdateForm { INF_TAU=0, ALPHA_BETA=1, INSTANT=2, DERIVED=3, CUSTOM_EXPR=4 };
    enum class Dependency { VOLTAGE, CALCIUM };

    std::string name;
    UpdateForm update_form = UpdateForm::INF_TAU;
    Dependency dependency = Dependency::VOLTAGE;
    double scale = 1.0;
    double initial_value = 0.0;

    // For INF_TAU and INSTANT forms
    BoltzmannParams inf;
    TauParams tau;

    // For ALPHA_BETA form
    RateFuncParams alpha;
    RateFuncParams beta;

    // For DERIVED form: X = a * (b + c * source_gate)
    int derived_source_gate = -1;
    double derived_a = 1.0;
    double derived_b = 0.0;
    double derived_c = 1.0;

    // For CUSTOM_EXPR form: bytecode programs executed by the pre-compiled C++ stack VM.
    // inf_vm / tau_vm: INF_TAU-style update  (x -> x_inf + (x - x_inf)*exp(-dt*scale/tau))
    // alpha_vm / beta_vm: ALPHA_BETA-style   (alpha/(alpha+beta) steady state)
    // dxdt_vm: arbitrary ODE  dx/dt = F(x, V)  — integrated with Euler
    //   PUSH_DEP pushes V (or Ca for calcium-dependent gates)
    //   PUSH_S   pushes the current gate state x
    VmExpr inf_vm, tau_vm, alpha_vm, beta_vm;
    VmExpr dxdt_vm;
};

struct CalciumSpec {
    bool enabled = false;
    bool use_nernst = false;
    double epsilon = 1e-4;
    double K_Ca = 15.0;
    double Ca_init = 0.1;
    double Ca_o = 2000.0;  // extracellular Ca (uM)
    double z = 2.0;
    double F = 96485.0;    // Faraday constant
    double R = 8314.0;     // gas constant (mJ/(mol*K))
    double T = 298.0;      // temperature (K)
    std::vector<int> source_channels;  // channel indices contributing to Ca influx
};

} // namespace hodgkin_huxley
