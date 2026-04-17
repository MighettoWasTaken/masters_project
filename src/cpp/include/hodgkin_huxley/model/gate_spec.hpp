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
    PUSH_S=17,                              // push kinetic synapse gating variable S (or self substance concentration)
    PUSH_A=18,                              // push auxiliary synapse state variable A
    PUSH_X=19,                              // push X_[operand] (substance concentration by index; 0 = self)
};

struct VmInstruction {
    VmOp    op      = VmOp::PUSH_CONST;
    int32_t operand = 0;  // const index (PUSH_CONST) or integer exponent (POW_INT) or substance index (PUSH_X)
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
    enum class Dependency {
        VOLTAGE,
        INTRACELLULAR  // replaces CALCIUM; intracellular_idx selects which substance
    };

    std::string name;
    UpdateForm update_form = UpdateForm::INF_TAU;
    Dependency dependency = Dependency::VOLTAGE;
    int intracellular_idx = 0;  // which IntracellularSpec provides X (default 0 = calcium by convention)
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
    //   PUSH_DEP pushes V (or X_[intracellular_idx] for intracellular-dependent gates)
    //   PUSH_S   pushes the current gate state x
    VmExpr inf_vm, tau_vm, alpha_vm, beta_vm;
    VmExpr dxdt_vm;
};

// =============================================================================
// Intracellular dynamics structs
// =============================================================================

// Forward declaration
struct IntracellularSpec;

/// Describes how one intracellular substance modulates channels, gates, or synapses.
struct IntracellularModulation {
    enum class Target {
        CHANNEL_G,       // g_eff = g * mod_vm(X) — mod_vm returns scale factor
        CHANNEL_EREV,    // E_rev = mod_vm(X)     — replaces channel E_rev
        GATE_INF_SHIFT,  // x_inf(V + shift) where shift = shift_scale*X or mod_vm(X)
        GATE_INF_SCALE,  // x_inf_eff = x_inf * mod_vm(X)
        GATE_TAU_SCALE,  // tau_eff = tau * mod_vm(X)
        GATE_INF_EXPR,   // x_inf fully replaced by mod_vm(V, X) — PUSH_DEP=V, PUSH_X=X
        SYNAPSE_G,       // scale all incoming synaptic g for neurons in this population
    };

    Target target;
    int    target_idx    = -1;  // channel index (CHANNEL_*) or gate index (GATE_*)
    int    substance_idx =  0;  // which IntracellularSpec provides X for the modulation

    // mod_vm semantics depend on target:
    //   CHANNEL_G, CHANNEL_EREV, GATE_INF_SHIFT, GATE_INF_SCALE, GATE_TAU_SCALE, SYNAPSE_G:
    //     PUSH_DEP → substance X concentration (ArrayXd)
    //     PUSH_X(n) → X_[n] concentration (for cascade mods)
    //   GATE_INF_EXPR:
    //     PUSH_DEP → V (voltage array)
    //     PUSH_S   → gate state (if needed)
    //     PUSH_X(n) → X_[n] concentration
    VmExpr mod_vm;

    // GATE_INF_SHIFT with scalar scale: if mod_vm is empty, shift = shift_scale * X
    double shift_scale = 0.0;
};

/// Describes one intracellular substance (calcium, dopamine, cAMP, IP3, etc.)
struct IntracellularSpec {
    std::string name;
    double      initial = 0.0;  // initial concentration

    /// ODE update form for d[X]/dt
    enum class UpdateForm {
        DECAY,               // d[X]/dt = -k_decay * X
        DRIVEN_DECAY,        // d[X]/dt = epsilon * (-I_src - k_decay * X)
        DRIVEN_DECAY_NERNST, // as DRIVEN_DECAY + standard Nernst E_rev update
        CUSTOM_EXPR,         // VM bytecode (see ode_vm below)
    };
    UpdateForm update_form = UpdateForm::CUSTOM_EXPR;

    // Scalar parameters for standard forms
    double epsilon = 1e-4;
    double k_decay = 15.0;
    std::vector<int> source_channels;  // channel indices contributing to I_src

    // Nernst parameters (DRIVEN_DECAY_NERNST standard form)
    bool   nernst_enabled = false;
    double nernst_Ca_o    = 2000.0;  // extracellular concentration (µM)
    double nernst_z       = 2.0;
    double nernst_R       = 8314.0;  // gas constant (mJ/(mol·K))
    double nernst_F       = 96485.0; // Faraday constant
    double nernst_T       = 298.0;   // temperature (K)

    // CUSTOM_EXPR Nernst: PUSH_DEP = X → returns E_rev
    VmExpr nernst_vm;

    // CUSTOM_EXPR ODE:
    //   PUSH_DEP → I_source sum (sum of source channel currents)
    //   PUSH_S   → this substance's concentration (reuses PUSH_S opcode)
    //   PUSH_X(n) → other substance concentration X_[n]
    VmExpr ode_vm;

    /// What this substance modulates each step
    std::vector<IntracellularModulation> modulations;
};

// =============================================================================
// Legacy CalciumSpec — kept for backward compatibility with old code paths.
// New code should use IntracellularSpec with UpdateForm::DRIVEN_DECAY_NERNST.
// =============================================================================
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

    /// Convert to the new IntracellularSpec representation.
    IntracellularSpec to_intracellular_spec() const {
        IntracellularSpec ic;
        ic.name    = "Ca";
        ic.initial = Ca_init;
        ic.epsilon = epsilon;
        ic.k_decay = K_Ca;
        ic.source_channels = source_channels;
        ic.nernst_enabled  = use_nernst;
        ic.nernst_Ca_o     = Ca_o;
        ic.nernst_z        = z;
        ic.nernst_R        = R;
        ic.nernst_F        = F;
        ic.nernst_T        = T;
        ic.update_form     = use_nernst
            ? IntracellularSpec::UpdateForm::DRIVEN_DECAY_NERNST
            : IntracellularSpec::UpdateForm::DRIVEN_DECAY;
        return ic;
    }
};

} // namespace hodgkin_huxley
