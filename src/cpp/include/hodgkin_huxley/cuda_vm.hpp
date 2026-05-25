#pragma once

#ifdef HH_USE_CUDA

#include "hodgkin_huxley/model/gate_spec.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace hodgkin_huxley {

struct DeviceVmProgram {
    VmInstruction* instructions = nullptr;
    double* constants = nullptr;
    int n_instructions = 0;
    int n_constants = 0;

    bool empty() const { return n_instructions == 0; }
};

inline void cuda_vm_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline DeviceVmProgram upload_vm_program(const VmExpr& expr) {
    DeviceVmProgram out;
    out.n_instructions = static_cast<int>(expr.instructions.size());
    out.n_constants = static_cast<int>(expr.constants.size());
    if (out.n_instructions > 0) {
        cuda_vm_check(
            cudaMalloc(reinterpret_cast<void**>(&out.instructions),
                       out.n_instructions * sizeof(VmInstruction)),
            "upload_vm_program cudaMalloc instructions");
        cuda_vm_check(
            cudaMemcpy(out.instructions, expr.instructions.data(),
                       out.n_instructions * sizeof(VmInstruction),
                       cudaMemcpyHostToDevice),
            "upload_vm_program cudaMemcpy instructions");
    }
    if (out.n_constants > 0) {
        cuda_vm_check(
            cudaMalloc(reinterpret_cast<void**>(&out.constants),
                       out.n_constants * sizeof(double)),
            "upload_vm_program cudaMalloc constants");
        cuda_vm_check(
            cudaMemcpy(out.constants, expr.constants.data(),
                       out.n_constants * sizeof(double),
                       cudaMemcpyHostToDevice),
            "upload_vm_program cudaMemcpy constants");
    }
    return out;
}

inline void free_vm_program(DeviceVmProgram& prog) {
    if (prog.instructions) cudaFree(prog.instructions);
    if (prog.constants) cudaFree(prog.constants);
    prog.instructions = nullptr;
    prog.constants = nullptr;
    prog.n_instructions = 0;
    prog.n_constants = 0;
}

template <int MAX_STACK = 64>
__device__ inline double eval_vm_program(
    const DeviceVmProgram& prog,
    double dep,
    double s_value,
    double a_value,
    const double* gate_values,
    int gate_count,
    const double* x_values,
    int x_count)
{
    if (prog.n_instructions == 0 || prog.instructions == nullptr) return 0.0;

    double stack[MAX_STACK];
    int sp = 0;

    auto push = [&](double v) {
        if (sp < MAX_STACK) stack[sp++] = v;
    };
    auto pop = [&]() {
        return (sp > 0) ? stack[--sp] : 0.0;
    };

    for (int pc = 0; pc < prog.n_instructions; ++pc) {
        const VmInstruction ins = prog.instructions[pc];
        switch (ins.op) {
            case VmOp::PUSH_DEP:
                push(dep);
                break;
            case VmOp::PUSH_CONST:
                push((ins.operand >= 0 && ins.operand < prog.n_constants)
                         ? prog.constants[ins.operand]
                         : 0.0);
                break;
            case VmOp::PUSH_GATE:
                push((gate_values && ins.operand >= 0 && ins.operand < gate_count)
                         ? gate_values[ins.operand]
                         : 0.0);
                break;
            case VmOp::PUSH_S:
                push(s_value);
                break;
            case VmOp::PUSH_A:
                push(a_value);
                break;
            case VmOp::PUSH_X:
                push((x_values && ins.operand >= 0 && ins.operand < x_count)
                         ? x_values[ins.operand]
                         : 0.0);
                break;
            case VmOp::ADD: {
                const double b = pop();
                const double a = pop();
                push(a + b);
                break;
            }
            case VmOp::MUL: {
                const double b = pop();
                const double a = pop();
                push(a * b);
                break;
            }
            case VmOp::NEG:
                push(-pop());
                break;
            case VmOp::RCP: {
                const double x = pop();
                push(1.0 / x);
                break;
            }
            case VmOp::POW_INT: {
                const double x = pop();
                push(pow(x, static_cast<double>(ins.operand)));
                break;
            }
            case VmOp::POW_HALF:
                push(sqrt(pop()));
                break;
            case VmOp::POW_GEN: {
                const double e = pop();
                const double x = pop();
                push(pow(x, e));
                break;
            }
            case VmOp::EXP:
                push(exp(pop()));
                break;
            case VmOp::LOG:
                push(log(pop()));
                break;
            case VmOp::TANH:
                push(tanh(pop()));
                break;
            case VmOp::SIN:
                push(sin(pop()));
                break;
            case VmOp::COS:
                push(cos(pop()));
                break;
            case VmOp::SQRT:
                push(sqrt(pop()));
                break;
            case VmOp::ABS:
                push(fabs(pop()));
                break;
        }
    }

    return (sp > 0) ? stack[sp - 1] : 0.0;
}

} // namespace hodgkin_huxley

#endif // HH_USE_CUDA
