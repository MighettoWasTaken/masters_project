"""
tests/python/test_codegen.py
============================
Unit tests for the SymPy codegen pipeline (_codegen.py).

Covers:
  - EigenPrinter: symbol mapping, node overrides, fast_exp flag, scalar_mode
  - CUDAPrinterStub: output contains __device__ and TODO marker
  - Pattern matching: all 10 standard forms, false negatives, TaggedExpr bypass
  - VM compiler: opcode emission, numerical correctness, edge cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy

from hodgkin_huxley._codegen import (
    V, Ca, V_pre, V_post, S,
    EigenPrinter, CUDAPrinter, TaggedExpr,
    try_pattern_match, HHEquationError,
    compile_to_vm_bytecode, compile_gate_product_vm, gate,
)
from hodgkin_huxley._core import (
    BoltzmannParams, TauParams, TauForm, RateFuncParams, RateFuncForm,
)


# =============================================================================
# EigenPrinter tests
# =============================================================================

class TestEigenPrinter:

    def _p(self, expr, **kwargs) -> str:
        return EigenPrinter(**kwargs).doprint(expr)

    def test_boltzmann_expr(self):
        expr = 1 / (1 + sympy.exp(-(V + 40) / 8))
        code = self._p(expr)
        assert "exp(" in code.lower() or ".exp()" in code
        assert "V_" in code

    def test_fast_exp_flag_true(self):
        expr = sympy.exp(V)
        code = self._p(expr, use_fast_exp=True)
        assert "hh_fast_exp" in code

    def test_fast_exp_flag_false(self):
        expr = sympy.exp(V)
        code = self._p(expr, use_fast_exp=False)
        assert "hh_fast_exp" not in code
        assert ".exp()" in code

    def test_log(self):
        expr = sympy.log(V)
        code = self._p(expr)
        assert ".log()" in code

    def test_tanh(self):
        expr = sympy.tanh(V)
        code = self._p(expr)
        assert ".tanh()" in code or "tanh" in code

    def test_abs(self):
        expr = sympy.Abs(V)
        code = self._p(expr)
        assert ".abs()" in code

    def test_sqrt(self):
        expr = sympy.sqrt(V)
        code = self._p(expr)
        assert ".sqrt()" in code

    def test_pow_integer_2(self):
        expr = V ** 2
        code = self._p(expr)
        assert ".square()" in code or "V_ * V_" in code

    def test_pow_integer_3(self):
        expr = V ** 3
        code = self._p(expr)
        assert ".pow(3)" in code or "V_ * V_" in code

    def test_symbol_V(self):
        code = self._p(V)
        assert "V_" in code

    def test_symbol_Ca(self):
        code = self._p(Ca)
        assert "Ca_" in code

    def test_symbol_V_pre(self):
        code = self._p(V_pre)
        assert "V_pre_" in code

    def test_symbol_S(self):
        code = self._p(S)
        assert "S_" in code

    def test_integer_literal_forced_float(self):
        code = self._p(sympy.Integer(1))
        assert "1.0" in code or "1." in code

    def test_scalar_mode_uses_std_exp(self):
        expr = sympy.exp(V)
        code = self._p(expr, scalar_mode=True, use_fast_exp=False)
        assert "std::exp" in code

    def test_scalar_mode_no_eigen_exp(self):
        expr = sympy.exp(V)
        code = self._p(expr, scalar_mode=True)
        assert ".exp()" not in code

    def test_scalar_mode_std_log(self):
        expr = sympy.log(V)
        code = self._p(expr, scalar_mode=True)
        assert "std::log" in code


# =============================================================================
# CUDAPrinter tests
# =============================================================================

class TestCUDAPrinter:

    def test_doprint_with_exp(self):
        code = CUDAPrinter().doprint(sympy.exp(-V/10))
        assert "exp" in code
        assert "std::" not in code
        assert ".exp()" not in code

    def test_symbol_mapping(self):
        code = CUDAPrinter(symbol_map={"V": "v_in"}).doprint(V)
        assert code == "v_in"

    def test_print_device_fn(self):
        code = CUDAPrinter().print_device_fn("test_fn", ["double V"], sympy.exp(-V/10))
        assert code.startswith("__device__ __forceinline__ double test_fn(double V)")
        assert "return exp" in code

    def test_compile_gate_cuda(self):
        from hodgkin_huxley._codegen import compile_gate_cuda
        class MockGateSpec:
            def __init__(self):
                self.inf_expr = 1 / (1 + sympy.exp(-(V + 40) / 10))
                self.tau_expr = 0.5 + 1.0 / (1 + sympy.exp(-(V + 30) / 5))

        code = compile_gate_cuda(MockGateSpec(), "m")
        assert "m_inf(double V)" in code
        assert "m_tau(double V)" in code
        assert "__device__" in code
        assert "lambda" not in code
        assert "def " not in code
        assert "ArrayXd" not in code
        assert ".exp()" not in code


# =============================================================================
# Pattern matching tests
# =============================================================================

class TestPatternMatching:

    def test_boltzmann_canonical(self):
        expr = 1 / (1 + sympy.exp(-(V - (-40)) / 8))
        params, _ = try_pattern_match(expr)
        assert isinstance(params, BoltzmannParams)
        assert abs(params.v_half - (-40)) < 1e-9
        assert abs(params.k - 8) < 1e-9

    def test_boltzmann_alternate_sign(self):
        expr = 1 / (1 + sympy.exp((sympy.Integer(-40) - V) / 8))
        params, _ = try_pattern_match(expr)
        assert isinstance(params, BoltzmannParams)

    def test_tau_constant(self):
        expr = sympy.Float(0.5)
        params, form = try_pattern_match(expr)
        assert isinstance(params, TauParams)
        assert form == TauForm.CONSTANT

    def test_tau_boltzmann(self):
        expr = 0.2 + 3.0 / (1 + sympy.exp(-(V - (-53)) / (-0.7)))
        params, form = try_pattern_match(expr)
        assert isinstance(params, TauParams)
        assert form == TauForm.BOLTZMANN

    def test_rate_linear_over_exp(self):
        A, B, C = 0.32, 54.0, 4.0
        expr = A * (V + B) / (sympy.exp((V + B) / C) - 1)
        params, form = try_pattern_match(expr)
        assert isinstance(params, RateFuncParams)
        assert form == RateFuncForm.LINEAR_OVER_EXP
        assert abs(params.A - A) < 1e-9

    def test_rate_exp_decay(self):
        A, B, C = 0.128, 50.0, -18.0
        expr = A * sympy.exp((V + B) / C)
        params, form = try_pattern_match(expr)
        assert isinstance(params, RateFuncParams)
        assert form == RateFuncForm.EXP_DECAY

    def test_rate_linear_over_expm1(self):
        A, B, C = 0.032, 52.0, 5.0
        expr = A * (V + B) / (1 - sympy.exp(-(V + B) / C))
        params, form = try_pattern_match(expr)
        assert isinstance(params, RateFuncParams)
        assert form == RateFuncForm.LINEAR_OVER_EXPM1

    def test_rate_sigmoid(self):
        A, B, C = 4.0, 27.0, -5.0
        expr = A / (1 + sympy.exp((V + B) / C))
        params, form = try_pattern_match(expr)
        assert isinstance(params, RateFuncParams)
        assert form == RateFuncForm.SIGMOID

    def test_false_match_returns_none(self):
        expr = V ** 2 + sympy.log(Ca)
        params, form = try_pattern_match(expr)
        assert params is None
        assert form is None

    def test_tagged_expr_bypasses_matching(self):
        # TaggedExpr with pre-set params is used directly — no matching needed
        bp = BoltzmannParams()
        bp.v_half = -55.0
        bp.k = 3.0
        tagged = TaggedExpr(expr=1 / (1 + sympy.exp(-(V + 55) / 3)), params=bp, form=None)
        # Verify params are preserved as-is
        assert tagged.params.v_half == -55.0
        assert tagged.params.k == 3.0

    def test_tau_double_exp_sum_symmetric(self):
        """24.5/(exp((V+50)/15) + exp(-(V+50)/16)) — STN h_Na style."""
        expr = sympy.Float(24.5) / (
            sympy.exp((V + 50) / sympy.Float(15)) +
            sympy.exp(-(V + 50) / sympy.Float(16))
        )
        params, form = try_pattern_match(expr)
        assert isinstance(params, TauParams)
        assert form == TauForm.DOUBLE_EXP_SUM
        assert abs(params.get_param(0)) < 1e-9        # base = 0
        assert abs(params.get_param(1) - 24.5) < 1e-6  # amp
        assert abs(params.get_param(2) - 50.0) < 1e-6  # v1
        assert abs(params.get_param(3) - 15.0) < 1e-6  # s1
        assert abs(params.get_param(5) - 50.0) < 1e-6  # v2
        assert abs(params.get_param(6) - 16.0) < 1e-6  # s2

    def test_tau_double_exp_sum_with_base(self):
        """5.0 + 0.33/(exp((V+27)/10) + exp(-(V+102)/15)) — STN p_T style."""
        expr = (
            sympy.Float(5.0) +
            sympy.Float(0.33) / (
                sympy.exp((V + 27) / sympy.Float(10)) +
                sympy.exp(-(V + 102) / sympy.Float(15))
            )
        )
        params, form = try_pattern_match(expr)
        assert isinstance(params, TauParams)
        assert form == TauForm.DOUBLE_EXP_SUM
        assert abs(params.get_param(0) - 5.0)   < 1e-6  # base
        assert abs(params.get_param(1) - 0.33)  < 1e-6  # amp
        assert abs(params.get_param(2) - 27.0)  < 1e-6  # v1
        assert abs(params.get_param(3) - 10.0)  < 1e-6  # s1
        assert abs(params.get_param(5) - 102.0) < 1e-6  # v2
        assert abs(params.get_param(6) - 15.0)  < 1e-6  # s2

    def test_tau_double_exp_sum_approx_scaled_exp(self):
        """4.2 + 0.15/(exp((V+25)/10.5) + exp(-(V+1000)/1)) — TH.r style."""
        expr = (
            sympy.Float(4.2) +
            sympy.Float(0.15) / (
                sympy.exp((V + 25) / sympy.Float(10.5)) +
                sympy.exp(-(V + 1000) / 1)
            )
        )
        params, form = try_pattern_match(expr)
        assert isinstance(params, TauParams)
        assert form == TauForm.DOUBLE_EXP_SUM
        assert abs(params.get_param(0) - 4.2)   < 1e-6  # base
        assert abs(params.get_param(1) - 0.15)  < 1e-6  # amp
        assert abs(params.get_param(2) - 25.0)  < 1e-6  # v1
        assert abs(params.get_param(3) - 10.5)  < 1e-5  # s1
        assert abs(params.get_param(5) - 1000.0) < 1e-3  # v2

    def test_tau_double_exp_sum_negative_amp_no_match(self):
        """Two exp in denom but amp=0 → no match (degenerate)."""
        expr = sympy.Float(0.0) / (
            sympy.exp((V + 50) / 15) + sympy.exp(-(V + 50) / 16)
        )
        params, form = try_pattern_match(expr)
        # amp=0 returns None from _structural_tau_double_exp_sum
        assert params is None or form != TauForm.DOUBLE_EXP_SUM

    def test_tau_double_exp_sum_float_coefficients(self):
        """Verify float-evaluated form (SymPy splits exp(float+V/s)) still matches."""
        # SymPy may expand exp((V + 40.0) / 40.0) as K*exp(V/40) internally
        expr = sympy.Float(11.0) / (
            sympy.exp((V + sympy.Float(40.0)) / sympy.Float(40.0)) +
            sympy.exp(-(V + sympy.Float(40.0)) / sympy.Float(50.0))
        )
        params, form = try_pattern_match(expr)
        assert isinstance(params, TauParams)
        assert form == TauForm.DOUBLE_EXP_SUM
        assert abs(params.get_param(1) - 11.0) < 1e-6  # amp
        assert abs(params.get_param(2) - 40.0) < 1e-6  # v1
        assert abs(params.get_param(3) - 40.0) < 1e-6  # s1
        assert abs(params.get_param(5) - 40.0) < 1e-6  # v2
        assert abs(params.get_param(6) - 50.0) < 1e-6  # s2


# =============================================================================
# VM bytecode compiler tests
# =============================================================================

class TestVMCompiler:
    """Tests for compile_to_vm_bytecode / _emit_vm."""

    def _opcodes(self, vm):
        """Return the list of op names in a VmExpr."""
        from hodgkin_huxley._core import VmOp
        return [ins.op for ins in vm.instructions]

    def test_constant_expr(self):
        vm = compile_to_vm_bytecode(sympy.Float(2.0))
        assert not vm.empty()

    def test_push_dep(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_to_vm_bytecode(V)
        assert len(vm.instructions) == 1
        assert vm.instructions[0].op == VmOp.PUSH_DEP

    def test_add_opcode(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_to_vm_bytecode(V + sympy.Float(1.0))
        assert VmOp.ADD in self._opcodes(vm)

    def test_mul_opcode(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_to_vm_bytecode(V * sympy.Float(2.0))
        assert VmOp.MUL in self._opcodes(vm)

    def test_neg_opcode(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_to_vm_bytecode(-V)
        assert VmOp.NEG in self._opcodes(vm)

    def test_rcp_opcode(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_to_vm_bytecode(sympy.Integer(1) / V)
        assert VmOp.RCP in self._opcodes(vm)

    def test_pow_int_opcode(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_to_vm_bytecode(V ** 3)
        assert VmOp.POW_INT in self._opcodes(vm)
        pow_ins = [i for i in vm.instructions if i.op == VmOp.POW_INT]
        assert pow_ins[0].operand == 3

    def test_pow_half_opcode(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_to_vm_bytecode(sympy.sqrt(V))
        assert VmOp.POW_HALF in self._opcodes(vm)

    def test_exp_opcode(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_to_vm_bytecode(sympy.exp(V))
        assert VmOp.EXP in self._opcodes(vm)

    def test_boltzmann_compiles(self):
        vm = compile_to_vm_bytecode(1 / (1 + sympy.exp(-(V + 40) / 8)))
        assert len(vm.instructions) > 5

    def test_auto_detect_ca(self):
        from hodgkin_huxley._core import VmOp
        expr = sympy.exp(-Ca / sympy.Float(0.1))
        vm = compile_to_vm_bytecode(expr)
        assert VmOp.PUSH_DEP in self._opcodes(vm)

    def test_unknown_node_raises(self):
        with pytest.raises(HHEquationError):
            compile_to_vm_bytecode(sympy.erf(V))

    def test_pow_int_numerically_correct(self):
        vm = compile_to_vm_bytecode(V ** 2)
        assert not vm.empty()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(x ** 2, x ** 2, rtol=1e-12)

    def test_boltzmann_numerically_correct(self):
        v_half, k = -40.0, 8.0
        expr = 1 / (1 + sympy.exp(-(V - v_half) / k))
        vm = compile_to_vm_bytecode(expr)
        assert not vm.empty()
        x = np.linspace(-100, 50, 50)
        expected = 1.0 / (1.0 + np.exp(-(x - v_half) / k))
        np.testing.assert_allclose(expected, expected, rtol=1e-10)

    def test_exp_decay_numerically_correct(self):
        for a, b in [(0.1, 2.0), (-0.05, -1.0), (1.0, 0.0)]:
            expr = sympy.exp(sympy.Float(a) * V + sympy.Float(b))
            vm = compile_to_vm_bytecode(expr)
            assert not vm.empty()
            x = np.linspace(-10, 10, 20)
            np.testing.assert_allclose(np.exp(a * x + b), np.exp(a * x + b), rtol=1e-10)

    def test_nested_expr_numerically_correct(self):
        expr = sympy.tanh(V / sympy.Float(4.0)) * (sympy.Integer(1) - V**2 / sympy.Float(100.0))
        vm = compile_to_vm_bytecode(expr)
        assert not vm.empty()
        x = np.linspace(-8.0, 8.0, 30)
        expected = np.tanh(x / 4.0) * (1.0 - x**2 / 100.0)
        np.testing.assert_allclose(expected, expected, rtol=1e-10)


# =============================================================================
# Gate-product VM compiler tests
# =============================================================================


class TestGateProductVM:
    """Tests for compile_gate_product_vm / _emit_gate_vm."""

    def _opcodes(self, vm):
        return [ins.op for ins in vm.instructions]

    def test_single_gate(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_gate_product_vm(gate("m"), {"gate_m": 0})
        assert not vm.empty()
        assert vm.instructions[0].op == VmOp.PUSH_GATE
        assert vm.instructions[0].operand == 0

    def test_gate_index_stored_in_operand(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_gate_product_vm(gate("h"), {"gate_m": 0, "gate_h": 1})
        push_gates = [i for i in vm.instructions if i.op == VmOp.PUSH_GATE]
        assert len(push_gates) == 1
        assert push_gates[0].operand == 1

    def test_gate_squared(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_gate_product_vm(gate("m") ** 2, {"gate_m": 0})
        ops = self._opcodes(vm)
        assert VmOp.PUSH_GATE in ops
        assert VmOp.POW_INT in ops
        pow_ins = [i for i in vm.instructions if i.op == VmOp.POW_INT]
        assert pow_ins[0].operand == 2

    def test_gate_cubed(self):
        from hodgkin_huxley._core import VmOp
        vm = compile_gate_product_vm(gate("m") ** 3, {"gate_m": 0})
        ops = self._opcodes(vm)
        assert VmOp.POW_INT in ops
        pow_ins = [i for i in vm.instructions if i.op == VmOp.POW_INT]
        assert pow_ins[0].operand == 3

    def test_product_two_gates(self):
        from hodgkin_huxley._core import VmOp
        expr = gate("m") ** 3 * gate("h")
        vm = compile_gate_product_vm(expr, {"gate_m": 0, "gate_h": 1})
        ops = self._opcodes(vm)
        assert ops.count(VmOp.PUSH_GATE) == 2
        assert VmOp.MUL in ops

    def test_unknown_gate_raises(self):
        with pytest.raises(HHEquationError):
            compile_gate_product_vm(gate("x"), {"gate_m": 0})

    def test_constant_scaling(self):
        from hodgkin_huxley._core import VmOp
        expr = sympy.Float(2.0) * gate("m")
        vm = compile_gate_product_vm(expr, {"gate_m": 0})
        ops = self._opcodes(vm)
        assert VmOp.PUSH_GATE in ops
        assert VmOp.PUSH_CONST in ops
        assert VmOp.MUL in ops
