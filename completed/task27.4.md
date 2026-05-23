# Task 27.4: SynArrays Remaining size_t + E_syn Narrowing

**Role:** Team lead  
**Status:** Completed  
**Depends on:** task27.2 (SynArrays narrowing baseline)

---

## What to implement

Narrow remaining `size_t` fields in `SynArrays` and narrow `E_syn` from double to float:

| Field | Before | After | Bytes saved/syn |
|---|---|---|---|
| `spec_idx` | size_t (8) | uint32_t (4) | 4 |
| `plast_spec_idx_arr` | size_t (8) | uint32_t (4) | 4 |
| `E_syn` | double (8) | float (4) | 4 |

**E_syn safety:** biological reversal potentials are -80/0/+40 mV — at most 3 significant
figures, well within float32's 7. The hot-loop expression `g * (E_syn - V)` promotes
the float E_syn to double for the subtraction (V is double), so arithmetic precision
is unchanged.

## Key files

- `src/cpp/include/hodgkin_huxley/network.hpp` — `SynArrays` struct field types
- `src/cpp/src/network.cpp` — push casts; `const float* E_syn_data` in all 4 hot-loop sites

## Contract

- All 4 hot-loop `E_syn` pointer declarations changed from `const double*` to `const float*`
- All tests pass unchanged
- Savings: 12 bytes × N_synapses
