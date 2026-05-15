# Task 22: Hines Solver + `MCPassivePool`

**Depends on:** task21 (`MorphologySpec` with `g_axial` available)  
**Unlocks:** task23 (`MCPool` reuses `HinesSolver`), task24 (Python API can expose passive pool for testing)

---

## What to implement

The passive cable equation and its O(C) tree solver. `MCPassivePool` is a minimal pool with no active channels — useful for verifying the solver against analytical cable theory before active gating is added in task23.

### Cable equation and Crank-Nicolson discretisation

For compartment `c` of neuron `n`, the lumped compartmental model gives:

```
C_m_c * A_c * dV_c/dt = -g_leak_c * A_c * (V_c - E_L)
                       + sum_{j in adj(c)} g_axial_{cj} * (V_j - V_c)
                       + I_ext_c
```

where:
- `A_c = pi * d_c * L_c * 1e-8` — membrane area in cm² (geometry in µm)
- `g_axial_{cj}` — axial coupling conductance between c and adjacent j (µS); for parent-child pairs, `g_axial_{cj} = min(g_axial[c], g_axial[j])` using the series combination at the junction (task22 uses the simpler `g_axial[child]` approximation; exact harmonic mean is a task23 refinement)
- `g_leak_c` — passive leak conductance density (µS/cm²); default 0.1 µS/cm²
- `E_L` — leak reversal, default −65 mV

Crank-Nicolson (θ = 0.5) gives the implicit system:

```
[C_m_c * A_c / dt + 0.5 * (g_leak_c * A_c + sum_j g_{cj})] * V_c^{n+1}
    - 0.5 * sum_j g_{cj} * V_j^{n+1}
= [C_m_c * A_c / dt - 0.5 * (g_leak_c * A_c + sum_j g_{cj})] * V_c^n
    + 0.5 * sum_j g_{cj} * V_j^n + I_ext_c
```

### Hines (1984) algorithm

The system matrix has tree sparsity. With the topological ordering invariant `parent_idx[c] < c`, the algorithm is:

**Forward elimination** (c = C-1 down to 1):
```
factor = off_diag[c] / diag[parent_idx[c]]
diag[parent_idx[c]] -= factor * off_diag[c]
rhs[parent_idx[c]]  -= factor * rhs[c]
```
where `off_diag[c] = -0.5 * g_axial[c]` (coupling between c and its parent).

**Solve root** (c = 0):
```
V[0] = rhs[0] / diag[0]
```

**Backward substitution** (c = 1 to C-1):
```
V[c] = (rhs[c] - off_diag[c] * V[parent_idx[c]]) / diag[c]
```

This is O(C) and handles arbitrary branching — branch points accumulate `factor` contributions from all children during the forward sweep because each child independently eliminates into its parent.

### `src/cpp/include/hodgkin_huxley/hines_solver.hpp` — new file

```cpp
#pragma once
#include "hodgkin_huxley/model/compartment_spec.hpp"
#include <vector>

namespace hodgkin_huxley {

class HinesSolver {
public:
    explicit HinesSolver(const MorphologySpec& morph);

    int n_comps() const { return n_comps_; }

    // Solve one time step for one neuron.
    //   V_prev[c]  : voltage at t^n (input)
    //   V_next[c]  : voltage at t^{n+1} (output, may alias V_prev)
    //   g_total[c] : sum of all ionic + leak conductances at c (µS)
    //   I_total[c] : sum of all current injections at c (nA)
    //   dt         : time step (ms)
    // g_total and I_total are the caller's responsibility to fill each step.
    void solve(const double* V_prev, double* V_next,
               const double* g_total, const double* I_total,
               double dt) const;

private:
    int                 n_comps_;
    std::vector<int>    parent_idx_;  // copy from MorphologySpec
    std::vector<double> Cm_A_;        // C_m_c * A_c (µF)
    std::vector<double> g_couple_;    // axial coupling g (µS), index = child comp

    // Reusable working arrays — mutable so solve() stays logically const.
    // Not thread-safe across concurrent calls on the same solver instance;
    // MCPool allocates one HinesSolver per pool, not per thread.
    mutable std::vector<double> diag_;
    mutable std::vector<double> off_diag_;
    mutable std::vector<double> rhs_;
};

} // namespace hodgkin_huxley
```

### `src/cpp/src/hines_solver.cpp` — new file

Constructor pre-computes `Cm_A_` and `g_couple_` so the hot path is allocation-free:

```cpp
HinesSolver::HinesSolver(const MorphologySpec& morph)
    : n_comps_(morph.n_comps()),
      parent_idx_(morph.parent_idx),
      diag_(morph.n_comps()),
      off_diag_(morph.n_comps(), 0.0),
      rhs_(morph.n_comps())
{
    Cm_A_.resize(n_comps_);
    g_couple_.resize(n_comps_, 0.0);
    for (int c = 0; c < n_comps_; ++c) {
        const auto& comp = morph.compartments[c];
        const double L_cm = comp.length_um  * 1e-4;
        const double d_cm = comp.diameter_um * 1e-4;
        // A_c in cm²; Cm in µF/cm² → Cm_A in µF
        Cm_A_[c] = comp.Cm * M_PI * d_cm * L_cm;
        if (c > 0) g_couple_[c] = morph.g_axial[c];
    }
}

void HinesSolver::solve(const double* V_prev, double* V_next,
                        const double* g_total, const double* I_total,
                        double dt) const
{
    // Build diagonal and RHS for Crank-Nicolson system.
    for (int c = 0; c < n_comps_; ++c) {
        const double Ca_dt = Cm_A_[c] / dt;
        const double gc    = g_couple_[c];  // coupling to parent (0 for root)
        // Sum coupling to children is accumulated in parent's diagonal below.
        diag_[c]    = Ca_dt + 0.5 * (g_total[c] + gc);
        off_diag_[c] = -0.5 * gc;
        rhs_[c]     = (Ca_dt - 0.5 * (g_total[c] + gc)) * V_prev[c]
                      + 0.5 * gc * (c > 0 ? V_prev[parent_idx_[c]] : 0.0)
                      + I_total[c];
    }
    // Add coupling-to-children contributions to parent diagonals.
    for (int c = 1; c < n_comps_; ++c) {
        diag_[parent_idx_[c]] += 0.5 * g_couple_[c];
        rhs_[parent_idx_[c]] += 0.5 * g_couple_[c] * V_prev[c];
    }

    // Forward elimination: leaves → root.
    for (int c = n_comps_ - 1; c >= 1; --c) {
        const int p = parent_idx_[c];
        const double factor = off_diag_[c] / diag_[p];
        diag_[p] -= factor * off_diag_[c];
        rhs_[p]  -= factor * rhs_[c];
    }

    // Solve root.
    V_next[0] = rhs_[0] / diag_[0];

    // Backward substitution: root → leaves.
    for (int c = 1; c < n_comps_; ++c) {
        V_next[c] = (rhs_[c] - off_diag_[c] * V_next[parent_idx_[c]]) / diag_[c];
    }
}
```

### `src/cpp/include/hodgkin_huxley/mc_passive_pool.hpp` — new file

```cpp
#pragma once
#include "hodgkin_huxley/pool/pool_base.hpp"
#include "hodgkin_huxley/hines_solver.hpp"
#include <Eigen/Dense>

namespace hodgkin_huxley {

// Passive (leak-only) multi-compartment pool.
// Used for solver validation and purely passive sub-tree models.
class MCPassivePool : public PoolBase {
public:
    explicit MCPassivePool(const NeuronModelSpec& spec);

    void   add(int n_neurons) override;
    void   step(double dt, const double* I_ext, int n_ext) override;
    bool   spiked(int neuron_idx) const override;
    double membrane_potential(int neuron_idx) const override;
    int    n_neurons() const override { return n_neurons_; }

    double V_comp(int neuron, int comp) const;
    int    n_comps() const { return solver_.n_comps(); }

    // PoolBase recording
    void scatter_V_into(double* buf, size_t n_rec, size_t t_rec) const override;

private:
    NeuronModelSpec      spec_;
    HinesSolver          solver_;
    Eigen::MatrixXd      V_;         // (n_neurons_, n_comps_), row = neuron
    std::vector<bool>    spiked_;
    int                  n_neurons_ = 0;

    // Per-step scratch — pre-allocated to n_comps_, no hot-loop allocation
    mutable std::vector<double> g_buf_;
    mutable std::vector<double> I_buf_;

    // Passive conductance and reversal per compartment (from spec channels if any,
    // otherwise uniform g_leak / E_L from spec parameters)
    std::vector<double> g_leak_;   // (n_comps_)  µS/cm² * A_c = µS
    double E_L_ = -65.0;           // mV
};

} // namespace hodgkin_huxley
```

`step()` fills `g_buf_[c] = g_leak_[c]` and `I_buf_[c] = I_ext_n[c]`, then calls `solver_.solve()` per neuron row. Spike detection: V_comp(n, 0) crosses 0 mV (soma threshold).

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/hines_solver.hpp` | New — `HinesSolver` class |
| `src/cpp/src/hines_solver.cpp` | New — constructor + `solve()` |
| `src/cpp/include/hodgkin_huxley/mc_passive_pool.hpp` | New — `MCPassivePool` |
| `src/cpp/src/mc_passive_pool.cpp` | New — `add()`, `step()`, accessors |
| `src/cpp/CMakeLists.txt` | Add `src/hines_solver.cpp`, `src/mc_passive_pool.cpp` |
| `src/python/bindings.cpp` | Bind `MCPassivePool` |

---

## Baseline tests (before PR to testing branch)

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] **Steady-state profile**: semi-infinite cable, constant current at soma, sealed distal end → voltage decays as `V(x) = V_0 * cosh((L-x)/λ) / cosh(L/λ)` with `λ = sqrt(d * Rm / (4 * Ra))`. 20-compartment cable, error < 1% of analytical.
- [ ] **Time constant**: single-compartment passive cell (1 comp), step current injection → `V(t) = V_inf * (1 - exp(-t/τ))` with `τ = Cm / g_leak`. Error < 0.5% at t = τ.
- [ ] **Sealed-end boundary**: no current leaks past the terminal compartment — verified by checking `g_axial[last] * (V[last] - V[last-1])` equals the injected current at steady state.
- [ ] **Branched morphology**: soma + 2 equal branches, same current at soma → identical voltage in both branches at all times.
- [ ] `HinesSolver::solve()` with `n_comps = 1` reduces to `V_next = (Cm_A/dt * V_prev + I) / (Cm_A/dt + g)` (single RC cell).

---

## Contract for downstream tasks

- `HinesSolver` is constructed once from a `const MorphologySpec&` and reused every step — no allocation in `solve()`.
- `solve(V_prev, V_next, g_total, I_total, dt)`: `g_total[c]` is the sum of ALL ionic conductances at c (leak + active channels); caller aggregates. `I_total[c]` is in nA.
- `V_next` may alias `V_prev` — the implementation does not read `V_prev` after writing `V_next` for a given index.
- task23's `MCPool` calls `solver_.solve()` once per neuron per step after summing active channel conductances into `g_buf_`.
