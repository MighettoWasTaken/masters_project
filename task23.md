# Task 23: `MCPool` — Active Multi-Compartment Pool

**Depends on:** task22 (`HinesSolver`, `MCPassivePool` for reference), task14 (intracellular dynamics logic to reuse)  
**Unlocks:** task24 (Python API and `RegionalNetwork` dispatch to `MCPool`)

---

## What to implement

Full active multi-compartment pool with per-compartment gate states, channel currents, and intracellular substance dynamics. Extends the solver from task22 with the gate/channel/intracellular step logic from `ComposablePool`, generalised over compartments.

### State layout

All state arrays are flat `std::vector<double>` indexed by stride helpers to avoid 3D matrix overhead.

```
V_            : Eigen::MatrixXd (N, C)
                row = neuron index, col = compartment index

gate_         : std::vector<double>, length = N * C * G_max
                index: gate_idx(n, c, g) = n*C*G_max + c*G_max + g
                G_max = max gate count across all compartments

X_            : std::vector<double>, length = N * C * S_max
                S_max = max substance count across all compartments

E_nernst_     : std::vector<double>, length = N * C * S_max
                (only populated for substance slots with nernst_enabled)

synapse_g_scale_ : Eigen::ArrayXd (N)
                per-neuron synaptic modulation factor (SYNAPSE_G target)
```

Stride helpers (private methods):
```cpp
int gate_idx(int n, int c, int g) const { return n*n_comps_*G_max_ + c*G_max_ + g; }
int subst_idx(int n, int c, int s) const { return n*n_comps_*S_max_ + c*S_max_ + s; }
```

### `src/cpp/include/hodgkin_huxley/mc_pool.hpp` — new file

```cpp
#pragma once
#include "hodgkin_huxley/pool/pool_base.hpp"
#include "hodgkin_huxley/hines_solver.hpp"
#include <Eigen/Dense>
#include <vector>

namespace hodgkin_huxley {

class MCPool : public PoolBase {
public:
    explicit MCPool(const NeuronModelSpec& spec);

    void   add(int n_neurons) override;
    void   step(double dt, const double* I_ext, int n_ext) override;
    bool   spiked(int neuron_idx) const override;
    double membrane_potential(int neuron_idx) const override;
    int    n_neurons() const override { return n_neurons_; }

    double V_comp(int neuron, int comp) const;
    int    n_comps() const { return n_comps_; }
    int    n_gates_at(int comp) const;
    int    n_substances_at(int comp) const;

    // Recording
    void scatter_V_into(double* buf, size_t n_rec, size_t t_rec) const override;
    void scatter_V_comp_into(int comp, double* buf, size_t n_rec, size_t t_rec) const;
    void scatter_substance_into(size_t s_idx, double* buf,
                                size_t n_rec, size_t t_rec) const override;
    void scatter_synapse_g_scale(double* buf) const override;

    // PoolBase CUDA interface stubs — always false (task17 MCPool CUDA is out of scope)
    bool is_cuda()    const override { return false; }
    int  device_id()  const override { return -1; }

private:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    NeuronModelSpec spec_;
    int             n_comps_;
    int             G_max_;   // max gates per compartment
    int             S_max_;   // max substances per compartment
    int             n_neurons_ = 0;

    // Per-compartment gate/substance counts (size = n_comps_)
    std::vector<int> n_gates_;
    std::vector<int> n_subst_;

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    Eigen::MatrixXd   V_;              // (N, C)
    std::vector<double> gate_;         // (N × C × G_max)
    std::vector<double> X_;            // (N × C × S_max)
    std::vector<double> E_nernst_;     // (N × C × S_max)
    Eigen::ArrayXd    synapse_g_scale_; // (N)
    std::vector<bool> spiked_;

    // -----------------------------------------------------------------------
    // Solver
    // -----------------------------------------------------------------------
    HinesSolver solver_;

    // Per-step scratch — pre-allocated, no hot-loop allocation
    std::vector<double> g_buf_;   // (C) total ionic conductance per compartment
    std::vector<double> I_buf_;   // (C) total current per compartment

    // -----------------------------------------------------------------------
    // Step helpers
    // -----------------------------------------------------------------------
    void step_gates(int n, double dt);
    void step_channels(int n, double* g_buf, double* I_buf) const;
    void step_intracellular(int n, double dt, const double* I_buf);
    void apply_modulations(int n);
    double eval_gate_product(int n, int c, const ChannelSpec& ch) const;
};

} // namespace hodgkin_huxley
```

### `src/cpp/src/mc_pool.cpp` — new file

#### `step()` — the hot loop

```cpp
void MCPool::step(double dt, const double* I_ext, int n_ext) {
    synapse_g_scale_.setOnes();

    for (int n = 0; n < n_neurons_; ++n) {
        // 1. Gate step (forward Euler on gate variables — same as ComposablePool)
        step_gates(n, dt);

        // 2. Channel currents → fill g_buf_[c] and I_buf_[c]
        std::fill(g_buf_.begin(), g_buf_.end(), 0.0);
        std::fill(I_buf_.begin(), I_buf_.end(), 0.0);
        // Add external current (I_ext laid out as [n * C + c])
        for (int c = 0; c < n_comps_; ++c)
            I_buf_[c] = (n_ext > 0) ? I_ext[n * n_comps_ + c] : 0.0;
        step_channels(n, g_buf_.data(), I_buf_.data());

        // 3. Cable solve → new V row
        double* V_row = V_.row(n).data();
        solver_.solve(V_row, V_row, g_buf_.data(), I_buf_.data(), dt);

        // 4. Intracellular dynamics + modulations
        step_intracellular(n, dt, I_buf_.data());
        apply_modulations(n);

        // 5. Spike detection at soma (compartment 0), threshold 0 mV
        const bool prev_above = spiked_[n];
        spiked_[n] = (V_row[0] >= 0.0);
        // Rising edge detection for output
        // (pool_base spike semantics: spiked() true on the step V crosses threshold)
    }
}
```

#### `step_gates(n, dt)`

Iterates `spec_.morphology.compartments[c].gates` for each compartment c. Gate update logic is identical to `ComposablePool::step()` — reuse the same `eval_inf_tau()` and `eval_alpha_beta()` helpers, but indexed by `gate_idx(n, c, g)` instead of a flat gate array.

#### `step_channels(n, g_buf, I_buf)`

For each compartment c and channel ch:
```cpp
const double A_c = M_PI * comp.diameter_um * 1e-4 * comp.length_um * 1e-4; // cm²
const double gp  = ch.g * A_c * eval_gate_product(n, c, ch);
const double E   = /* channel E_rev, or E_nernst_[subst_idx(n,c,nernst_idx)] if set */;
g_buf[c] += gp;
I_buf[c] += gp * E;  // I = g*(E - V) → split into g*E and -g*V terms
                      // solver receives g_total and I_total = g*E + I_ext;
                      // the -g*V term is handled implicitly in HinesSolver diagonal
```

Note: `I_total[c]` passed to `solver_.solve()` must be `sum(gp*E)` + I_ext — the `−g*V` term is absorbed into the diagonal via `g_total[c]`.

#### `step_intracellular(n, dt, I_ch_buf)`

For each compartment c and substance s in `comp.intracellular`:
- Reuse `ComposablePool` ODE evaluation logic, but read/write `X_[subst_idx(n, c, s)]` and `E_nernst_[subst_idx(n, c, s)]`
- Source channel currents `I_ch_buf[c]` are the current cache from `step_channels`

#### `apply_modulations(n)`

For each compartment c, substance s, modulation `mod`:
- `SYNAPSE_G`: `synapse_g_scale_[n] *= eval_mod_vm(mod.mod_vm, X_[subst_idx(n,c,s)], ...)`
- All other targets (CHANNEL_G, GATE_INF_SHIFT, etc.) are applied inline during the next step's `step_channels` / `step_gates` via pre-computed modifier arrays (same deferred pattern as `ComposablePool`)

---

## Key files

| File | Change |
|---|---|
| `src/cpp/include/hodgkin_huxley/mc_pool.hpp` | New — `MCPool` class |
| `src/cpp/src/mc_pool.cpp` | New — all step methods |
| `src/cpp/CMakeLists.txt` | Add `src/mc_pool.cpp` |
| `src/python/bindings.cpp` | Bind `MCPool`, expose `V_comp`, `n_comps`, `scatter_V_comp_into` |

---

## Baseline tests (before PR to testing branch)

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] **Single-compartment parity**: `MCPool` with `n_comps=1` and standard HH channels produces action potentials matching `ComposablePool` output to within 0.1 mV across a 200 ms simulation
- [ ] **Backpropagating AP**: soma (comp 0) with Na/K channels, dendrite (comp 1) passive — somatic spike produces a voltage transient in dendrite within 2 ms, attenuated to < 50% of soma peak
- [ ] **Compartment isolation**: channel active only in comp 1 does not affect comp 0 voltage in the absence of axial coupling (set `g_axial[1] = 0` manually)
- [ ] **Intracellular per-compartment**: CaT channel only in dendrite; Ca accumulation measured only in `X_[subst_idx(n, 1, 0)]`, not in compartment 0
- [ ] **SYNAPSE_G modulation**: dopamine substance with `SYNAPSE_G` target → `synapse_g_scale_` deviates from 1.0 after simulation; verified via `scatter_synapse_g_scale()`
- [ ] `V_comp(n, c)` returns correct values; `scatter_V_comp_into(c, ...)` populates buffer correctly

---

## Contract for downstream tasks

- `MCPool::membrane_potential(n)` returns `V_comp(n, 0)` (soma) for `PoolBase` compatibility — synaptic current routing in `Network` uses this for voltage-gated synapses.
- `scatter_V_comp_into(comp, buf, n_rec, t_rec)` is the recording hook used by task24's compartment-resolved recording.
- `synapse_g_scale_` is reset to 1.0 at the start of each `step()` before modulations are applied, matching `ComposablePool` convention from task14.
- `I_ext` layout for multi-compartment: flat `[n * C + c]` — task24's `RegionalNetwork` must supply `I_ext` with this layout when driving an `MCPool`.
