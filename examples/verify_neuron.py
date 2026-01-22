#!/usr/bin/env python3
"""
Comprehensive verification of HH neuron behavior.

Generates multiple figures to verify:
1. Single neuron membrane potential with constant current
2. Gating variables over time
3. Comparison of integration methods
4. F-I curve (firing rate vs current)
5. Phase plane analysis
6. Current clamp series
7. Parameter sensitivity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hodgkin_huxley import HHNeuron, IntegrationMethod, HHParameters


def setup_output_dir():
    """Create output directory for figures."""
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    return figs_dir


def count_spikes(trace, threshold=0.0):
    """Count action potentials using threshold crossing."""
    above = np.array(trace) > threshold
    crossings = np.diff(above.astype(int))
    return np.sum(crossings == 1)


def plot_membrane_potential(figs_dir):
    """Plot membrane potential with constant current injection."""
    print("Generating membrane potential plot...")

    duration = 100.0
    dt = 0.01
    I_ext = 10.0

    neuron = HHNeuron()
    trace = neuron.simulate(duration=duration, dt=dt, I_ext=I_ext)
    time = np.arange(0, duration, dt)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, trace, 'b-', linewidth=0.8)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='0 mV threshold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title(f'HH Neuron Membrane Potential (I_ext = {I_ext} uA/cm^2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    fig.tight_layout()
    fig.savefig(figs_dir / "01_membrane_potential.png", dpi=150)
    plt.close(fig)

    print(f"  Spikes detected: {count_spikes(trace)}")
    print(f"  Voltage range: [{min(trace):.1f}, {max(trace):.1f}] mV")


def plot_gating_variables(figs_dir):
    """Plot gating variables during action potential."""
    print("Generating gating variables plot...")

    duration = 20.0
    dt = 0.01
    I_ext = 15.0

    neuron = HHNeuron()
    time_points = []
    V_trace = []
    m_trace = []
    h_trace = []
    n_trace = []

    num_steps = int(duration / dt)
    for i in range(num_steps):
        state = neuron.state
        time_points.append(i * dt)
        V_trace.append(state.V)
        m_trace.append(state.m)
        h_trace.append(state.h)
        n_trace.append(state.n)
        neuron.step(dt, I_ext)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Membrane potential
    axes[0].plot(time_points, V_trace, 'b-', linewidth=1)
    axes[0].set_ylabel('V (mV)')
    axes[0].set_title('Membrane Potential and Gating Variables')
    axes[0].grid(True, alpha=0.3)

    # Gating variables
    axes[1].plot(time_points, m_trace, 'r-', label='m (Na activation)', linewidth=1)
    axes[1].plot(time_points, h_trace, 'g-', label='h (Na inactivation)', linewidth=1)
    axes[1].plot(time_points, n_trace, 'b-', label='n (K activation)', linewidth=1)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Gating Variable')
    axes[1].legend(loc='right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(figs_dir / "02_gating_variables.png", dpi=150)
    plt.close(fig)


def plot_integration_comparison(figs_dir):
    """Compare Euler and RK4 integration methods."""
    print("Generating integration method comparison...")

    duration = 50.0
    dt_coarse = 0.05
    dt_fine = 0.001
    I_ext = 15.0

    # Reference (RK4 with very fine dt)
    ref = HHNeuron()
    ref.integration_method = IntegrationMethod.RK4
    trace_ref = ref.simulate(duration, dt_fine, I_ext)
    time_ref = np.arange(0, duration, dt_fine)

    # Euler with coarse dt
    euler = HHNeuron()
    euler.integration_method = IntegrationMethod.EULER
    trace_euler = euler.simulate(duration, dt_coarse, I_ext)
    time_euler = np.arange(0, duration, dt_coarse)

    # RK4 with coarse dt
    rk4 = HHNeuron()
    rk4.integration_method = IntegrationMethod.RK4
    trace_rk4 = rk4.simulate(duration, dt_coarse, I_ext)
    time_rk4 = np.arange(0, duration, dt_coarse)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Full traces
    axes[0].plot(time_ref, trace_ref, 'k-', label=f'Reference (RK4, dt={dt_fine})', linewidth=0.8, alpha=0.7)
    axes[0].plot(time_euler, trace_euler, 'r--', label=f'Euler (dt={dt_coarse})', linewidth=1.5)
    axes[0].plot(time_rk4, trace_rk4, 'b--', label=f'RK4 (dt={dt_coarse})', linewidth=1.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Membrane Potential (mV)')
    axes[0].set_title('Integration Method Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoomed view of first spike
    axes[1].plot(time_ref, trace_ref, 'k-', label='Reference', linewidth=0.8)
    axes[1].plot(time_euler, trace_euler, 'r--', label='Euler', linewidth=1.5, marker='o', markersize=3)
    axes[1].plot(time_rk4, trace_rk4, 'b--', label='RK4', linewidth=1.5, marker='s', markersize=3)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Membrane Potential (mV)')
    axes[1].set_title('Zoomed View (First Action Potential)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 15)

    fig.tight_layout()
    fig.savefig(figs_dir / "03_integration_comparison.png", dpi=150)
    plt.close(fig)

    # Print error metrics
    print(f"  Peak voltage - Reference: {max(trace_ref):.2f} mV")
    print(f"  Peak voltage - Euler:     {max(trace_euler):.2f} mV")
    print(f"  Peak voltage - RK4:       {max(trace_rk4):.2f} mV")


def plot_fi_curve(figs_dir):
    """Plot firing rate vs injected current (F-I curve)."""
    print("Generating F-I curve...")

    duration = 1000.0
    dt = 0.01
    currents = np.arange(0, 30, 1)
    firing_rates = []

    for I_ext in currents:
        neuron = HHNeuron()
        trace = neuron.simulate(duration, dt, I_ext)
        spikes = count_spikes(trace)
        rate = spikes / (duration / 1000)  # Hz
        firing_rates.append(rate)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(currents, firing_rates, 'bo-', linewidth=1.5, markersize=5)
    ax.set_xlabel('Injected Current (uA/cm^2)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('F-I Curve (Frequency vs Current)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(currents))
    ax.set_ylim(0, max(firing_rates) * 1.1 if max(firing_rates) > 0 else 10)

    fig.tight_layout()
    fig.savefig(figs_dir / "04_fi_curve.png", dpi=150)
    plt.close(fig)

    # Find rheobase (minimum current for spiking)
    rheobase_idx = next((i for i, r in enumerate(firing_rates) if r > 0), None)
    if rheobase_idx is not None:
        print(f"  Rheobase (approx): {currents[rheobase_idx]} uA/cm^2")
    print(f"  Max firing rate: {max(firing_rates):.1f} Hz at {currents[np.argmax(firing_rates)]} uA/cm^2")


def plot_phase_plane(figs_dir):
    """Plot V-n phase plane."""
    print("Generating phase plane plot...")

    duration = 100.0
    dt = 0.01
    I_ext = 10.0

    neuron = HHNeuron()
    V_trace = []
    n_trace = []

    num_steps = int(duration / dt)
    for _ in range(num_steps):
        state = neuron.state
        V_trace.append(state.V)
        n_trace.append(state.n)
        neuron.step(dt, I_ext)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by time
    points = np.array([V_trace, n_trace]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection
    norm = plt.Normalize(0, duration)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.linspace(0, duration, len(V_trace)))
    lc.set_linewidth(1)
    line = ax.add_collection(lc)

    ax.set_xlim(min(V_trace) - 5, max(V_trace) + 5)
    ax.set_ylim(min(n_trace) - 0.05, max(n_trace) + 0.05)
    ax.set_xlabel('Membrane Potential V (mV)')
    ax.set_ylabel('K+ Activation n')
    ax.set_title('Phase Plane (V vs n)')

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Time (ms)')

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(figs_dir / "05_phase_plane.png", dpi=150)
    plt.close(fig)


def plot_current_clamp_series(figs_dir):
    """Plot responses to different current levels."""
    print("Generating current clamp series...")

    duration = 100.0
    dt = 0.01
    currents = [0, 5, 7, 10, 15, 20]

    fig, axes = plt.subplots(len(currents), 1, figsize=(12, 10), sharex=True)

    for ax, I_ext in zip(axes, currents):
        neuron = HHNeuron()
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)

        ax.plot(time, trace, 'b-', linewidth=0.8)
        ax.set_ylabel('V (mV)')
        ax.set_title(f'I = {I_ext} uA/cm^2', loc='right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-90, 50)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Current Clamp Series', fontsize=12, y=1.02)

    fig.tight_layout()
    fig.savefig(figs_dir / "06_current_clamp_series.png", dpi=150)
    plt.close(fig)


def plot_parameter_sensitivity(figs_dir):
    """Plot sensitivity to conductance parameters."""
    print("Generating parameter sensitivity plot...")

    duration = 50.0
    dt = 0.01
    I_ext = 10.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Vary g_Na
    ax = axes[0, 0]
    for g_Na in [60, 90, 120, 150, 180]:
        params = HHParameters()
        params.g_Na = g_Na
        neuron = HHNeuron(params)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)
        ax.plot(time, trace, label=f'g_Na={g_Na}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title('Effect of g_Na')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vary g_K
    ax = axes[0, 1]
    for g_K in [18, 27, 36, 45, 54]:
        params = HHParameters()
        params.g_K = g_K
        neuron = HHNeuron(params)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)
        ax.plot(time, trace, label=f'g_K={g_K}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title('Effect of g_K')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vary g_L
    ax = axes[1, 0]
    for g_L in [0.1, 0.2, 0.3, 0.5, 1.0]:
        params = HHParameters()
        params.g_L = g_L
        neuron = HHNeuron(params)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)
        ax.plot(time, trace, label=f'g_L={g_L}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title('Effect of g_L')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vary C_m
    ax = axes[1, 1]
    for C_m in [0.5, 0.75, 1.0, 1.5, 2.0]:
        params = HHParameters()
        params.C_m = C_m
        neuron = HHNeuron(params)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)
        ax.plot(time, trace, label=f'C_m={C_m}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title('Effect of C_m')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Parameter Sensitivity Analysis', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(figs_dir / "07_parameter_sensitivity.png", dpi=150)
    plt.close(fig)


def main():
    print("=" * 60)
    print("HH Neuron Verification Suite")
    print("=" * 60)

    figs_dir = setup_output_dir()
    print(f"\nOutput directory: {figs_dir}\n")

    plot_membrane_potential(figs_dir)
    plot_gating_variables(figs_dir)
    plot_integration_comparison(figs_dir)
    plot_fi_curve(figs_dir)
    plot_phase_plane(figs_dir)
    plot_current_clamp_series(figs_dir)
    plot_parameter_sensitivity(figs_dir)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"See: {figs_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
