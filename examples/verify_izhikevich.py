#!/usr/bin/env python3
"""
Comprehensive verification of Izhikevich neuron behavior.

Generates multiple figures to verify:
1. Single neuron membrane potential with constant current
2. All preset neuron types comparison
3. Fast spiking vs regular spiking comparison
4. F-I curve (firing rate vs current)
5. V-u phase plane
6. Current clamp series
7. Parameter sensitivity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hodgkin_huxley import (
    IzhikevichNeuron,
    IzhikevichType,
    IzhikevichParameters,
)


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
    print("Generating Izhikevich membrane potential plot...")

    duration = 200.0
    dt = 0.1
    I_ext = 10.0

    neuron = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
    trace = neuron.simulate(duration=duration, dt=dt, I_ext=I_ext)
    time = np.arange(0, duration, dt)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, trace, 'b-', linewidth=0.8)
    ax.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Spike threshold (30 mV)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title(f'Izhikevich Neuron (Regular Spiking) - I_ext = {I_ext}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    fig.tight_layout()
    fig.savefig(figs_dir / "iz_01_membrane_potential.png", dpi=150)
    plt.close(fig)

    print(f"  Spikes detected: {count_spikes(trace, threshold=0)}")
    print(f"  Voltage range: [{min(trace):.1f}, {max(trace):.1f}] mV")


def plot_all_preset_types(figs_dir):
    """Plot all preset neuron types side by side."""
    print("Generating all preset types comparison...")

    duration = 500.0
    dt = 0.1
    I_ext = 10.0

    preset_types = [
        (IzhikevichType.REGULAR_SPIKING, "Regular Spiking (RS)"),
        (IzhikevichType.FAST_SPIKING, "Fast Spiking (FS)"),
        (IzhikevichType.INTRINSICALLY_BURSTING, "Intrinsically Bursting (IB)"),
        (IzhikevichType.CHATTERING, "Chattering (CH)"),
        (IzhikevichType.LOW_THRESHOLD_SPIKING, "Low Threshold Spiking (LTS)"),
    ]

    fig, axes = plt.subplots(len(preset_types), 1, figsize=(14, 12), sharex=True)

    for ax, (neuron_type, name) in zip(axes, preset_types):
        neuron = IzhikevichNeuron(neuron_type)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)

        ax.plot(time, trace, 'b-', linewidth=0.6)
        ax.set_ylabel('V (mV)')
        ax.set_title(name, loc='left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-90, 40)

        spikes = count_spikes(trace, threshold=0)
        ax.text(0.98, 0.95, f'{spikes} spikes', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='red')

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle(f'Izhikevich Neuron Preset Types (I_ext = {I_ext})', fontsize=12, y=1.01)

    fig.tight_layout()
    fig.savefig(figs_dir / "iz_02_preset_types.png", dpi=150)
    plt.close(fig)


def plot_fs_vs_rs(figs_dir):
    """Compare fast spiking and regular spiking neurons."""
    print("Generating FS vs RS comparison...")

    duration = 500.0
    dt = 0.1
    I_ext = 15.0

    rs = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
    fs = IzhikevichNeuron(IzhikevichType.FAST_SPIKING)

    trace_rs = rs.simulate(duration, dt, I_ext)
    trace_fs = fs.simulate(duration, dt, I_ext)
    time = np.arange(0, duration, dt)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(time, trace_rs, 'b-', linewidth=0.6)
    axes[0].set_ylabel('V (mV)')
    axes[0].set_title(f'Regular Spiking - {count_spikes(trace_rs)} spikes')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-90, 40)

    axes[1].plot(time, trace_fs, 'r-', linewidth=0.6)
    axes[1].set_ylabel('V (mV)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title(f'Fast Spiking - {count_spikes(trace_fs)} spikes')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-90, 40)

    fig.suptitle(f'Fast Spiking vs Regular Spiking (I_ext = {I_ext})', fontsize=12)

    fig.tight_layout()
    fig.savefig(figs_dir / "iz_03_fs_vs_rs.png", dpi=150)
    plt.close(fig)

    print(f"  RS spikes: {count_spikes(trace_rs)}")
    print(f"  FS spikes: {count_spikes(trace_fs)}")


def plot_fi_curve(figs_dir):
    """Plot firing rate vs injected current (F-I curve) for different types."""
    print("Generating F-I curves...")

    duration = 1000.0
    dt = 0.1
    currents = np.arange(0, 25, 1)

    neuron_types = [
        (IzhikevichType.REGULAR_SPIKING, "RS", 'b'),
        (IzhikevichType.FAST_SPIKING, "FS", 'r'),
        (IzhikevichType.LOW_THRESHOLD_SPIKING, "LTS", 'g'),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for neuron_type, label, color in neuron_types:
        firing_rates = []
        for I_ext in currents:
            neuron = IzhikevichNeuron(neuron_type)
            trace = neuron.simulate(duration, dt, I_ext)
            spikes = count_spikes(trace, threshold=0)
            rate = spikes / (duration / 1000)  # Hz
            firing_rates.append(rate)

        ax.plot(currents, firing_rates, f'{color}o-', label=label, linewidth=1.5, markersize=4)

    ax.set_xlabel('Injected Current')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('F-I Curves for Different Izhikevich Neuron Types')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(currents))

    fig.tight_layout()
    fig.savefig(figs_dir / "iz_04_fi_curve.png", dpi=150)
    plt.close(fig)


def plot_phase_plane(figs_dir):
    """Plot V-u phase plane."""
    print("Generating phase plane plot...")

    duration = 200.0
    dt = 0.1
    I_ext = 10.0

    neuron = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
    V_trace = []
    u_trace = []

    num_steps = int(duration / dt)
    for _ in range(num_steps):
        state = neuron.state
        V_trace.append(state.v)
        u_trace.append(state.u)
        neuron.step(dt, I_ext)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by time
    points = np.array([V_trace, u_trace]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection
    norm = plt.Normalize(0, duration)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.linspace(0, duration, len(V_trace)))
    lc.set_linewidth(1)
    line = ax.add_collection(lc)

    ax.set_xlim(min(V_trace) - 5, max(V_trace) + 5)
    ax.set_ylim(min(u_trace) - 2, max(u_trace) + 2)
    ax.set_xlabel('Membrane Potential v (mV)')
    ax.set_ylabel('Recovery Variable u')
    ax.set_title('Phase Plane (v vs u) - Regular Spiking')

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Time (ms)')

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(figs_dir / "iz_05_phase_plane.png", dpi=150)
    plt.close(fig)


def plot_current_clamp_series(figs_dir):
    """Plot responses to different current levels."""
    print("Generating current clamp series...")

    duration = 200.0
    dt = 0.1
    currents = [0, 3, 5, 8, 12, 18]

    fig, axes = plt.subplots(len(currents), 1, figsize=(12, 10), sharex=True)

    for ax, I_ext in zip(axes, currents):
        neuron = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)

        ax.plot(time, trace, 'b-', linewidth=0.8)
        ax.set_ylabel('V (mV)')
        ax.set_title(f'I = {I_ext}', loc='right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-90, 40)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Izhikevich Current Clamp Series (Regular Spiking)', fontsize=12, y=1.02)

    fig.tight_layout()
    fig.savefig(figs_dir / "iz_06_current_clamp_series.png", dpi=150)
    plt.close(fig)


def plot_parameter_sensitivity(figs_dir):
    """Plot sensitivity to Izhikevich parameters."""
    print("Generating parameter sensitivity plot...")

    duration = 200.0
    dt = 0.1
    I_ext = 10.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Vary parameter a (time scale of recovery)
    ax = axes[0, 0]
    for a in [0.01, 0.02, 0.05, 0.1, 0.2]:
        params = IzhikevichParameters()
        params.a = a
        neuron = IzhikevichNeuron(parameters=params)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)
        ax.plot(time, trace, label=f'a={a}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title('Effect of a (recovery time scale)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vary parameter b (sensitivity of u to v)
    ax = axes[0, 1]
    for b in [0.1, 0.2, 0.25, 0.3]:
        params = IzhikevichParameters()
        params.b = b
        neuron = IzhikevichNeuron(parameters=params)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)
        ax.plot(time, trace, label=f'b={b}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title('Effect of b (sensitivity of u to v)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vary parameter c (after-spike reset of v)
    ax = axes[1, 0]
    for c in [-70, -65, -60, -55, -50]:
        params = IzhikevichParameters()
        params.c = c
        neuron = IzhikevichNeuron(parameters=params)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)
        ax.plot(time, trace, label=f'c={c}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title('Effect of c (after-spike reset value)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vary parameter d (after-spike reset of u)
    ax = axes[1, 1]
    for d in [2, 4, 6, 8, 10]:
        params = IzhikevichParameters()
        params.d = d
        neuron = IzhikevichNeuron(parameters=params)
        trace = neuron.simulate(duration, dt, I_ext)
        time = np.arange(0, duration, dt)
        ax.plot(time, trace, label=f'd={d}', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title('Effect of d (after-spike u increment)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Izhikevich Parameter Sensitivity Analysis', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(figs_dir / "iz_07_parameter_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_neuron_comparison(figs_dir):
    """Compare HH and Izhikevich neuron responses."""
    print("Generating HH vs Izhikevich comparison...")

    from hodgkin_huxley import HHNeuron

    duration = 100.0
    I_ext = 10.0

    # HH neuron (smaller dt for accuracy)
    hh = HHNeuron()
    trace_hh = hh.simulate(duration, dt=0.01, I_ext=I_ext)
    time_hh = np.arange(0, duration, 0.01)

    # Izhikevich neuron
    iz = IzhikevichNeuron(IzhikevichType.REGULAR_SPIKING)
    trace_iz = iz.simulate(duration, dt=0.1, I_ext=I_ext)
    time_iz = np.arange(0, duration, 0.1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(time_hh, trace_hh, 'b-', linewidth=0.6)
    axes[0].set_ylabel('V (mV)')
    axes[0].set_title(f'Hodgkin-Huxley Neuron - {count_spikes(trace_hh)} spikes')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-90, 50)

    axes[1].plot(time_iz, trace_iz, 'r-', linewidth=0.6)
    axes[1].set_ylabel('V (mV)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title(f'Izhikevich Neuron (RS) - {count_spikes(trace_iz)} spikes')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-90, 50)

    fig.suptitle(f'Hodgkin-Huxley vs Izhikevich Comparison (I_ext = {I_ext})', fontsize=12)

    fig.tight_layout()
    fig.savefig(figs_dir / "iz_08_hh_vs_iz_comparison.png", dpi=150)
    plt.close(fig)


def main():
    print("=" * 60)
    print("Izhikevich Neuron Verification Suite")
    print("=" * 60)

    figs_dir = setup_output_dir()
    print(f"\nOutput directory: {figs_dir}\n")

    plot_membrane_potential(figs_dir)
    plot_all_preset_types(figs_dir)
    plot_fs_vs_rs(figs_dir)
    plot_fi_curve(figs_dir)
    plot_phase_plane(figs_dir)
    plot_current_clamp_series(figs_dir)
    plot_parameter_sensitivity(figs_dir)
    plot_neuron_comparison(figs_dir)

    print("\n" + "=" * 60)
    print("All Izhikevich figures generated successfully!")
    print(f"See: {figs_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
