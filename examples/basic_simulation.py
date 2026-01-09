#!/usr/bin/env python3
"""
Basic Hodgkin-Huxley Neuron Simulation Example

This example demonstrates basic usage of the library:
1. Creating neurons and networks
2. Running simulations
3. Analyzing results

For comprehensive neuron verification plots, see verify_neuron.py instead.
"""

import numpy as np
from pathlib import Path

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed - skipping visualization")

from hodgkin_huxley import HHNeuron, Network, Parameters


def get_figs_dir():
    """Get the figs directory, creating it if needed."""
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    return figs_dir


def single_neuron_example():
    """Simulate a single Hodgkin-Huxley neuron."""
    print("=== Single Neuron Simulation ===\n")

    # Create a neuron with default parameters
    neuron = HHNeuron()
    print(f"Initial state: {neuron}")
    print(f"Resting potential: {neuron.V:.2f} mV")

    # Simulation parameters
    duration = 100.0  # ms
    dt = 0.01         # ms
    I_ext = 10.0      # uA/cm^2

    # Run simulation
    print(f"\nRunning simulation for {duration} ms with I_ext = {I_ext} uA/cm^2...")
    trace = neuron.simulate(duration=duration, dt=dt, I_ext=I_ext)

    # Analyze results
    print(f"Simulation complete. {len(trace)} time points recorded.")
    print(f"Voltage range: [{trace.min():.2f}, {trace.max():.2f}] mV")

    # Count spikes (threshold crossings)
    threshold = 0.0  # mV
    crossings = np.where(np.diff(trace > threshold))[0]
    num_spikes = len(crossings) // 2
    print(f"Number of action potentials: {num_spikes}")

    # No plot here - see verify_neuron.py for comprehensive plots
    return trace


def current_injection_example():
    """Demonstrate different current injection levels."""
    print("\n=== Current-Frequency Relationship ===\n")

    duration = 500.0  # ms
    dt = 0.01         # ms
    currents = [5, 7, 10, 15, 20]  # uA/cm^2

    results = []
    for I_ext in currents:
        neuron = HHNeuron()
        trace = neuron.simulate(duration=duration, dt=dt, I_ext=I_ext)

        # Count spikes
        threshold = 0.0
        crossings = np.where(np.diff(trace > threshold))[0]
        num_spikes = len(crossings) // 2
        freq = num_spikes / (duration / 1000)  # Hz

        results.append((I_ext, num_spikes, freq))
        print(f"I_ext = {I_ext:2d} uA/cm²: {num_spikes:2d} spikes ({freq:.1f} Hz)")

    return results


def network_example():
    """Simulate a small network of neurons."""
    print("\n=== Network Simulation ===\n")

    # Create a network with 3 neurons
    net = Network(3)
    print(f"Created network: {net}")

    # Add excitatory connections: 0 -> 1, 1 -> 2
    net.add_synapse(pre_idx=0, post_idx=1, weight=0.1, E_syn=0.0, tau=2.0)
    net.add_synapse(pre_idx=1, post_idx=2, weight=0.1, E_syn=0.0, tau=2.0)
    print(f"Added synapses: {net.num_synapses}")

    # Simulation parameters
    duration = 200.0  # ms
    dt = 0.01         # ms
    num_steps = int(duration / dt)

    # Create input currents - only stimulate neuron 0
    I_ext = np.zeros((3, num_steps))
    I_ext[0, :] = 10.0  # Constant current to neuron 0
    print(I_ext)

    print(f"\nRunning network simulation for {duration} ms...")
    traces = net.simulate(duration=duration, dt=dt, I_ext=I_ext)

    print(f"Simulation complete. Shape: {traces.shape}")
    for i in range(3):
        print(f"  Neuron {i}: V range = [{traces[i].min():.1f}, {traces[i].max():.1f}] mV")

    if HAS_MATPLOTLIB:
        figs_dir = get_figs_dir()
        time = np.arange(0, duration, dt)
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, ax in enumerate(axes):
            ax.plot(time, traces[i], color=colors[i], linewidth=0.8)
            ax.set_ylabel(f'Neuron {i}\n(mV)')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (ms)')
        axes[0].set_title('Network Simulation (Chain: 0 -> 1 -> 2)')

        plt.tight_layout()
        fig.savefig(figs_dir / 'network_example.png', dpi=150)
        plt.close(fig)
        print(f"Plot saved to '{figs_dir / 'network_example.png'}'")

    return traces


def custom_parameters_example():
    """Demonstrate custom neuron parameters."""
    print("\n=== Custom Parameters Example ===\n")

    # Create custom parameters
    params = Parameters()
    print("Default parameters:")
    print(f"  g_Na = {params.g_Na} mS/cm²")
    print(f"  g_K  = {params.g_K} mS/cm²")
    print(f"  g_L  = {params.g_L} mS/cm²")

    # Modify sodium conductance
    params.g_Na = 150.0  # Increased from 120
    print(f"\nModified g_Na to {params.g_Na} mS/cm²")

    neuron = HHNeuron(params)
    trace = neuron.simulate(duration=50.0, dt=0.01, I_ext=10.0)

    print(f"Max voltage with modified g_Na: {trace.max():.2f} mV")

    return trace


if __name__ == "__main__":
    single_neuron_example()
    current_injection_example()
    network_example()
    custom_parameters_example()

    print("\n=== All examples complete! ===")
