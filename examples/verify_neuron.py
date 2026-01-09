#!/usr/bin/env python3
"""
Verify single neuron membrane potential behavior with constant current input.

This script tests that the HHNeuron produces expected action potential traces
when stimulated with constant current injection.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hodgkin_huxley import HHNeuron, IntegrationMethod


def main():
    # Output directory
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)

    # Simulation parameters
    duration = 100.0  # ms
    dt = 0.01         # ms
    I_ext = 10.0      # uA/cm^2 (sufficient to trigger action potentials)

    # Create neuron with default parameters (RK4 integration)
    neuron = HHNeuron()

    print(f"Neuron initial state:")
    print(f"  Membrane potential: {neuron.V:.2f} mV")
    try:
        print(f"  Integration method: {neuron.integration_method}")
    except AttributeError:
        print(f"  Integration method: RK4 (default)")
    print(f"\nSimulation parameters:")
    print(f"  Duration: {duration} ms")
    print(f"  Time step: {dt} ms")
    print(f"  External current: {I_ext} uA/cm^2")

    # Run simulation
    trace = neuron.simulate(duration=duration, dt=dt, I_ext=I_ext)
    time = np.arange(0, duration, dt)

    # Analyze results
    v_min, v_max = min(trace), max(trace)
    print(f"\nResults:")
    print(f"  Voltage range: [{v_min:.2f}, {v_max:.2f}] mV")

    # Count action potentials (threshold crossings going up)
    threshold = 0.0
    trace_arr = np.array(trace)
    above = trace_arr > threshold
    crossings_up = np.where(np.diff(above.astype(int)) == 1)[0]
    num_spikes = len(crossings_up)
    print(f"  Action potentials: {num_spikes}")

    if num_spikes > 0:
        firing_rate = num_spikes / (duration / 1000)
        print(f"  Firing rate: {firing_rate:.1f} Hz")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, trace, 'b-', linewidth=0.8)
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold (0 mV)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title(f'HH Neuron Verification (I_ext = {I_ext} uA/cm², method = RK4)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = figs_dir / "verify_neuron_membrane_potential.png"
    fig.savefig(output_path, dpi=150)
    print(f"\nFigure saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
