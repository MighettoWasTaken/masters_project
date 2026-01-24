#!/usr/bin/env python3
"""
Network Visualization Tool

Generates visual representations of neural networks showing:
- Neurons as vertices (colored by firing rate)
- Synapses as edges (styled by type: excitatory/inhibitory)
- Stimulation indicators
- Network activity statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Using basic layout.")

from hodgkin_huxley import (
    Network,
    HHNeuron,
    IzhikevichNeuron,
    IzhikevichType,
    NetworkNeuronType,
)


@dataclass
class NetworkVisualization:
    """Container for network visualization data."""
    num_neurons: int
    neuron_types: List[str]  # "HH" or "Izhikevich"
    synapses: List[Tuple[int, int, float, float]]  # (pre, post, weight, E_syn)
    firing_rates: List[float]
    stimulation: Dict[int, float]  # neuron_idx -> current
    title: str = "Neural Network"
    neuron_labels: Optional[List[str]] = None  # Custom labels for each neuron (e.g., "RS", "FS", "E1")


def count_spikes(trace: np.ndarray, threshold: float = 0.0) -> int:
    """Count spikes using upward threshold crossings."""
    above = trace > threshold
    crossings = np.diff(above.astype(int))
    return int(np.sum(crossings == 1))


def simulate_and_analyze(
    net: Network,
    duration: float,
    dt: float,
    I_ext: np.ndarray,
    threshold: float = 0.0
) -> Tuple[List[float], np.ndarray]:
    """
    Simulate network and compute firing rates.

    Returns:
        firing_rates: List of firing rates (Hz) for each neuron
        traces: Voltage traces for all neurons
    """
    traces = net.simulate(duration, dt, I_ext)
    traces = np.array(traces)

    firing_rates = []
    for i in range(net.num_neurons):
        spikes = count_spikes(traces[i], threshold)
        rate = spikes / (duration / 1000.0)  # Convert to Hz
        firing_rates.append(rate)

    return firing_rates, traces


def visualize_network(
    vis_data: NetworkVisualization,
    output_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Generate a visualization of the neural network.

    Args:
        vis_data: NetworkVisualization containing network structure and activity
        output_path: Path to save figure (optional)
        show: Whether to display the figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
    ax_network = axes[0]
    ax_info = axes[1]

    n = vis_data.num_neurons

    # Create graph layout
    if HAS_NETWORKX:
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for pre, post, weight, E_syn in vis_data.synapses:
            G.add_edge(pre, post, weight=weight, E_syn=E_syn)

        # Use spring layout for nice positioning
        if n <= 10:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)
    else:
        # Fallback: circular layout
        pos = {}
        for i in range(n):
            angle = 2 * np.pi * i / n
            pos[i] = (np.cos(angle), np.sin(angle))

    # Normalize firing rates for color mapping
    max_rate = max(vis_data.firing_rates) if max(vis_data.firing_rates) > 0 else 1
    norm = Normalize(vmin=0, vmax=max_rate)
    cmap = plt.cm.Reds

    # Draw edges (synapses)
    for pre, post, weight, E_syn in vis_data.synapses:
        x_pre, y_pre = pos[pre]
        x_post, y_post = pos[post]

        # Determine synapse type (excitatory vs inhibitory)
        is_excitatory = E_syn > -60  # E_syn > V_rest is excitatory

        # Edge style based on type
        if is_excitatory:
            color = 'green'
            style = '-'
            alpha = 0.4 + 0.4 * min(weight / 10.0, 1.0)
        else:
            color = 'blue'
            style = '--'
            alpha = 0.4 + 0.4 * min(weight / 10.0, 1.0)

        # Line width based on weight
        linewidth = 0.5 + 2.0 * min(weight / 10.0, 1.0)

        # Draw arrow
        ax_network.annotate(
            '',
            xy=(x_post, y_post),
            xytext=(x_pre, y_pre),
            arrowprops=dict(
                arrowstyle='-|>',
                color=color,
                alpha=alpha,
                linestyle=style,
                linewidth=linewidth,
                shrinkA=15,
                shrinkB=15,
                mutation_scale=10 + 5 * min(weight / 5.0, 1.0)
            )
        )

    # Draw nodes (neurons)
    for i in range(n):
        x, y = pos[i]
        rate = vis_data.firing_rates[i]
        neuron_type = vis_data.neuron_types[i]

        # Node color based on firing rate
        node_color = cmap(norm(rate))

        # Node shape based on neuron type
        if neuron_type == "HH":
            marker = 'o'  # Circle for HH
            size = 800
        else:
            marker = 's'  # Square for Izhikevich
            size = 700

        # Draw neuron
        ax_network.scatter(
            [x], [y],
            s=size,
            c=[node_color],
            marker=marker,
            edgecolors='black',
            linewidths=2,
            zorder=3
        )

        # Add neuron index label
        ax_network.annotate(
            str(i),
            (x, y),
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            zorder=4
        )

        # Mark stimulated neurons with a ring
        if i in vis_data.stimulation:
            stim_current = vis_data.stimulation[i]
            ring_size = 1200 + 200 * min(stim_current / 20.0, 1.0)
            ax_network.scatter(
                [x], [y],
                s=ring_size,
                facecolors='none',
                edgecolors='orange',
                linewidths=3,
                zorder=2
            )
            # Add stimulation label
            ax_network.annotate(
                f'{stim_current:.0f}',
                (x, y + 0.15),
                ha='center',
                va='bottom',
                fontsize=8,
                color='orange',
                fontweight='bold'
            )

    # Add colorbar for firing rate
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_network, shrink=0.6, pad=0.02)
    cbar.set_label('Firing Rate (Hz)', fontsize=10)

    # Network plot styling
    ax_network.set_xlim(-1.5, 1.5)
    ax_network.set_ylim(-1.5, 1.5)
    ax_network.set_aspect('equal')
    ax_network.axis('off')
    ax_network.set_title(vis_data.title, fontsize=14, fontweight='bold')

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='black', label='HH Neuron (circle)'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Izhikevich (square)'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Excitatory synapse'),
        plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Inhibitory synapse'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='orange', markersize=15, markeredgewidth=2,
                   label='Stimulated neuron'),
    ]
    ax_network.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Info panel
    ax_info.axis('off')

    # Build neuron type summary
    hh_count = sum(1 for t in vis_data.neuron_types if t == 'HH')
    iz_count = sum(1 for t in vis_data.neuron_types if t == 'Izhikevich')

    info_text = f"""Network Statistics
{'='*30}

Neurons: {n}
  - HH: {hh_count}
  - Izhikevich: {iz_count}

Synapses: {len(vis_data.synapses)}
  - Excitatory: {sum(1 for _, _, _, e in vis_data.synapses if e > -60)}
  - Inhibitory: {sum(1 for _, _, _, e in vis_data.synapses if e <= -60)}

Stimulated neurons: {len(vis_data.stimulation)}
"""

    # Add neuron key if custom labels are provided
    if vis_data.neuron_labels:
        info_text += f"""
Neuron Key:
{'='*30}
"""
        for i in range(n):
            if i < len(vis_data.neuron_labels):
                info_text += f"  {i}: {vis_data.neuron_labels[i]}\n"

    info_text += f"""
Firing Rates:
{'='*30}
"""

    for i in range(n):
        rate_str = f"{vis_data.firing_rates[i]:6.1f} Hz"
        stim_str = ""
        if i in vis_data.stimulation:
            stim_str = f" (I={vis_data.stimulation[i]:.0f})"
        info_text += f"  [{i}]: {rate_str}{stim_str}\n"

    avg_rate = np.mean(vis_data.firing_rates)
    max_rate_idx = np.argmax(vis_data.firing_rates)
    info_text += f"""
{'='*30}
Average rate: {avg_rate:.1f} Hz
Most active: Neuron {max_rate_idx} ({vis_data.firing_rates[max_rate_idx]:.1f} Hz)
"""

    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_network_visualization(
    net: Network,
    synapses: List[Tuple[int, int, float, float]],
    stimulation: Dict[int, float],
    duration: float = 500.0,
    dt: float = 0.01,
    title: str = "Neural Network",
    neuron_labels: Optional[List[str]] = None
) -> NetworkVisualization:
    """
    Create visualization data by simulating the network.

    Args:
        net: Network object
        synapses: List of (pre, post, weight, E_syn) tuples
        stimulation: Dict mapping neuron index to stimulation current
        duration: Simulation duration in ms
        dt: Time step in ms
        title: Title for the visualization
        neuron_labels: Optional list of custom labels for each neuron
                      (e.g., ["E1", "E2", "I1"] or ["RS", "FS", "IB"])

    Returns:
        NetworkVisualization object with simulation results
    """
    num_neurons = net.num_neurons
    num_steps = int(duration / dt)

    # Get neuron types
    neuron_types = [net.neuron_type(i) for i in range(num_neurons)]

    # Create external current array
    I_ext = np.zeros((num_neurons, num_steps))
    for idx, current in stimulation.items():
        I_ext[idx, :] = current

    # Simulate and get firing rates
    # Use threshold=0 for HH, threshold=30 for Izhikevich
    firing_rates, traces = simulate_and_analyze(net, duration, dt, I_ext, threshold=0)

    return NetworkVisualization(
        num_neurons=num_neurons,
        neuron_types=neuron_types,
        synapses=synapses,
        firing_rates=firing_rates,
        stimulation=stimulation,
        title=title,
        neuron_labels=neuron_labels
    )


# =============================================================================
# Example Networks
# =============================================================================

def example_feedforward_chain():
    """Feedforward chain: 0 -> 1 -> 2 -> 3"""
    print("\n1. Feedforward Chain (HH neurons)")

    net = Network(4)
    synapses = []

    # Add excitatory chain
    for i in range(3):
        weight, E_syn, tau = 0.1, 0.0, 2.0
        net.add_synapse(i, i+1, weight, E_syn, tau)
        synapses.append((i, i+1, weight, E_syn))

    stimulation = {0: 15.0}

    vis_data = create_network_visualization(
        net, synapses, stimulation,
        duration=500.0,
        title="Feedforward Chain (HH)"
    )

    return vis_data


def example_divergent_network():
    """One neuron driving multiple targets"""
    print("\n2. Divergent Network (HH neurons)")

    net = Network(5)
    synapses = []

    # Neuron 0 drives neurons 1-4
    for i in range(1, 5):
        weight, E_syn, tau = 10.0, 0.0, 2.0
        net.add_synapse(0, i, weight, E_syn, tau)
        synapses.append((0, i, weight, E_syn))

    stimulation = {0: 15.0}

    vis_data = create_network_visualization(
        net, synapses, stimulation,
        duration=500.0,
        title="Divergent Network (1-to-many)"
    )

    return vis_data


def example_convergent_network():
    """Multiple neurons converging on one target"""
    print("\n3. Convergent Network (HH neurons)")

    net = Network(5)
    synapses = []

    # Neurons 0-3 drive neuron 4
    for i in range(4):
        weight, E_syn, tau = 3.0, 0.0, 2.0
        net.add_synapse(i, 4, weight, E_syn, tau)
        synapses.append((i, 4, weight, E_syn))

    stimulation = {0: 12.0, 1: 12.0, 2: 12.0, 3: 12.0}

    vis_data = create_network_visualization(
        net, synapses, stimulation,
        duration=500.0,
        title="Convergent Network (many-to-1)"
    )

    return vis_data


def example_recurrent_ei_network():
    """Small E-I network with recurrent connections"""
    print("\n4. E-I Recurrent Network (HH neurons)")

    net = Network(6)
    synapses = []

    # Excitatory neurons: 0, 1, 2, 3
    # Inhibitory neurons: 4, 5
    neuron_labels = ["E1", "E2", "E3", "E4", "I1", "I2"]

    # E -> E connections (weak)
    for pre in [0, 1]:
        for post in [2, 3]:
            weight, E_syn, tau = 2.0, 0.0, 2.0
            net.add_synapse(pre, post, weight, E_syn, tau)
            synapses.append((pre, post, weight, E_syn))

    # E -> I connections
    for pre in [0, 1, 2, 3]:
        for post in [4, 5]:
            weight, E_syn, tau = 4.0, 0.0, 2.0
            net.add_synapse(pre, post, weight, E_syn, tau)
            synapses.append((pre, post, weight, E_syn))

    # I -> E connections (inhibitory)
    for pre in [4, 5]:
        for post in [0, 1, 2, 3]:
            weight, E_syn, tau = 3.0, -80.0, 5.0
            net.add_synapse(pre, post, weight, E_syn, tau)
            synapses.append((pre, post, weight, E_syn))

    stimulation = {0: 12.0, 1: 12.0}

    vis_data = create_network_visualization(
        net, synapses, stimulation,
        duration=500.0,
        title="E-I Recurrent Network",
        neuron_labels=neuron_labels
    )

    return vis_data


def example_mixed_neuron_types():
    """Network with both HH and Izhikevich neurons"""
    print("\n5. Mixed Neuron Types (HH + Izhikevich)")

    net = Network()
    synapses = []

    # Add HH neurons (0, 1)
    net.add_hh_neuron()
    net.add_hh_neuron()

    # Add Izhikevich neurons (2: RS, 3: FS, 4: IB)
    net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)
    net.add_izhikevich_neuron(IzhikevichType.FAST_SPIKING)
    net.add_izhikevich_neuron(IzhikevichType.INTRINSICALLY_BURSTING)

    # Labels indicating model and type
    neuron_labels = ["HH-1", "HH-2", "Iz-RS", "Iz-FS", "Iz-IB"]

    # HH -> Izhikevich connections
    for post in [2, 3, 4]:
        weight, E_syn, tau = 8.0, 0.0, 2.0
        net.add_synapse(0, post, weight, E_syn, tau)
        synapses.append((0, post, weight, E_syn))

    # Izhikevich -> HH connections
    for pre in [2, 3]:
        weight, E_syn, tau = 5.0, 0.0, 2.0
        net.add_synapse(pre, 1, weight, E_syn, tau)
        synapses.append((pre, 1, weight, E_syn))

    # FS inhibits IB
    weight, E_syn, tau = 4.0, -80.0, 5.0
    net.add_synapse(3, 4, weight, E_syn, tau)
    synapses.append((3, 4, weight, E_syn))

    stimulation = {0: 15.0, 2: 8.0}

    vis_data = create_network_visualization(
        net, synapses, stimulation,
        duration=500.0,
        title="Mixed Network (HH + Izhikevich)",
        neuron_labels=neuron_labels
    )

    return vis_data


def example_winner_take_all():
    """Mutual inhibition network (winner-take-all)"""
    print("\n6. Winner-Take-All Network")

    net = Network(4)
    synapses = []

    # Labels showing competition ranking by drive strength
    neuron_labels = ["HH-Hi", "HH-Med", "HH-Low", "HH-Min"]

    # All-to-all inhibition (except self)
    for pre in range(4):
        for post in range(4):
            if pre != post:
                weight, E_syn, tau = 4.0, -80.0, 5.0
                net.add_synapse(pre, post, weight, E_syn, tau)
                synapses.append((pre, post, weight, E_syn))

    # Asymmetric stimulation - neuron 0 gets most
    stimulation = {0: 15.0, 1: 12.0, 2: 10.0, 3: 8.0}

    vis_data = create_network_visualization(
        net, synapses, stimulation,
        duration=500.0,
        title="Winner-Take-All (Mutual Inhibition)",
        neuron_labels=neuron_labels
    )

    return vis_data


def example_izhikevich_variety():
    """Network showcasing different Izhikevich neuron types"""
    print("\n7. Izhikevich Neuron Variety")

    net = Network()
    synapses = []

    # Add different Izhikevich types
    net.add_izhikevich_neuron(IzhikevichType.REGULAR_SPIKING)    # 0
    net.add_izhikevich_neuron(IzhikevichType.FAST_SPIKING)       # 1
    net.add_izhikevich_neuron(IzhikevichType.INTRINSICALLY_BURSTING)  # 2
    net.add_izhikevich_neuron(IzhikevichType.CHATTERING)         # 3
    net.add_izhikevich_neuron(IzhikevichType.LOW_THRESHOLD_SPIKING)  # 4

    # Labels for Izhikevich neuron types
    neuron_labels = ["RS", "FS", "IB", "CH", "LTS"]

    # Connect RS -> all others
    for post in [1, 2, 3, 4]:
        weight, E_syn, tau = 5.0, 0.0, 2.0
        net.add_synapse(0, post, weight, E_syn, tau)
        synapses.append((0, post, weight, E_syn))

    # FS inhibits IB and CH
    for post in [2, 3]:
        weight, E_syn, tau = 3.0, -80.0, 5.0
        net.add_synapse(1, post, weight, E_syn, tau)
        synapses.append((1, post, weight, E_syn))

    # Stimulate all with different currents
    stimulation = {0: 10.0, 1: 10.0, 2: 8.0, 3: 8.0, 4: 5.0}

    vis_data = create_network_visualization(
        net, synapses, stimulation,
        duration=500.0,
        title="Izhikevich Neuron Types",
        neuron_labels=neuron_labels
    )

    return vis_data


def setup_output_dir():
    """Create output directory for figures."""
    figs_dir = Path(__file__).parent / "figs"
    figs_dir.mkdir(exist_ok=True)
    return figs_dir


def main():
    print("=" * 60)
    print("Neural Network Visualization Suite")
    print("=" * 60)

    figs_dir = setup_output_dir()
    print(f"\nOutput directory: {figs_dir}\n")

    # Generate all example visualizations
    examples = [
        ("network_01_feedforward_chain.png", example_feedforward_chain),
        ("network_02_divergent.png", example_divergent_network),
        ("network_03_convergent.png", example_convergent_network),
        ("network_04_ei_recurrent.png", example_recurrent_ei_network),
        ("network_05_mixed_types.png", example_mixed_neuron_types),
        ("network_06_winner_take_all.png", example_winner_take_all),
        ("network_07_izhikevich_variety.png", example_izhikevich_variety),
    ]

    for filename, example_fn in examples:
        vis_data = example_fn()
        visualize_network(vis_data, output_path=figs_dir / filename)

    print("\n" + "=" * 60)
    print("All network visualizations generated successfully!")
    print(f"See: {figs_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
