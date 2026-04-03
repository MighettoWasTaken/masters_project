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
    IzhikevichType, RegionalNetwork, NeuronModelSpec, SynapseSpec, RecordingConfig,
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
    neuron_labels: Optional[List[str]] = None  # Custom labels for each neuron


def count_spikes(trace: np.ndarray, threshold: float = 0.0) -> int:
    """Count spikes using upward threshold crossings."""
    above = trace > threshold
    crossings = np.diff(above.astype(int))
    return int(np.sum(crossings == 1))


def _pop_info(rn: RegionalNetwork) -> Dict[str, Tuple[int, int]]:
    """Return {pop_name: (global_start, count)} in insertion order."""
    return {
        name: (rn._rnet.population_start(name), rn._rnet.population_size(name))
        for name in rn.population_names()
    }


def simulate_and_analyze(
    rn: RegionalNetwork,
    duration: float,
    dt: float,
    I_ext_dict: dict,
) -> Tuple[List[float], np.ndarray]:
    """
    Simulate network and compute firing rates.

    Returns:
        firing_rates: List of firing rates (Hz) in global neuron order
        traces: Voltage traces (n_neurons, n_steps) in global neuron order
    """
    result = rn.simulate(duration, dt, I_ext_dict,
                         record=RecordingConfig(["V", "firing_rate"]))
    firing_rates = []
    trace_rows = []
    for pop_name in rn.population_names():
        firing_rates.extend(result[pop_name]["firing_rate"].tolist())
        trace_rows.append(result[pop_name]["V"])
    traces = np.vstack(trace_rows) if trace_rows else np.empty((0, 0))
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

        if n <= 10:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)
    else:
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

        is_excitatory = E_syn > -60

        if is_excitatory:
            color = 'green'
            style = '-'
            alpha = 0.4 + 0.4 * min(weight / 10.0, 1.0)
        else:
            color = 'blue'
            style = '--'
            alpha = 0.4 + 0.4 * min(weight / 10.0, 1.0)

        linewidth = 0.5 + 2.0 * min(weight / 10.0, 1.0)

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

        node_color = cmap(norm(rate))

        if neuron_type == "HH":
            marker = 'o'
            size = 800
        else:
            marker = 's'
            size = 700

        ax_network.scatter(
            [x], [y],
            s=size,
            c=[node_color],
            marker=marker,
            edgecolors='black',
            linewidths=2,
            zorder=3
        )

        ax_network.annotate(
            str(i),
            (x, y),
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            zorder=4
        )

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
            ax_network.annotate(
                f'{stim_current:.0f}',
                (x, y + 0.15),
                ha='center',
                va='bottom',
                fontsize=8,
                color='orange',
                fontweight='bold'
            )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_network, shrink=0.6, pad=0.02)
    cbar.set_label('Firing Rate (Hz)', fontsize=10)

    ax_network.set_xlim(-1.5, 1.5)
    ax_network.set_ylim(-1.5, 1.5)
    ax_network.set_aspect('equal')
    ax_network.axis('off')
    ax_network.set_title(vis_data.title, fontsize=14, fontweight='bold')

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

    ax_info.axis('off')

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
    rn: RegionalNetwork,
    synapses: List[Tuple[int, int, float, float]],
    stimulation: Dict[int, float],
    neuron_types: List[str],
    duration: float = 500.0,
    dt: float = 0.01,
    title: str = "Neural Network",
    neuron_labels: Optional[List[str]] = None
) -> NetworkVisualization:
    """
    Create visualization data by simulating the network.

    Args:
        rn: RegionalNetwork to simulate
        synapses: List of (pre, post, weight, E_syn) tuples with global neuron indices
        stimulation: Dict mapping global neuron index to stimulation current
        neuron_types: List of "HH" or "Izhikevich" per global neuron index
        duration: Simulation duration in ms
        dt: Time step in ms
        title: Title for the visualization
        neuron_labels: Optional list of custom labels for each neuron
    """
    info = _pop_info(rn)
    n_steps = int(duration / dt)

    # Build per-population I_ext from global stimulation dict
    I_ext_dict = {name: np.zeros((size, n_steps)) for name, (_, size) in info.items()}
    for global_idx, current in stimulation.items():
        for name, (start, size) in info.items():
            if start <= global_idx < start + size:
                I_ext_dict[name][global_idx - start, :] = current
                break

    firing_rates, _ = simulate_and_analyze(rn, duration, dt, I_ext_dict)

    return NetworkVisualization(
        num_neurons=sum(size for _, (_, size) in info.items()),
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

    rn = RegionalNetwork()
    rn.add_population("E", 4, model=NeuronModelSpec.hh_default())
    rn.connect("E", "E", lambda ns, nd: [(i, i + 1) for i in range(3)],
               weight=0.1, synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))

    neuron_types = ["HH"] * 4
    synapses = [(i, i + 1, 0.1, 0.0) for i in range(3)]
    stimulation = {0: 15.0}

    return create_network_visualization(
        rn, synapses, stimulation, neuron_types,
        duration=500.0, title="Feedforward Chain (HH)"
    )


def example_divergent_network():
    """One neuron driving multiple targets"""
    print("\n2. Divergent Network (HH neurons)")

    rn = RegionalNetwork()
    rn.add_population("E", 5, model=NeuronModelSpec.hh_default())
    rn.connect("E", "E", lambda ns, nd: [(0, i) for i in range(1, 5)],
               weight=10.0, synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))

    neuron_types = ["HH"] * 5
    synapses = [(0, i, 10.0, 0.0) for i in range(1, 5)]
    stimulation = {0: 15.0}

    return create_network_visualization(
        rn, synapses, stimulation, neuron_types,
        duration=500.0, title="Divergent Network (1-to-many)"
    )


def example_convergent_network():
    """Multiple neurons converging on one target"""
    print("\n3. Convergent Network (HH neurons)")

    rn = RegionalNetwork()
    rn.add_population("E", 5, model=NeuronModelSpec.hh_default())
    rn.connect("E", "E", lambda ns, nd: [(i, 4) for i in range(4)],
               weight=3.0, synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))

    neuron_types = ["HH"] * 5
    synapses = [(i, 4, 3.0, 0.0) for i in range(4)]
    stimulation = {0: 12.0, 1: 12.0, 2: 12.0, 3: 12.0}

    return create_network_visualization(
        rn, synapses, stimulation, neuron_types,
        duration=500.0, title="Convergent Network (many-to-1)"
    )


def example_recurrent_ei_network():
    """Small E-I network with recurrent connections"""
    print("\n4. E-I Recurrent Network (HH neurons)")

    # E population: global 0-3, I population: global 4-5
    rn = RegionalNetwork()
    rn.add_population("E", 4, model=NeuronModelSpec.hh_default())
    rn.add_population("I", 2, model=NeuronModelSpec.hh_default())

    # E -> E (weak): local 0,1 -> 2,3
    rn.connect("E", "E", lambda ns, nd: [(0, 2), (0, 3), (1, 2), (1, 3)],
               weight=2.0, synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
    # E -> I (all-to-all)
    rn.connect("E", "I", "all_to_all", weight=4.0,
               synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
    # I -> E (inhibitory, all-to-all)
    rn.connect("I", "E", "all_to_all", weight=3.0,
               synapse=SynapseSpec.exponential(E_syn=-80.0, tau=5.0))

    neuron_labels = ["E1", "E2", "E3", "E4", "I1", "I2"]
    neuron_types = ["HH"] * 6

    synapses = []
    for pre in [0, 1]:
        for post in [2, 3]:
            synapses.append((pre, post, 2.0, 0.0))
    for pre in range(4):
        for post in [4, 5]:
            synapses.append((pre, post, 4.0, 0.0))
    for pre in [4, 5]:
        for post in range(4):
            synapses.append((pre, post, 3.0, -80.0))

    stimulation = {0: 12.0, 1: 12.0}

    return create_network_visualization(
        rn, synapses, stimulation, neuron_types,
        duration=500.0, title="E-I Recurrent Network",
        neuron_labels=neuron_labels
    )


def example_mixed_neuron_types():
    """Network with both HH and Izhikevich neurons"""
    print("\n5. Mixed Neuron Types (HH + Izhikevich)")

    # HH: global 0-1; Iz-RS: global 2; Iz-FS: global 3; Iz-IB: global 4
    rn = RegionalNetwork()
    rn.add_population("HH", 2, model=NeuronModelSpec.hh_default())
    rn.add_population("IZ_RS", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING))
    rn.add_population("IZ_FS", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING))
    rn.add_population("IZ_IB", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.INTRINSICALLY_BURSTING))

    # HH[0] -> all Izhikevich
    for dst in ["IZ_RS", "IZ_FS", "IZ_IB"]:
        rn.connect("HH", dst, lambda ns, nd: [(0, 0)], weight=8.0,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
    # IZ_RS[0] and IZ_FS[0] -> HH[1]
    for src in ["IZ_RS", "IZ_FS"]:
        rn.connect(src, "HH", lambda ns, nd: [(0, 1)], weight=5.0,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
    # IZ_FS inhibits IZ_IB
    rn.connect("IZ_FS", "IZ_IB", "all_to_all", weight=4.0,
               synapse=SynapseSpec.exponential(E_syn=-80.0, tau=5.0))

    neuron_labels = ["HH-1", "HH-2", "Iz-RS", "Iz-FS", "Iz-IB"]
    neuron_types = ["HH", "HH", "Izhikevich", "Izhikevich", "Izhikevich"]
    synapses = (
        [(0, post, 8.0, 0.0) for post in [2, 3, 4]] +
        [(pre, 1, 5.0, 0.0) for pre in [2, 3]] +
        [(3, 4, 4.0, -80.0)]
    )
    stimulation = {0: 15.0, 2: 8.0}

    return create_network_visualization(
        rn, synapses, stimulation, neuron_types,
        duration=500.0, title="Mixed Network (HH + Izhikevich)",
        neuron_labels=neuron_labels
    )


def example_winner_take_all():
    """Mutual inhibition network (winner-take-all)"""
    print("\n6. Winner-Take-All Network")

    rn = RegionalNetwork()
    rn.add_population("E", 4, model=NeuronModelSpec.hh_default())
    rn.connect("E", "E", "all_to_all", weight=4.0,
               synapse=SynapseSpec.exponential(E_syn=-80.0, tau=5.0))

    neuron_labels = ["HH-Hi", "HH-Med", "HH-Low", "HH-Min"]
    neuron_types = ["HH"] * 4
    synapses = [(pre, post, 4.0, -80.0)
                for pre in range(4) for post in range(4) if pre != post]
    stimulation = {0: 15.0, 1: 12.0, 2: 10.0, 3: 8.0}

    return create_network_visualization(
        rn, synapses, stimulation, neuron_types,
        duration=500.0, title="Winner-Take-All (Mutual Inhibition)",
        neuron_labels=neuron_labels
    )


def example_izhikevich_variety():
    """Network showcasing different Izhikevich neuron types"""
    print("\n7. Izhikevich Neuron Variety")

    # RS: global 0, FS: 1, IB: 2, CH: 3, LTS: 4
    rn = RegionalNetwork()
    rn.add_population("RS",  1, model=NeuronModelSpec.izhikevich(IzhikevichType.REGULAR_SPIKING))
    rn.add_population("FS",  1, model=NeuronModelSpec.izhikevich(IzhikevichType.FAST_SPIKING))
    rn.add_population("IB",  1, model=NeuronModelSpec.izhikevich(IzhikevichType.INTRINSICALLY_BURSTING))
    rn.add_population("CH",  1, model=NeuronModelSpec.izhikevich(IzhikevichType.CHATTERING))
    rn.add_population("LTS", 1, model=NeuronModelSpec.izhikevich(IzhikevichType.LOW_THRESHOLD_SPIKING))

    # RS -> all others (excitatory)
    for dst in ["FS", "IB", "CH", "LTS"]:
        rn.connect("RS", dst, "all_to_all", weight=5.0,
                   synapse=SynapseSpec.exponential(E_syn=0.0, tau=2.0))
    # FS inhibits IB and CH
    for dst in ["IB", "CH"]:
        rn.connect("FS", dst, "all_to_all", weight=3.0,
                   synapse=SynapseSpec.exponential(E_syn=-80.0, tau=5.0))

    neuron_labels = ["RS", "FS", "IB", "CH", "LTS"]
    neuron_types = ["Izhikevich"] * 5
    synapses = (
        [(0, post, 5.0, 0.0) for post in [1, 2, 3, 4]] +
        [(1, post, 3.0, -80.0) for post in [2, 3]]
    )
    stimulation = {0: 10.0, 1: 10.0, 2: 8.0, 3: 8.0, 4: 5.0}

    return create_network_visualization(
        rn, synapses, stimulation, neuron_types,
        duration=500.0, title="Izhikevich Neuron Types",
        neuron_labels=neuron_labels
    )


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
