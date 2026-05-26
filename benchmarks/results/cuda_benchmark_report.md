# CUDA Benchmark Report

## Run context

- Python: `3.13.2`
- Platform: `Windows-11-10.0.26200-SP0`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`
- Trials per data point: `3`
- Time step: `0.05 ms`


## What was benchmarked

- `hh_single` checks the composable neuron path without synapse overhead.
- `hh_dense` stresses the dense connected CUDA synapse path.
- `custom_modulated` exercises the 17.5 to 17.9 bridge: custom gate VM logic plus a nonlinear intracellular ODE and `SYNAPSE_G` modulation on the GPU runtime.

## Speedup tables

### HH default single population

One composable HH population with no synapses.

| Size | Total neurons | Total synapses | CPU median (s) | GPU median (s) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 2048 | 0 | 0.3262 | 0.8936 | 0.37x |
| 4096 | 4096 | 0 | 1.1778 | 1.1423 | 1.03x |
| 8192 | 8192 | 0 | 2.4312 | 1.6197 | 1.50x |
| 16384 | 16384 | 0 | 5.1136 | 3.1922 | 1.60x |

### HH default dense projection

Two HH populations with dense all-to-all AMPA projection.

| Size | Total neurons | Total synapses | CPU median (s) | GPU median (s) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 16384 | 0.7326 | 1.1437 | 0.64x |
| 256 | 512 | 65536 | 2.7921 | 1.3509 | 2.07x |
| 384 | 768 | 147456 | 6.3172 | 1.4623 | 4.32x |
| 512 | 1024 | 262144 | 11.3465 | 1.8449 | 6.15x |

### Custom gate plus synapse-g modulation

HH pre-population driving a post-population with a custom gate, a nonlinear intracellular ODE, and SYNAPSE_G modulation.

| Size | Total neurons | Total synapses | CPU median (s) | GPU median (s) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 16384 | 0.8572 | 1.6479 | 0.52x |
| 256 | 512 | 65536 | 2.8352 | 1.7658 | 1.61x |
| 384 | 768 | 147456 | 6.7757 | 1.8784 | 3.61x |
| 512 | 1024 | 262144 | 12.0990 | 2.3231 | 5.21x |

## Takeaways

- GPU overhead is visible on the smaller runs, especially when the network is not large enough to hide host-device movement and recording costs.
- The dense synapse case shows the clearest CUDA benefit as size grows because the GPU can keep the repeated synapse work on device.
- The custom gate plus modulation case also speeds up at larger sizes, which is the practical confirmation that the linked 17.5 and 17.9 path is not only correct but worth benchmarking.
