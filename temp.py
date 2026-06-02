import time, hodgkin_huxley as hh
from hodgkin_huxley import RecordingConfig
from examples.benchmark_complexity_sweep import _build_network
from benchmarks.ctxbgth_model import build_network
cfg=RecordingConfig(["spikes"],spike_threshold=-10.0)
net=_build_network(4, 8, 10); net.to(hh.Device.cuda(0))   # tier 4 = STN, 8 pools×10
net.simulate(20.,0.05,{"P0":10.0},record=cfg)
t=time.perf_counter(); net.simulate(200.,0.05,{"P0":10.0},record=cfg); print("8×STN(10):", time.perf_counter()-t)
