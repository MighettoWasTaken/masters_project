import numpy as np, hodgkin_huxley as hh
from hodgkin_huxley import RegionalNetwork, RecordingConfig
from benchmarks.ctxbgth_model import _make_gpe_spec, _make_stn_spec

def build(mk, on_gpu):
    spec, ca = mk()
    rn = RegionalNetwork(); rn.add_population("P", 4, model=spec)
    rn.add_intracellular(ca, "P")
    if on_gpu: rn.to(hh.Device.cuda(0))
    return rn

for name, mk in [("GPe", _make_gpe_spec), ("STN", _make_stn_spec)]:
    cfg = RecordingConfig(["V"])
    vc = build(mk, False).simulate(20.0, 0.01, {"P": 5.0}, record=cfg)["P"]["V"]
    vg = build(mk, True ).simulate(20.0, 0.01, {"P": 5.0}, record=cfg)["P"]["V"]
    d = np.abs(np.asarray(vc) - np.asarray(vg)).max()
    print(f"{name}: max|Vcpu-Vgpu| = {d:.2e}   {'OK' if d < 1e-6 else 'MISMATCH'}")

from benchmarks.ctxbgth_model import build_network
cfg = RecordingConfig(["spikes"], spike_threshold=-10.0); I={"TH":1.2,"GPe":3.0,"GPi":3.0}
g1 = build_network(n=10); g1.to(hh.Device.cuda(0))
g2 = build_network(n=10); g2.to(hh.Device.cuda(0))
o1 = g1.simulate(100.,0.01,I,record=cfg); o2 = g2.simulate(100.,0.01,I,record=cfg)
print("GPU run1 vs run2 GPe:", [len(s) for s in o1['GPe']['spikes']][:5], [len(s) for s in o2['GPe']['spikes']][:5])
