# Task 17.10: Python Device API + pybind11 Bindings

**Role:** Team lead  
**Status:** Not started  
**Depends on:** 17.1 (Device struct), 17.3 (RegionalNetwork::to(Device))  
**Unlocks:** 17.11 (tests use Python API), 17.12 (benchmarks use Python API)

---

## What to implement

### `src/python/bindings.cpp` — bind `Device`

```cpp
py::class_<Device>(m, "Device")
    .def_static("cpu",  &Device::cpu,  "CPU device")
    .def_static("cuda", &Device::cuda, py::arg("index") = 0, "CUDA device by index")
    .def_readonly("type",  &Device::type)
    .def_readonly("index", &Device::index)
    .def("__repr__", [](const Device& d){ return d.str(); })
    .def("__eq__", &Device::operator==);

py::enum_<Device::Type>(m, "DeviceType")
    .value("CPU",  Device::Type::CPU)
    .value("CUDA", Device::Type::CUDA);

m.def("cuda_device_count", &cuda_device_count,
      "Returns number of available CUDA devices (0 if no CUDA build).");
m.def("cuda_is_available", &cuda_is_available);
```

### `src/python/bindings.cpp` — bind `RegionalNetwork::to()`

On the `_RegionalNetwork` binding:

```cpp
.def("to", &RegionalNetwork::to, py::arg("device"),
     "Move all pool state to the given device. Call before simulate().")
.def("current_device", &RegionalNetwork::current_device)
```

### `src/hodgkin_huxley/_network/__init__.py` — Python-side `.to()`

```python
def to(self, device: "Device") -> "RegionalNetwork":
    """
    Move simulation to device. Returns self for chaining.

    rn.to(hh.Device.cuda(0))
    rn.to(hh.Device.cpu())
    """
    if device.type == DeviceType.CUDA and not cuda_is_available():
        raise RuntimeError(
            "CUDA device requested but this build was not compiled with HH_USE_CUDA "
            "or no CUDA devices are present."
        )
    self._rnet.to(device)
    return self
```

### `src/hodgkin_huxley/__init__.py`

Add to imports from `._core`:
```python
Device,
DeviceType,
cuda_device_count,
cuda_is_available,
```

Add to `__all__`:
```python
"Device",
"DeviceType",
"cuda_device_count",
"cuda_is_available",
```

### `src/hodgkin_huxley/_core.pyi` — type stubs

```python
class DeviceType(enum.Enum):
    CPU: DeviceType
    CUDA: DeviceType

class Device:
    type: DeviceType
    index: int
    @staticmethod
    def cpu() -> Device: ...
    @staticmethod
    def cuda(index: int = 0) -> Device: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

def cuda_device_count() -> int: ...
def cuda_is_available() -> bool: ...
```

Add `.to(device: Device) -> RegionalNetwork` and `.current_device() -> Device` stubs to `RegionalNetwork`.

### Convenience string parsing (optional but recommended)

Add a module-level `device(spec: str) -> Device` function in `__init__.py`:

```python
def device(spec: str) -> Device:
    """Parse 'cpu', 'cuda', 'cuda:0', 'cuda:1', ... into a Device."""
    if spec == "cpu":
        return Device.cpu()
    if spec.startswith("cuda"):
        idx = int(spec.split(":")[-1]) if ":" in spec else 0
        return Device.cuda(idx)
    raise ValueError(f"Unknown device spec: {spec!r}")
```

Add `"device"` to `__all__`.

---

## Key files

| File | Change |
|---|---|
| `src/python/bindings.cpp` | Bind `Device`, `DeviceType`, `cuda_device_count`, `cuda_is_available`, `to()`, `current_device()` |
| `src/hodgkin_huxley/_network/__init__.py` | Python `to()` with availability check |
| `src/hodgkin_huxley/__init__.py` | Export `Device`, `DeviceType`, `cuda_device_count`, `cuda_is_available`, `device` |
| `src/hodgkin_huxley/_core.pyi` | Stubs for all new bindings |

---

## Contract for downstream tasks

- Tests (17.11) use `hh.Device.cuda(0)`, `rn.to(hh.Device.cuda(0))`, `hh.cuda_is_available()`.
- Benchmarks (17.12) use `hh.device("cuda:0")` shorthand.
- On non-CUDA builds, `Device.cuda()` must construct without error; `rn.to(Device.cuda(0))` raises `RuntimeError` at the Python level with a clear message.
