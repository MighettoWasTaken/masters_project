# Task 17.4: Python Device API + pybind11 Bindings

**Role:** Team lead  
**Status:** Not started  
**Depends on:** 17.1 (Device struct — core bindings already done), 17.3 (RegionalNetwork::to(Device))  
**Unlocks:** 17.6, 17.7, 17.8, 17.9, 17.10 (all CUDA pool tasks need Python-level testing from day one)

---

## Note on 17.1 overlap

Task 17.1 already bound `Device`, `cuda_device_count`, and `cuda_is_available` in `bindings.cpp` and exported them from `__init__.py` so that `test_device.py` could run immediately. This task completes the remaining Python API surface:

- `rn.to(device)` / `rn.current_device()` on `RegionalNetwork`
- `hh.device("cuda:0")` string-parsing helper
- `_core.pyi` type stubs
- `DeviceType` enum export

Do not re-bind what 17.1 already bound — check `bindings.cpp` and `__init__.py` first.

---

## What to implement

### `src/python/bindings.cpp` — bind `RegionalNetwork::to()`

On the `_RegionalNetwork` binding (after 17.3 adds `RegionalNetwork::to()` to the C++ class):

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
    if device.type == hh.Device.Type.CUDA and not hh.cuda_is_available():
        raise RuntimeError(
            "CUDA device requested but this build was not compiled with HH_USE_CUDA "
            "or no CUDA devices are present."
        )
    self._rnet.to(device)
    return self

def current_device(self) -> "Device":
    return self._rnet.current_device()
```

### `src/hodgkin_huxley/__init__.py` — `device()` helper

```python
def device(spec: str) -> "Device":
    """Parse 'cpu', 'cuda', 'cuda:0', 'cuda:1', ... into a Device."""
    if spec == "cpu":
        return Device.cpu()
    if spec.startswith("cuda"):
        idx = int(spec.split(":")[-1]) if ":" in spec else 0
        return Device.cuda(idx)
    raise ValueError(f"Unknown device spec: {spec!r}")
```

Add `"device"` to `__all__`.

### `src/hodgkin_huxley/_core.pyi` — type stubs

```python
class Device:
    class Type(enum.Enum):
        CPU: Device.Type
        CUDA: Device.Type
    type: Device.Type
    index: int
    @staticmethod
    def cpu() -> Device: ...
    @staticmethod
    def cuda(index: int = 0) -> Device: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

def cuda_device_count() -> int: ...
def cuda_is_available() -> bool: ...
def device(spec: str) -> Device: ...
```

Add `.to(device: Device) -> RegionalNetwork` and `.current_device() -> Device` to the `RegionalNetwork` stub.

---

## Key files

| File | Change |
|---|---|
| `src/python/bindings.cpp` | Add `to()`, `current_device()` to `RegionalNetwork` binding |
| `src/hodgkin_huxley/_network/__init__.py` | Python `to()` + `current_device()` with availability check |
| `src/hodgkin_huxley/__init__.py` | Add `device()` helper + `"device"` to `__all__` |
| `src/hodgkin_huxley/_core.pyi` | Complete stubs for all Device API |

---

## Baseline tests (before PR to testing branch)

- [ ] `pip install -e .` completes without error
- [ ] `pytest tests/python/ -x -q` — all existing tests pass
- [ ] `hh.device("cpu") == hh.Device.cpu()` — string helper works
- [ ] `hh.device("cuda:1").index == 1`
- [ ] `rn.to(hh.Device.cpu())` on a minimal network — no exception, `rn.current_device() == hh.Device.cpu()`
- [ ] `rn.to(hh.Device.cuda(0))` raises `RuntimeError` on non-CUDA build

---

## Contract for downstream tasks

- All CUDA pool tasks (17.6–17.10) write per-task Python tests using `hh.Device.cuda(0)` and `rn.to(...)`.
- `hh.device("cuda:0")` is the shorthand used in benchmarks (17.12).
- On non-CUDA builds, `rn.to(Device.cuda(0))` must raise `RuntimeError` with a clear message — no silent wrong answer.
