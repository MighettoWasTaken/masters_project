"""Verify CUDA environment and confirm GPU kernel execution.

Run with:
    python examples/verify_cuda.py

Exit code 0 = all GPUs passed. Non-zero = at least one failure.
"""
import sys
import hodgkin_huxley as hh

_ERRORS = {
    -1.0: (
        "Library built without CUDA support.",
        "Rebuild with CUDA Toolkit on PATH: `pip install -e . --no-build-isolation`.\n"
        "  Windows: ensure VS 2022 Build Tools are installed and SKBUILD_CMAKE_GENERATOR=Ninja is set.",
    ),
    1.0: (
        "cudaSetDevice() failed — could not select this device.",
        "Check that the device index is valid and the driver is not in exclusive mode.",
    ),
    2.0: (
        "cudaMalloc() failed — GPU memory allocation error.",
        "The GPU may be out of memory. Try closing other GPU applications.",
    ),
    3.0: (
        "Kernel launch failed (no binary for this GPU architecture).",
        "Rebuild so CMake compiles for this GPU: delete the build cache and reinstall.\n"
        "  `Remove-Item -Recurse -Force _skbuild; pip install -e . --no-build-isolation`",
    ),
    4.0: (
        "Kernel execution failed (cudaDeviceSynchronize error).",
        "The kernel crashed on the GPU. Check your CUDA driver version matches the toolkit.",
    ),
    5.0: (
        "cudaMemcpy (device→host) failed.",
        "The GPU ran the kernel but could not copy results back. "
        "Try updating your NVIDIA driver.",
    ),
}


def _smoke_status(code: float) -> tuple[bool, str]:
    """Return (passed, human_readable_message)."""
    if code == 0.0:
        return True, "PASS"
    if code in _ERRORS:
        reason, hint = _ERRORS[code]
        return False, f"FAIL — {reason}\n    Hint: {hint}"
    if code >= 6.0:
        bad_index = int(code - 6.0)
        return False, (
            f"FAIL — kernel wrote wrong value at index {bad_index}.\n"
            "    This indicates a GPU computation error. Try updating your NVIDIA driver."
        )
    return False, f"FAIL — unknown error code {code}"


def main() -> int:
    n = hh.cuda_device_count()
    print(f"CUDA devices found: {n}")

    if not hh.cuda_is_available():
        _, hint = _ERRORS[-1.0]
        print(f"CUDA not available.\n  Hint: {hint}")
        return 1

    all_passed = True
    for i in range(n):
        name = hh.cuda_device_name(i)
        code = hh.cuda_smoke_test(i)
        passed, msg = _smoke_status(code)
        print(f"  [{i}] {name}")
        print(f"       smoke_test: {msg}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
