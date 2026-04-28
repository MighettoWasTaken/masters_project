"""
pytest configuration for hodgkin-huxley tests.

Sets EIGEN3_INCLUDE_DIR so that JIT compilation of custom SymPy expressions
(KineticSynapseModel.custom_expr, CUSTOM_EXPR gate compilation) can locate the
Eigen headers that were downloaded by CMake's FetchContent during the build.
"""

import os
import pathlib
import sys

# Block gmpy2 before pytest imports any test file that uses sympy.
# On Windows some conda environments ship a gmpy2 build whose DLL has a missing
# entry point (STATUS_ENTRYPOINT_NOT_FOUND), which causes a fatal process crash
# rather than a catchable ImportError. Blocking it here — before collection —
# ensures mpmath falls back to pure Python regardless of import order in tests.
if sys.platform == "win32":
    sys.modules.setdefault("gmpy2", None)  # type: ignore[assignment]

# Path to Eigen headers fetched by CMake FetchContent during `pip install -e .`
_repo_root = pathlib.Path(__file__).parent.parent.parent
_eigen_path = _repo_root / "build" / "_deps" / "eigen-src"

if _eigen_path.is_dir() and "EIGEN3_INCLUDE_DIR" not in os.environ:
    os.environ["EIGEN3_INCLUDE_DIR"] = str(_eigen_path)
