"""
pytest configuration for hodgkin-huxley tests.

Sets EIGEN3_INCLUDE_DIR so that JIT compilation of custom SymPy expressions
(KineticSynapseModel.custom_expr, CUSTOM_EXPR gate compilation) can locate the
Eigen headers that were downloaded by CMake's FetchContent during the build.
"""

import os
import pathlib

# Path to Eigen headers fetched by CMake FetchContent during `pip install -e .`
_repo_root = pathlib.Path(__file__).parent.parent.parent
_eigen_path = _repo_root / "build" / "_deps" / "eigen-src"

if _eigen_path.is_dir() and "EIGEN3_INCLUDE_DIR" not in os.environ:
    os.environ["EIGEN3_INCLUDE_DIR"] = str(_eigen_path)
