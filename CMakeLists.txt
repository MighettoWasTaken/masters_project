cmake_minimum_required(VERSION 3.15...3.27)
project(hodgkin_huxley VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard (C++14 for broader compiler compatibility)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for Release builds.
# CUDA files are excluded via COMPILE_LANGUAGE generator expressions — nvcc
# does not accept bare MSVC/GCC flags and will fatal-error if they reach it.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX,C>:/O2>
            $<$<COMPILE_LANGUAGE:CXX,C>:/fp:fast>
            $<$<COMPILE_LANGUAGE:CXX,C>:/arch:AVX2>
        )
    elseif(MINGW)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX,C>:-O3>
            $<$<COMPILE_LANGUAGE:CXX,C>:-ffast-math>
        )
    else()
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX,C>:-O3>
            $<$<COMPILE_LANGUAGE:CXX,C>:-march=native>
            $<$<COMPILE_LANGUAGE:CXX,C>:-ffast-math>
        )
    endif()
endif()

# Fetch Eigen 3.4 (header-only, SIMD-vectorized math)
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
# Eigen has a heavy CMakeLists; skip tests/docs
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# OpenMP (optional — enables pool-level parallelism in step_all)
find_package(OpenMP)

# CUDA detection — check_language(CUDA) probes the compiler before committing;
# guards against CUDAToolkit being found but the toolset integration failing
# (common with pip-fetched cmake on Windows).
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        # "native" compiles only for the GPU(s) present on this machine (CMake 3.24+).
        # Fallback covers Volta through Ada Lovelace for older CMake.
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90 CACHE STRING "CUDA architectures" FORCE)
        endif()
    endif()
endif()

# Option to build tests
option(BUILD_TESTS "Build C++ tests" OFF)

# Add the C++ library
add_subdirectory(src/cpp)

# Python bindings
pybind11_add_module(_core MODULE src/python/bindings.cpp)
target_link_libraries(_core PRIVATE hodgkin_huxley_core)
target_include_directories(_core PRIVATE src/cpp/include)

# Install the Python module
install(TARGETS _core DESTINATION hodgkin_huxley)

# Build tests if requested
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
