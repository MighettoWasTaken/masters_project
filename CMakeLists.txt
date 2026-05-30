cmake_minimum_required(VERSION 3.15...3.27)
project(hodgkin_huxley VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard (C++14 for broader compiler compatibility)
set(CMAKE_CXX_STANDARD 17)
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

# Fetch Eigen 3.4 headers if a system package is unavailable.
# We intentionally avoid adding Eigen's own CMake project as a subdirectory:
# Eigen is header-only for our use, and its optional BLAS/LAPACK sub-builds
# pull in extra toolchain requirements (including Fortran on Windows) that we
# do not need for this project.
find_package(Eigen3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
  endif()

  add_library(eigen_headers INTERFACE)
  target_include_directories(eigen_headers INTERFACE
        ${eigen_SOURCE_DIR}
    )
  add_library(Eigen3::Eigen ALIAS eigen_headers)
endif()

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# OpenMP (optional — enables pool-level parallelism in step_all)
find_package(OpenMP)

# CUDA detection — seed a likely nvcc path before check_language(CUDA).
# On Windows, find_package(CUDAToolkit) often fails first when nvcc is not
# already visible to CMake, so we probe the compiler before depending on the
# toolkit package result.
set(_hh_nvcc_candidate "")
if(NOT CMAKE_CUDA_COMPILER)
  if(DEFINED ENV{CUDACXX} AND EXISTS "$ENV{CUDACXX}")
    set(_hh_nvcc_candidate "$ENV{CUDACXX}")
  elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc.exe")
    set(_hh_nvcc_candidate "$ENV{CUDA_PATH}/bin/nvcc.exe")
  elseif(EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe")
    set(_hh_nvcc_candidate
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe")
  endif()

  if(_hh_nvcc_candidate)
    set(CMAKE_CUDA_COMPILER "${_hh_nvcc_candidate}" CACHE FILEPATH
            "Path to NVIDIA nvcc compiler" FORCE)
  endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit QUIET)
  # "native" compiles only for the GPU(s) present on this machine (CMake 3.24+).
  # Fallback covers Volta through Ada Lovelace for older CMake.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90 CACHE STRING "CUDA architectures" FORCE)
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
