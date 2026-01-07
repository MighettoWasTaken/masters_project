cmake_minimum_required(VERSION 3.15...3.27)
project(hodgkin_huxley VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard (C++14 for broader compiler compatibility)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2 /fp:fast)
    elseif(MINGW)
        # Avoid -march=native on MinGW due to SEH/AVX bugs in older versions
        add_compile_options(-O3 -ffast-math)
    else()
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
endif()

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

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
