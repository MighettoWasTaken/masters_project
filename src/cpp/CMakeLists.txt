# Core C++ library
add_library(hodgkin_huxley_core STATIC
    src/neuron_base.cpp
    src/neuron.cpp
    src/izhikevich.cpp
    src/synapse_base.cpp
    src/synapse_spec.cpp
    src/hh_pool.cpp
    src/iz_pool.cpp
    src/network.cpp
    src/regional_network.cpp
    src/ion_channels.cpp
    src/composable_neuron.cpp
    src/composable_pool.cpp
    src/dbs_stimulator.cpp
    src/network/pool_manager.cpp
    src/device.cpp
)

target_include_directories(hodgkin_huxley_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hodgkin_huxley_core PUBLIC Eigen3::Eigen)

# OpenMP is disabled on MinGW/GCC<6 due to ICE with -fopenmp + C++14 generic lambdas.
# Enabled on Linux (GCC/Clang), macOS (AppleClang), and MSVC.
if(OpenMP_CXX_FOUND AND NOT MINGW)
    target_link_libraries(hodgkin_huxley_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(hodgkin_huxley_core PRIVATE HH_USE_OPENMP)
endif()

if(CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    target_sources(hodgkin_huxley_core PRIVATE
        src/cuda_hh_pool.cu
        src/cuda_iz_pool.cu
        src/cuda_composable_pool.cu
        src/cuda_synapse.cu
    )
    target_link_libraries(hodgkin_huxley_core PUBLIC CUDA::cudart)
    target_compile_definitions(hodgkin_huxley_core PRIVATE HH_USE_CUDA)
    # Separable compilation (-rdc=true) is not needed — kernels are
    # self-contained within their own .cu files and don't call across them.
endif()

# Enable position independent code for shared library linking
set_target_properties(hodgkin_huxley_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
