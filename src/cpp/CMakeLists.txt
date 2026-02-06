# Core C++ library
add_library(hodgkin_huxley_core STATIC
    src/neuron_base.cpp
    src/neuron.cpp
    src/izhikevich.cpp
    src/synapse.cpp
    src/hh_pool.cpp
    src/iz_pool.cpp
    src/network.cpp
)

target_include_directories(hodgkin_huxley_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hodgkin_huxley_core PUBLIC Eigen3::Eigen)

# Enable position independent code for shared library linking
set_target_properties(hodgkin_huxley_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
