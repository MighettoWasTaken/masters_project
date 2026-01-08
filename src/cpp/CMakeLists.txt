# Core C++ library
add_library(hodgkin_huxley_core STATIC
    src/neuron.cpp
    src/network.cpp
)

target_include_directories(hodgkin_huxley_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Enable position independent code for shared library linking
set_target_properties(hodgkin_huxley_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
