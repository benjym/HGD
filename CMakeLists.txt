cmake_minimum_required(VERSION 3.15)
project(move_voids_module LANGUAGES CXX)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Define module with sources
pybind11_add_module(d2q4_cpp HGD/motion/d2q4.cpp HGD/motion/helpers.cpp)

# Link Eigen library
target_link_libraries(d2q4_cpp PRIVATE Eigen3::Eigen)

# Set optimization and math flags
target_compile_options(d2q4_cpp PRIVATE -O3 -march=native) # -ffast-math) # -ffast-math breaks nan check operations!!!!

# Include Eigen headers
target_include_directories(d2q4_cpp PRIVATE ${EIGEN3_INCLUDE_DIRS})

# Link OpenMP correctly
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    target_link_libraries(d2q4_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compiler-specific settings
if(APPLE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        message(STATUS "Using AppleClang on macOS.")
        target_compile_options(d2q4_cpp PRIVATE -Xpreprocessor -fopenmp)
        target_link_directories(d2q4_cpp PRIVATE /usr/local/opt/libomp/lib)
        target_include_directories(d2q4_cpp PRIVATE /usr/local/opt/libomp/include)

    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Using Homebrew GCC on macOS.")
        target_include_directories(d2q4_cpp SYSTEM PRIVATE
            /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include
            /Library/Developer/CommandLineTools/usr/include/c++/v1
            /usr/local/include
        )
        target_link_directories(d2q4_cpp PRIVATE /usr/local/opt/libomp/lib)
        target_include_directories(d2q4_cpp PRIVATE /usr/local/opt/libomp/include)
    endif()
endif()

# Ensure the module is placed inside HGD/motion/
set_target_properties(d2q4_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/HGD/motion
)
