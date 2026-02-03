cmake_minimum_required(VERSION 3.15)
project(move_voids_module LANGUAGES CXX)

# Find required packages
find_package(pybind11 REQUIRED)

# Create platform-agnostic core library
add_library(core STATIC HGD/motion/core.cpp HGD/motion/stress.cpp)

# === 【關鍵修改】加入這一行來解決 -fPIC 錯誤 ===
set_target_properties(core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# ============================================

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/HGD/motion)
target_compile_features(core PUBLIC cxx_std_14)

# Python extension
pybind11_add_module(d2q4_cpp HGD/motion/bindings_py.cpp)
target_link_libraries(d2q4_cpp PRIVATE core)
target_compile_definitions(d2q4_cpp PRIVATE EIGEN_DONT_ALIGN_STATICALLY EIGEN_DONT_PARALLELIZE)

# Set optimization and math flags
target_compile_options(d2q4_cpp PRIVATE -O3 -march=native)

# Ensure the module is placed inside HGD/motion/
set_target_properties(d2q4_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/HGD/motion
)