cmake_minimum_required(VERSION 3.15)
project(move_voids_module LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(d2q4_cpp HGD/motion/d2q4.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")

# Ensure the module is placed inside HGD/motion/
set_target_properties(d2q4_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/HGD/motion)

# Add error messages if desired
add_compile_definitions(PYBIND11_DETAILED_ERROR_MESSAGES)
