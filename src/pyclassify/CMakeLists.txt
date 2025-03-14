cmake_minimum_required(VERSION 3.12)

# This is a comment.
project(QR_binding VERSION 1.0
  DESCRIPTION "QR method"
  LANGUAGES CXX)


# Set C++ standard and compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compile options
add_compile_options(-O3 -Wall -Werror -Wpedantic)
find_package(pybind11 REQUIRED)
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})

pybind11_add_module(QR_cpp QR_bindings.cpp)