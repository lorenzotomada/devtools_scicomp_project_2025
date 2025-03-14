cmake_minimum_required(VERSION 3.12)

project(QR_binding VERSION 1.0
  DESCRIPTION "QR method"
  LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/pyclassify)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/pyclassify)

set(PYBIND11_FINDPYTHON ON)

# To find pybind11 on Ulysses:
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# Needed for compilation (again, Ulysses... :/ )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
 
add_compile_options(-O3 -Wall -Werror -Wpedantic)

pybind11_add_module(QR_cpp ${CMAKE_SOURCE_DIR}/src/pyclassify/QR_bindings.cpp)
target_compile_features(QR_cpp PUBLIC cxx_std_17)
