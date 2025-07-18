cmake_minimum_required(VERSION 3.12)

project(QR_binding VERSION 1.0
	DESCRIPTION "Python bindings of some helper functions written in C++."
  LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/pyclassify)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/pyclassify)

include(FetchContent)
# Fetch Eigen 3.4.0 (or whatever version you need)
FetchContent_Declare(
  eigen
  URL  https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(eigen)

set(PYBIND11_FINDPYTHON ON)

# To find pybind11 on Ulysses:
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED CONFIG)

# Needed for compilation (again, Ulysses... :/ )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
 
add_compile_options(-O3 -Wall -Werror -Wpedantic)

pybind11_add_module(cxx_utils ${CMAKE_SOURCE_DIR}/src/pyclassify/cxx_utils.cpp)

target_link_libraries(cxx_utils   PRIVATE    Eigen3::Eigen  )
set_target_properties(cxx_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

target_compile_features(cxx_utils PUBLIC cxx_std_17)
install(TARGETS cxx_utils LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/src/pyclassify)
