cmake_minimum_required(VERSION 3.18)

set(VCPKG_MANIFEST_MODE ON)
set(VCPKG_TARGET_TRIPLET x64-linux)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(planning LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_program(CCACHE "ccache")
if(CCACHE)
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif(CCACHE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_ROOT_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DFASTOR_ENABLE_RUNTIME_CHECKS=0 -march=native -ffast-math")

if (SKBUILD)
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_path(DBG_MACRO_INCLUDE_DIRS "dbg.h")

execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(Fastor)

add_subdirectory(planning)
