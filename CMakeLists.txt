cmake_minimum_required(VERSION 3.15)
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

add_subdirectory(Fastor)
add_subdirectory(dbg-macro)

add_subdirectory(planning)
add_subdirectory(demo)
