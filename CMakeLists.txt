# This file is part of TURNBACKLU.
#
# Copyright (c) 2024 Kai Pfeiffer
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.9)

project(TURNBACKLU)

# set(CMAKE_BUILD_TYPE Release CACHE STRING "Release RelWithDebInfo Debug DEFAULT: Release" FORCE)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")

option(SAFEGUARD "Enable safeguarding functions" OFF)
option(COMPILE_TESTS "compile the tests" OFF)
option(TIMEMEASUREMENTS "compile the tests" OFF)
option(MULTITHREADING "Enable multithreading via OPENMP" OFF)

add_compile_options(
  -O3 -g -Wall -Wpedantic -fPIC -std=c++14 -Wno-terminate
  )

################################################################################
if(SPARSE)
    add_definitions(-DSPARSE)
endif()
if(SAFEGUARD)
    #a number of safety tests is performed:
    # - nullspace test
    # - dLagrange multiplier test
    add_definitions(-DSAFEGUARD)
endif()
if(TIMEMEASUREMENTS)
    # solver timings
    add_definitions(-DTIMEMEASUREMENTS)
endif()
if (MULTITHREADING)
    add_definitions(-DMULTITHREADING)
endif()

file(GLOB SRC_TurnbackLU turnbackLU/*.cpp)
file(GLOB INC_TurnbackLU turnbackLU/*.h)

add_library(turnbackLU SHARED
    ${SRC_TurnbackLU}
  )

################################################################################

find_package (Eigen3 3.2.10 REQUIRED NO_MODULE)
target_link_libraries(turnbackLU Eigen3::Eigen)
target_link_libraries(turnbackLU clusol lapack blas)
find_package (eigenlusol REQUIRED)
target_link_libraries(turnbackLU eigenlusol)
if(MULTITHREADING)
    find_package(OpenMP)
    target_link_libraries(turnbackLU OpenMP::OpenMP_CXX)
endif()

################################################################################

set_target_properties(turnbackLU PROPERTIES SOVERSION 0)

install(
    TARGETS turnbackLU
    EXPORT "turnbackLUConfig"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)
install(FILES ${INC_TurnbackLU} DESTINATION "include/turnbackLU")

export(TARGETS turnbackLU
    NAMESPACE turnbackLU::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/turnbackLUConfig.cmake"
)
install(EXPORT
    turnbackLUConfig
    DESTINATION "/usr/local/lib/cmake/turnbackLU/"
    NAMESPACE turnbackLU::
)

if(COMPILE_TESTS)
         add_subdirectory(tests)
endif()
