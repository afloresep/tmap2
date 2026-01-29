cmake_minimum_required(VERSION 3.15)
project(tmap_ogdf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TMAP_OPTIONAL_OGDF "Build OGDF layout extension if available" ON)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find OGDF - try CONFIG first, then manual
set(OGDF_FOUND_METHOD "")

find_package(OGDF CONFIG QUIET)
if(OGDF_FOUND)
    set(OGDF_FOUND_METHOD "cmake-config")
endif()

# Fallback: environment variable (supports both installed and build-dir layouts)
if(NOT OGDF_FOUND AND DEFINED ENV{OGDF_DIR})
    set(OGDF_DIR "$ENV{OGDF_DIR}")

    # Try installed layout first: ${OGDF_DIR}/include, ${OGDF_DIR}/lib
    if(EXISTS "${OGDF_DIR}/include/ogdf" AND EXISTS "${OGDF_DIR}/lib")
        set(OGDF_INCLUDE_DIR "${OGDF_DIR}/include")
        set(OGDF_LIB_DIR "${OGDF_DIR}/lib")
        set(OGDF_FOUND_METHOD "env-var-installed")
    # Try build directory layout: ${OGDF_DIR}/include, ${OGDF_DIR}/build
    elseif(EXISTS "${OGDF_DIR}/include/ogdf" AND EXISTS "${OGDF_DIR}/build")
        set(OGDF_INCLUDE_DIR "${OGDF_DIR}/include")
        set(OGDF_LIB_DIR "${OGDF_DIR}/build")
        # OGDF build generates config_autogen.h in build/include/ogdf-release/
        if(EXISTS "${OGDF_DIR}/build/include/ogdf-release")
            set(OGDF_BUILD_INCLUDE_DIR "${OGDF_DIR}/build/include/ogdf-release")
        endif()
        set(OGDF_FOUND_METHOD "env-var-build")
    endif()

    if(OGDF_LIB_DIR)
        find_library(OGDF_LIBRARY NAMES OGDF ogdf PATHS ${OGDF_LIB_DIR} NO_DEFAULT_PATH)
        find_library(COIN_LIBRARY NAMES COIN coin PATHS ${OGDF_LIB_DIR} NO_DEFAULT_PATH)

        if(OGDF_LIBRARY)
            set(OGDF_FOUND TRUE)
        endif()
    endif()
endif()

# Build extension if OGDF found
if(TMAP_OPTIONAL_OGDF AND OGDF_FOUND)
    message(STATUS "")
    message(STATUS "=== OGDF Configuration ===")
    message(STATUS "  Found via: ${OGDF_FOUND_METHOD}")
    if(OGDF_INCLUDE_DIR)
        message(STATUS "  Include: ${OGDF_INCLUDE_DIR}")
    endif()
    if(OGDF_BUILD_INCLUDE_DIR)
        message(STATUS "  Build include: ${OGDF_BUILD_INCLUDE_DIR}")
    endif()
    if(OGDF_LIBRARY)
        message(STATUS "  Library: ${OGDF_LIBRARY}")
    endif()
    if(COIN_LIBRARY)
        message(STATUS "  COIN: ${COIN_LIBRARY}")
    endif()
    message(STATUS "")

    add_subdirectory(cpp)
else()
    message(STATUS "")
    message(WARNING "OGDF not found - layout extension will NOT be built.")
    message(WARNING "To enable: install OGDF and set OGDF_DIR=/path/to/ogdf")
    message(WARNING "  macOS: brew install ogdf")
    message(WARNING "  Linux: apt install libogdf-dev (or build from source)")
    message(STATUS "")
endif()
