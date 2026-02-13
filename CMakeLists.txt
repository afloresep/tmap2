cmake_minimum_required(VERSION 3.15)
project(tmap_ogdf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

option(TMAP_BUILD_LAYOUT "Build OGDF-backed layout extension." ON)

if(TMAP_BUILD_LAYOUT)
    # ---------------------------------------------------------------------------
    # Build OGDF from bundled submodule (single supported path)
    # ---------------------------------------------------------------------------
    set(TMAP_OGDF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/ogdf")

    if(NOT EXISTS "${TMAP_OGDF_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "Bundled OGDF source was not found at extern/ogdf.\n"
            "Initialize submodules and reinstall:\n"
            "  git submodule update --init --recursive extern/ogdf\n"
            "  pip install -e .\n"
            "For core-only installs without layout:\n"
            "  pip install -e . --config-settings=cmake.define.TMAP_BUILD_LAYOUT=OFF"
        )
    endif()

    message(STATUS "Building OGDF from bundled source: ${TMAP_OGDF_SOURCE_DIR}")

    # Keep OGDF static and avoid enabling docs by accident.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(DOC_INSTALL OFF CACHE BOOL "" FORCE)

    add_subdirectory(extern/ogdf EXCLUDE_FROM_ALL)

    if(NOT TARGET OGDF)
        message(FATAL_ERROR "OGDF target was not created by extern/ogdf.")
    endif()

    add_subdirectory(cpp)
else()
    message(STATUS "TMAP_BUILD_LAYOUT=OFF: skipping OGDF layout extension build.")
endif()
