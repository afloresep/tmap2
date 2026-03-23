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
    # Locate OGDF: prefer bundled submodule, fall back to FetchContent download
    # ---------------------------------------------------------------------------
    set(TMAP_OGDF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/ogdf")

    if(EXISTS "${TMAP_OGDF_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "Building OGDF from bundled source: ${TMAP_OGDF_SOURCE_DIR}")
    else()
        message(STATUS "Bundled OGDF not found — downloading via FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            ogdf
            GIT_REPOSITORY https://github.com/ogdf/ogdf.git
            GIT_TAG        403b4df616227f096c42465cf53bdc918a2f63bb
            GIT_SHALLOW    FALSE
        )
        FetchContent_MakeAvailable(ogdf)
        set(TMAP_OGDF_SOURCE_DIR "${ogdf_SOURCE_DIR}")
    endif()

    # Keep OGDF static and avoid enabling docs by accident.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(DOC_INSTALL OFF CACHE BOOL "" FORCE)

    if(NOT TARGET OGDF)
        add_subdirectory("${TMAP_OGDF_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ogdf" EXCLUDE_FROM_ALL)
    endif()

    if(NOT TARGET OGDF)
        message(FATAL_ERROR "OGDF target was not created.")
    endif()

    add_subdirectory(cpp)
else()
    message(STATUS "TMAP_BUILD_LAYOUT=OFF: skipping OGDF layout extension build.")
endif()
