# OGDF layout extension for TMAP

pybind11_add_module(_tmap_ogdf
    src/layout.cpp
    src/bindings.cpp
)

# Include directories
target_include_directories(_tmap_ogdf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OGDF from bundled submodule.
if(NOT TARGET OGDF)
    message(FATAL_ERROR "OGDF target is required but not available.")
endif()
target_link_libraries(_tmap_ogdf PRIVATE OGDF)

# Install to tmap/layout/
install(TARGETS _tmap_ogdf LIBRARY DESTINATION .)
