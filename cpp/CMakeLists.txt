# OGDF layout extension for TMAP

pybind11_add_module(_tmap_ogdf
    src/layout.cpp
    src/bindings.cpp
)

# Include directories
target_include_directories(_tmap_ogdf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Handle OGDF found via find_package vs manual
if(TARGET OGDF::OGDF)
    # Modern CMake config
    target_link_libraries(_tmap_ogdf PRIVATE OGDF::OGDF)
else()
    # Manual paths
    target_include_directories(_tmap_ogdf PRIVATE ${OGDF_INCLUDE_DIR})
    if(OGDF_BUILD_INCLUDE_DIR)
        target_include_directories(_tmap_ogdf PRIVATE ${OGDF_BUILD_INCLUDE_DIR})
    endif()
    target_link_libraries(_tmap_ogdf PRIVATE ${OGDF_LIBRARY})
    if(COIN_LIBRARY)
        target_link_libraries(_tmap_ogdf PRIVATE ${COIN_LIBRARY})
    endif()
endif()

# Install to tmap/layout/
install(TARGETS _tmap_ogdf LIBRARY DESTINATION .)
