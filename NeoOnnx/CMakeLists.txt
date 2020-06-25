cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(NeoOnnx LANGUAGES CXX)

if(USE_FINE_OBJECTS)
    message(FATAL_ERROR "NeoOnnx does not support FineObjects")
endif()

set(PROJECT_VERSION ${FINE_VERSION_MAJOR}.${FINE_VERSION_MINOR}.${FINE_VERSION_PATCH})

# Build NeoOnnx as shared library.
option(NeoOnnx_BUILD_SHARED "Build NeoOnnx as shared library." ON)

add_subdirectory(src)

if(NeoML_INSTALL)
    install(DIRECTORY include/NeoOnnx DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
