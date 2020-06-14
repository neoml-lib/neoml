cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source! Please build out-of-source directory!")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

include(Settings)
include(Version)

set(NeoML_VERSION ${FINE_VERSION_MAJOR}.${FINE_VERSION_MINOR}.${FINE_VERSION_PATCH})

project(NeoML VERSION ${NeoML_VERSION} LANGUAGES CXX)

get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

# Use FineObjects to build NeoML
option(USE_FINE_OBJECTS "Use FineObjects to build NeoML." OFF)

# Build NeoML tests
option(NeoML_BUILD_TESTS "Enable and build all of NeoML's tests." ON)

# Build NeoML as shared library.
option(NeoML_BUILD_SHARED "Build NeoML as shared library." ON)

# Install NeoML or not
option(NeoML_INSTALL "Install NeoML" ON)

if(USE_FINE_OBJECTS)
    include(FineInstall)
else()
    include(GNUInstallDirs)
endif()

# Set cmake default variables and some usefull variables
set_global_variables()

if(USE_FINE_OBJECTS)
    find_package(FineObjects)
    if(NOT FineObj_FOUND)
        message(FATAL_ERROR "FineObjects is not found!")
    endif()
endif()

# Note, that NeoML target builds NeoMathEngine
add_subdirectory(src)

# Tests
if(NeoML_BUILD_TESTS AND NOT IS_SUBPROJECT)
    enable_testing()
    if(USE_FINE_OBJECTS)
        add_subdirectory(${FINE_ROOT}/NeoMLTest/test ${CMAKE_BINARY_DIR}/test)
    endif()
endif()
