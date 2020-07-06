cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source! Please build out-of-source directory!")
endif()

project(NeoML LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
if(USE_FINE_OBJECTS)
    file(TO_CMAKE_PATH $ENV{ROOT} ROOT_PATH)
    list(APPEND CMAKE_MODULE_PATH ${ROOT_PATH}/FineObjects/Cmake)
endif()

include(Settings)
include(Version)
if(USE_FINE_OBJECTS)
    include(FineInstall)
else()
    include(GNUInstallDirs)
endif()

set(PROJECT_VERSION ${FINE_VERSION_MAJOR}.${FINE_VERSION_MINOR}.${FINE_VERSION_PATCH})

get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

# Use FineObjects to build NeoML
option(USE_FINE_OBJECTS "Use FineObjects to build NeoML." OFF)

# Build NeoML tests
option(NeoML_BUILD_TESTS "Enable and build all of NeoML's tests." ON)

# Build NeoML as shared library.
option(NeoML_BUILD_SHARED "Build NeoML as shared library." ON)

# Install NeoML or not
option(NeoML_INSTALL "Install NeoML" ON)

# Build NeoOnnx library
option(NeoOnnx_BUILD "Build NeoOnnx" ON)
if(USE_FINE_OBJECTS)
    set(NeoOnnx_BUILD OFF CACHE BOOL "" FORCE) 
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

# NeoOnnx
if(NeoOnnx_BUILD)
    add_subdirectory(../NeoOnnx ${CMAKE_BINARY_DIR}/NeoOnnx)
endif()

# Tests
if(NeoML_BUILD_TESTS AND NOT IS_SUBPROJECT)
    enable_testing()
    if(USE_FINE_OBJECTS)
        add_subdirectory(${ROOT_PATH}/NeoMLTest/test ${CMAKE_BINARY_DIR}/test)
    endif()
endif()

# Install
if(NeoML_INSTALL)
    set(INSTALLED_TARGETS NeoML NeoMathEngine)
    if(NeoOnnx_BUILD)
        list(APPEND INSTALLED_TARGETS NeoOnnx)
    endif()
    
    if(USE_FINE_OBJECTS)
        fine_install(TARGETS ${INSTALLED_TARGETS})
    else()
        install(
            TARGETS ${INSTALLED_TARGETS} FineObjLite
            EXPORT NeoMLConfig
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

        install(DIRECTORY include/NeoML DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

        install(EXPORT NeoMLConfig
            NAMESPACE Neo::
            DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
        )
    endif()
endif()
