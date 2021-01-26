cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(PythonWrapper LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

include(Version)
include(Settings)

set_global_variables()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

find_path(pybind11_INCLUDE_DIR
    NAMES pybind11/pybind11.h
    DOC "The directory where pybind11 includes reside")
if (NOT (pybind11_INCLUDE_DIR STREQUAL ""))
    set(ENV{pybind11_DIR} ${pybind11_INCLUDE_DIR})
endif()
find_package(pybind11 REQUIRED)

# Build NeoML
set(NeoML_INSTALL OFF CACHE BOOL "" FORCE)
set(NeoML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NeoOnnx_BUILD OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_BINARY_DIR}/NeoML)

# Build python module
pybind11_add_module(PythonWrapper
    src/common.cpp
    src/PyArgmaxLayer.cpp
    src/PyAttentionDecoderLayer.cpp
    src/PyBaseConvLayer.cpp
    src/PyChannelwiseConvLayer.cpp
    src/PyConcatChannelsLayer.cpp
    src/PyConvLayer.cpp
    src/PyClustering.cpp
    src/PyDnn.cpp
    src/PyDnnBlob.cpp
    src/PyEltwiseLayer.cpp
    src/PyFullyConnectedLayer.cpp
    src/PyInitializer.cpp
    src/PyLayer.cpp
    src/PyLossLayer.cpp
    src/PyLstmLayer.cpp
    src/PyMathEngine.cpp
    src/PyMemoryFile.cpp
    src/PyMultichannelLookupLayer.cpp
    src/PyPoolingLayer.cpp
    src/PyRandom.cpp
    src/PyReLULayer.cpp
    src/PySinkLayer.cpp
    src/PySolver.cpp
    src/PySourceLayer.cpp
    src/PyTrainingModel.cpp
    src/PyWrapper.cpp
)

target_include_directories(PythonWrapper PRIVATE src)

target_link_libraries(PythonWrapper PRIVATE NeoML)

set_target_properties(PythonWrapper PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
    set_target_properties(PythonWrapper PROPERTIES INSTALL_RPATH "@loader_path")
endif()

install(TARGETS PythonWrapper DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES $<TARGET_FILE:NeoMathEngine> $<TARGET_FILE:NeoML> DESTINATION ${CMAKE_INSTALL_PREFIX})
if(TARGET NeoMathEngineAvx)
    install(FILES $<TARGET_FILE:NeoMathEngineAvx> DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

