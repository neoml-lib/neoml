cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(PythonWrapper LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

include(Version)
include(Settings)
include(FetchContent)

set_global_variables()

set(Python3_FIND_VERSION "3" CACHE STRING "Python3 find version.")

find_package(Python3 ${Python3_FIND_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.6.2
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Build NeoML
set(NeoML_INSTALL OFF CACHE BOOL "" FORCE)
set(NeoML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NeoOnnx_BUILD OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_BINARY_DIR}/NeoML)

# Build python module
pybind11_add_module(PythonWrapper
    src/common.cpp
    src/PyAutoDiff.cpp
    src/PyAccumulativeLookupLayer.cpp
    src/PyAccuracyLayer.cpp
    src/PyActivationLayer.cpp
    src/PyAddToObjectLayer.cpp
    src/PyArgmaxLayer.cpp
    src/PyAttentionDecoderLayer.cpp
    src/PyBaseConvLayer.cpp
    src/PyBatchNormalizationLayer.cpp
    src/PyBertConvLayer.cpp
    src/PyBinarizationLayer.cpp
    src/PyBroadcastLayer.cpp
    src/PyBytePairEncoder.cpp
    src/PyConcatLayer.cpp
    src/PyCastLayer.cpp
    src/PyConvLayer.cpp
    src/PyClustering.cpp
    src/PyCrfLayer.cpp
    src/PyCtcLayer.cpp
    src/PyCumSumLayer.cpp
    src/PyCustomLossLayer.cpp
    src/PyDataLayer.cpp
    src/PyDifferentialEvolution.cpp
    src/PyDotProductLayer.cpp
    src/PyDropoutLayer.cpp
    src/PyDnnDistributed.cpp
    src/PyDnn.cpp
    src/PyDnnBlob.cpp
    src/PyEltwiseLayer.cpp
    src/PyFullyConnectedLayer.cpp
    src/PyInitializer.cpp
    src/PyImageConversionLayer.cpp
    src/PyIndRnnLayer.cpp
    src/PyIrnnLayer.cpp
    src/PyGruLayer.cpp
    src/PyLayer.cpp
    src/PyLogicalLayer.cpp
    src/PyLossLayer.cpp
    src/PyLrnLayer.cpp
    src/PyLstmLayer.cpp
    src/PyMathEngine.cpp
    src/PyMatrixMultiplicationLayer.cpp
    src/PyMemoryFile.cpp
    src/PyMultichannelLookupLayer.cpp
    src/PyMultiheadAttentionLayer.cpp
    src/PyObjectNormalizationLayer.cpp
    src/PyPCA.cpp
    src/PyPoolingLayer.cpp
    src/PyPositionalEmbeddingLayer.cpp
    src/PyPrecisionRecallLayer.cpp
    src/PyQrnnLayer.cpp
    src/PyRandom.cpp
    src/PyReorgLayer.cpp
    src/PyRepeatSequenceLayer.cpp
    src/PyScatterGatherLayers.cpp
    src/PySequenceSumLayer.cpp
    src/PySoftmaxLayer.cpp
    src/PySplitLayer.cpp
    src/PySinkLayer.cpp
    src/PySolver.cpp
    src/PySourceLayer.cpp
    src/PySpaceAndDepthLayer.cpp
    src/PySubSequenceLayer.cpp
    src/PyTiedEmbeddingsLayer.cpp
    src/PyTrainingModel.cpp
    src/PyTransformerLayer.cpp
    src/PyTransformLayer.cpp
    src/PyTransposeLayer.cpp
    src/PyUpsampling2DLayer.cpp
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

