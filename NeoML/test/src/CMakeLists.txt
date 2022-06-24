project(NeoMLTestSrc)

add_library(${PROJECT_NAME} INTERFACE)

target_sources(${PROJECT_NAME} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/AutoDiffTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TestFixture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TestFixture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestParams.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TestParams.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MlTestCommon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusteringTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnLayersSerializationTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnSerializationTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/InferencePerformanceMultiThreadingTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FloatVectorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseFloatMatrixTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RandomProblem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RandomProblem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ClassificationAndRegressionTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LAMBSolverTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PCATest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GradientBoostingTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnBlobTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnSolverTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CtcTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnDistributedTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BpeTest.cpp
)

set_target_properties( ${PROJECT_NAME} PROPERTIES
    UNITY_BUILD_MODE BATCH
    UNITY_BUILD_BATCH_SIZE ${NeoML_UNITY_BUILD_BATCH_SIZE}
)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
