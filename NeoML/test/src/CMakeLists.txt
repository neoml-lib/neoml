project(NeoMLTestSrc)

add_library(${PROJECT_NAME} INTERFACE)

target_sources(${PROJECT_NAME} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/AutoDiffTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BpeTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClassificationAndRegressionTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusteringTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnBatchNormFusionTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnBlobTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnChannelwiseWith1x1BlockTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnCtcTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnDistributedTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnDropoutTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnLambSolverTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnLayersSerializationTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnLoraTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnMobileNetV2BlockTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnMobileNetV3BlockTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnOnnxLayerTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnOptimizationTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnParameterLayerTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnReferenceTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnRowwiseTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnSerializationTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnSimpleTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnSolverTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnSpaceToDepthTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnTiedEmbeddingTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DnnTransformerSourceMaskTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FloatVectorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GradientBoostingTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GraphGeneratorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/InferencePerformanceMultiThreadingTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PCATest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RandomProblem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseFloatMatrixTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestFixture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestParams.cpp
)

set_target_properties( ${PROJECT_NAME} PROPERTIES
    UNITY_BUILD_MODE BATCH
    UNITY_BUILD_BATCH_SIZE ${NeoML_UNITY_BUILD_BATCH_SIZE}
)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
