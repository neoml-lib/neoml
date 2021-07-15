cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(NeoMLTestJni)

set(NeoML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../..)

list(APPEND CMAKE_MODULE_PATH ${NeoML_DIR}/cmake)

include(Utils)

set(NeoMathEngine_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(NeoML_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(NeoOnnx_BUILD OFF CACHE BOOL "" FORCE)
add_subdirectory(${NeoML_DIR}/NeoML ${CMAKE_BINARY_DIR}/NeoML)

add_subdirectory(${NeoML_DIR}/NeoML/test/src ${CMAKE_BINARY_DIR}/NeoML/test/src)

add_library(${PROJECT_NAME} SHARED
    NeoMLTestJni.cpp
)

add_gtest_target()

target_link_libraries(${PROJECT_NAME} PRIVATE NeoMLTestSrc NeoML gtest)

configure_target(${PROJECT_NAME})
