cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(InferenceTestJni)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../cmake)

include(Utils)

set(NeoMathEngine_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.. ${CMAKE_BINARY_DIR}/NeoMathEngine)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../test/src ${CMAKE_BINARY_DIR}/test/src)

add_library(${PROJECT_NAME} SHARED
    InferenceTestJni.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE InferenceTests CommonTestTools NeoMathEngine gtest)

configure_target(${PROJECT_NAME})
