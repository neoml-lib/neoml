cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(NeoInferenceJni)

add_library(${PROJECT_NAME} SHARED NeoInferenceJni.cpp)

set(NeoProxy_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../NeoProxy ${CMAKE_BINARY_DIR}/NeoProxy)

string(TOUPPER ${CMAKE_SYSTEM_NAME} UPPERCASE_CMAKE_SYSTEM_NAME)
target_compile_definitions(${PROJECT_NAME} PUBLIC _${UPPERCASE_CMAKE_SYSTEM_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../include)

target_link_libraries(${PROJECT_NAME} PRIVATE NeoProxy)
