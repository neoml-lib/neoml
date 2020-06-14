cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(NeoInference)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

include(Version)
include(Settings)
include(GNUInstallDirs)

set_global_variables()

add_library(${PROJECT_NAME} SHARED
    NeoInference.mm NeoInference.h
)

target_include_directories(${PROJECT_NAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_options(${PROJECT_NAME} PRIVATE -fmodules -fobjc-arc)

# TODO Use find_package instead add_subdirectory
set(NeoProxy_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(NeoProxy_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../NeoProxy ${CMAKE_BINARY_DIR}/NeoProxy)

string(TOUPPER ${CMAKE_SYSTEM_NAME} UPPERCASE_CMAKE_SYSTEM_NAME)
target_compile_definitions(${PROJECT_NAME} PUBLIC _${UPPERCASE_CMAKE_SYSTEM_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(${PROJECT_NAME} PRIVATE NeoProxy)

configure_target(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/NeoInference.h
)

get_target_property(METAL_LIB NeoMetalLib OUTPUT_NAME)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${METAL_LIB} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copy NeoMetalLib to NeoInference framework."
)

install(TARGETS ${PROJECT_NAME} FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR})
