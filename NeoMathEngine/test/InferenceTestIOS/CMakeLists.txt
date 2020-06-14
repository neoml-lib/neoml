cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(inference.test.app)

set(APP_NAME inference.test)
set(BUNDLE_IDENTIFIER com.neomathengine.${APP_NAME})

set(DEVELOPMENT_TEAM_ID "Q58QAQTUTZ" CACHE STRING "Development team id.")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)

include(Utils)

# Build NeoMathEngine
set(NeoMathEngine_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_BINARY_DIR}/NeoMathEngine)

# Add tests libraries
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/test/src)

get_target_property(METAL_LIB NeoMetalLib OUTPUT_NAME)
set_property(SOURCE ${METAL_LIB} PROPERTY GENERATED ON)

set(APP_RESOURCES Main.storyboard LaunchScreen.storyboard ${METAL_LIB})

add_executable(${APP_NAME} MACOSX_BUNDLE
    AppDelegate.h
    ViewController.h
    
    AppDelegate.m
    ViewController.mm
    main.m
    
    ${APP_RESOURCES}
)

target_compile_options(${APP_NAME} PRIVATE -fobjc-arc)

find_library(UIKIT UIKit)
find_library(FOUNDATION Foundation)

target_link_libraries(${APP_NAME} PRIVATE
    InferenceTests
    CommonTestTools
    NeoMathEngine
    gtest
    ${UIKIT}
    ${FOUNDATION}
)

set_target_properties(${APP_NAME} PROPERTIES
    RESOURCE "${APP_RESOURCES}"
    MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist
)

set_xcode_property(${APP_NAME} PRODUCT_BUNDLE_IDENTIFIER "${BUNDLE_IDENTIFIER}" All)

set_xcode_property(${APP_NAME} CODE_SIGN_IDENTITY "iPhone Developer" All)
set_xcode_property(${APP_NAME} DEVELOPMENT_TEAM ${DEVELOPMENT_TEAM_ID} All)

configure_target(${APP_NAME})
