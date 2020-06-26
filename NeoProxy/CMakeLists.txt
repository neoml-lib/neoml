cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(NeoProxy LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

include(Version)
include(Settings)
include(GNUInstallDirs)

option(NeoProxy_BUILD_SHARED "Build NeoProxy as shared library." ON)

option(NeoProxy_INSTALL "Install NeoProxy" ON)

set_global_variables()

if(NeoProxy_BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED
        include/NeoProxy/NeoProxy.h
        src/NeoProxy.cpp
    )
else()
    add_library(${PROJECT_NAME} STATIC
        include/NeoProxy/NeoProxy.h
        src/NeoProxy.cpp
    )
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# adding NeoML, NeoMathEngine and NeoOnnx
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../NeoML ${CMAKE_BINARY_DIR}/NeoML)

target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_NEOPROXY)
if(NOT NeoProxy_BUILD_SHARED)
    target_compile_definitions(${PROJECT_NAME} PUBLIC STATIC_NEOPROXY)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE NeoML NeoOnnx)

configure_target(${PROJECT_NAME})

# Install
if(NeoProxy_INSTALL)
    if(USE_FINE_OBJECTS)
        fine_unexport_symbols(${PROJECT_NAME})
        fine_install(TARGETS ${PROJECT_NAME})
    else()
        install(
            TARGETS ${PROJECT_NAME}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

        install(DIRECTORY include/NeoProxy DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif()
endif()
