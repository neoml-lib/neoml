cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(NeoOnnx)

if(USE_FINE_OBJECTS)
    message(FATAL_ERROR "NeoOnnx does not support FineObjects")
endif()

include(FetchContent)

# build protobuf
FetchContent_Declare(
    Protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
    GIT_TAG v3.11.4
    PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/protobuf.patch
)

set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "")
set(protobuf_BUILD_CONFORMANCE OFF CACHE BOOL "")
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "")
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
FetchContent_GetProperties(Protobuf)

if(NOT protobuf_POPULATED)
    FetchContent_Populate(Protobuf)
    set(CMAKE_CXX_FLAGS_backup ${CMAKE_CXX_FLAGS})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "IOS")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shorten-64-to-32")
    endif()

    add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR})

    # fix known protobuf build bug on Android
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        target_link_libraries(libprotobuf log)
    endif()

    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_backup})
endif()

set(PROJECT_VERSION ${FINE_VERSION_MAJOR}.${FINE_VERSION_MINOR}.${FINE_VERSION_PATCH})

# Build NeoOnnx as shared library.
option(NeoOnnx_BUILD_SHARED "Build NeoOnnx as shared library." ON)

# Install NeoOnnx.
if(DEFINED NeoML_INSTALL)
    option(NeoOnnx_INSTALL "Install NeoOnnx." ${NeoML_INSTALL})
else()
    set(NeoOnnx_INSTALL ON)
endif()

add_subdirectory(src)
