cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if(NeoOnnx_BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED common.cpp)
else()
    add_library(${PROJECT_NAME} STATIC common.cpp)
endif()

add_library(NeoML::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set_target_properties( ${PROJECT_NAME} PROPERTIES
    UNITY_BUILD_MODE BATCH
    UNITY_BUILD_BATCH_SIZE ${NeoML_UNITY_BUILD_BATCH_SIZE}
)

target_sources( ${PROJECT_NAME} PRIVATE
    GraphInput.cpp
    GraphInitializer.cpp
    GraphOutput.cpp
    NeoOnnx.cpp
    NeoOnnxImport.cpp
    NeoOnnxCheck.cpp
    LayerOperator.cpp
    LayerUtils.cpp
    Operator.cpp
    Operators/ActivationOperator.cpp
    Operators/ArgMaxOperator.cpp
    Operators/BatchNormalizationOperator.cpp
    Operators/CastOperator.cpp
    Operators/ConcatOperator.cpp
    Operators/ConstantOfShapeOperator.cpp
    Operators/ConstantOperator.cpp
    Operators/ConvOperator.cpp
    Operators/ConvTransposeOperator.cpp
    Operators/CumSumOperator.cpp
    Operators/DropoutOperator.cpp
    Operators/EltwiseOperator.cpp
    Operators/ExpandOperator.cpp
    Operators/FlattenOperator.cpp
    Operators/GatherOperator.cpp
    Operators/GemmOperator.cpp
    Operators/GlobalPoolOperator.cpp
    Operators/IdentityOperator.cpp
    Operators/InstanceNormalizationOperator.cpp
    Operators/LogicalOperator.cpp
    Operators/LrnOperator.cpp
    Operators/LstmOperator.cpp
    Operators/MatMulOperator.cpp
    Operators/NonZeroOperator.cpp
    Operators/OneHotOperator.cpp
    Operators/PadOperator.cpp
    Operators/PoolOperator.cpp
    Operators/RangeOperator.cpp
    Operators/ResizeOperator.cpp
    Operators/ReshapeOperator.cpp
    Operators/ScatterOperator.cpp
    Operators/ShapeOperator.cpp
    Operators/SliceOperator.cpp
    Operators/SoftmaxOperator.cpp
    Operators/SplitOperator.cpp
    Operators/SqueezeOperator.cpp
    Operators/TransposeOperator.cpp
    Operators/UnsqueezeOperator.cpp
    Operators/UpsampleOperator.cpp
    TensorUtils.cpp

    ../include/NeoOnnx/NeoOnnx.h
    ../include/NeoOnnx/NeoOnnxDefs.h
    ../include/NeoOnnx/NeoOnnxImport.h
    AttributeGetters.h
    GraphInput.h
    GraphInitializer.h
    GraphOutput.h
    LayerOperator.h
    LayerUtils.h
    NeoOnnxCheck.h
    Operator.h
    Operators/ActivationOperator.h
    Operators/ArgMaxOperator.h
    Operators/BatchNormalizationOperator.h
    Operators/CastOperator.h
    Operators/ConcatOperator.h
    Operators/ConstantOfShapeOperator.h
    Operators/ConstantOperator.h
    Operators/ConvOperator.h
    Operators/ConvTransposeOperator.h
    Operators/CumSumOperator.h
    Operators/DropoutOperator.h
    Operators/EltwiseOperator.h
    Operators/ExpandOperator.h
    Operators/FlattenOperator.h
    Operators/GatherOperator.h
    Operators/GemmOperator.h
    Operators/GlobalPoolOperator.h
    Operators/IdentityOperator.h
    Operators/InstanceNormalizationOperator.h
    Operators/LogicalOperator.h
    Operators/LrnOperator.h
    Operators/LstmOperator.h
    Operators/MatMulOperator.h
    Operators/NonZeroOperator.h
    Operators/OneHotOperator.h
    Operators/PadOperator.h
    Operators/PoolOperator.h
    Operators/RangeOperator.h
    Operators/ReshapeOperator.h
    Operators/ResizeOperator.h
    Operators/ScatterOperator.h
    Operators/ShapeOperator.h
    Operators/SliceOperator.h
    Operators/SoftmaxOperator.h
    Operators/SplitOperator.h
    Operators/SqueezeOperator.h
    Operators/TransposeOperator.h
    Operators/UnsqueezeOperator.h
    Operators/UpsampleOperator.h
    Tensor.h
    TensorLayout.h
    TensorUtils.h
)

if(ANDROID)
    set(CMAKE_LIBRARY_ARCHITECTURE ${ANDROID_ABI})
elseif(IOS)
    set(CMAKE_LIBRARY_ARCHITECTURE ${IOS_ARCH})
endif()

configure_target(${PROJECT_NAME})

if(DEFINED ENV{Protobuf_ROOT} AND (IOS OR ANDROID))
    list(APPEND CMAKE_FIND_ROOT_PATH $ENV{Protobuf_ROOT})
endif()

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
    set(CMAKE_UNITY_BUILD OFF)
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "")
set(protobuf_MSVC_STATIC_RUNTIME "${STATIC_NEOML}" CACHE BOOL "")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "")
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "")
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "")

if(USE_FINE_OBJECTS)
    add_subdirectory("${FINE_ROOT}/ThirdParty/protobuf/cmake"
        "${CMAKE_BINARY_DIR}/protobuf" EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG v3.11.4
    )

    FetchContent_GetProperties(protobuf)

    if(NOT protobuf_POPULATED)
        FetchContent_Populate(protobuf)
        add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR} EXCLUDE_FROM_ALL)    
    endif()
endif()

if(LINUX)
    target_compile_options(libprotobuf PRIVATE -Wno-stringop-overflow)
    target_compile_options(libprotobuf-lite PRIVATE -Wno-stringop-overflow)
endif()

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
    set(CMAKE_UNITY_BUILD ${NeoML_UNITY_BUILD})
endif()

if(USE_FINE_OBJECTS)
    target_link_libraries(${PROJECT_NAME} PRIVATE FineObjects)

    # Todo: properly link FineStlStaticPart if any problems occure
    if(MSVC)
        target_link_libraries(${PROJECT_NAME} PRIVATE "msvcprt$<$<CONFIG:Debug>:d>")
    endif()
endif()

set(PROTO_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpp_proto/${CMAKE_CFG_INTDIR})
set(PROTO_SRCS ${PROTO_DIR}/onnx.pb.cc)
set(PROTO_HDRS ${PROTO_DIR}/onnx.pb.h)

# On some systems protoc can't create the output directory
add_custom_target(create_proto_dir COMMAND ${CMAKE_COMMAND} -E make_directory ${PROTO_DIR})
add_custom_command(
    OUTPUT ${PROTO_HDRS} ${PROTO_SRCS}
    DEPENDS protobuf::protoc create_proto_dir
    COMMAND protobuf::protoc
        "--cpp_out=${PROTO_DIR}"
        "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto"
        "${CMAKE_CURRENT_SOURCE_DIR}/proto/onnx.proto"
)

target_sources(${PROJECT_NAME} PRIVATE ${PROTO_HDRS} ${PROTO_SRCS})
set_property(SOURCE ${PROTO_HDRS} ${PROTO_SRCS} PROPERTY SKIP_UNITY_BUILD_INCLUSION ON)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W2)
    target_compile_options(${PROJECT_NAME} PRIVATE /MD$<$<CONFIG:Debug>:d>)
    # Because of the code generated by protobuf
    target_compile_options(${PROJECT_NAME} PRIVATE /wd4946 /wd4251)
elseif(NOT WIN32)
    set_property(SOURCE ${PROTO_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-pedantic -Wno-array-bounds ")
    if(USE_FINE_OBJECTS)
        target_compile_options(${PROJECT_NAME} PRIVATE -Wno-nonportable-include-path)
    endif()
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/NeoOnnx/src>
        $<BUILD_INTERFACE:${PROTO_DIR}>
)

target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_NEOONNX)
if(NOT NeoOnnx_BUILD_SHARED)
    target_compile_definitions(${PROJECT_NAME} PUBLIC STATIC_NEOONNX)
endif()

if(NeoOnnx_BUILD_SHARED AND NOT WIN32)
    set(EXPORT_SYMBOLS
        _ZN7NeoOnnx12LoadFromOnnxEPKcRKNS_15CImportSettingsERN5NeoML4CDnnERNS_18CImportedModelInfoE
        _ZN7NeoOnnx12LoadFromOnnxEPKviRKNS_15CImportSettingsERN5NeoML4CDnnERNS_18CImportedModelInfoE)
    if(LINUX OR ANDROID)
        set(EXPORT_SYMBOLS_STR "${EXPORT_SYMBOLS};")
        configure_file(../NeoOnnx.version.in NeoOnnx.version)
        target_link_options(${PROJECT_NAME} PRIVATE -Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/NeoOnnx.version)
    elseif(DARWIN OR IOS)
            foreach(SYM ${EXPORT_SYMBOLS})
                target_link_options(${PROJECT_NAME} PRIVATE -Wl,-exported_symbol,_${SYM})
            endforeach() 
    endif()
endif()

if(NOT TARGET protobuf::libprotobuf)
    message(FATAL_ERROR "Protobuf not found!")
endif()

# Fix for a known protobuf Windows dll issue https://github.com/protocolbuffers/protobuf/issues/2502
if((Protobuf_LIBRARY MATCHES ".*dll") OR (Protobuf_LIBRARY MATCHES ".*DLL"))
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROTOBUF_USE_DLLS)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE protobuf::libprotobuf PUBLIC NeoML)
if(ANDROID)
    target_link_libraries(${PROJECT_NAME} PRIVATE log)
endif()

if(NOT USE_FINE_OBJECTS)
    target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)
endif()

if(WIN32)
    if(USE_FINE_OBJECTS)
        target_include_directories(${PROJECT_NAME} PRIVATE
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${FINE_ROOT}/Build/Inc ${FINE_ROOT}/FineObjects ${MESSAGES_DIR}>>
        )
    else()
        target_include_directories(${PROJECT_NAME} PRIVATE 
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${CMAKE_CURRENT_SOURCE_DIR}/../../Build/Inc>>
        )
    endif()
    
    enable_language(RC)
    target_sources(${PROJECT_NAME} PRIVATE ../NeoOnnx.rc)
endif()
