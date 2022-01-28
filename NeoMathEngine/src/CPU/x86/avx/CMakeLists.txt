project(NeoMathEngineAvx)

add_library(${PROJECT_NAME} SHARED common.cpp)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

target_sources(${PROJECT_NAME}
    PRIVATE

    # Sources
    ./src/AvxMathEngine.cpp
    ./src/MatrixMultiplyingInterleaved/AvxMatrixMultiplying.cpp

    # Headers
    ./common.h
    ./src/BlobConvolution.h
    ./src/BlobConvolution.inl
    ./src/BlobConvolution_jit.inl
    ./src/BlobConvolution_jit_FltCnt_3.inl
    ./src/BlobConvolution_jit_FltCnt_6.inl
    ./src/BlobConvolution_jit_FltCnt_8.inl
    ./src/BlobConvolution_jit_FltCnt_16.inl
    ./src/BlobConvolution_jit_FltCnt_18.inl
    ./src/BlobConvolution_jit_FltCnt_24.inl
    ./src/BlobConvolution_jit_FltCnt_32.inl
    ./src/AvxCommon.h
    ./src/JitCommon.h
    ./src/MatrixMultiplyingInterleaved/Interleavers/Interleavers.h
    ./src/MatrixMultiplyingInterleaved/MicroKernels/Kernel_AVX_6x16.h
    ./src/MatrixMultiplyingInterleaved/MicroKernels/Kernel_AVX_6x8.h
    ./src/MatrixMultiplyingInterleaved/MicroKernels/Kernel_AVX_6x4.h
    ./src/MatrixMultiplyingInterleaved/MicroKernels/Kernel_AVX_6x2.h
    ./src/MatrixMultiplyingInterleaved/MicroKernels/Kernel_AVX_6x1.h
)

string(TOUPPER ${CMAKE_SYSTEM_NAME} UPPERCASE_CMAKE_SYSTEM_NAME)
target_compile_definitions(${PROJECT_NAME} PUBLIC _${UPPERCASE_CMAKE_SYSTEM_NAME})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../MatrixMultiplyingInterleavedCommon>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/MatrixMultiplyingInterleaved>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/MatrixMultiplyingInterleaved/Interleavers>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/MatrixMultiplyingInterleaved/MicroKernels>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

# Add some definitions
target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        "$<$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:_RELEASE>"
        "$<$<CONFIG:Debug>:_DEBUG>"
        "$<$<CONFIG:Release>:_FINAL>"
)

target_link_libraries(${PROJECT_NAME} PRIVATE NeoMathEngine)

# OpenMP
if(NOT DARWIN)
    link_openmp(${PROJECT_NAME})
endif()

# AVX
if(WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX)
elseif(LINUX OR DARWIN)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx -mfma>)
endif()

# Win resources
if(WIN32)
        if(USE_FINE_OBJECTS)
        target_include_directories(${PROJECT_NAME} PRIVATE 
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${FINE_ROOT}/Build/Inc ${FINE_ROOT}/FineObjects>>
        )
    else()
        target_include_directories(${PROJECT_NAME} PRIVATE 
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${CMAKE_CURRENT_SOURCE_DIR}/../../../../../Build/Inc>>
        )
    endif()
    enable_language(RC)
    target_sources(${PROJECT_NAME} PRIVATE ./NeoMathEngineAvx.rc)
endif()

# Install
if(NeoMathEngine_INSTALL)
    if(USE_FINE_OBJECTS)
        fine_install(TARGETS ${PROJECT_NAME} NOARCHIVE)
    else()
        install(
            TARGETS ${PROJECT_NAME}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
    endif()
endif()
