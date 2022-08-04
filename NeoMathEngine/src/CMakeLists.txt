set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NeoMathEngine_BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED common.cpp)
else()
    add_library(${PROJECT_NAME} STATIC common.cpp)
endif()

set(CPU_COMMON_SOURCES
    # Sources
    CPU/CpuMathEngineBlas.cpp
    CPU/CpuMathEngineDnn3dConv.cpp
    CPU/CpuMathEngineDnnConv.cpp
    CPU/CpuMathEngineDnnCtc.cpp
    CPU/CpuMathEngineDnnChannelwiseConv.cpp
    CPU/CpuMathEngineDnnDropout.cpp
    CPU/CpuMathEngineDnnLrn.cpp
    CPU/CpuMathEngineDnnLstm.cpp
    CPU/CpuMathEngineDnn.cpp
    CPU/CpuMathEngineDnnPooling.cpp
    CPU/CpuMathEngineDnnRleConv.cpp
    CPU/CpuMathEngineDnnTimeConv.cpp
    CPU/CpuMathEngine.cpp
    CPU/CpuMathEngineVectorMath.cpp
    CPU/CpuMathEngineDnnDistributed.cpp
    CrtAllocatedObject.cpp
    DllLoader.cpp
    MathEngineDeviceStackAllocator.cpp
    MathEngineDnnDropout.cpp
    MathEngine.cpp
    MathEngineHostStackAllocator.cpp
    MemoryPool.cpp
    ThreadPool.cpp
    common.cpp
)

target_sources(${PROJECT_NAME}
    PRIVATE
    ${CPU_COMMON_SOURCES}
    # Headers
    common.h
    MathEngineAllocator.h
    MathEngineCommon.h
    MathEngineDeviceStackAllocator.h
    MathEngineDll.h
    MathEngineDnnConv.h
    MathEngineDnnDropout.h
    MathEngineDnnLrn.h
    MathEngineDnnPoolings.h
    MathEngineHostStackAllocator.h
    MemoryHandleInternal.h
    MemoryPool.h
    RawMemoryManager.h
    CPU/CpuMathEngine.h
    CPU/CpuRandom.h
    CPU/CpuMathEnginePrivate.h
    CPU/CpuMathEngineOmp.h
    CPU/CpuMathEngineDnnDistributed.h
    CPU/CpuMathEngineDnnLstm.h
    CPU/CpuExecutionScope.h
    CPU/CpuFunctorCommon.h

    CPU/MatrixMultiplyingInterleavedCommon/CpuMemoryHelper.h
    CPU/MatrixMultiplyingInterleavedCommon/MatrixMultiplier.h
    CPU/MatrixMultiplyingInterleavedCommon/MatrixMultiplying.h
    CPU/MatrixMultiplyingInterleavedCommon/Interleavers/InterleaverBase.h
    CPU/MatrixMultiplyingInterleavedCommon/MicroKernels/MicroKernelBase.h

    ../include/NeoMathEngine/NeoMathEngineDefs.h
    ../include/NeoMathEngine/BlobDesc.h
    ../include/NeoMathEngine/BlobType.h
    ../include/NeoMathEngine/LookupData.h
    ../include/NeoMathEngine/MemoryHandle.h
    ../include/NeoMathEngine/MemoryHandle.inl
    ../include/NeoMathEngine/SparseMatrixDesc.h
    ../include/NeoMathEngine/NeoMathEngine.h
    ../include/NeoMathEngine/NeoMathEngineException.h
    ../include/NeoMathEngine/SimdMathEngine.h
    ../include/NeoMathEngine/ThreadPool.h
)

add_library(NeoML::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set_target_properties( ${PROJECT_NAME} PROPERTIES
    UNITY_BUILD_MODE GROUP
)
set_property(SOURCE ${CPU_COMMON_SOURCES} PROPERTY UNITY_GROUP 1)
set_property(SOURCE MathEngineDeviceStackAllocator.cpp PROPERTY SKIP_UNITY_BUILD_INCLUSION ON)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

source_group("Header Files\\NeoMathEngine" REGULAR_EXPRESSION "^.*NeoMathEngine/.+\.h|inl$")
source_group("CPU" REGULAR_EXPRESSION "CPU\/.*")

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CPU>
)

if(NOT USE_FINE_OBJECTS)
    target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)
endif()

string(TOUPPER ${CMAKE_SYSTEM_NAME} UPPERCASE_CMAKE_SYSTEM_NAME)
target_compile_definitions(${PROJECT_NAME} PUBLIC _${UPPERCASE_CMAKE_SYSTEM_NAME})

# OS specific settings
if(ANDROID OR LINUX)
    target_sources(${PROJECT_NAME}
    PRIVATE
        CPU/PerformanceCountersCpuLinux.cpp
    )
endif()

# CPU specific settings
if((DARWIN AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*") OR (ANDROID AND ANDROID_ABI MATCHES "^arm.*") OR (IOS AND IOS_ARCH MATCHES "^arm.*"))
    set(CPU_ARM_SOURCES
        CPU/arm/CpuArmMathEngineBlas.cpp
        CPU/arm/CpuArmMathEngineDnn.cpp
        CPU/arm/CpuArmMathEngineVectorMath.cpp
        CPU/arm/CpuArmMathEngineDnn3dConv.cpp
        CPU/arm/CpuArmMathEngineMatrixMultiplying.cpp
    )
    target_sources(${PROJECT_NAME} 
    PRIVATE
        ${CPU_ARM_SOURCES}
        CPU/arm/CpuArm.h
        CPU/arm/CpuArmFunctors.h
        CPU/arm/CpuArmMathEngineBlasPrivate.h
        CPU/arm/CpuArmMathEngineVectorMathPrivate.h
        CPU/arm/MatrixMultiplyingInterleaved/Interleavers/Interleaver_ARM64.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM32NEON_6x8.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_1x12.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_1x4.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_4x1.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_4x12.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_4x4.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_8x1.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_8x12.h
        CPU/arm/MatrixMultiplyingInterleaved/MicroKernels/Kernel_ARM64_8x4.h
    )
    set_property(SOURCE ${CPU_ARM_SOURCES} PROPERTY UNITY_GROUP 1)
    source_group("CPU\\arm" REGULAR_EXPRESSION "CPU\/arm\/.*")
    target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CPU/arm>)
else()
    set(CPU_X86_SOURCES
        CPU/x86/CpuX86MathEngineBlas.cpp
        CPU/x86/CpuX86MathEngineBlasMkl.cpp
        CPU/x86/CpuX86MathEngineDnn.cpp
        CPU/x86/CpuX86MathEngineVectorMath.cpp
        CPU/x86/CpuX86MathEngineVectorMathMkl.cpp
        CPU/x86/CpuX86MathEngineDnn3dConv.cpp
    )
    target_sources(${PROJECT_NAME} 
    PRIVATE
        ${CPU_X86_SOURCES}
        CPU/x86/CpuX86.h
        CPU/x86/CpuX86Functors.h
        CPU/x86/CpuX86MathEngineBlasPrivate.h
        CPU/x86/CpuX86MathEngineVectorMathPrivate.h
    )
    set_property(SOURCE ${CPU_X86_SOURCES} PROPERTY UNITY_GROUP 1)
    source_group("CPU\\x86" REGULAR_EXPRESSION "CPU\/x86\/.*")
    target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CPU/x86>)
    if(NEOML_USE_AVX)
        target_sources(${PROJECT_NAME}
        PRIVATE
            CPU/x86/AvxDll.cpp
            CPU/x86/AvxDll.h
        )
        target_compile_definitions(${PROJECT_NAME} PRIVATE NEOML_USE_AVX)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/CPU/x86/avx)
        set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "\$ORIGIN")
    endif()
endif()

# GPU specific source and settings
if((WIN32 OR LINUX) AND CMAKE_SIZEOF_VOID_P EQUAL 8)

    if(USE_FINE_OBJECTS)
        # The CUDA must exist at $ENV{ROOT}/ThirdParty
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{ROOT}/ThirdParty/CUDA/${CMAKE_SYSTEM_NAME}/nvcc)
        file(TO_CMAKE_PATH "${CUDA_TOOLKIT_ROOT_DIR}" CUDA_TOOLKIT_ROOT_DIR)

        set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

        if(WIN32)
            set(CMAKE_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}.exe")
        endif()

        if(LINUX)
            set(NCCL_ROOT_DIR $ENV{ROOT}/ThirdParty/CUDA/Linux/nccl)
        endif()
    endif() # USE_FINE_OBJECTS

    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        # Using CUDA
        enable_language(CUDA)
        
        # extracting CUDA version
        string(REPLACE "." ";" NEOML_CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
        list(GET NEOML_CUDA_VERSION_LIST 0 NEOML_CUDA_VERSION_MAJOR)
        list(GET NEOML_CUDA_VERSION_LIST 1 NEOML_CUDA_VERSION_MINOR)

        if(USE_FINE_OBJECTS AND (NOT "${NEOML_CUDA_VERSION_MAJOR}" STREQUAL "11" OR NOT "${NEOML_CUDA_VERSION_MINOR}" STREQUAL "2"))
            message(FATAL_ERROR "CUDA 11.2 is required for build with FineObjects (found: ${NEOML_CUDA_VERSION_MAJOR}.${NEOML_CUDA_VERSION_MINOR})")
        elseif(NOT USE_FINE_OBJECTS AND NOT "${NEOML_CUDA_VERSION_MAJOR}" STREQUAL "11")
            message(FATAL_ERROR "CUDA 11 is required for build (found: ${NEOML_CUDA_VERSION_MAJOR}.${NEOML_CUDA_VERSION_MINOR})")
        else()
            target_compile_definitions(${PROJECT_NAME} PRIVATE NEOML_USE_CUDA)
            target_compile_options(${PROJECT_NAME} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_35,code=compute_35>
                $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>
                $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>
                $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>
                $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
            target_sources(${PROJECT_NAME}
                PRIVATE
                    GPU/CUDA/CublasDll.cpp
                    GPU/CUDA/CudaCommon.cpp
                    GPU/CUDA/CudaDevice.cpp
                    GPU/CUDA/CudaMathEngine.cpp
                    GPU/CUDA/CusparseDll.cpp
                    GPU/CUDA/NcclDll.cpp
                    # CUDA Sources
                    GPU/CUDA/CudaMathEngineBlas.cu
                    GPU/CUDA/CudaMathEngineCublas.cu
                    GPU/CUDA/CudaMathEngineCusparse.cu
                    GPU/CUDA/CudaMathEngineDnn3dConv.cu
                    GPU/CUDA/CudaMathEngineDnnChannelwiseConv.cu
                    GPU/CUDA/CudaMathEngineDnnConv.cu
                    GPU/CUDA/CudaMathEngineDnnCtc.cu
                    GPU/CUDA/CudaMathEngineDnn.cu
                    GPU/CUDA/CudaMathEngineDnnDropout.cu
                    GPU/CUDA/CudaMathEngineDnnLrn.cu
                    GPU/CUDA/CudaMathEngineDnnLstm.cu
                    GPU/CUDA/CudaMathEngineDnnPoolings.cu
                    GPU/CUDA/CudaMathEngineDnnRleConv.cu
                    GPU/CUDA/CudaMathEngineDnnTimeConv.cu
                    GPU/CUDA/CudaMathEngine.cu
                    GPU/CUDA/CudaMathEngineVectorMath.cu
                    GPU/CUDA/CudaMathEngineDnnDistributed.cpp
                    # Headers
                    GPU/CUDA/CublasDll.h
                    GPU/CUDA/CublasFunctions.h
                    GPU/CUDA/NcclDll.h
                    GPU/CUDA/NcclFunctions.h
                    GPU/CUDA/CudaAssert.h
                    GPU/CUDA/CudaBlobDesc.h
                    GPU/CUDA/CudaCommon.h
                    GPU/CUDA/CudaDevice.h
                    GPU/CUDA/CudaMathEngineDnnConvs.h
                    GPU/CUDA/CudaMathEngineDnnPoolings.h
                    GPU/CUDA/CudaMathEngine.h
                    GPU/CUDA/CusparseDll.h
                    GPU/CUDA/CusparseFunctions.h
                    GPU/CUDA/Kernels/CudaBlasKernels.h
                    GPU/CUDA/Kernels/CudaDnn3dConvKernels.h
                    GPU/CUDA/Kernels/CudaDnn3dPoolingKernels.h
                    GPU/CUDA/Kernels/CudaDnnChannelwiseConvKernels.h
                    GPU/CUDA/Kernels/CudaDnnConvKernels.h
                    GPU/CUDA/Kernels/CudaDnnCtcKernels.h
                    GPU/CUDA/Kernels/CudaDnnDropoutKernels.h
                    GPU/CUDA/Kernels/CudaDnnGlobalPoolingKernels.h
                    GPU/CUDA/Kernels/CudaDnnGlobalTimePoolingKernels.h
                    GPU/CUDA/Kernels/CudaDnnKernels.h
                    GPU/CUDA/Kernels/CudaDnnLrnKernels.h
                    GPU/CUDA/Kernels/CudaDnnPoolingKernels.h
                    GPU/CUDA/Kernels/CudaDnnRleConvKernels.h
                    GPU/CUDA/Kernels/CudaDnnTimeConvKernels.h
                    GPU/CUDA/Kernels/CudaDnnTimePoolingKernels.h
                    GPU/CUDA/Kernels/CudaGrid.h
                    GPU/CUDA/Kernels/CudaRandom.h
                    GPU/CUDA/Kernels/CudaReduce.h
                    GPU/CUDA/Kernels/CudaVectorMathKernels.h
            )
            target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/GPU/CUDA>)
            target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>)
            source_group("GPU\\CUDA" REGULAR_EXPRESSION "GPU\/CUDA\/[A-Za-z0-9_]+\.(h|cpp|cu)")
            source_group("GPU\\CUDA\\Kernels" REGULAR_EXPRESSION "GPU\/CUDA\/Kernels\/.+")

            find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS ${NCCL_ROOT_DIR})

	        if (NOT "${NCCL_INCLUDE_DIR}" STREQUAL "NCCL_INCLUDE_DIR-NOTFOUND")
                message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIR})")
                target_compile_definitions(${PROJECT_NAME} PUBLIC -D NEOML_USE_NCCL)
                target_include_directories(${PROJECT_NAME} PRIVATE ${NCCL_INCLUDE_DIR})
            endif ()
        endif()
    else()
        message(STATUS "No CUDA support.")
    endif()
    
elseif(IOS)
    # Apple metal
    if(NeoMathEngine_ENABLE_METAL)

        target_compile_definitions(${PROJECT_NAME} PRIVATE NEOML_USE_METAL)

        set(IOS_SOURCES
            GPU/Metal/MetalMathEngine.h
            GPU/Metal/MetalMathEngine.mm
            GPU/Metal/MetalMathEngineVectorMath.mm
            GPU/Metal/MetalMathEngineBlas.mm
            GPU/Metal/MetalMathEngineDnn.mm
            GPU/Metal/MetalMathEngineDnnConvs.mm
            GPU/Metal/MetalMathEngineDnnDropout.mm
            GPU/Metal/MetalMathEngineDnnPoolings.mm
            GPU/Metal/MetalMathEngineDnnLrn.mm
            GPU/Metal/MetalMathEngineDnnLstm.mm
            GPU/Metal/MetalKernel.h
            GPU/Metal/MetalKernel.mm
            GPU/Metal/MetalCommandQueue.h
            GPU/Metal/MetalCommandQueue.mm
        )

        set_source_files_properties(${IOS_SOURCES} PROPERTIES COMPILE_FLAGS -fmodules)
        target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/GPU/Metal>)
        target_sources(${PROJECT_NAME} PRIVATE ${IOS_SOURCES})
    
        add_subdirectory(GPU/Metal/NeoMetalLib ${CMAKE_BINARY_DIR}/NeoMetalLib)
        add_dependencies(${PROJECT_NAME} NeoMetalLib)
    
        get_target_property(METAL_LIB NeoMetalLib OUTPUT_NAME)
    
        set_property(SOURCE ${METAL_LIB} PROPERTY GENERATED ON)
        target_sources(${PROJECT_NAME} PRIVATE ${METAL_LIB})
    
        if(NeoMathEngine_BUILD_SHARED)
            set_target_properties(${PROJECT_NAME} PROPERTIES RESOURCE ${METAL_LIB})
        endif()
    endif()
endif()

if(NeoMathEngine_ENABLE_VULKAN)
    if(WIN32 OR ANDROID)
        # Vulkan
        if(USE_FINE_OBJECTS)
            set(ENV{VULKAN_SDK} ${FINE_ROOT}/ThirdParty/Vulkan/${CMAKE_HOST_SYSTEM_NAME})
        endif()
        
        find_path(Vulkan_INCLUDE_DIR
            NAMES vulkan/vulkan.h
            PATHS "$ENV{VULKAN_SDK}/Include"
                    "$ENV{VULKAN_SDK}/include"
        )
        
        find_program(GLSL_COMPILER 
            NAMES glslangValidator
            PATHS "$ENV{VULKAN_SDK}/Bin"
                    "$ENV{VULKAN_SDK}/bin"
        )
        
        if(Vulkan_INCLUDE_DIR AND GLSL_COMPILER)
            message(STATUS "Found Vulkan: TRUE")

            if(USE_FINE_OBJECTS AND CMAKE_HOST_SYSTEM_NAME MATCHES "Linux|Darwin")
                execute_process(COMMAND chmod +x ${GLSL_COMPILER})
            endif()

            target_compile_definitions(${PROJECT_NAME} PRIVATE NEOML_USE_VULKAN VK_NO_PROTOTYPES)
            
            target_include_directories(${PROJECT_NAME}
                PRIVATE 
                    GPU/Vulkan
                    "${CMAKE_CURRENT_BINARY_DIR}/GPU/Vulkan"
                    ${Vulkan_INCLUDE_DIR}
            )
            
            set(VULKAN_SOURCES
                GPU/Vulkan/VulkanCommandQueue.cpp
                GPU/Vulkan/VulkanDll.cpp
                GPU/Vulkan/VulkanImage.cpp
                GPU/Vulkan/VulkanMathEngineBlas.cpp
                GPU/Vulkan/VulkanMathEngineDnnConvs.cpp
                GPU/Vulkan/VulkanMathEngineDnn.cpp
                GPU/Vulkan/VulkanMathEngineDnnPoolings.cpp
                GPU/Vulkan/VulkanMathEngineDnnLrn.cpp
                GPU/Vulkan/VulkanMathEngineDnnLstm.cpp
                GPU/Vulkan/VulkanMathEngine.cpp
                GPU/Vulkan/VulkanMathEngineVectorMath.cpp
                GPU/Vulkan/VulkanShader.cpp
                GPU/Vulkan/VulkanMemory.cpp
            )
            target_sources(${PROJECT_NAME}
                PRIVATE
                    ${VULKAN_SOURCES}
                    GPU/Vulkan/VulkanCommandQueue.h
                    GPU/Vulkan/VulkanDll.h
                    GPU/Vulkan/VulkanImage.h
                    GPU/Vulkan/VulkanMathEngine.h
                    GPU/Vulkan/VulkanShader.h
                    GPU/Vulkan/VulkanMemory.h
                    GPU/Vulkan/shaders/common/CommonStruct.h
            )
            set_property(SOURCE ${VULKAN_SOURCES} PROPERTY UNITY_GROUP 2)
            source_group("GPU\\Vulkan" REGULAR_EXPRESSION "GPU\/Vulkan\/[A-Za-z0-9_]+\.(h|cpp)")
            
            add_subdirectory(GPU/Vulkan/shaders)
            
            add_dependencies(${PROJECT_NAME} VulkanShaders)
        else()
            message(STATUS "Found Vulkan: FALSE")
        endif()
    endif()
endif()

# Add some definitions
target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        "$<$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:_RELEASE>"
        "$<$<CONFIG:Debug>:_DEBUG>"
        "$<$<CONFIG:Release>:_FINAL>"
    PRIVATE 
        BUILD_NEOMATHENGINE
)

if(NOT NeoMathEngine_BUILD_SHARED)
    target_compile_definitions(${PROJECT_NAME} PUBLIC STATIC_NEOMATHENGINE)
endif()


# OpenMP
if(NOT IOS AND NOT DARWIN)
    link_openmp(${PROJECT_NAME})
endif()

# SSE2
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_compile_options(${PROJECT_NAME} PRIVATE /arch:SSE2)
elseif(((LINUX OR DARWIN) AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86.*") OR (ANDROID AND ANDROID_ABI MATCHES "^x86.*") OR (IOS_ARCH MATCHES "^x86.*"))
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msse2>)
endif()

# Arm neon
if(ANDROID AND ANDROID_ABI STREQUAL "armeabi-v7a")
    target_compile_options(${PROJECT_NAME} PRIVATE -mfpu=neon)
endif()

# MKL
if(NOT ANDROID AND NOT IOS AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86.*)|(amd64)|(AMD64)")

    if(USE_FINE_OBJECTS)
        set(ENV{MKLROOT} ${FINE_ROOT}/ThirdParty/MKL/${CMAKE_SYSTEM_NAME})
    endif()

    set(MKL_USE_STATIC_LIBS ON)
    find_package(MKL)

    if(MKL_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE MKL::Libs)
        target_compile_definitions(${PROJECT_NAME} PRIVATE NEOML_USE_MKL)
    elseif(USE_FINE_OBJECTS)
        message(FATAL_ERROR "MKL not found!")
    endif()
endif()

# Win resources
if(WIN32)
    if(USE_FINE_OBJECTS)
        target_include_directories(${PROJECT_NAME} PRIVATE 
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${FINE_ROOT}/Build/Inc ${FINE_ROOT}/FineObjects>>
        )
        target_compile_definitions(${PROJECT_NAME} PUBLIC NEOMATHENGINE_USE_FINEOBJ)
    else()
        target_include_directories(${PROJECT_NAME} PRIVATE 
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${CMAKE_CURRENT_SOURCE_DIR}/../../Build/Inc>>
        )
    endif()

    enable_language(RC)
    target_sources(${PROJECT_NAME} PRIVATE ../NeoMathEngine.rc)
endif()

configure_target(${PROJECT_NAME})

# Install
if(NeoMathEngine_INSTALL)
    if(USE_FINE_OBJECTS)
        fine_install(${PROJECT_NAME})
        if(WIN32 AND CMAKE_CUDA_COMPILER)
            file(GLOB CUDA_LIBS ${FINE_ROOT}/ThirdParty/CUDA/Windows/redist/x64/*.dll)
            install(FILES ${CUDA_LIBS} DESTINATION ${FINE_ROOT}/Win${FINE_BUILD_TYPE}.x64)
        elseif(LINUX AND CMAKE_CUDA_COMPILER)
            file(GLOB CUDA_LIBS ${FINE_ROOT}/ThirdParty/CUDA/Linux/redist/x64/*.so.11 ${FINE_ROOT}/ThirdParty/CUDA/Linux/redist/x64/libnccl.so)
            install(FILES ${CUDA_LIBS} DESTINATION ${FINE_ROOT}/X.${CMAKE_SYSTEM_NAME}.${FINE_BUILD_TYPE}/x86_64)
        endif()
    else()
        install(DIRECTORY ../include/NeoMathEngine DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

        if(CMAKE_PROJECT_NAME STREQUAL "NeoMathEngine")
            install(
                TARGETS NeoMathEngine
                EXPORT NeoMathEngineConfig
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
        
            install(EXPORT NeoMathEngineConfig
                NAMESPACE NeoML::
                DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
            )
        endif()
    endif()
endif()