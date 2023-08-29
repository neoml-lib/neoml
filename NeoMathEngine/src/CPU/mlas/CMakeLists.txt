# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(MLAS_AMX_SUPPORTED FALSE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
  # match assembler version, AMX instructions are supported from 2.38
  if (CMAKE_ASM_COMPILER_ID STREQUAL "GNU")
    execute_process(
        COMMAND as --version
        OUTPUT_VARIABLE _as_version
    )
    # 2.38 or later
    if (_as_version MATCHES "GNU.[Aa]ssembler.*(2\\.38|2\\.39|2\\.[4-9][0-9]|[3-9]\\.[0-9][0-9])")
        set(MLAS_AMX_SUPPORTED TRUE)
    endif()
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(MLAS_AMX_SUPPORTED TRUE)
endif()


#
# All hardware agnostic source files here
# hardware specific files would cause trouble in
# multi-target build
#
add_library(onnxruntime_mlas STATIC
  lib/platform.cpp
  lib/threading.cpp
  lib/sgemm.cpp
  lib/halfgemm.cpp
  lib/qgemm.cpp
  lib/qdwconv.cpp
  lib/convolve.cpp
  lib/convsym.cpp
  lib/pooling.cpp
  lib/transpose.cpp
  lib/reorder.cpp
  lib/snchwc.cpp
  lib/activate.cpp
  lib/logistic.cpp
  lib/tanh.cpp
  lib/erf.cpp
  lib/compute.cpp
  lib/quantize.cpp
  lib/qgemm_kernel_default.cpp
  lib/qladd.cpp
  lib/qlmul.cpp
  lib/qpostprocessor.cpp
  lib/qlgavgpool.cpp
  lib/qdwconv_kernelsize.cpp
)
# onnxruntime_configure_target(${target_name})
set_property(TARGET onnxruntime_mlas PROPERTY CXX_STANDARD 17)
target_compile_definitions(onnxruntime_mlas PUBLIC BUILD_MLAS_NO_ONNXRUNTIME)

if(MLAS_AMX_SUPPORTED)
  target_compile_definitions(onnxruntime_mlas PRIVATE MLAS_AMX_SUPPORTED)
else()
  message(WARNING "AMX instructions NOT supported due to lack of compiler tool chain!")
endif()

set(ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas)

if(MSVC)
  if (CMAKE_GENERATOR_PLATFORM)
    # Multi-platform generator
    set(onnxruntime_target_platform ${CMAKE_GENERATOR_PLATFORM})
  else()
    set(onnxruntime_target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()
  if (onnxruntime_target_platform STREQUAL "ARM64")
    set(onnxruntime_target_platform "ARM64")
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.26.0")
      enable_language(ASM_MARMASM)
    else()
      enable_language(ASM_MASM)
    endif()
  elseif (onnxruntime_target_platform STREQUAL "ARM64EC")
    set(onnxruntime_target_platform "ARM64EC")
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.26.0")
      enable_language(ASM_MARMASM)
    else()
      enable_language(ASM_MASM)
    endif()
  elseif (onnxruntime_target_platform STREQUAL "ARM" OR CMAKE_GENERATOR MATCHES "ARM")
    set(onnxruntime_target_platform "ARM")
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.26.0")
      enable_language(ASM_MARMASM)
    else()
      enable_language(ASM_MASM)
    endif()
  elseif (onnxruntime_target_platform STREQUAL "x64" OR onnxruntime_target_platform STREQUAL "x86_64" OR onnxruntime_target_platform STREQUAL "AMD64" OR CMAKE_GENERATOR MATCHES "Win64")
    set(onnxruntime_target_platform "x64")
    enable_language(ASM_MASM)
  elseif (onnxruntime_target_platform STREQUAL "Win32" OR onnxruntime_target_platform STREQUAL "x86" OR onnxruntime_target_platform STREQUAL "i386" OR onnxruntime_target_platform STREQUAL "i686")
    set(onnxruntime_target_platform "x86")
    enable_language(ASM_MASM)
    if (NOT onnxruntime_BUILD_WEBASSEMBLY)
      message("Enabling SAFESEH for x86 build")
      set(CMAKE_ASM_MASM_FLAGS "${CMAKE_ASM_MASM_FLAGS} /safeseh")
    endif()
  else()
    message(FATAL_ERROR "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()

#TODO: set MASM flags properly
function(setup_mlas_source_for_windows)

  #
  # Sources common for all platforms.
  #
  target_sources(onnxruntime_mlas PRIVATE
    lib/activate_fp16.cpp
    lib/dwconv.cpp
    lib/pooling_fp16.cpp
  )

  #The onnxruntime_target_platform variable was added by Windows AI team in onnxruntime_common.cmake
  #Don't use it for other platforms.
  if((onnxruntime_target_platform STREQUAL "ARM64") OR (onnxruntime_target_platform STREQUAL "ARM64EC"))
    set(PREPROCESS_ARMASM_FLAGS "")
    set(ARMASM_FLAGS "")

    if(onnxruntime_target_platform STREQUAL "ARM64")
      target_sources(onnxruntime_mlas PRIVATE
        lib/halfgemm_kernel_neon.cpp
        lib/qgemm_kernel_neon.cpp
        lib/qgemm_kernel_udot.cpp
        lib/qgemm_kernel_sdot.cpp
      )

      set(mlas_platform_preprocess_srcs
        lib/arm64/ConvSymS8KernelDot.asm
        lib/arm64/ConvSymS8KernelDotLd64.asm
        lib/arm64/ConvSymU8KernelDot.asm
        lib/arm64/ConvSymS8KernelNeon.asm
        lib/arm64/ConvSymU8KernelNeon.asm
        lib/arm64/DepthwiseQConvSymS8KernelNeon.asm
        lib/arm64/DepthwiseQConvSymU8KernelNeon.asm
        lib/arm64/DepthwiseQConvKernelSize9Neon.asm
        lib/arm64/HalfGemmKernelNeon.asm
        lib/arm64/QgemmU8X8KernelNeon.asm
        lib/arm64/QgemmS8S8KernelNeon.asm
        lib/arm64/QgemmU8X8KernelUdot.asm
        lib/arm64/QgemmS8S8KernelSdot.asm
        lib/arm64/SgemmKernelNeon.asm
        lib/arm64/SgemvKernelNeon.asm
        lib/arm64/SymQgemmS8KernelNeon.asm
        lib/arm64/SymQgemmS8KernelSDot.asm
        lib/arm64/SymQgemmS8KernelSDotLd64.asm
      )
    else()
      target_sources(onnxruntime_mlas PRIVATE
        lib/qgemm_kernel_neon.cpp
      )

      set(mlas_platform_preprocess_srcs
        lib/arm64ec/QgemmU8X8KernelNeon.asm
        lib/arm64ec/SgemmKernelNeon.asm
      )

      string(APPEND PREPROCESS_ARMASM_FLAGS " /arm64EC")
      string(APPEND ARMASM_FLAGS " -machine ARM64EC")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      string(APPEND ARMASM_FLAGS " -g")
    endif()

    # Remove double quotes from flag strings.
    separate_arguments(PREPROCESS_ARMASM_FLAGS NATIVE_COMMAND "${PREPROCESS_ARMASM_FLAGS}")
    separate_arguments(ARMASM_FLAGS NATIVE_COMMAND "${ARMASM_FLAGS}")

    # Run the C precompiler on each input before the assembler.
    foreach(asm_filename ${mlas_platform_preprocess_srcs})
      get_filename_component(asm_filename_base ${asm_filename} NAME_WLE)
      set(preprocess_filename ${CMAKE_CURRENT_BINARY_DIR}/${asm_filename_base}.i)
      set(obj_filename ${CMAKE_CURRENT_BINARY_DIR}/${asm_filename_base}.obj)
      add_custom_command(
        OUTPUT ${obj_filename}
          COMMAND
              cl.exe ${PREPROCESS_ARMASM_FLAGS} /P ${asm_filename} /Fi${preprocess_filename}
          COMMAND
              armasm64.exe ${ARMASM_FLAGS} ${preprocess_filename} ${obj_filename}
        DEPENDS ${asm_filename}
        BYPRODUCTS ${preprocess_filename}
      )
      target_sources(onnxruntime_mlas PRIVATE ${obj_filename})
    endforeach()
  elseif(onnxruntime_target_platform STREQUAL "ARM")
    target_sources(onnxruntime_mlas PRIVATE
      lib/arm/sgemmc.cpp
    )
  elseif(onnxruntime_target_platform STREQUAL "x64")

    file(GLOB_RECURSE mlas_platform_srcs_avx CONFIGURE_DEPENDS
      "lib/intrinsics/avx/*.cpp"
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "/arch:AVX")

    file(GLOB_RECURSE mlas_platform_srcs_avx2 CONFIGURE_DEPENDS
      "lib/intrinsics/avx2/*.cpp"
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "/arch:AVX2")

    target_sources(onnxruntime_mlas PRIVATE
      lib/dgemm.cpp
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      lib/qgemm_kernel_amx.cpp
      lib/qgemm_kernel_avx2.cpp
      lib/qgemm_kernel_sse.cpp
      lib/qgemm_kernel_sse41.cpp
      lib/intrinsics/avx512/quantize_avx512f.cpp
      lib/amd64/QgemmU8S8KernelAmx.asm
      lib/amd64/QgemmU8S8KernelAvx2.asm
      lib/amd64/QgemmU8U8KernelAvx2.asm
      lib/amd64/QgemmU8X8KernelAvx2.asm
      lib/amd64/QgemmU8X8KernelAvx512Core.asm
      lib/amd64/QgemvU8S8KernelAvx2.asm
      lib/amd64/QgemvU8S8KernelAvx512Core.asm
      lib/amd64/QgemvU8S8KernelAvx512Vnni.asm
      lib/amd64/QgemvU8S8KernelAvxVnni.asm
      lib/amd64/ConvSymKernelAvx2.asm
      lib/amd64/ConvSymKernelAvx512Core.asm
      lib/amd64/DgemmKernelSse2.asm
      lib/amd64/DgemmKernelAvx.asm
      lib/amd64/DgemmKernelFma3.asm
      lib/amd64/DgemmKernelAvx512F.asm
      lib/amd64/SgemmKernelSse2.asm
      lib/amd64/SgemmKernelAvx.asm
      lib/amd64/SgemmKernelM1Avx.asm
      lib/amd64/SgemmKernelFma3.asm
      lib/amd64/SgemmKernelAvx512F.asm
      lib/amd64/SconvKernelSse2.asm
      lib/amd64/SconvKernelAvx.asm
      lib/amd64/SconvKernelFma3.asm
      lib/amd64/SconvKernelAvx512F.asm
      lib/amd64/SpoolKernelSse2.asm
      lib/amd64/SpoolKernelAvx.asm
      lib/amd64/SpoolKernelAvx512F.asm
      lib/amd64/sgemma.asm
      lib/amd64/cvtfp16a.asm
      lib/amd64/SoftmaxKernelAvx.asm
      lib/amd64/TransKernelFma3.asm
      lib/amd64/TransKernelAvx512F.asm
      lib/amd64/LogisticKernelFma3.asm
      lib/amd64/TanhKernelFma3.asm
      lib/amd64/ErfKernelFma3.asm
    )
  else()
    target_sources(onnxruntime_mlas PRIVATE
      lib/qgemm_kernel_sse.cpp
      lib/qgemm_kernel_sse41.cpp
      lib/i386/SgemmKernelSse2.asm
      lib/i386/SgemmKernelAvx.asm
    )
  endif()
endfunction()

if (onnxruntime_BUILD_WEBASSEMBLY)
  if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    file(GLOB_RECURSE mlas_platform_srcs
      "lib/wasm_simd/*.cpp"
    )
    set(mlas_platform_srcs
      ${mlas_platform_srcs}
      lib/qgemm_kernel_wasmsimd.cpp
    )
  else()
    file(GLOB_RECURSE mlas_platform_srcs
      "lib/scalar/*.cpp"
    )
  endif()
  target_sources(onnxruntime_mlas PRIVATE ${mlas_platform_srcs})
elseif(MSVC)
  setup_mlas_source_for_windows()
else()

    if(APPLE)
        get_target_property(ONNXRUNTIME_MLAS_OSX_ARCH onnxruntime_mlas OSX_ARCHITECTURES)

        if(NOT ONNXRUNTIME_MLAS_OSX_ARCH)
         set(ONNXRUNTIME_MLAS_OSX_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
        endif()
        foreach(OSX_ARCH ${ONNXRUNTIME_MLAS_OSX_ARCH})
        if (OSX_ARCH STREQUAL "arm64")
            set(ARM64 TRUE)
        elseif (OSX_ARCH STREQUAL "arm64e")
            set(ARM64 TRUE)
        elseif (OSX_ARCH STREQUAL "arm")
            set(ARM TRUE)
        elseif (OSX_ARCH STREQUAL "x86_64")
            set(X86_64 TRUE)
        elseif (OSX_ARCH STREQUAL "i386")
            set(X86 TRUE)
        endif()
        endforeach()
    elseif(ANDROID)
        if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
          set(ARM TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
          set(ARM64 TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
          set(X86_64 TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
          set(X86 TRUE)
        endif()
    else()
        #Linux/FreeBSD/PowerPC/...
        #The value of CMAKE_SYSTEM_PROCESSOR should be from `uname -m`
        #Example values:
        #arm64v8/ubuntu -> aarch64
        #arm32v6/alpine -> armv7l
        #arm32v7/centos -> armv7l
        #ppc64le/debian -> ppc64le
        #s390x/ubuntu -> s390x
        #ppc64le/busybox -> ppc64le
        #arm64v8/ubuntu -> aarch64
        #Android: armv7-a aarch64 i686 x86_64
        #chasun: I don't think anyone uses 'arm64'
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
          set(ARM64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
          set(ARM TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
          set(ARM64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc.*|ppc.*)")
          set(POWER TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
          set(X86 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
          set(X86_64 TRUE)
        endif()
    endif()

    if(APPLE)
      get_target_property(ONNXRUNTIME_MLAS_MACOSX_ARCH onnxruntime_mlas OSX_ARCHITECTURES)
    endif()
    list(LENGTH ONNXRUNTIME_MLAS_MACOSX_ARCH  ONNXRUNTIME_MLAS_MACOSX_ARCH_LENGH)
    if(ONNXRUNTIME_MLAS_MACOSX_ARCH_LENGH GREATER 1)
        set(ONNXRUNTIME_MLAS_MULTI_ARCH TRUE)
    endif()
    #If ONNXRUNTIME_MLAS_MULTI_ARCH is true, we need to go through every if branch below
    #and split MLAS to multiple static libraries.
    #Otherwise, it works like if(...) elseif(...) elseif(...) endif()
    set(MLAS_SOURCE_IS_NOT_SET 1)
    if(ARM)
        enable_language(ASM)

        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -mfpu=neon")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")

        set(mlas_platform_srcs
          lib/aarch32/QgemmU8X8KernelNeon.S
          lib/arm/sgemmc.cpp
          lib/qgemm_kernel_neon.cpp
        )
        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(ARM64 AND MLAS_SOURCE_IS_NOT_SET )
        enable_language(ASM)
        set(mlas_platform_srcs
          lib/aarch64/ConvSymS8KernelDot.S
          lib/aarch64/ConvSymS8KernelDotLd64.S
          lib/aarch64/ConvSymU8KernelDot.S
          lib/aarch64/ConvSymS8KernelNeon.S
          lib/aarch64/ConvSymU8KernelNeon.S
          lib/aarch64/DepthwiseQConvSymS8KernelNeon.S
          lib/aarch64/DepthwiseQConvSymU8KernelNeon.S
          lib/aarch64/DepthwiseQConvKernelSize9Neon.S
          lib/aarch64/QgemmU8X8KernelNeon.S
          lib/aarch64/QgemmS8S8KernelNeon.S
          lib/aarch64/QgemmU8X8KernelUdot.S
          lib/aarch64/QgemmS8S8KernelSdot.S
          lib/aarch64/SgemmKernelNeon.S
          lib/aarch64/SgemvKernelNeon.S
          lib/aarch64/SymQgemmS8KernelNeon.S
          lib/aarch64/SymQgemmS8KernelSdot.S
          lib/aarch64/SymQgemmS8KernelSdotLd64.S
          lib/qgemm_kernel_neon.cpp
          lib/qgemm_kernel_udot.cpp
          lib/qgemm_kernel_sdot.cpp
        )
        if (NOT APPLE)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            lib/aarch64/HalfGemmKernelNeon.S
            lib/activate_fp16.cpp
            lib/dwconv.cpp
            lib/halfgemm_kernel_neon.cpp
            lib/pooling_fp16.cpp
          )
          set_source_files_properties(lib/aarch64/HalfGemmKernelNeon.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(lib/activate_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(lib/dwconv.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(lib/pooling_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
        endif()

        if(ONNXRUNTIME_MLAS_MULTI_ARCH)
            onnxruntime_add_static_library(onnxruntime_mlas_arm64 ${mlas_platform_srcs})
            set_target_properties(onnxruntime_mlas_arm64 PROPERTIES OSX_ARCHITECTURES "arm64")
            list(APPEND ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas_arm64)
            set(mlas_platform_srcs )
        else()
            set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(POWER AND MLAS_SOURCE_IS_NOT_SET)
        set(mlas_platform_srcs
          lib/power/SgemmKernelPower.cpp
          lib/dgemm.cpp
          lib/power/DgemmKernelPower.cpp
          lib/power/QuantizePower.cpp
        )
        set_source_files_properties(lib/power/SgemmKernelPower.cpp PROPERTIES COMPILE_FLAGS "-DSINGLE")

        check_cxx_compiler_flag("-mcpu=power9" HAS_POWER9)
        if (HAS_POWER9)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            lib/power/QuantizePowerVSX.cpp
          )
          set_source_files_properties(lib/power/QuantizePowerVSX.cpp PROPERTIES COMPILE_FLAGS "-mcpu=power9")
        endif()

        check_cxx_compiler_flag("-mcpu=power10" HAS_POWER10)
        if(HAS_POWER10)
          set(CMAKE_REQUIRED_FLAGS "-mcpu=power10")
          check_cxx_source_compiles("
            #include <altivec.h>
            int main() {
              __vector_quad acc0;
              __builtin_mma_xxsetaccz (&acc0);
              return 0;
            }"
            COMPILES_P10
          )
          if(COMPILES_P10)
            check_cxx_source_compiles("
              #include <sys/auxv.h>
              int main() {
                unsigned long hwcap2 = getauxval(AT_HWCAP2);
                bool HasP10 = ((hwcap2 & PPC_FEATURE2_MMA) && (hwcap2 & PPC_FEATURE2_ARCH_3_1));
                return 0;
              }"
              HAS_P10_RUNTIME
            )
            if (HAS_P10_RUNTIME)
              set_source_files_properties(lib/platform.cpp PROPERTIES COMPILE_FLAGS "-DPOWER10")
              set_source_files_properties(lib/qgemm.cpp PROPERTIES COMPILE_FLAGS "-DPOWER10")
            endif()
            set(mlas_platform_srcs_power10
              lib/power/SgemmKernelPOWER10.cpp
              lib/power/DgemmKernelPOWER10.cpp
              lib/power/qgemm_kernel_power10.cpp
            )
            set_source_files_properties(lib/power/SgemmKernelPOWER10.cpp PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10 -DSINGLE")
            set_source_files_properties(lib/power/DgemmKernelPOWER10.cpp PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10")
            set_source_files_properties(lib/power/qgemm_kernel_power10.cpp PROPERTIES COMPILE_FLAGS "-O3 -mcpu=power10")
            set(mlas_platform_srcs
              ${mlas_platform_srcs}
              ${mlas_platform_srcs_power10}
            )
          endif()
        endif()
        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(X86 AND MLAS_SOURCE_IS_NOT_SET)
        enable_language(ASM)

        set(mlas_platform_srcs_sse2
          lib/qgemm_kernel_sse.cpp
          lib/x86/SgemmKernelSse2.S
        )
        set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

        set(mlas_platform_srcs_avx
          lib/x86/SgemmKernelAvx.S
        )
        set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

        set(mlas_platform_srcs
          ${mlas_platform_srcs_sse2}
          ${mlas_platform_srcs_avx}
        )

        # In r23, NDK remove __x86.get_pc_thunk.* from libatomic. Add our own
        # implementation to avoid external dependency.
        if(ANDROID)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            lib/x86/x86.get_pc_thunk.S
          )
        endif()

        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(X86_64 AND MLAS_SOURCE_IS_NOT_SET)
        enable_language(ASM)

        # Forward the flags for the minimum target platform version from the C
        # compiler to the assembler. This works around CMakeASMCompiler.cmake.in
        # not including the logic to set this flag for the assembler.
        set(CMAKE_ASM${ASM_DIALECT}_OSX_DEPLOYMENT_TARGET_FLAG "${CMAKE_C_OSX_DEPLOYMENT_TARGET_FLAG}")

        # The LLVM assembler does not support the .arch directive to enable instruction
        # set extensions and also doesn't support AVX-512F instructions without
        # turning on support via command-line option. Group the sources by the
        # instruction set extension and explicitly set the compiler flag as appropriate.

        set(mlas_platform_srcs_sse2
          lib/qgemm_kernel_sse.cpp
          lib/x86_64/DgemmKernelSse2.S
          lib/x86_64/SgemmKernelSse2.S
          lib/x86_64/SgemmTransposePackB16x4Sse2.S
          lib/x86_64/SconvKernelSse2.S
          lib/x86_64/SpoolKernelSse2.S
        )
        set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

        set(mlas_platform_srcs_avx
          lib/x86_64/DgemmKernelAvx.S
          lib/x86_64/SgemmKernelAvx.S
          lib/x86_64/SgemmKernelM1Avx.S
          lib/x86_64/SgemmKernelM1TransposeBAvx.S
          lib/x86_64/SgemmTransposePackB16x4Avx.S
          lib/x86_64/SconvKernelAvx.S
          lib/x86_64/SpoolKernelAvx.S
          lib/x86_64/SoftmaxKernelAvx.S
          lib/intrinsics/avx/min_max_elements.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

        set(mlas_platform_srcs_avx2
          lib/x86_64/QgemmU8S8KernelAvx2.S
          lib/x86_64/QgemvU8S8KernelAvx2.S
          lib/x86_64/QgemmU8U8KernelAvx2.S
          lib/x86_64/QgemvU8S8KernelAvxVnni.S
          lib/x86_64/QgemmU8X8KernelAvx2.S
          lib/x86_64/ConvSymKernelAvx2.S
          lib/x86_64/DgemmKernelFma3.S
          lib/x86_64/SgemmKernelFma3.S
          lib/x86_64/SconvKernelFma3.S
          lib/x86_64/TransKernelFma3.S
          lib/x86_64/LogisticKernelFma3.S
          lib/x86_64/TanhKernelFma3.S
          lib/x86_64/ErfKernelFma3.S
          lib/intrinsics/avx2/qladd_avx2.cpp
          lib/intrinsics/avx2/qdwconv_avx2.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

        set(mlas_platform_srcs_avx512f
          lib/x86_64/DgemmKernelAvx512F.S
          lib/x86_64/SgemmKernelAvx512F.S
          lib/x86_64/SconvKernelAvx512F.S
          lib/x86_64/SpoolKernelAvx512F.S
          lib/x86_64/TransKernelAvx512F.S
          lib/intrinsics/avx512/quantize_avx512f.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

        set(mlas_platform_srcs_avx512core
          lib/x86_64/QgemvU8S8KernelAvx512Core.S
          lib/x86_64/QgemvU8S8KernelAvx512Vnni.S
          lib/x86_64/QgemmU8X8KernelAvx512Core.S
          lib/x86_64/ConvSymKernelAvx512Core.S
        )
        set_source_files_properties(${mlas_platform_srcs_avx512core} PROPERTIES COMPILE_FLAGS "-mavx512bw -mavx512dq -mavx512vl")

        set(mlas_platform_srcs
          lib/activate_fp16.cpp
          lib/dwconv.cpp
          lib/dgemm.cpp
          lib/pooling_fp16.cpp
          lib/qgemm_kernel_avx2.cpp
          ${mlas_platform_srcs_sse2}
          ${mlas_platform_srcs_avx}
          ${mlas_platform_srcs_avx2}
          ${mlas_platform_srcs_avx512f}
          ${mlas_platform_srcs_avx512core}
        )

        if(MLAS_AMX_SUPPORTED)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            lib/qgemm_kernel_amx.cpp
            lib/x86_64/QgemmU8S8KernelAmx.S
          )
          set_source_files_properties(lib/qgemm_kernel_amx.cpp PROPERTIES COMPILE_FLAGS "-mamx-tile -mamx-int8 -mavx2 -mavx512bw -mavx512dq -mavx512vl")
          set_source_files_properties(lib/x86_64/QgemmU8S8KernelAmx.S PROPERTIES COMPILE_FLAGS "-mamx-tile -mamx-int8 -mavx2 -mavx512bw -mavx512dq -mavx512vl")
        endif()

        if(ONNXRUNTIME_MLAS_MULTI_ARCH)
          onnxruntime_add_static_library(onnxruntime_mlas_x86_64 ${mlas_platform_srcs})
          set_target_properties(onnxruntime_mlas_x86_64 PROPERTIES OSX_ARCHITECTURES "x86_64")
          list(APPEND ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas_x86_64)
          set(mlas_platform_srcs )
        else()
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH AND MLAS_SOURCE_IS_NOT_SET)
        file(GLOB_RECURSE mlas_platform_srcs
          "lib/scalar/*.cpp")
    endif()
    target_sources(onnxruntime_mlas PRIVATE ${mlas_platform_srcs})
endif()

foreach(mlas_target ${ONNXRUNTIME_MLAS_LIBS})
    target_include_directories(${mlas_target} PRIVATE inc lib)
endforeach()
set_target_properties(onnxruntime_mlas PROPERTIES FOLDER "ONNXRuntime")
if (WIN32)
  target_compile_options(onnxruntime_mlas PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/wd6385>" "$<$<COMPILE_LANGUAGE:CXX>:/wd4127>")
  if (onnxruntime_ENABLE_STATIC_ANALYSIS)
    target_compile_options(onnxruntime_mlas PRIVATE  "$<$<COMPILE_LANGUAGE:CXX>:/analyze:stacksize 131072">)
  endif()
endif()

#if (NOT onnxruntime_BUILD_SHARED_LIB)
#    install(TARGETS onnxruntime_mlas
#            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
#            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
#            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
#            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
#endif()
