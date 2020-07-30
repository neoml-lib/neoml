project(VulkanShaders)

set(SHADER_SOURCES
    VectorFillScalar.comp 
    VectorReLU.comp
    VectorReLU4.comp
    VectorReLUDiff.comp 
    Transpose.comp 
    BlobMaxPooling.comp 
    BlobMeanPooling.comp
    VectorELU.comp 
    VectorELUDiff.comp 
    VectorELUDiffOp.comp 
    BlobConvolution3x3s1d1.comp
    VectorLeakyReLU.comp 
    VectorLeakyReLUDiff.comp 
    VectorHSwish.comp
    VectorHSwishDiff.comp
    VectorEltwiseMax.comp 
    VectorEltwiseMin.comp
    VectorAbs.comp 
    VectorAbsDiff.comp 
    VectorHinge.comp 
    VectorHingeDiff.comp 
    SumMatrixRows.comp
    SumMatrixColumns.comp
    VectorSquaredHinge.comp 
    VectorSquaredHingeDiff.comp 
    VectorHuber.comp 
    VectorHardTanh.comp
    VectorHardTanhDiff.comp 
    VectorHardSigmoid.comp 
    VectorHardSigmoidDiff.comp 
    VectorHardSigmoidDiffOp.comp
    VectorExp.comp 
    VectorLog.comp 
    VectorBernulliKLDerivative.comp  
    VectorEltwiseDivide.comp
    SetVectorToMatrixRows.comp
    BlobConvolution3x3s1d1.comp 
    MultiplyMatrixByDiagMatrix.comp
    MultiplyDiagMatrixByMatrix.comp 
    BlobConvolution.comp  
    BlobConvolution8.comp
    BlobChannelwiseConvolution.comp 
    BlobChannelwiseConvolution3x3s1.comp 
    BlobChannelwiseConvolution3x3s2.comp
    VectorAddFloat4.comp
    VectorAddFloat1.comp
    VectorAddValue.comp
    VectorAddInt.comp 
    VectorSum.comp 
    VectorEqual.comp
    VectorSub.comp 
    VectorMultiplyAndAdd.comp 
    VectorMultiplyAndSub.comp 
    VectorMultiply.comp
    VectorEltwisePower.comp 
    VectorSqrt.comp 
    VectorInv.comp 
    VectorMinMax.comp 
    VectorL1DiffAdd.comp
    VectorSigmoid.comp 
    VectorSigmoidDiff.comp 
    VectorSigmoidDiffOp.comp 
    RowMultiplyMatrixByMatrix.comp
    VectorTanh.comp 
    VectorTanhDiff.comp 
    VectorTanhDiffOp.comp 
    VectorPower.comp 
    VectorPowerDiff.comp 
    VectorPowerDiffOp.comp 
    VectorDotProduct.comp 
    VectorFillBernoulli.comp
    BlobConvolutionBackward.comp
    PrepareBlobWithPaddingBuffers.comp
    VectorEltwiseLogSumExp.comp 
    SetVectorToMatrixElements.comp 
    LookupAndSum.comp 
    Upsampling2DForward.comp
    FindMaxValueInRows.comp 
    FindMaxValueInRowsNoIndices.comp 
    FindMaxValueInColumns.comp
    FindMaxValueInColumnsNoIndices.comp
    FindMinValueInColumns.comp
    BlobGlobalMaxPooling.comp
    Blob3dMaxPoolingNoIndices.comp 
    Blob3dMeanPooling.comp 
    BlobMaxOverTimePoolingNoIndices.comp
    MultiplyMatrixByMatrix.comp
    MultiplySparseMatrixByTransposedMatrix.comp
    MultiplyTransposedMatrixBySparseMatrix.comp
    BatchMultiplyMatrixByMatrixBorders.comp
    BatchMultiplyMatrixByTransposedMatrix.comp 
    BatchMultiplyMatrixByTransposedMatrixBorders.comp
    BatchMultiplyTransposedMatrixByMatrix.comp 
    BatchMultiplyTransposedMatrixByMatrixBorders.comp
    MultiplyDiagMatrixByMatrixAndAdd.comp
    BatchInitAddMultiplyMatrixByTransposedMatrix.comp
    BatchInitMultiplyMatrixByTransposedMatrixBorders.comp
    Blob3dConvolution.comp 
    Blob3dConvolutionBackward.comp
    AddMatrixElementsToVector.comp 
    AddMatrixElementsToVectorEx.comp
    AddVectorToMatrixColumnsFloat.comp
    AddVectorToMatrixColumnsInt.comp 
    BatchAddVectorToMatrixRows.comp 
    BlobResizeImage.comp 
    MatrixLogSumExpByRows.comp 
    MatrixSoftmaxByRows.comp
    MatrixSoftmaxByColumns.comp
    EnumBinarizationFloat.comp 
    EnumBinarizationInt.comp
    BitSetBinarization.comp
    BlobSpatialDropout.comp
    BuildIntegerHist.comp
    VectorFindMaxValueInSet.comp
    VectorFindMaxValueInSetNoIndices.comp
    MatrixSpreadRowsFloatAdd.comp
    MatrixSpreadRowsFloat.comp
    MatrixSpreadRowsInt.comp
    BlobGetSubSequence.comp
    BlobGetSubSequenceNoIndices.comp
    BlobConvertFromRLE.comp
    BlobSplitByDim.comp
    BlobMergeByDim.comp
    VectorMultichannelLookupAndCopyFloat.comp
    VectorMultichannelCopyFloat.comp
    VectorMultichannelLookupAndCopyInt.comp
    VectorMultichannelCopyInt.comp
    BlobTimeConvolutionPrepare.comp
    PrepareBlobForConvolution.comp
    BlobReorgFloat.comp
    BlobReorgInt.comp
)

set(IB_SHADER_SOURCES
    Matrix2InterleavedAdreno.comp
    MultiplyMatrixInterleavedAdreno.comp
    AddVectorToMatrixRowsAdreno.comp
    SetVectorToMatrixRowsAdreno.comp
    PrepareBlobForConvolutionAdreno.comp
    BlobConvolutionAdreno.comp 
    BlobConvolution3x3s1d1Adreno.comp
    BlobConvolution8Adreno.comp
    PrepareFilter3x3ForConvolutionAdreno.comp
    BlobChannelwiseConvolutionAdreno.comp
    BlobConvolutionBackwardAdreno.comp
    VectorToImage.comp
    PrepareFilterForConvolutionBackwardAdreno.comp
    MultiplyMatrixByDiagMatrixAdreno.comp
    MultiplyDiagMatrixByMatrixAdreno.comp
    BlobConvolution3x3s1d1Adreno.comp
    PrepareBlobWithPaddingAdreno.comp
    MultiplyMatrixInterleavedBoardersAdreno.comp
    AddVectorToMatrixColumnsFloatAdreno.comp
)

# Directories for common files and result files
set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")
set(RESULTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")

file(MAKE_DIRECTORY ${RESULTS_DIR})

# Common files for all shaders
set(HEADER_FILE "${COMMON_DIR}/CommonHeader.comph")
set(STRUCTS_FILE "${COMMON_DIR}/CommonStruct.h")
set(FOOTER_FILE "${COMMON_DIR}/CommonFooter.comph")

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
    string(REPLACE / \\ HEADER_FILE "${HEADER_FILE}")
    string(REPLACE / \\ STRUCTS_FILE "${STRUCTS_FILE}")
    string(REPLACE / \\ FOOTER_FILE "${FOOTER_FILE}")
endif()

set(RESULTS)

# Macro for compiling 1 shader.
macro(register_shader SHADER IS_IB)
    set(SHADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}")
    if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
        string(REPLACE / \\ SHADER_FILE "${SHADER_FILE}")
    endif()

    string(REPLACE ".comp" "" SHADER_NAME ${SHADER})

    set(GLSL_DEFINES -D_GLSL_ -D_SHADER_NAME_=${SHADER_NAME} -DPARAM_STRUCT_NAME=C${SHADER_NAME}Param)
    set(VAR_NAME "Shader_${SHADER_NAME}")
    string(REPLACE ".comp" ".h" COMPILED_SHADER ${SHADER})

    if(${IS_IB})
        set(GLSL_DEFINES ${GLSL_DEFINES})
        set(RESULT_FILE ${RESULTS_DIR}/${COMPILED_SHADER})
        set(VAR_NAME "Shader_${SHADER_NAME}")
        set(FULL_SOURCE_FILE "${SHADER_NAME}_source.tmp")
    else()
        set(RESULT_FILE ${RESULTS_DIR}/${COMPILED_SHADER})
        set(VAR_NAME "Shader_${SHADER_NAME}")
        set(FULL_SOURCE_FILE "${SHADER_NAME}_source.tmp")
    endif()

    if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
        add_custom_command(
            OUTPUT ${FULL_SOURCE_FILE}
            COMMAND type "${HEADER_FILE}" "${STRUCTS_FILE}" "${FOOTER_FILE}" "${SHADER_FILE}" > "${FULL_SOURCE_FILE}" 2>nul
            DEPENDS ${SHADER_FILE} ${HEADER_FILE} ${STRUCTS_FILE} ${FOOTER_FILE}
        )
    else()
        add_custom_command(
            OUTPUT ${FULL_SOURCE_FILE}
            COMMAND cat "${HEADER_FILE}" "${STRUCTS_FILE}" "${FOOTER_FILE}" "${SHADER_FILE}" > "${FULL_SOURCE_FILE}"
            DEPENDS ${SHADER_FILE} ${HEADER_FILE} ${STRUCTS_FILE} ${FOOTER_FILE}
        )
    endif()

    add_custom_command(
        OUTPUT ${RESULT_FILE} 
        COMMAND ${CMAKE_COMMAND} -E time "${GLSL_COMPILER}" -V100 ${GLSL_DEFINES} --vn ${VAR_NAME} -o "${RESULT_FILE}" -S comp "${FULL_SOURCE_FILE}"
        DEPENDS ${FULL_SOURCE_FILE}
    )

    list(APPEND RESULTS ${RESULT_FILE})
endmacro()

# Non image based shaders.
foreach(SHADER ${SHADER_SOURCES})
    register_shader(${SHADER} FALSE)
endforeach()

# Image based shaders.
foreach(SHADER ${IB_SHADER_SOURCES})
    register_shader(${SHADER} TRUE)
endforeach()

# Set the dependency of every resulting shader.
add_custom_target(${PROJECT_NAME} DEPENDS ${RESULTS})
