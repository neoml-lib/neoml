if(NeoML_BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED common.cpp)
else()
    add_library(${PROJECT_NAME} STATIC common.cpp)
endif()

add_library(NeoML::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_sources( ${PROJECT_NAME} PRIVATE
    ArchiveFile.cpp
    NeoML.cpp
    Random.cpp
    Dnn/BaseLayer.cpp
    Dnn/Dnn.cpp
    Dnn/DnnBlob.cpp
    Dnn/DnnInitializer.cpp
    Dnn/DnnSolver.cpp
    Dnn/DnnSparseMatrix.cpp
    Dnn/Layers/3dConvLayer.cpp
    Dnn/Layers/3dPoolingLayer.cpp
    Dnn/Layers/3dTransposedConvLayer.cpp
    Dnn/Layers/AccumulativeLookupLayer.cpp
    Dnn/Layers/AccuracyLayer.cpp
    Dnn/Layers/ActivationLayers.cpp
    Dnn/Layers/ArgmaxLayer.cpp
    Dnn/Layers/AttentionLayer.cpp
    Dnn/Layers/BackLinkLayer.cpp
    Dnn/Layers/BaseConvLayer.cpp
    Dnn/Layers/BaseInPlaceLayer.cpp
    Dnn/Layers/BatchNormalizationLayer.cpp
    Dnn/Layers/AddToObjectLayer.cpp
    Dnn/Layers/BinaryCrossEntropyLayer.cpp
    Dnn/Layers/BinaryFocalLossLayer.cpp
    Dnn/Layers/CenterLossLayer.cpp
    Dnn/Layers/ChannelwiseConvLayer.cpp
    Dnn/Layers/CompositeLayer.cpp
    Dnn/Layers/ConcatLayer.cpp
    Dnn/Layers/ConcatObjectLayer.cpp
    Dnn/Layers/ConvLayer.cpp
    Dnn/Layers/CrfLayer.cpp
    Dnn/Layers/CrossEntropyLossLayer.cpp
    Dnn/Layers/CtcLayer.cpp
    Dnn/Layers/DotProductLayer.cpp
    Dnn/Layers/DropoutLayer.cpp
    Dnn/Layers/EltwiseLayer.cpp
    Dnn/Layers/EnumBinarizationLayer.cpp
    Dnn/Layers/EuclideanLossLayer.cpp
    Dnn/Layers/FocalLossLayer.cpp
    Dnn/Layers/FullyConnectedLayer.cpp
    Dnn/Layers/FullyConnectedSourceLayer.cpp
    Dnn/Layers/GELULayer.cpp
    Dnn/Layers/GlobalMaxPoolingLayer.cpp
    Dnn/Layers/GlobalMeanPoolingLayer.cpp
    Dnn/Layers/GruLayer.cpp
    Dnn/Layers/HingeLossLayer.cpp
    Dnn/Layers/ImageAndPixelConversionLayer.cpp
    Dnn/Layers/ImageResizeLayer.cpp
    Dnn/Layers/LossLayer.cpp
    Dnn/Layers/LstmLayer.cpp
    Dnn/Layers/MatrixMultiplicationLayer.cpp
    Dnn/Layers/MaxOverTimePoolingLayer.cpp
    Dnn/Layers/ModelWrapperLayer.cpp
    Dnn/Layers/MultichannleLookupLayer.cpp
    Dnn/Layers/MultiheadAttentionLayer.cpp
    Dnn/Layers/MultiHingeLossLayer.cpp
    Dnn/Layers/ObjectNormalizationLayer.cpp
    Dnn/Layers/PoolingLayer.cpp
    Dnn/Layers/PositionalEmbeddingLayer.cpp
    Dnn/Layers/PrecisionRecallLayer.cpp
    Dnn/Layers/QualityControlLayer.cpp
    Dnn/Layers/RecurrentLayer.cpp
    Dnn/Layers/ReorgLayer.cpp
    Dnn/Layers/RepeatSequenceLayer.cpp
    Dnn/Layers/RleConvLayer.cpp
    Dnn/Layers/SequenceSumLayer.cpp
    Dnn/Layers/SinkLayer.cpp
    Dnn/Layers/SoftmaxLayer.cpp
    Dnn/Layers/SourceLayer.cpp
    Dnn/Layers/SplitLayer.cpp
    Dnn/Layers/SubSequenceLayer.cpp
    Dnn/Layers/TimeConvLayer.cpp
    Dnn/Layers/TransformLayer.cpp
    Dnn/Layers/TransposedConvLayer.cpp
    Dnn/Layers/TransposeLayer.cpp
    Dnn/Layers/Upsampling2DLayer.cpp
    TraditionalML/AIModel.cpp
    TraditionalML/ClassificationProbability.cpp
    TraditionalML/ClusterCenter.cpp
    TraditionalML/CommonCluster.cpp
    TraditionalML/CrossValidation.cpp
    TraditionalML/CrossValidationSubProblem.h
    TraditionalML/DecisionTreeClassificationModel.cpp
    TraditionalML/DecisionTreeClassificationModel.h
    TraditionalML/DecisionTreeNodeBase.cpp
    TraditionalML/DecisionTreeNodeBase.h
    TraditionalML/DecisionTreeNodeClassificationStatistic.cpp
    TraditionalML/DecisionTreeNodeClassificationStatistic.h
    TraditionalML/DecisionTreeNodeStatisticBase.h
    TraditionalML/DecisionTreeTrainingModel.cpp
    TraditionalML/DifferentialEvolution.cpp
    TraditionalML/EMClustering.cpp
    TraditionalML/FeatureSelection.cpp
    TraditionalML/FirstComeClustering.cpp
    TraditionalML/FloatVector.cpp
    TraditionalML/Function.cpp
    TraditionalML/FunctionEvaluation.cpp
    TraditionalML/GradientBoost.cpp
    TraditionalML/GradientBoostFastHistProblem.cpp
    TraditionalML/GradientBoostFastHistProblem.h
    TraditionalML/GradientBoostFastHistTreeBuilder.cpp
    TraditionalML/GradientBoostFastHistTreeBuilder.h
    TraditionalML/GradientBoostFullProblem.cpp
    TraditionalML/GradientBoostFullProblem.h
    TraditionalML/GradientBoostFullTreeBuilder.cpp
    TraditionalML/GradientBoostFullTreeBuilder.h
    TraditionalML/GradientBoostModel.cpp
    TraditionalML/GradientBoostModel.h
    TraditionalML/GradientBoostQSEnsemble.cpp
    TraditionalML/GradientBoostQSEnsemble.h
    TraditionalML/GradientBoostQuickScorer.cpp
    TraditionalML/GradientBoostVectorSetStatistics.h
    TraditionalML/HierarchicalClustering.cpp
    TraditionalML/IsoDataClustering.cpp
    TraditionalML/KMeansClustering.cpp
    TraditionalML/LinearBinaryClassifierBuilder.cpp
    TraditionalML/LinearBinaryModel.cpp
    TraditionalML/LinearBinaryModel.h
    TraditionalML/MemoryProblem.cpp
    TraditionalML/OneVersusAll.cpp
    TraditionalML/OneVersusAllModel.cpp
    TraditionalML/OneVersusAllModel.h
    TraditionalML/PlattScalling.cpp
    TraditionalML/ProblemWrappers.cpp
    TraditionalML/ProblemWrappers.h
    TraditionalML/ProblemWrappers.inl
    TraditionalML/RegressionTreeModel.cpp
    TraditionalML/RegressionTreeModel.h
    TraditionalML/Score.cpp
    TraditionalML/Shuffler.cpp
    TraditionalML/SMOptimizer.cpp
    TraditionalML/SparseFloatMatrix.cpp
    TraditionalML/SparseFloatVector.cpp
    TraditionalML/StratifiedCrossValidationSubProblem.cpp
    TraditionalML/StratifiedCrossValidationSubProblem.h
    TraditionalML/SubProblem.h
    TraditionalML/SvmBinaryClassifierBuilder.cpp
    TraditionalML/SvmBinaryModel.cpp
    TraditionalML/SvmBinaryModel.h
    TraditionalML/TrustRegionNewtonOptimizer.cpp

    # Headers
    ../include/NeoML/ArchiveFile.h
    ../include/NeoML/NeoML.h
    ../include/NeoML/NeoMLCommon.h
    ../include/NeoML/NeoMLDefs.h
    ../include/NeoML/Platforms.h
    ../include/NeoML/Random.h
    ../include/NeoML/Dnn/Dnn.h
    ../include/NeoML/Dnn/Dnn.inl
    ../include/NeoML/Dnn/DnnBlob.h
    ../include/NeoML/Dnn/DnnInitializer.h
    ../include/NeoML/Dnn/DnnSolver.h
    ../include/NeoML/Dnn/DnnSparseMatrix.h
    ../include/NeoML/Dnn/Layers/3dConvLayer.h
    ../include/NeoML/Dnn/Layers/3dPoolingLayer.h
    ../include/NeoML/Dnn/Layers/3dTransposedConvLayer.h
    ../include/NeoML/Dnn/Layers/AccumulativeLookupLayer.h
    ../include/NeoML/Dnn/Layers/AccuracyLayer.h
    ../include/NeoML/Dnn/Layers/ActivationLayers.h
    ../include/NeoML/Dnn/Layers/ArgmaxLayer.h
    ../include/NeoML/Dnn/Layers/AttentionLayer.h
    ../include/NeoML/Dnn/Layers/BackLinkLayer.h
    ../include/NeoML/Dnn/Layers/BaseInPlaceLayer.h
    ../include/NeoML/Dnn/Layers/BatchNormalizationLayer.h
    ../include/NeoML/Dnn/Layers/AddToObjectLayer.h
    ../include/NeoML/Dnn/Layers/BinaryFocalLossLayer.h
    ../include/NeoML/Dnn/Layers/CenterLossLayer.h
    ../include/NeoML/Dnn/Layers/ChannelwiseConvLayer.h
    ../include/NeoML/Dnn/Layers/CompositeLayer.h
    ../include/NeoML/Dnn/Layers/ConcatLayer.h
    ../include/NeoML/Dnn/Layers/ConvLayer.h
    ../include/NeoML/Dnn/Layers/CrfLayer.h
    ../include/NeoML/Dnn/Layers/CtcLayer.h
    ../include/NeoML/Dnn/Layers/DotProductLayer.h
    ../include/NeoML/Dnn/Layers/DropoutLayer.h
    ../include/NeoML/Dnn/Layers/EltwiseLayer.h
    ../include/NeoML/Dnn/Layers/EnumBinarizationLayer.h
    ../include/NeoML/Dnn/Layers/FocalLossLayer.h
    ../include/NeoML/Dnn/Layers/FullyConnectedLayer.h
    ../include/NeoML/Dnn/Layers/FullyConnectedSourceLayer.h
    ../include/NeoML/Dnn/Layers/GELULayer.h
    ../include/NeoML/Dnn/Layers/GlobalMaxPoolingLayer.h
    ../include/NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h
    ../include/NeoML/Dnn/Layers/GruLayer.h
    ../include/NeoML/Dnn/Layers/ImageAndPixelConversionLayer.h
    ../include/NeoML/Dnn/Layers/ImageResizeLayer.h
    ../include/NeoML/Dnn/Layers/LossLayer.h
    ../include/NeoML/Dnn/Layers/LstmLayer.h
    ../include/NeoML/Dnn/Layers/MatrixMultiplicationLayer.h
    ../include/NeoML/Dnn/Layers/MaxOverTimePoolingLayer.h
    ../include/NeoML/Dnn/Layers/ModelWrapperLayer.h
    ../include/NeoML/Dnn/Layers/MultichannelLookupLayer.h
    ../include/NeoML/Dnn/Layers/MultiheadAttentionLayer.h
    ../include/NeoML/Dnn/Layers/MultiHingeLossLayer.h
    ../include/NeoML/Dnn/Layers/ObjectNormalizationLayer.h
    ../include/NeoML/Dnn/Layers/PoolingLayer.h
    ../include/NeoML/Dnn/Layers/PositionalEmbeddingLayer.h
    ../include/NeoML/Dnn/Layers/PrecisionRecallLayer.h
    ../include/NeoML/Dnn/Layers/QualityControlLayer.h
    ../include/NeoML/Dnn/Layers/RecurrentLayer.h
    ../include/NeoML/Dnn/Layers/ReorgLayer.h
    ../include/NeoML/Dnn/Layers/RepeatSequenceLayer.h
    ../include/NeoML/Dnn/Layers/SequenceSumLayer.h
    ../include/NeoML/Dnn/Layers/SinkLayer.h
    ../include/NeoML/Dnn/Layers/SoftmaxLayer.h
    ../include/NeoML/Dnn/Layers/SourceLayer.h
    ../include/NeoML/Dnn/Layers/SplitLayer.h
    ../include/NeoML/Dnn/Layers/SubSequenceLayer.h
    ../include/NeoML/Dnn/Layers/TimeConvLayer.h
    ../include/NeoML/Dnn/Layers/TransformLayer.h
    ../include/NeoML/Dnn/Layers/TransposedConvLayer.h
    ../include/NeoML/Dnn/Layers/TransposeLayer.h
    ../include/NeoML/Dnn/Layers/Upsampling2DLayer.h
    ../include/NeoML/TraditionalML/ClassificationProbability.h
    ../include/NeoML/TraditionalML/ClassificationResult.h
    ../include/NeoML/TraditionalML/ClusterCenter.h
    ../include/NeoML/TraditionalML/Clustering.h
    ../include/NeoML/TraditionalML/CommonCluster.h
    ../include/NeoML/TraditionalML/CrossValidation.h
    ../include/NeoML/TraditionalML/DecisionTreeTrainingModel.h
    ../include/NeoML/TraditionalML/DifferentialEvolution.h
    ../include/NeoML/TraditionalML/EMClustering.h
    ../include/NeoML/TraditionalML/FeatureSelection.h
    ../include/NeoML/TraditionalML/FirstComeClustering.h
    ../include/NeoML/TraditionalML/FloatVector.h
    ../include/NeoML/TraditionalML/Function.h
    ../include/NeoML/TraditionalML/FunctionEvaluation.h
    ../include/NeoML/TraditionalML/GradientBoost.h
    ../include/NeoML/TraditionalML/GradientBoostQuickScorer.h
    ../include/NeoML/TraditionalML/GraphGenerator.h
    ../include/NeoML/TraditionalML/HierarchicalClustering.h
    ../include/NeoML/TraditionalML/IsoDataClustering.h
    ../include/NeoML/TraditionalML/KMeansClustering.h
    ../include/NeoML/TraditionalML/LdGraph.h
    ../include/NeoML/TraditionalML/LinearBinaryClassifierBuilder.h
    ../include/NeoML/TraditionalML/MatchingGenerator.h
    ../include/NeoML/TraditionalML/MemoryProblem.h
    ../include/NeoML/TraditionalML/Model.h
    ../include/NeoML/TraditionalML/OneVersusAll.h
    ../include/NeoML/TraditionalML/PlattScalling.h
    ../include/NeoML/TraditionalML/Problem.h
    ../include/NeoML/TraditionalML/Score.h
    ../include/NeoML/TraditionalML/Shuffler.h
    ../include/NeoML/TraditionalML/SimpleGenerator.h
    ../include/NeoML/TraditionalML/SMOptimizer.h
    ../include/NeoML/TraditionalML/SparseFloatMatrix.h
    ../include/NeoML/TraditionalML/SparseFloatVector.h
    ../include/NeoML/TraditionalML/SvmBinaryClassifierBuilder.h
    ../include/NeoML/TraditionalML/TrainingModel.h
    ../include/NeoML/TraditionalML/TrustRegionNewtonOptimizer.h
    ../include/NeoML/TraditionalML/VariableMatrix.h
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/TraditionalML>
)

if(USE_FINE_OBJECTS)
    target_link_libraries(${PROJECT_NAME} PRIVATE FineObjects)

    target_compile_definitions(${PROJECT_NAME} PUBLIC NEOML_USE_FINEOBJ)

    set(MESSAGES_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.Messages)
    set(MESSAGES_SRC ${MESSAGES_DIR}/NeoMLMessages.cpp)
    add_custom_target(
        CompileMessage
        COMMAND ${FINE_MESSAGE_COMPILER} -R ${MESSAGES_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../NeoMLMessages.msg
        WORKING_DIRECTORY ${FINE_MESSAGE_COMPILER_DIR}
        COMMENT "Generating messages."
    )
    add_dependencies(${PROJECT_NAME} CompileMessage)
    
    set_property(SOURCE ${MESSAGES_SRC} PROPERTY GENERATED ON)
    target_sources(${PROJECT_NAME} PRIVATE ${MESSAGES_SRC})
else()
    add_subdirectory(../include/NeoML/FineObjLite ${CMAKE_BINARY_DIR}/NeoML/include/FineObjLite)
    target_link_libraries(${PROJECT_NAME} PUBLIC FineObjLite)
    source_group("Header Files\\FineObjLite" REGULAR_EXPRESSION "^.*FOL\.h$")
    target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)
endif()

# Win resources
if(WIN32)
	if(USE_FINE_OBJECTS)
		target_include_directories(${PROJECT_NAME} PRIVATE
			$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${CMAKE_CURRENT_SOURCE_DIR}/../../../Build/Inc ${FINE_ROOT}/FineObjects ${MESSAGES_DIR}>>
		)
	else()
		target_include_directories(${PROJECT_NAME} PRIVATE 
			$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${CMAKE_CURRENT_SOURCE_DIR}/../../Build/Inc>>
		)
	endif()

    enable_language(RC)
    target_sources(${PROJECT_NAME} PRIVATE ../NeoML.rc)
endif()

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

add_subdirectory(../../NeoMathEngine ${CMAKE_BINARY_DIR}/NeoMathEngine)

target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_NEOML)
if(NOT NeoML_BUILD_SHARED)
    target_compile_definitions(${PROJECT_NAME} PUBLIC STATIC_NEOML)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4 AND NOT ANDROID)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _FILE_OFFSET_BITS=64)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC NeoMathEngine)
if(ANDROID)
    target_link_libraries(${PROJECT_NAME} PRIVATE android)
endif()

configure_target(${PROJECT_NAME})

if(USE_FINE_OBJECTS)
    fine_unexport_symbols(${PROJECT_NAME})
endif()
