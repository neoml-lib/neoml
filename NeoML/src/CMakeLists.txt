if(NeoML_BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED common.cpp)
else()
    add_library(${PROJECT_NAME} STATIC common.cpp)
endif()

add_library(NeoML::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set(NeoML_SOURCES_COMPACT
    ArchiveFile.cpp
    NeoML.cpp
    Random.cpp
    Dnn/AutoDiff.cpp
    Dnn/AutoDiffFunctions.cpp
    Dnn/BaseLayer.cpp
    Dnn/Dnn.cpp
    Dnn/DnnBlob.cpp
    Dnn/DnnInitializer.cpp
    Dnn/DnnReferenceRegister.cpp
    Dnn/DnnSparseMatrix.cpp
    Dnn/Layers/3dConvLayer.cpp
    Dnn/Layers/ActivationLayers.cpp
    Dnn/Layers/AddToObjectLayer.cpp
    Dnn/Layers/BackLinkLayer.cpp
    Dnn/Layers/BaseConvLayer.cpp
    Dnn/Layers/BaseInPlaceLayer.cpp
    Dnn/Layers/BatchNormalizationLayer.cpp
    Dnn/Layers/BinaryCrossEntropyLayer.cpp
    Dnn/Layers/BroadcastLayer.cpp
    Dnn/Layers/CastLayer.cpp
    Dnn/Layers/ChannelwiseConvLayer.cpp
    Dnn/Layers/CompositeLayer.cpp
    Dnn/Layers/ConcatLayer.cpp
    Dnn/Layers/ConcatObjectLayer.cpp
    Dnn/Layers/ConvLayer.cpp
    Dnn/Layers/CrossEntropyLossLayer.cpp
    Dnn/Layers/DataLayer.cpp
    Dnn/Layers/DropoutLayer.cpp
    Dnn/Layers/EltwiseLayer.cpp
    Dnn/Layers/EuclideanLossLayer.cpp
    Dnn/Layers/FullyConnectedLayer.cpp
    Dnn/Layers/GELULayer.cpp
    Dnn/Layers/GlobalMeanPoolingLayer.cpp
    Dnn/Layers/HingeLossLayer.cpp
    Dnn/Layers/IndRnnLayer.cpp
    Dnn/Layers/L1LossLayer.cpp
    Dnn/Layers/LstmLayer.cpp
    Dnn/Layers/MatrixMultiplicationLayer.cpp
    Dnn/Layers/MobileNetV2BlockLayer.cpp
    Dnn/Layers/MultichannelLookupLayer.cpp
    Dnn/Layers/MultiheadAttentionLayer.cpp
    Dnn/Layers/MultiHingeLossLayer.cpp
    Dnn/Layers/ObjectNormalizationLayer.cpp
    Dnn/Layers/ParameterLayer.cpp
    Dnn/Layers/PoolingLayer.cpp
    Dnn/Layers/RecurrentLayer.cpp
    Dnn/Layers/RleConvLayer.cpp
    Dnn/Layers/QrnnLayer.cpp
    Dnn/Layers/QualityControlLayer.cpp
    Dnn/Layers/SinkLayer.cpp
    Dnn/Layers/SoftmaxLayer.cpp
    Dnn/Layers/SourceLayer.cpp
    Dnn/Layers/SplitLayer.cpp
    Dnn/Layers/TimeConvLayer.cpp
    Dnn/Layers/TransformLayer.cpp
    Dnn/Layers/TransposedConvLayer.cpp
    Dnn/Layers/TransposeLayer.cpp
    TraditionalML/AIModel.cpp
    TraditionalML/ClassificationProbability.cpp
    TraditionalML/CrossValidation.cpp
    TraditionalML/CrossValidationSubProblem.cpp
    TraditionalML/DecisionTreeNodeBase.cpp
    TraditionalML/DecisionTreeNodeClassificationStatistic.cpp
    TraditionalML/DecisionTree.cpp
    TraditionalML/FeatureSelection.cpp
    TraditionalML/FloatVector.cpp
    TraditionalML/Function.cpp
    TraditionalML/FunctionEvaluation.cpp
    TraditionalML/GradientBoostFastHistProblem.cpp
    TraditionalML/GradientBoostFastHistTreeBuilder.cpp
    TraditionalML/GradientBoostFullProblem.cpp
    TraditionalML/GradientBoostFullTreeBuilder.cpp
    TraditionalML/GradientBoostThreadTask.cpp
    TraditionalML/GradientBoostQSEnsemble.cpp
    TraditionalML/Linear.cpp
    TraditionalML/LinkedRegressionTree.cpp
    TraditionalML/MemoryProblem.cpp
    TraditionalML/OneVersusAll.cpp
    TraditionalML/OneVersusOne.cpp
    TraditionalML/PlattScalling.cpp
    TraditionalML/ProblemWrappers.cpp
    TraditionalML/Score.cpp
    TraditionalML/Shuffler.cpp
    TraditionalML/SparseFloatMatrix.cpp
    TraditionalML/StratifiedCrossValidationSubProblem.cpp
    TraditionalML/TrustRegionNewtonOptimizer.cpp
    NeoML.natvis
)

set(NeoML_SOURCES
    ${NeoML_SOURCES_COMPACT}
    Dnn/DnnDistributed.cpp
    Dnn/DnnLora.cpp
    Dnn/DnnOptimization.cpp
    Dnn/Layers/3dPoolingLayer.cpp
    Dnn/Layers/3dTransposedConvLayer.cpp
    Dnn/Layers/AccumulativeLookupLayer.cpp
    Dnn/Layers/AccuracyLayer.cpp
    Dnn/Layers/ArgmaxLayer.cpp
    Dnn/Layers/AttentionLayer.cpp
    Dnn/Layers/BertConvLayer.cpp
    Dnn/Layers/BinaryFocalLossLayer.cpp
    Dnn/Layers/CenterLossLayer.cpp
    Dnn/Layers/ChannelwiseWith1x1Layer.cpp
    Dnn/Layers/CrfLayer.cpp
    Dnn/Layers/CtcLayer.cpp
    Dnn/Layers/CumSumLayer.cpp
    Dnn/Layers/DepthToSpaceLayer.cpp
    Dnn/Layers/DotProductLayer.cpp
    Dnn/Layers/EnumBinarizationLayer.cpp
    Dnn/Layers/FocalLossLayer.cpp
    Dnn/Layers/FullyConnectedSourceLayer.cpp
    Dnn/Layers/GlobalMaxPoolingLayer.cpp
    Dnn/Layers/GlobalSumPoolingLayer.cpp
    Dnn/Layers/GrnLayer.cpp
    Dnn/Layers/GruLayer.cpp
    Dnn/Layers/ImageAndPixelConversionLayer.cpp
    Dnn/Layers/ImageResizeLayer.cpp
    Dnn/Layers/InterpolationLayer.cpp
    Dnn/Layers/IrnnLayer.cpp
    Dnn/Layers/LogicalLayers.cpp
    Dnn/Layers/LoraFullyConnectedLayer.cpp
    Dnn/Layers/LrnLayer.cpp
    Dnn/Layers/MaxOverTimePoolingLayer.cpp
    Dnn/Layers/MobileNetV3BlockLayer.cpp
    Dnn/Layers/ModelWrapperLayer.cpp
    Dnn/Layers/ObjectNormalizationLayer.cpp
    Dnn/Layers/Onnx/OnnxEltwiseLayer.cpp
    Dnn/Layers/Onnx/OnnxCastLayer.cpp
    Dnn/Layers/Onnx/OnnxConcatLayer.cpp
    Dnn/Layers/Onnx/OnnxConstantOfShapeLayer.cpp
    Dnn/Layers/Onnx/OnnxConvTransposeLayer.cpp
    Dnn/Layers/Onnx/OnnxExpandLayer.cpp
    Dnn/Layers/Onnx/OnnxGatherLayer.cpp
    Dnn/Layers/Onnx/OnnxLayerBase.cpp
    Dnn/Layers/Onnx/OnnxLayers.cpp
    Dnn/Layers/Onnx/OnnxNonZeroLayer.cpp
    Dnn/Layers/Onnx/OnnxOneHotLayer.cpp
    Dnn/Layers/Onnx/OnnxRangeLayer.cpp
    Dnn/Layers/Onnx/OnnxReshapeLayer.cpp
    Dnn/Layers/Onnx/OnnxResizeLayer.cpp
    Dnn/Layers/Onnx/OnnxShapeLayer.cpp
    Dnn/Layers/Onnx/OnnxShapeToBlobLayer.cpp
    Dnn/Layers/Onnx/OnnxSliceLayer.cpp
    Dnn/Layers/Onnx/OnnxSourceHelper.cpp
    Dnn/Layers/Onnx/OnnxSplitLayer.cpp
    Dnn/Layers/Onnx/OnnxTransformHelper.cpp
    Dnn/Layers/Onnx/OnnxTransposeHelper.cpp
    Dnn/Layers/PositionalEmbeddingLayer.cpp
    Dnn/Layers/PrecisionRecallLayer.cpp
    Dnn/Layers/ProjectionPoolingLayer.cpp
    Dnn/Layers/ReorgLayer.cpp
    Dnn/Layers/RepeatSequenceLayer.cpp
    Dnn/Layers/RowwiseOperationChainLayer.cpp
    Dnn/Layers/ScatterGatherLayers.cpp
    Dnn/Layers/SequenceSumLayer.cpp
    Dnn/Layers/SpaceToDepthLayer.cpp
    Dnn/Layers/SubSequenceLayer.cpp
    Dnn/Layers/TiedEmbeddingsLayer.cpp
    Dnn/Layers/TransformerLayer.cpp
    Dnn/Layers/TransformerSourceMaskLayer.cpp
    Dnn/Layers/Upsampling2DLayer.cpp
    Dnn/Optimization/BatchNormFusionOptimizer.cpp
    Dnn/Optimization/ChannelwiseWith1x1Optimizer.cpp
    Dnn/Optimization/Graph.cpp
    Dnn/Optimization/MobileNetV2Optimizer.cpp
    Dnn/Optimization/MobileNetV3Optimizer.cpp
    Dnn/Optimization/OptimizerFunctions.cpp
    Dnn/Rowwise/Activation.cpp
    Dnn/Rowwise/ChannelwiseConv.cpp
    Dnn/Rowwise/ChannelwiseWith1x1.cpp
    Dnn/Rowwise/Conv.cpp
    Dnn/Rowwise/ImageResize.cpp
    Dnn/Rowwise/MobileNetV2.cpp
    Dnn/Rowwise/Pooling.cpp
    Dnn/Rowwise/RowwiseOperation.cpp
    TraditionalML/BytePairEncoderTrainer.cpp
    TraditionalML/ClusterCenter.cpp
    TraditionalML/CommonCluster.cpp
    TraditionalML/DifferentialEvolution.cpp
    TraditionalML/FirstComeClustering.cpp
    TraditionalML/HierarchicalClustering.cpp
    TraditionalML/IsoDataClustering.cpp
    TraditionalML/KMeansClustering.cpp
    TraditionalML/NaiveHierarchicalClustering.cpp
    TraditionalML/NnChainHierarchicalClustering.cpp
    TraditionalML/PCA.cpp
    TraditionalML/SMOptimizer.cpp
    TraditionalML/SubwordDecoder.cpp
    TraditionalML/SubwordEncoder.cpp
    TraditionalML/SubwordEncoderTrainer.cpp
    TraditionalML/Svm.cpp
    TraditionalML/SvmKernel.cpp
    TraditionalML/UnigramTrainer.cpp
    TraditionalML/UnigramTools.cpp
    TraditionalML/Utf8Tools.cpp
    TraditionalML/WordDictionary.cpp
)

set(NeoML_NON_UNITY_SOURCES_COMPACT
    Dnn/DnnSolver.cpp
    Dnn/Layers/LossLayer.cpp
    Dnn/Layers/LstmLayer.cpp
    TraditionalML/CompactRegressionTree.cpp
    TraditionalML/DecisionTreeClassificationModel.cpp
    TraditionalML/GradientBoost.cpp
    TraditionalML/GradientBoostModel.cpp
    TraditionalML/GradientBoostQuickScorer.cpp
    TraditionalML/LinearBinaryModel.cpp
    TraditionalML/LinkedRegressionTree.cpp
    TraditionalML/OneVersusAllModel.cpp
    TraditionalML/OneVersusOneModel.cpp
    TraditionalML/SparseFloatVector.cpp
)

set(NeoML_NON_UNITY_SOURCES
    ${NeoML_NON_UNITY_SOURCES_COMPACT}
    TraditionalML/BytePairEncoder.cpp
    TraditionalML/SvmBinaryModel.cpp
    TraditionalML/UnigramEncoder.cpp
)

set(NeoML_HEADERS_COMPACT
    TraditionalML/CompactRegressionTree.h
    TraditionalML/DecisionTreeClassificationModel.h
    TraditionalML/DecisionTreeNodeBase.h
    TraditionalML/DecisionTreeNodeClassificationStatistic.h
    TraditionalML/DecisionTreeNodeStatisticBase.h
    TraditionalML/GradientBoostFastHistProblem.h
    TraditionalML/GradientBoostFastHistTreeBuilder.h
    TraditionalML/GradientBoostFullProblem.h
    TraditionalML/GradientBoostFullTreeBuilder.h
    TraditionalML/GradientBoostModel.h
    TraditionalML/GradientBoostQSEnsemble.h
    TraditionalML/GradientBoostStatisticsSingle.h
    TraditionalML/GradientBoostStatisticsMulti.h
    TraditionalML/GradientBoostThreadTask.h
    TraditionalML/LinearBinaryModel.h
    TraditionalML/LinkedRegressionTree.h
    TraditionalML/OneVersusAllModel.h
    TraditionalML/OneVersusOneModel.h
    TraditionalML/ProblemWrappers.h
    TraditionalML/ProblemWrappers.inl
    TraditionalML/RegressionTree.h
    TraditionalML/SerializeCompact.h

    # Headers
    ../include/NeoML/ArchiveFile.h
    ../include/NeoML/NeoML.h
    ../include/NeoML/NeoMLCommon.h
    ../include/NeoML/NeoMLDefs.h
    ../include/NeoML/Random.h
    ../include/NeoML/Dnn/AutoDiff.h
    ../include/NeoML/Dnn/AutoDiffFunctions.h
    ../include/NeoML/Dnn/Dnn.h
    ../include/NeoML/Dnn/Dnn.inl
    ../include/NeoML/Dnn/DnnBlob.h
    ../include/NeoML/Dnn/DnnInitializer.h
    ../include/NeoML/Dnn/DnnLambdaHolder.h
    ../include/NeoML/Dnn/DnnSolver.h
    ../include/NeoML/Dnn/DnnSparseMatrix.h
    ../include/NeoML/Dnn/Layers/3dConvLayer.h
    ../include/NeoML/Dnn/Layers/ActivationLayers.h
    ../include/NeoML/Dnn/Layers/AddToObjectLayer.h
    ../include/NeoML/Dnn/Layers/BackLinkLayer.h
    ../include/NeoML/Dnn/Layers/BaseInPlaceLayer.h
    ../include/NeoML/Dnn/Layers/BatchNormalizationLayer.h
    ../include/NeoML/Dnn/Layers/BroadcastLayer.h
    ../include/NeoML/Dnn/Layers/CastLayer.h
    ../include/NeoML/Dnn/Layers/ChannelwiseConvLayer.h
    ../include/NeoML/Dnn/Layers/CompositeLayer.h
    ../include/NeoML/Dnn/Layers/ConcatLayer.h
    ../include/NeoML/Dnn/Layers/ConvLayer.h
    ../include/NeoML/Dnn/Layers/DataLayer.h
    ../include/NeoML/Dnn/Layers/DropoutLayer.h
    ../include/NeoML/Dnn/Layers/EltwiseLayer.h
    ../include/NeoML/Dnn/Layers/FullyConnectedLayer.h
    ../include/NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h
    ../include/NeoML/Dnn/Layers/IndRnnLayer.h
    ../include/NeoML/Dnn/Layers/LossLayer.h
    ../include/NeoML/Dnn/Layers/LstmLayer.h
    ../include/NeoML/Dnn/Layers/MatrixMultiplicationLayer.h
    ../include/NeoML/Dnn/Layers/MobileNetV2BlockLayer.h
    ../include/NeoML/Dnn/Layers/MultichannelLookupLayer.h
    ../include/NeoML/Dnn/Layers/MultiheadAttentionLayer.h
    ../include/NeoML/Dnn/Layers/ObjectNormalizationLayer.h
    ../include/NeoML/Dnn/Layers/ParameterLayer.h
    ../include/NeoML/Dnn/Layers/PoolingLayer.h
    ../include/NeoML/Dnn/Layers/RecurrentLayer.h
    ../include/NeoML/Dnn/Layers/QrnnLayer.h
    ../include/NeoML/Dnn/Layers/QualityControlLayer.h
    ../include/NeoML/Dnn/Layers/SinkLayer.h
    ../include/NeoML/Dnn/Layers/SoftmaxLayer.h
    ../include/NeoML/Dnn/Layers/SourceLayer.h
    ../include/NeoML/Dnn/Layers/SplitLayer.h
    ../include/NeoML/Dnn/Layers/TimeConvLayer.h
    ../include/NeoML/Dnn/Layers/TransformLayer.h
    ../include/NeoML/Dnn/Layers/TransposedConvLayer.h
    ../include/NeoML/Dnn/Layers/TransposeLayer.h

    ../include/NeoML/TraditionalML/ClassificationProbability.h
    ../include/NeoML/TraditionalML/ClassificationResult.h
    ../include/NeoML/TraditionalML/CrossValidationSubProblem.h
    ../include/NeoML/TraditionalML/CrossValidation.h
    ../include/NeoML/TraditionalML/DecisionTree.h
    ../include/NeoML/TraditionalML/FeatureSelection.h
    ../include/NeoML/TraditionalML/FloatVector.h
    ../include/NeoML/TraditionalML/Function.h
    ../include/NeoML/TraditionalML/FunctionEvaluation.h
    ../include/NeoML/TraditionalML/GradientBoost.h
    ../include/NeoML/TraditionalML/GradientBoostQuickScorer.h
    ../include/NeoML/TraditionalML/Linear.h
    ../include/NeoML/TraditionalML/MemoryProblem.h
    ../include/NeoML/TraditionalML/Model.h
    ../include/NeoML/TraditionalML/OneVersusAll.h
    ../include/NeoML/TraditionalML/OneVersusOne.h
    ../include/NeoML/TraditionalML/PlattScalling.h
    ../include/NeoML/TraditionalML/Problem.h
    ../include/NeoML/TraditionalML/Score.h
    ../include/NeoML/TraditionalML/Shuffler.h
    ../include/NeoML/TraditionalML/SimpleGenerator.h
    ../include/NeoML/TraditionalML/SparseFloatMatrix.h
    ../include/NeoML/TraditionalML/SparseFloatVector.h
    ../include/NeoML/TraditionalML/SparseVectorIterator.h
    ../include/NeoML/TraditionalML/StratifiedCrossValidationSubProblem.h
    ../include/NeoML/TraditionalML/TrainingModel.h
    ../include/NeoML/TraditionalML/TrustRegionNewtonOptimizer.h
    ../include/NeoML/TraditionalML/VariableMatrix.h
    ../include/NeoML/TraditionalML/VectorIterator.h
)

set(NeoML_HEADERS
    ${NeoML_HEADERS_COMPACT}
    Dnn/Layers/MobileNetBlockUtils.h
    Dnn/Optimization/BatchNormFusionOptimizer.h
    Dnn/Optimization/ChannelwiseWith1x1Optimizer.h
    Dnn/Optimization/MobileNetV2Optimizer.h
    Dnn/Optimization/MobileNetV3Optimizer.h
    Dnn/Optimization/OptimizerFunctions.h
    TraditionalML/BytePairEncoder.h
    TraditionalML/BytePairEncoderTrainer.h
    TraditionalML/NaiveHierarchicalClustering.h
    TraditionalML/NnChainHierarchicalClustering.h
    TraditionalML/SMOptimizer.h
    TraditionalML/SvmBinaryModel.h
    TraditionalML/SubwordDecoder.h
    TraditionalML/UnigramEncoder.h
    TraditionalML/UnigramTrainer.h
    TraditionalML/UnigramTools.h
    TraditionalML/Utf8Tools.h

    # Headers
    ../include/NeoML/Dnn/DnnDistributed.h
    ../include/NeoML/Dnn/DnnLora.h
    ../include/NeoML/Dnn/DnnOptimization.h
    ../include/NeoML/Dnn/Layers/3dPoolingLayer.h
    ../include/NeoML/Dnn/Layers/3dTransposedConvLayer.h
    ../include/NeoML/Dnn/Layers/AccumulativeLookupLayer.h
    ../include/NeoML/Dnn/Layers/AccuracyLayer.h
    ../include/NeoML/Dnn/Layers/ArgmaxLayer.h
    ../include/NeoML/Dnn/Layers/AttentionLayer.h
    ../include/NeoML/Dnn/Layers/BertConvLayer.h
    ../include/NeoML/Dnn/Layers/BinaryFocalLossLayer.h
    ../include/NeoML/Dnn/Layers/CenterLossLayer.h
    ../include/NeoML/Dnn/Layers/ChannelwiseWith1x1Layer.h
    ../include/NeoML/Dnn/Layers/CrfLayer.h
    ../include/NeoML/Dnn/Layers/CtcLayer.h
    ../include/NeoML/Dnn/Layers/CumSumLayer.h
    ../include/NeoML/Dnn/Layers/DepthToSpaceLayer.h
    ../include/NeoML/Dnn/Layers/DotProductLayer.h
    ../include/NeoML/Dnn/Layers/EnumBinarizationLayer.h
    ../include/NeoML/Dnn/Layers/FocalLossLayer.h
    ../include/NeoML/Dnn/Layers/FullyConnectedSourceLayer.h
    ../include/NeoML/Dnn/Layers/GlobalMaxPoolingLayer.h
    ../include/NeoML/Dnn/Layers/GlobalSumPoolingLayer.h
    ../include/NeoML/Dnn/Layers/GrnLayer.h
    ../include/NeoML/Dnn/Layers/GruLayer.h
    ../include/NeoML/Dnn/Layers/ImageAndPixelConversionLayer.h
    ../include/NeoML/Dnn/Layers/ImageResizeLayer.h
    ../include/NeoML/Dnn/Layers/InterpolationLayer.h
    ../include/NeoML/Dnn/Layers/IrnnLayer.h
    ../include/NeoML/Dnn/Layers/LogicalLayers.h
    ../include/NeoML/Dnn/Layers/LoraFullyConnectedLayer.h
    ../include/NeoML/Dnn/Layers/LrnLayer.h
    ../include/NeoML/Dnn/Layers/MaxOverTimePoolingLayer.h
    ../include/NeoML/Dnn/Layers/MobileNetV3BlockLayer.h
    ../include/NeoML/Dnn/Layers/ModelWrapperLayer.h
    ../include/NeoML/Dnn/Layers/MultiHingeLossLayer.h
    ../include/NeoML/Dnn/Layers/ObjectNormalizationLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxCastLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxConcatLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxConstantOfShapeLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxConvTransposeLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxExpandLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxGatherLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxLayerBase.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxLayers.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxNonZeroLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxOneHotLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxRangeLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxResizeLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxShapeLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxShapeToBlobLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxSliceLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxSourceHelper.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxSplitLayer.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h
    ../include/NeoML/Dnn/Layers/Onnx/OnnxTransposeHelper.h
    ../include/NeoML/Dnn/Layers/PoolingLayer.h
    ../include/NeoML/Dnn/Layers/PositionalEmbeddingLayer.h
    ../include/NeoML/Dnn/Layers/PrecisionRecallLayer.h
    ../include/NeoML/Dnn/Layers/ProjectionPoolingLayer.h
    ../include/NeoML/Dnn/Layers/ReorgLayer.h
    ../include/NeoML/Dnn/Layers/RepeatSequenceLayer.h
    ../include/NeoML/Dnn/Layers/RowwiseOperationChainLayer.h
    ../include/NeoML/Dnn/Layers/ScatterGatherLayers.h
    ../include/NeoML/Dnn/Layers/SequenceSumLayer.h
    ../include/NeoML/Dnn/Layers/SpaceToDepthLayer.h
    ../include/NeoML/Dnn/Layers/SubSequenceLayer.h
    ../include/NeoML/Dnn/Layers/TiedEmbeddingsLayer.h
    ../include/NeoML/Dnn/Layers/TransformerLayer.h
    ../include/NeoML/Dnn/Layers/TransformerSourceMaskLayer.h
    ../include/NeoML/Dnn/Layers/Upsampling2DLayer.h

    ../include/NeoML/Dnn/Optimization/Graph.h

    ../include/NeoML/Dnn/Rowwise/Activation.h
    ../include/NeoML/Dnn/Rowwise/ChannelwiseConv.h
    ../include/NeoML/Dnn/Rowwise/ChannelwiseWith1x1.h
    ../include/NeoML/Dnn/Rowwise/Conv.h
    ../include/NeoML/Dnn/Rowwise/ImageResize.h
    ../include/NeoML/Dnn/Rowwise/MobileNetV2.h
    ../include/NeoML/Dnn/Rowwise/Pooling.h
    ../include/NeoML/Dnn/Rowwise/RowwiseOperation.h

    ../include/NeoML/TraditionalML/ClusterCenter.h
    ../include/NeoML/TraditionalML/Clustering.h
    ../include/NeoML/TraditionalML/CommonCluster.h
    ../include/NeoML/TraditionalML/DifferentialEvolution.h
    ../include/NeoML/TraditionalML/FirstComeClustering.h
    ../include/NeoML/TraditionalML/GraphGenerator.h
    ../include/NeoML/TraditionalML/HierarchicalClustering.h
    ../include/NeoML/TraditionalML/IsoDataClustering.h
    ../include/NeoML/TraditionalML/KMeansClustering.h
    ../include/NeoML/TraditionalML/LdGraph.h
    ../include/NeoML/TraditionalML/MatchingGenerator.h
    ../include/NeoML/TraditionalML/PCA.h
    ../include/NeoML/TraditionalML/SubwordEncoder.h
    ../include/NeoML/TraditionalML/SubwordEncoderTrainer.h
    ../include/NeoML/TraditionalML/Svm.h
    ../include/NeoML/TraditionalML/SvmKernel.h
    ../include/NeoML/TraditionalML/WordDictionary.h
)

set_target_properties( ${PROJECT_NAME} PROPERTIES
    UNITY_BUILD_MODE BATCH
    UNITY_BUILD_BATCH_SIZE ${NeoML_UNITY_BUILD_BATCH_SIZE}
)

if(NOT IOS AND NOT ANDROID)
    set_property(SOURCE ${NeoML_NON_UNITY_SOURCES} PROPERTY SKIP_UNITY_BUILD_INCLUSION ON)
    target_sources( ${PROJECT_NAME} PRIVATE
        ${NeoML_SOURCES}
        ${NeoML_NON_UNITY_SOURCES}
        ${NeoML_HEADERS}
    )
else()
    set_property(SOURCE ${NeoML_NON_UNITY_SOURCES_COMPACT} PROPERTY SKIP_UNITY_BUILD_INCLUSION ON)
    target_sources( ${PROJECT_NAME} PRIVATE
        ${NeoML_SOURCES_COMPACT}
        ${NeoML_NON_UNITY_SOURCES_COMPACT}
        ${NeoML_HEADERS_COMPACT}
    )
endif()

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
    add_custom_command(
        OUTPUT ${MESSAGES_SRC}
        COMMAND ${FINE_MESSAGE_COMPILER} -R ${MESSAGES_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../NeoMLMessages.msg
        MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/../NeoMLMessages.msg
        WORKING_DIRECTORY ${FINE_MESSAGE_COMPILER_DIR}
        COMMENT "Generating messages."
    )
    
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
			$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${FINE_ROOT}/Build/Inc ${FINE_ROOT}/FineObjects ${MESSAGES_DIR}>>
		)
	else()
		target_include_directories(${PROJECT_NAME} PRIVATE 
			$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:RC>:${CMAKE_CURRENT_SOURCE_DIR}/../../Build/Inc>>
		)
	endif()

    enable_language(RC)
    target_sources(${PROJECT_NAME} PRIVATE ../NeoML.rc)
endif()

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

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
