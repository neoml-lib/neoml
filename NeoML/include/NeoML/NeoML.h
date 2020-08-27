/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/TraditionalML/SparseFloatVector.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/ClusterCenter.h>
#include <NeoML/TraditionalML/CommonCluster.h>
#include <NeoML/TraditionalML/Clustering.h>
#include <NeoML/TraditionalML/FirstComeClustering.h>
#include <NeoML/TraditionalML/IsoDataClustering.h>
#include <NeoML/TraditionalML/KMeansClustering.h>
#include <NeoML/TraditionalML/EMClustering.h>
#include <NeoML/TraditionalML/HierarchicalClustering.h>
#include <NeoML/TraditionalML/MemoryProblem.h>
#include <NeoML/TraditionalML/LinearBinaryClassifierBuilder.h>
#include <NeoML/TraditionalML/DecisionTreeTrainingModel.h>
#include <NeoML/TraditionalML/OneVersusAll.h>
#include <NeoML/TraditionalML/SMOptimizer.h>
#include <NeoML/TraditionalML/SvmBinaryClassifierBuilder.h>
#include <NeoML/TraditionalML/PlattScalling.h>
#include <NeoML/TraditionalML/DifferentialEvolution.h>
#include <NeoML/Dnn/DnnBlob.h>
#include <NeoML/Dnn/DnnSparseMatrix.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>
#include <NeoML/Dnn/Layers/SourceLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>
#include <NeoML/Dnn/Layers/LossLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedSourceLayer.h>
#include <NeoML/Dnn/Layers/PoolingLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/ModelWrapperLayer.h>
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Layers/ObjectNormalizationLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/ImageResizeLayer.h>
#include <NeoML/Dnn/Layers/BackLinkLayer.h>
#include <NeoML/TraditionalML/LdGraph.h>
#include <NeoML/TraditionalML/SimpleGenerator.h>
#include <NeoML/TraditionalML/GraphGenerator.h>
#include <NeoML/TraditionalML/FeatureSelection.h>
#include <NeoML/TraditionalML/MatchingGenerator.h>
#include <NeoML/TraditionalML/VariableMatrix.h>
#include <NeoML/TraditionalML/Score.h>
#include <NeoML/TraditionalML/Shuffler.h>
#include <NeoML/TraditionalML/GradientBoost.h>
#include <NeoML/TraditionalML/GradientBoostQuickScorer.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/RecurrentLayer.h>
#include <NeoML/Dnn/Layers/SubSequenceLayer.h>
#include <NeoML/Dnn/Layers/EnumBinarizationLayer.h>
#include <NeoML/Dnn/Layers/SoftmaxLayer.h>
#include <NeoML/Dnn/Layers/GlobalMeanPoolingLayer.h>
#include <NeoML/Dnn/Layers/GlobalMaxPoolingLayer.h>
#include <NeoML/Dnn/Layers/LstmLayer.h>
#include <NeoML/Dnn/Layers/ReorgLayer.h>
#include <NeoML/Dnn/Layers/GruLayer.h>
#include <NeoML/Dnn/DnnSolver.h>
#include <NeoML/Dnn/DnnInitializer.h>
#include <NeoML/Dnn/Layers/MultichannelLookupLayer.h>
#include <NeoML/Dnn/Layers/MaxOverTimePoolingLayer.h>
#include <NeoML/Dnn/Layers/3dConvLayer.h>
#include <NeoML/Dnn/Layers/3dPoolingLayer.h>
#include <NeoML/Dnn/Layers/TimeConvLayer.h>
#include <NeoML/Dnn/Layers/TransposedConvLayer.h>
#include <NeoML/Dnn/Layers/3dTransposedConvLayer.h>
#include <NeoML/Dnn/Layers/AttentionLayer.h>
#include <NeoML/Dnn/Layers/SequenceSumLayer.h>
#include <NeoML/Dnn/Layers/CrfLayer.h>
#include <NeoML/Dnn/Layers/TransformLayer.h>
#include <NeoML/Dnn/Layers/TransposeLayer.h>
#include <NeoML/Dnn/Layers/MultiHingeLossLayer.h>
#include <NeoML/Dnn/Layers/Upsampling2DLayer.h>
#include <NeoML/Dnn/Layers/ChannelwiseConvLayer.h>
#include <NeoML/Dnn/Layers/AccumulativeLookupLayer.h>
#include <NeoML/Dnn/Layers/QualityControlLayer.h>
#include <NeoML/Dnn/Layers/AccuracyLayer.h>
#include <NeoML/Dnn/Layers/PrecisionRecallLayer.h>
#include <NeoML/Dnn/Layers/ArgmaxLayer.h>
#include <NeoML/Dnn/Layers/CenterLossLayer.h>
#include <NeoML/Dnn/Layers/FocalLossLayer.h>
#include <NeoML/Dnn/Layers/BinaryFocalLossLayer.h>
#include <NeoML/Dnn/Layers/ImageAndPixelConversionLayer.h>
#include <NeoML/Dnn/Layers/RepeatSequenceLayer.h>
#include <NeoML/Dnn/Layers/DotProductLayer.h>
#include <NeoML/Dnn/Layers/CtcLayer.h>
#include <NeoML/Dnn/Layers/AddToObjectLayer.h>
#include <NeoML/Dnn/Layers/MatrixMultiplicationLayer.h>
#include <NeoML/Dnn/Layers/MultiheadAttentionLayer.h>
#include <NeoML/Dnn/Layers/PositionalEmbeddingLayer.h>
#include <NeoML/Dnn/Layers/GELULayer.h>
#include <NeoML/ArchiveFile.h>

#ifndef NO_NEOML_NAMESPACE
using namespace NeoML;
#endif
