/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <common.h>
#pragma hdrstop

#include "PyClustering.h"
#include "PyTrainingModel.h"
#include "PyMathEngine.h"
#include "PyDnn.h"
#include "PyDnnBlob.h"
#include "PyDnnDistributed.h"
#include "PyAutoDiff.h"
#include "PyLayer.h"
#include "PyAccumulativeLookupLayer.h"
#include "PyAccuracyLayer.h"
#include "PyActivationLayer.h"
#include "PyAddToObjectLayer.h"
#include "PyArgmaxLayer.h"
#include "PyAttentionDecoderLayer.h"
#include "PyBatchNormalizationLayer.h"
#include "PyBaseConvLayer.h"
#include "PyBertConvLayer.h"
#include "PyBinarizationLayer.h"
#include "PyBroadcastLayer.h"
#include "PyCastLayer.h"
#include "PyConvLayer.h"
#include "PyConcatLayer.h"
#include "PyCrfLayer.h"
#include "PyCtcLayer.h"
#include "PyCumSumLayer.h"
#include "PyCustomLossLayer.h"
#include "PyDataLayer.h"
#include "PyDotProductLayer.h"
#include "PyDropoutLayer.h"
#include "PyFullyConnectedLayer.h"
#include "PyImageConversionLayer.h"
#include "PyIndRnnLayer.h"
#include "PyIrnnLayer.h"
#include "PyGruLayer.h"
#include "PyLogicalLayer.h"
#include "PyLrnLayer.h"
#include "PyMultichannelLookupLayer.h"
#include "PyMatrixMultiplicationLayer.h"
#include "PyMultiheadAttentionLayer.h"
#include "PyObjectNormalizationLayer.h"
#include "PyPositionalEmbeddingLayer.h"
#include "PyPrecisionRecallLayer.h"
#include "PyQrnnLayer.h"
#include "PyReorgLayer.h"
#include "PyRepeatSequenceLayer.h"
#include "PyScatterGatherLayers.h"
#include "PySequenceSumLayer.h"
#include "PySoftmaxLayer.h"
#include "PySpaceAndDepthLayer.h"
#include "PySplitLayer.h"
#include "PySubSequenceLayer.h"
#include "PyTransformerLayer.h"
#include "PyTransformLayer.h"
#include "PyTransposeLayer.h"
#include "PyUpsampling2DLayer.h"
#include "PyLossLayer.h"
#include "PyLstmLayer.h"
#include "PyBaseConvLayer.h"
#include "PyConvLayer.h"
#include "PyEltwiseLayer.h"
#include "PyPoolingLayer.h"
#include "PySourceLayer.h"
#include "PySinkLayer.h"
#include "PySolver.h"
#include "PyRandom.h"
#include "PyInitializer.h"
#include "PyTiedEmbeddingsLayer.h"
#include "PyDifferentialEvolution.h"
#include "PyPCA.h"
#include "PyBytePairEncoder.h"

PYBIND11_MODULE(PythonWrapper, m) {

	InitializeClustering( m );

	InitializeTrainingModel( m );

	InitializePCA( m );

	InitializeMathEngine( m );

	InitializeDnn( m );

	InitializeBlob( m );

	InitializeLayer( m );
	InitializeAccumulativeLookupLayer( m );
	InitializeAccuracyLayer( m );
	InitializeActivationLayer( m );
	InitializeAddToObjectLayer( m );
	InitializeArgmaxLayer( m );
	InitializeAttentionDecoderLayer( m );
	InitializeBaseConvLayer( m );
	InitializeBatchNormalizationLayer( m );
	InitializeBertConvLayer( m );
	InitializeBinarizationLayer( m );
	InitializeBroadcastLayer( m );
	InitializeCastLayer( m );
	InitializeConvLayer( m );
	InitializeConcatLayer( m );
	InitializeCrfLayer( m );
	InitializeCtcLayer( m );
	InitializeCumSumLayer( m );
	InitializeDataLayer( m );
	InitializeEltwiseLayer( m );
	InitializeDotProductLayer( m );
	InitializeDropoutLayer( m );
	InitializeDistributedTraining( m );
	InitializeFullyConnectedLayer( m );
	InitializeImageConversionLayer( m );
	InitializeIndRnnLayer( m );
	InitializeIrnnLayer( m );
	InitializeGruLayer( m );
	InitializeLogicalLayer( m );
	InitializeLossLayer( m );
	InitializeLrnLayer( m );
	InitializeCustomLossLayer( m );
	InitializeLstmLayer( m );
	InitializeMatrixMultiplicationLayer( m );
	InitializeMultichannelLookupLayer( m );
	InitializeMultiheadAttentionLayer( m );
	InitializeObjectNormalizationLayer( m );
	InitializePoolingLayer( m );
	InitializePositionalEmbeddingLayer( m );
	InitializePrecisionRecallLayer( m );
	InitializeQrnnLayer( m );
	InitializeReorgLayer( m );
	InitializeRepeatSequenceLayer( m );
	InitializeScatterGatherLayers( m );
	InitializeSequenceSumLayer( m );
	InitializeSoftmaxLayer( m );
	InitializeSplitLayer( m );
	InitializeSubSequenceLayer( m );
	InitializeTransformerLayer( m );
	InitializeTransformLayer( m );
	InitializeTransposeLayer( m );
	InitializeTiedEmbeddingsLayer( m );
	InitializeUpsampling2DLayer( m );
	InitializeSourceLayer( m );
	InitializeSpaceAndDepthLayer( m );
	InitializeSinkLayer( m );

	InitializeSolver( m );
	InitializeTape( m );

	InitializeRandom( m );

	InitializeInitializer( m );
	InitializeDifferentialEvolution( m );

	InitializeBytePairEncoder( m );
}
