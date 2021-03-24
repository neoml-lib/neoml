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
#include "PyLayer.h"
#include "PyAccumulativeLookupLayer.h"
#include "PyAccuracyLayer.h"
#include "PyActivationLayer.h"
#include "PyAddToObjectLayer.h"
#include "PyArgmaxLayer.h"
#include "PyAttentionDecoderLayer.h"
#include "PyBatchNormalizationLayer.h"
#include "PyBaseConvLayer.h"
#include "PyBinarizationLayer.h"
#include "PyConvLayer.h"
#include "PyConcatLayer.h"
#include "PyCrfLayer.h"
#include "PyCtcLayer.h"
#include "PyDotProductLayer.h"
#include "PyDropoutLayer.h"
#include "PyFullyConnectedLayer.h"
#include "PyImageConversionLayer.h"
#include "PyIrnnLayer.h"
#include "PyGruLayer.h"
#include "PyMultichannelLookupLayer.h"
#include "PyMatrixMultiplicationLayer.h"
#include "PyMultiheadAttentionLayer.h"
#include "PyObjectNormalizationLayer.h"
#include "PyPositionalEmbeddingLayer.h"
#include "PyPrecisionRecallLayer.h"
#include "PyQrnnLayer.h"
#include "PyReorgLayer.h"
#include "PyRepeatSequenceLayer.h"
#include "PySequenceSumLayer.h"
#include "PySoftmaxLayer.h"
#include "PySplitLayer.h"
#include "PySubSequenceLayer.h"
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

PYBIND11_MODULE(PythonWrapper, m) {

	InitializeClustering( m );

	InitializeTrainingModel( m );

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
	InitializeBinarizationLayer( m );
	InitializeConvLayer( m );
	InitializeConcatLayer( m );
	InitializeCrfLayer( m );
	InitializeCtcLayer( m );
	InitializeEltwiseLayer( m );
	InitializeDotProductLayer( m );
	InitializeDropoutLayer( m );
	InitializeFullyConnectedLayer( m );
	InitializeImageConversionLayer( m );
	InitializeIrnnLayer( m );
	InitializeGruLayer( m );
	InitializeLossLayer( m );
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
	InitializeSequenceSumLayer( m );
	InitializeSoftmaxLayer( m );
	InitializeSplitLayer( m );
	InitializeSubSequenceLayer( m );
	InitializeTransformLayer( m );
	InitializeTransposeLayer( m );
	InitializeTiedEmbeddingsLayer( m );
	InitializeUpsampling2DLayer( m );
	InitializeSourceLayer( m );
	InitializeSinkLayer( m );

	InitializeSolver( m );

	InitializeRandom( m );

	InitializeInitializer( m );
}

