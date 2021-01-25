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
#include "PyArgmaxLayer.h"
#include "PyAttentionDecoderLayer.h"
#include "PyBaseConvLayer.h"
#include "PyConvLayer.h"
#include "PyChannelwiseConvLayer.h"
#include "PyConcatChannelsLayer.h"
#include "PyMultichannelLookupLayer.h"
#include "PyLossLayer.h"
#include "PyBaseConvLayer.h"
#include "PyConvLayer.h"
#include "PyChannelwiseConvLayer.h"
#include "PyEltwiseLayer.h"
#include "PyPoolingLayer.h"
#include "PySourceLayer.h"
#include "PySinkLayer.h"
#include "PySolver.h"
#include "PyRandom.h"
#include "PyInitializer.h"

PYBIND11_MODULE(PythonWrapper, m) {

	InitializeClustering( m );

	InitializeTrainingModel( m );

	InitializeMathEngine( m );

	InitializeDnn( m );

	InitializeBlob( m );

	InitializeLayer( m );
	InitializeArgmaxLayer( m );
	InitializeAttentionDecoderLayer( m );
	InitializeBaseConvLayer( m );
	InitializeConvLayer( m );
	InitializeChannelwiseConvLayer( m );
	InitializeConcatChannelsLayer( m );
	InitializeEltwiseLayer( m );
	InitializeLossLayer( m );
	InitializeMultichannelLookupLayer( m );
	InitializePoolingLayer( m );
	InitializeSourceLayer( m );
	InitializeSinkLayer( m );

	InitializeSolver( m );

	InitializeRandom( m );

	InitializeInitializer( m );
}

