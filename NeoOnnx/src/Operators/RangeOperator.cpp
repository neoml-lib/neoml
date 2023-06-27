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

#include "../common.h"
#pragma hdrstop

#include "RangeOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxRangeLayer.h>

namespace NeoOnnx {

CRangeOperator::CRangeOperator( const onnx::NodeProto& range, int opsetVersion ) :
	CLayerOperator( range, opsetVersion )
{
	// v11 - original
	CheckOnnxProtocol( OpsetVersion >= 11, "Range operator was introduced in opset v11", *this );
	CheckNeoOnnxSupport( OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 3, "operator must have 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CRangeOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoUserInputs( inputs );

	CPtr<COnnxRangeLayer> layer = new COnnxRangeLayer( dnn.GetMathEngine() );
	layer->SetName( Name() );
	for( int i = 0; i < 3; ++i ) {
		CPtr<const CShapeTensor> currInput = AsShapeTensor( *inputs[i], Name() + "_Source#" + Str( i ), dnn );
		layer->Connect( i, *currInput->Layer(), currInput->OutputIndex() );
	}
	dnn.AddLayer( *layer );

	outputs.Add( new CUserTensor( CTensorLayout::IOLayout( 1 ), CLayerOutput( layer, 0 ) ) );
}

} // namespace NeoOnnx
