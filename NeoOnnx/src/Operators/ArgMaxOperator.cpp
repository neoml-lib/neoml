/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "onnx.pb.h"

#include "ArgMaxOperator.h"
#include "NeoOnnxCheck.h"

namespace NeoOnnx {

CArgMaxOperator::CArgMaxOperator( const onnx::NodeProto& argMax, int opsetVersion ) :
	CLayerOperator( argMax, opsetVersion )
{
	// v1 - original
	// v11 - negative axis attribute values are supported
	// v12 - select_last_index attribute is added
	// v13 - bfloat16 data type is supported
	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CArgMaxOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional" );
	CPtr<const CUserTensor> input = AsUserTensor( *inputs[0], Name() + "_data", dnn );

	// In ONNX ArgMax supports any data type
	// In NeoML CArgMaxLayer supports only float input
	CBaseLayer* outputLayer = Cast( CT_Float )( Name() + "_cast", CDnnLayerLink( input->Layer(), input->OutputIndex() ) );

	int axis = 0;
	GetAttribute( "axis", axis );
	if( axis < 0 ) {
		axis += input->DimCount();
	}
	outputLayer = Argmax( input->Layout()[axis] )( Name(), outputLayer );

	CTensorShape outputShape;
	input->Shape().CopyTo( outputShape );
	outputShape[axis] = 1;

	CTensorLayout outputLayout = input->Layout();

	int keepDims = 1;
	GetAttribute( "keepdims", keepDims );
	if( keepDims == 0 ) {
		outputLayout.DeleteAt( axis );
		outputShape.DeleteAt( axis );
	}

	outputs.Add( new CUserTensor( outputShape, outputLayout, CLayerOutput( outputLayer, 0 ) ) );
}

} // namespace NeoOnnx
