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

#include "../common.h"
#pragma hdrstop

#include "ConcatOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxConcatLayer.h>

namespace NeoOnnx {

CConcatOperator::CConcatOperator( const onnx::NodeProto& concat, int opsetVersion ) :
	CLayerOperator( concat, opsetVersion )
{
	// v1 - original
	// v4 - supported new data types and axis becomes required attributes
	// v11 - supported negative axis index
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() >= 1, "operator must have at least 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CConcatOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	const int axis = getAxis( inputs[0]->DimCount() );
	const CTensorLayout& inputLayout = inputs[0]->Layout();
	const bool returnUserTensor = HasUserInput( inputs );

	CPtr<COnnxConcatLayer> concat = new COnnxConcatLayer( dnn.GetMathEngine() );
	concat->SetName( Name() );
	concat->SetConcatDim( inputLayout[axis] );

	int connectionIndex = 0;
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		CLayerOutput layerOutput;
		if( returnUserTensor ) {
			layerOutput = AsUserTensor( *ConvertTensor( *inputs[inputIndex], inputLayout ),
				Name() + "_Source" + Str( inputIndex ), dnn )->LayerOutput();
		} else {
			layerOutput = AsShapeTensor( *ConvertTensor( *inputs[inputIndex], inputLayout ),
				Name() + "_Source" + Str( inputIndex ), dnn )->LayerOutput();
		}
		concat->Connect( connectionIndex++, *layerOutput.Layer, layerOutput.OutputIndex );
	}

	dnn.AddLayer( *concat );

	if( returnUserTensor ) {
		outputs.Add( new CUserTensor( inputLayout, CLayerOutput( concat, 0 ) ) );
	} else {
		NeoPresume( inputs[0]->Type() != TTensorType::User );
		CTensorShape outputShape;
		GetTensorShape( *inputs[0], outputShape );

		// Calculate total size of axis along all inputs
		int& concatDimSize = outputShape[axis];
		for( int i = 1; i < inputs.Size(); ++i ) {
			if( inputs[i]->Type() == TTensorType::Data ) {
				concatDimSize += dynamic_cast<const CDataTensor&>( *inputs[i] ).DimSize( axis );
			} else {
				concatDimSize += dynamic_cast<const CShapeTensor&>( *inputs[i] ).Shape()[axis];
			}
		}

		outputs.Add( new CShapeTensor( inputLayout, outputShape, CLayerOutput( concat, 0 ) ) );
	}
}

// Returns non-negative axis index along which Concat is performing
// Takes into account opset version and number of input dimensions
int CConcatOperator::getAxis( int dimCount ) const
{
	int axis = 1;
	if( OpsetVersion < 4 ) {
		GetAttribute( "axis", axis );
	} else {
		CheckOnnxProtocol( GetAttribute( "axis", axis ), "axis attribute is missing", *this );
		if( axis < 0 ) {
			axis += dimCount;
		}
	}
	return axis;
}

} // namespace NeoOnnx
