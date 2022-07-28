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
	int firstInput = NotFound;
	int inputCount = 0;
	for( int i = 0; i < inputs.Size(); ++i ) {
		if( inputs[i] != nullptr ) {
			if( firstInput == NotFound ) {
				firstInput = i;
			}
			inputCount++;
		}
	}

	if( inputCount == 0 ) {
		outputs.Add( nullptr );
		return;
	}

	if( inputCount == 1 ) {
		outputs.Add( inputs[firstInput] );
		return;
	}

	const int dimCount = inputs[firstInput]->DimCount();

	int axis = 1;
	if( OpsetVersion < 4 ) {
		GetAttribute( "axis", axis );
	} else {
		CheckOnnxProtocol( GetAttribute( "axis", axis ), "axis attribute is missing", *this );
		if( axis < 0 ) {
			axis += dimCount;
		}
	}

	const CTensorLayout& inputLayout = inputs[firstInput]->Layout();
	CPtr<CBaseLayer> concat = createLayer( inputLayout[axis], dnn.GetMathEngine() );
	concat->SetName( Name() );

	CTensorShape outputShape;
	inputs[firstInput]->Shape().CopyTo( outputShape );
	outputShape[axis] = 0;

	int connectionIndex = 0;
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		if( inputs[inputIndex] == nullptr ) {
			continue;
		}
		CPtr<const CUserTensor> preparedInput = AsUserTensor( *ConvertTensor( *inputs[inputIndex], inputLayout ),
			Name() + "_Source" + Str( inputIndex ), dnn );
		concat->Connect( connectionIndex++, *preparedInput->Layer(), preparedInput->OutputIndex() );
		outputShape[axis] += inputs[inputIndex]->Shape()[axis];
	}

	dnn.AddLayer( *concat );

	outputs.Add( new CUserTensor( outputShape, inputLayout, CLayerOutput( concat, 0 ) ) );
}

// Creates corresponding CConcat*Layer
CPtr<CBaseLayer> CConcatOperator::createLayer( TBlobDim concatDim, IMathEngine& mathEngine ) const
{
	switch( concatDim ) {
		case BD_BatchWidth:
			return new CConcatBatchWidthLayer( mathEngine );
		case BD_Height:
			return new CConcatHeightLayer( mathEngine );
		case BD_Width:
			return new CConcatWidthLayer( mathEngine );
		case BD_Depth:
			return new CConcatDepthLayer( mathEngine );
		case BD_Channels:
			return new CConcatChannelsLayer( mathEngine );
		case BD_BatchLength:
			return new CConcatBatchLengthLayer( mathEngine );
		case BD_ListSize:
			return new CConcatListSizeLayer( mathEngine );
		default:
			CheckNeoOnnxSupport( false, "unsupported Concat dimension", *this );
	}

	return nullptr;
}

} // namespace NeoOnnx
