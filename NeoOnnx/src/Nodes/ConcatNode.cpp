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

#include "ConcatNode.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConcatNode::CConcatNode( const onnx::NodeProto& concat, int opsetVersion ) :
	COpNode( concat, opsetVersion )
{
	// v1 - original
	// v4 - supported new data types and axis becomes required attributes
	// v11 - supported negative axis index
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", concat );

	CheckOnnxProtocol( InputCount() > 1, "node must have more than 1 inputs", concat );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", concat );
}

void CConcatNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CheckNeoOnnxInternal( inputs[0] != nullptr, "Unknown input", OnnxNode );
	const int dimCount = inputs[0]->Shape().Size();

	int axis = 1;
	if( OpsetVersion < 4 ) {
		axis = Attributes.GetOptionalInt( "axis", 1 );
	} else {
		axis = Attributes.GetRequiredInt( "axis" );
		if( axis < 0 ) {
			CheckOnnxProtocol( OpsetVersion >= 11, "negative axis is supported since v11", OnnxNode );
			axis += dimCount;
		}
	}

	const CTensorLayout& inputLayout = inputs[0]->Layout();
	TBlobDim concatDim;
	if( inputLayout.DimType == DT_Onnx ) {
		concatDim = static_cast<TBlobDim>( axis );
	} else {
		concatDim = inputLayout.OnnxOrder[axis];
	}

	CPtr<CBaseLayer> concat = createLayer( concatDim, dnn.GetMathEngine() );
	concat->SetName( Name() );

	CTensorShape outputShape;
	inputs[0]->Shape().CopyTo( outputShape );
	outputShape[axis] = 0;

	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		CheckNeoOnnxInternal( inputs[inputIndex] != nullptr, "Unknown input", OnnxNode );
		CPtr<const CUserTensor> preparedInput = prepareInput( inputs, inputIndex, inputLayout, dnn );
		concat->Connect( inputIndex, *preparedInput->Layer(), preparedInput->OutputIndex() );
		outputShape[axis] += inputs[inputIndex]->Shape()[axis];
	}
	
	outputs[0] = new CUserTensor( outputShape, inputLayout, CLayerOutput( concat, 0 ) );
}

// Creates corresponding CConcat*Layer
CPtr<CBaseLayer> CConcatNode::createLayer( TBlobDim concatDim, IMathEngine& mathEngine ) const
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
			CheckNeoOnnxSupport( false, "unsupported Concat dimension", OnnxNode );
	}

	return nullptr;
}

// Prepares input for concatenation
CPtr<const CUserTensor> CConcatNode::prepareInput( const CObjectArray<const CTensorBase>& inputs,
	int inputIndex, const CTensorLayout& layout, CDnn& dnn ) const
{
	// Converting input into required layout
	CPtr<const CTensorBase> convertedInput = ConvertTensor( *inputs[inputIndex], layout );

	if( !convertedInput->IsCalculated() ) {
		return dynamic_cast<const CUserTensor*>( convertedInput.Ptr() );
	}

	// If tensor contains data we need to add it to the net via CSourceLayer
	CPtr<CDnnBlob> data = dynamic_cast<const CDataTensor*>( convertedInput.Ptr() )->Data()->GetCopy();
	CPtr<CSourceLayer> source = new CSourceLayer( dnn.GetMathEngine() );
	source->SetName( Name() + "_input_" + Str( inputIndex ) );
	dnn.AddLayer( *source );
	source->SetBlob( data );
	
	return new CUserTensor( convertedInput->Shape(), convertedInput->Layout(), CLayerOutput( source, 0 ) );
}

} // namespace NeoOnnx
