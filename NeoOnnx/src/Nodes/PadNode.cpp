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

#include "PadNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

void CalculatePadding( const CString& autoPad, const CTensorShape& inputShape, const CTensorShape& kernelShape,
	CFastArray<int, 8>& pads )
{
	const int padDims = static_cast<int>( kernelShape.Size() );
	const int skipDims = static_cast<int>( inputShape.Size() ) - padDims;

	for( int padDimIndex = 0; padDimIndex < padDims; ++padDimIndex ) {
		const int totalPadSize = kernelShape[padDimIndex] - 1;
		if( autoPad == "SAME_LOWER" ) {
			pads[padDimIndex] = ( totalPadSize + 1 ) / 2;
		} else {
			pads[padDimIndex] = totalPadSize / 2;
		}
		pads[padDims + padDimIndex] = totalPadSize - pads[padDimIndex];
	}
}

CPtr<CBaseLayer> CreatePaddingLayer( IMathEngine& mathEngine, const CString& layerName, const CTensorDim& dim,
	const CFastArray<int, 8>& pads, float padValue, const onnx::NodeProto& onnxNode )
{
	// Pool and conv operators storing pads only for N-2 tensor dimensions (leaving out batch and channels)
	// On the other side Pad operator is storing pads for every tensor dimension

	// Number of padded dimensions
	const int paddedDims = pads.Size() / 2;
	// Index of first padded dimension
	const int padDimIndex = dim.Size() - paddedDims;

	CPtr<CImageResizeLayer> imageResize = new CImageResizeLayer( mathEngine );
	imageResize->SetName( layerName );
	imageResize->SetDefalutValue( padValue );

	for( int i = 0; i < paddedDims; ++i ) {
		if( pads[i] == 0 && pads[i + paddedDims] == 0 ) {
			continue;
		}

		switch( dim[padDimIndex + i] ) {
			case BD_Height:
				imageResize->SetDelta( CImageResizeLayer::IS_Top, pads[i] );
				imageResize->SetDelta( CImageResizeLayer::IS_Bottom, pads[paddedDims + i] );
				break;
			case BD_Width:
				imageResize->SetDelta( CImageResizeLayer::IS_Left, pads[i] );
				imageResize->SetDelta( CImageResizeLayer::IS_Right, pads[paddedDims + i] );
				break;
			default:
				CheckNeoOnnxSupport( false, "Can't pad dimension " + Str( dim[padDimIndex + i]), onnxNode );
		}
	}

	return imageResize.Ptr();
}

//---------------------------------------------------------------------------------------------------------------------

CPadNode::CPadNode( int nodeIndex, const onnx::NodeProto& pad, int opsetVersion ) :
	COpNode( nodeIndex, pad, opsetVersion ),
	mode( Attributes.GetOptionalString( "mode", "constant" ) ),
	value( 0.f )
{
	// In v1 pads are provided by 'paddings' attribute and pad value is provided by 'value' attribute 
	// In v2 pads are provided by 'pads' attribute and pad value is provided by 'value' attribute 
	// In v11 pads and pad value are provided by additional inputs
	if( opsetVersion < 11 ) {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", pad );
		Attributes.GetRequiredIntArray( opsetVersion == 1 ? "paddings" : "pads", pads );
		value = Attributes.GetOptionalFloat( "value", 0.f );
	} else {
		CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "node must have 2 or 3 inputs", pad );
	}

	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", pad );
	CheckNeoOnnxSupport( mode == "constant", "Pad with non-constant mode", pad );
}

void CPadNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "constant data", OnnxNode );

	if( OpsetVersion >= 11 ) {
		CheckNeoOnnxSupport( tensors[Input[1]].Data != nullptr, "non-constant pad sizes", OnnxNode );
		const CDnnBlob& padsBlob = *tensors[Input[1]].Data;
		CheckOnnxProtocol( padsBlob.GetDataType() == CT_Int, "non-integer pad sizes", OnnxNode );
		pads.SetSize( padsBlob.GetDataSize() );
		padsBlob.CopyTo( pads.GetPtr() );
		if( InputCount() == 3 ) {
			CheckNeoOnnxSupport( tensors[Input[2]].Data != nullptr, "non-constant pad value", OnnxNode );
			const CDnnBlob& valueBlob = *tensors[Input[2]].Data;
			if( valueBlob.GetDataType() == CT_Float ) {
				value = valueBlob.GetData<float>().GetValue();
			} else {
				value = static_cast<float>( valueBlob.GetData<int>().GetValue() );
			}
		}
	}

	const CTensorShape& inputShape = tensors[Input[0]].Shape;
	CheckOnnxProtocol( pads.Size() == inputShape.Size() * 2, "wrong size of pads array", OnnxNode );
	CTensorShape& outputShape = tensors[Output[0]].Shape;
	outputShape.SetSize( inputShape.Size() );
	for( int i = 0; i < inputShape.Size(); ++i ) {
		outputShape[i] = inputShape[i] + pads[i] + pads[i + pads.Size() / 2];
	}
}

void CPadNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"labeling input dimensions failed", OnnxNode );
	}
}

void CPadNode::AddLayers( const CGraph& /* graph */, const CTensorCache& /* tensors */, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CBaseLayer> padding = CreatePaddingLayer( dnn.GetMathEngine(), "NeoMLLayer" + Str( dnn.GetLayerCount() ),
		dims[Input[0]], pads, value, OnnxNode );
	padding->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[1]].OutputIndex );
	dnn.AddLayer( *padding );

	neoMLLinks[Output[0]] = CNeoMLLink( padding, 0 );
}

} // namespace NeoOnnx
