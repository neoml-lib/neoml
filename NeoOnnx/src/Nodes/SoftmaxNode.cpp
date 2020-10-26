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

#include "SoftmaxNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSoftmaxNode::CSoftmaxNode( int nodeIndex, const onnx::NodeProto& softmax, int opsetVersion ) :
	COpNode( nodeIndex, softmax, opsetVersion ),
	axis( Attributes.GetOptionalInt( "axis", 1 ) )
{
	// The differences between versions are in negative axis support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", softmax );

	// Negative axis index supported since v11
	CheckOnnxProtocol( axis >= 0 || opsetVersion >= 11, "negative axis index", softmax );
	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", softmax );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", softmax );
}

void CSoftmaxNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );
}

void CSoftmaxNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
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

void CSoftmaxNode::AddLayers( const CGraph& /* graph */, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	const CTensorShape& shape = tensors[Input[0]].Shape;
	CTensorDim outputDim;
	getOutputDim( shape, dims, outputDim );

	CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer( dnn.GetMathEngine() );
	softmax->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );
	softmax->SetNormalizationArea( getArea( shape, outputDim ) );
	softmax->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );

	dnn.AddLayer( *softmax );
	neoMLLinks[Output[0]] = CNeoMLLink( softmax, 0 );
}

// Returns NeoML softmax area applicable to given tensor shape and dim
CSoftmaxLayer::TNormalizationArea CSoftmaxNode::getArea( const CTensorShape& shape, const CTensorDim& dim ) const
{
	// Softmax will be applied to all axes since axisIndex
	const int axisIndex = axis >= 0 ? axis : axis + shape.Size();

	// Bit masks of axes of batch (softmax won't be applied) and object (softmax will be applied)
	int objectAxes = 0;
	int batchAxes = 0;
	for( int i = 0; i < shape.Size(); ++i ) {
		if( shape[i] != 1 ) {
			if( i < axisIndex ) {
				batchAxes |= ( 1 << dim[i] );
			} else {
				objectAxes |= ( 1<< dim[i] );
			}
		}
	}

	// Bit masks of possible softmax areas in NeoML
	const int channelMask = 1 << BD_Channels;
	const int objectSizeMask = channelMask | ( 1 << BD_Height ) | ( 1 << BD_Width ) | ( 1 << BD_Depth );
	const int listSizeMask = 1 << BD_ListSize;
	const int batchLengthMask = 1 << BD_BatchLength;

	const CFastArray<int, 4> masks = { channelMask, objectSizeMask, listSizeMask, batchLengthMask };
	const CFastArray<CSoftmaxLayer::TNormalizationArea, 4> areas = { CSoftmaxLayer::NA_Channel,
		CSoftmaxLayer::NA_ObjectSize, CSoftmaxLayer::NA_ListSize, CSoftmaxLayer::NA_BatchLength };

	for( int i = 0; i < masks.Size(); ++i ) {
		if( ( masks[i] & objectAxes ) == objectAxes && ( masks[i] & batchAxes ) == 0 ) {
			return areas[i];
		}
	}

	CheckNeoOnnxSupport( false, "unsupported softmax axes", OnnxNode );
	return CSoftmaxLayer::NA_Count;
}

void CSoftmaxNode::getOutputDim( const CTensorShape& shape, const CDimCache& dims, CTensorDim& outputDim ) const
{
	if( !dims[Output[0]].IsEmpty() ) {
		dims[Output[0]].CopyTo( outputDim );
	}
	CTensorDim batchDims = { BD_BatchLength, BD_BatchWidth, BD_ListSize };
	CTensorDim objectDims = { BD_Channels, BD_Depth, BD_Height, BD_Width };

	const int axisIndex = axis >= 0 ? axis : axis + shape.Size();
	CheckNeoOnnxSupport( axisIndex <= 3 && shape.Size() - axisIndex <= 4, "too many dims to softmax", OnnxNode );

	outputDim.SetSize( shape.Size() );
	for( int i = 0; i < shape.Size(); ++i ) {
		outputDim[i] = i < axisIndex ? batchDims[i] : objectDims[i - axisIndex];
	}
}

} // namespace NeoOnnx
