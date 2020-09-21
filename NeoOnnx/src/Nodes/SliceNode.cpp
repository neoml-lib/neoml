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

#include "SliceNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSliceNode::CSliceNode( int nodeIndex, const onnx::NodeProto& slice, int opsetVersion ) :
	COpNode( nodeIndex, slice, opsetVersion )
{
	// Newer versions are using inputs instead of attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= 10, "opset version", slice );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", slice );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", slice );

	Attributes.GetRequiredIntArray( "starts", starts );
	Attributes.GetRequiredIntArray( "ends", ends );
	Attributes.GetRequiredIntArray( "axes", axes );

	CheckNeoOnnxSupport( starts.Size() == 1, "slice along multiple axes", slice );
	CheckNeoOnnxSupport( ends.Size() == 1, "slice along multiple axes", slice );
	CheckNeoOnnxSupport( axes.Size() == 1, "slice along multiple axes", slice );
}

void CSliceNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& mathEngine )
{
	const CTensor& inputTensor = tensors[Input[0]];

	CTensorShape& outputShape = tensors[Output[0]].Shape;
	inputTensor.Shape.CopyTo( outputShape );

	if( starts[0] < 0 ) {
		starts[0] += inputTensor.Shape[axes[0]];
	}
	if( ends[0] < 0 ) {
		ends[0] += inputTensor.Shape[axes[0]];
		if( starts[0] == ends[0] ) {
			++ends[0];
		}
	}

	outputShape[axes[0]] = ends[0] - starts[0];

	if( inputTensor.Data == nullptr ) {
		return;
	}

	TBlobType outputBlobType = inputTensor.Data->GetDataType();

	CBlobDesc desc = inputTensor.Data->GetDesc();
	int outputBlobIndex = starts[0] == 0 ? 0 : 1;
	CObjectArray<CDnnBlob> parts;

	if( starts[0] > 0 ) {
		desc.SetDimSize( axes[0], starts[0] );
		parts.Add( CDnnBlob::CreateBlob( mathEngine, outputBlobType, desc ) );
	}

	desc.SetDimSize( axes[0], ends[0] - starts[0] );
	parts.Add( CDnnBlob::CreateBlob( mathEngine, outputBlobType, desc ) );

	if( ends[0] < inputTensor.Shape[axes[0]] ) {
		desc.SetDimSize( axes[0], inputTensor.Shape[axes[0]] - ends[0] );
		parts.Add( CDnnBlob::CreateBlob( mathEngine, outputBlobType, desc ) );
	}

	CDnnBlob::SplitByDim( mathEngine, static_cast< TBlobDim >( axes[0] ), inputTensor.Data, parts );
	tensors[Output[0]].Data = parts[outputBlobIndex];
}

void CSliceNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( tensors[Output[0]].Data != nullptr ) {
		return;
	}

	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"labeling output dimensions failed", OnnxNode );
	}

	if( !dims[Output[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Input[0]].Shape, dims[Output[0]], dims[Input[0]] ),
			"marking inputTensor dimensions failed", OnnxNode );
	}
}

void CSliceNode::AddLayers( const CGraph& graph, const CTensorCache& tensors, const CDimCache& dims,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	if( tensors[Output[0]].Data != nullptr ) {
		return;
	}

	IMathEngine& mathEngine = dnn.GetMathEngine();

	const TBlobDim splitDim = ( dims[Output[0]] )[axes[0]];
	// There is no layer in NeoML to split along BD_ListSize
	CheckNeoOnnxSupport( splitDim != BD_ListSize, "slice along BD_ListSize", OnnxNode );

	const int splitDimSize = tensors[Input[0]].Shape[axes[0]];

	if( starts[0] < 0 ) {
		starts[0] += splitDimSize;
	}

	if( ends[0] < 0 ) {
		ends[0] += splitDimSize;
		if( starts[0] == ends[0] ) {
			++ends[0];
		}
	}

	if( splitDim == BD_BatchLength ) {
		addSubSequenceLayer( starts[0], ends[0], dnn, neoMLLinks );
	} else {
		addSplitLayer( splitDim, starts[0], ends[0], splitDimSize, dnn, neoMLLinks );
	}
}

// Adds CSubSequeceLayer to the dnn
void CSliceNode::addSubSequenceLayer( int start, int end, CDnn& dnn, CNeoMLLinkCache& neoMLLinks )
{
	CPtr<CSubSequenceLayer> subseq = new CSubSequenceLayer( dnn.GetMathEngine() );
	subseq->SetStartPos( start );
	subseq->SetLength( end - start );

	subseq->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );
	subseq->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *subseq );
	neoMLLinks[Output[0]] = CNeoMLLink( subseq, 0 );
}

// Adds sink layer for index'th output of inputLayer
static void addSinkLayer( CBaseLayer& inputLayer, int index, CDnn& dnn )
{
	CPtr<CSinkLayer> sink = new CSinkLayer( dnn.GetMathEngine() );
	sink->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );
	sink->Connect( 0, inputLayer, index );
	dnn.AddLayer( *sink );
}

// Adds split layer along splitDim to the dnn
// Also adds sink layers to split outputs which won't be used later
void CSliceNode::addSplitLayer( TBlobDim splitDim, int start, int end, int dimSize,
	CDnn& dnn, CNeoMLLinkCache& neoMLLinks )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CBaseSplitLayer> split;
	switch( splitDim ) {
		case BD_BatchWidth:
			split = new CSplitBatchWidthLayer( dnn.GetMathEngine() );
			break;
		case BD_Height:
			split = new CSplitHeightLayer( dnn.GetMathEngine() );
			break;
		case BD_Width:
			split = new CSplitWidthLayer( dnn.GetMathEngine() );
			break;
		case BD_Depth:
			split = new CSplitDepthLayer( dnn.GetMathEngine() );
			break;
		case BD_Channels:
			split = new CSplitChannelsLayer( dnn.GetMathEngine() );
			break;
		default:
			CheckNeoOnnxInternal( false, "unknown split dimension", OnnxNode );
	}

	int actualOutputIndex = 0; // Split layer's output containing slice operator result
	CArray<int> outputCounts;

	if( start != 0 ) {
		actualOutputIndex = 1; // Required slice is not in the #0 output of split layer
		outputCounts.Add( start );
	}

	outputCounts.Add( end - start );
	split->SetOutputCounts( outputCounts );
	split->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );
	split->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *split );
	neoMLLinks[Output[0]] = CNeoMLLink( split, actualOutputIndex );

	if( actualOutputIndex == 1 ) {
		addSinkLayer( *split, 0, dnn ); // Sink layer for the first output
	}
	if( end != dimSize ) {
		addSinkLayer( *split, actualOutputIndex + 1, dnn ); // Sink layer for the last output
	}
}

} // namespace NeoOnnx
