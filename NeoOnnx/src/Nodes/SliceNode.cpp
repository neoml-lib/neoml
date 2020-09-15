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
	// TODO: Add support for multi-axes slice.
	CheckNeoOnnxSupport( starts.Size() == 1, "starts.Size() > 1", slice );
	CheckNeoOnnxSupport( ends.Size() == 1, "ends.Size() > 1", slice );
	CheckNeoOnnxSupport( axes.Size() == 1, "axes.Size() > 1", slice );
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

	CDnnBlob::SplitByDim( mathEngine, static_cast<TBlobDim>( axes[0] ), inputTensor.Data, parts );
	tensors[Output[0]].Data = parts[outputBlobIndex];
}

void CSliceNode::MarkTensorDims( const CTensorCache& tensors, CDimCache& dims )
{
	if( tensors[Output[0]].Data != nullptr ) {
		return;
	}

	if( !dims[Input[0]].IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( tensors[Output[0]].Shape, dims[Input[0]], dims[Output[0]] ),
			"marking output dimensions failed", OnnxNode );
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

	const TBlobDim concatDim = ( dims[Output[0]] )[axes[0]];

	// TODO: add the rest dims support
	CheckNeoOnnxSupport( concatDim == BD_BatchLength, "concat along dim other than BD_BatchLength", OnnxNode );
	CPtr<CSubSequenceLayer> subseq = new CSubSequenceLayer( mathEngine );
	subseq->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	if( starts[0] < 0 ) {
		starts[0] += tensors[Input[0]].Shape[axes[0]];
	}

	if( ends[0] < 0 ) {
		ends[0] += tensors[Input[0]].Shape[axes[0]];
		if( starts[0] == ends[0] ) {
			++ends[0];
		}
	}

	subseq->SetStartPos( starts[0] );
	subseq->SetLength( ends[0] - starts[0] );

	subseq->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );

	dnn.AddLayer( *subseq );

	neoMLLinks[Output[0]] = CNeoMLLink( subseq, 0 );
}

} // namespace NeoOnnx
