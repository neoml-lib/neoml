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

#include "proto/onnx.pb.h"

namespace NeoOnnx {

CSliceNode::CSliceNode( const onnx::NodeProto& slice ) :
	CNode( slice )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", slice );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", slice );

	attributes.GetRequiredIntArray( "starts", starts );
	attributes.GetRequiredIntArray( "ends", ends );
	attributes.GetRequiredIntArray( "axes", axes );
	// TODO: Add support for multi-axes slice.
	CheckNeoOnnxSupport( starts.Size() == 1, "starts.Size() > 1", slice );
	CheckNeoOnnxSupport( ends.Size() == 1, "ends.Size() > 1", slice );
	CheckNeoOnnxSupport( axes.Size() == 1, "axes.Size() > 1", slice );
}

void CSliceNode::CalcOutputShape()
{
	CTensor& input = InputTensor( 0 );

	CTensorShape& outputShape = output[0].Shape;
	input.Shape.CopyTo( outputShape );

	if( starts[0] < 0 ) {
		starts[0] += input.Shape[axes[0]];
	}
	if( ends[0] < 0 ) {
		ends[0] += input.Shape[axes[0]];
		if( starts[0] == ends[0] ) {
			++ends[0];
		}
	}

	outputShape[axes[0]] = ends[0] - starts[0];
}

void CSliceNode::CalcOutputData()
{
	CTensor& input = InputTensor( 0 );
	if( input.Data == nullptr ) {
		return;
	}

	IMathEngine& mathEngine = input.Data->GetMathEngine();
	TBlobType outputBlobType = input.Data->GetDataType();

	CBlobDesc desc = input.Data->GetDesc();
	int outputBlobIndex = starts[0] == 0 ? 0 : 1;
	CObjectArray<CDnnBlob> parts;

	if( starts[0] > 0 ) {
		desc.SetDimSize( axes[0], starts[0] );
		parts.Add( CDnnBlob::CreateBlob( mathEngine, outputBlobType, desc ) );
	}

	desc.SetDimSize( axes[0], ends[0] - starts[0] );
	parts.Add( CDnnBlob::CreateBlob( mathEngine, outputBlobType, desc ) );

	if( ends[0] < input.Shape[axes[0]] ) {
		desc.SetDimSize( axes[0], input.Shape[axes[0]] - ends[0] );
		parts.Add( CDnnBlob::CreateBlob( mathEngine, outputBlobType, desc ) );
	}

	CDnnBlob::SplitByDim( mathEngine, static_cast<TBlobDim>( axes[0] ), input.Data, parts );
	output[0].Data = parts[outputBlobIndex];
}

void CSliceNode::MarkTensorDims()
{
	if( output[0].Data != nullptr ) {
		return;
	}

	if( !InputTensor( 0 ).Dim.IsEmpty() ) {
		CheckNeoOnnxInternal( output[0].SetTensorDim( InputTensor( 0 ).Dim ),
			"marking output dimensions failed", onnxNode );
	}

	if( !output[0].Dim.IsEmpty() ) {
		CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( output[0].Dim ),
			"marking input dimensions failed", onnxNode );
	}
}

void CSliceNode::AddLayers( CDnn& dnn )
{
	if( output[0].Data != nullptr ) {
		return;
	}

	IMathEngine& mathEngine = dnn.GetMathEngine();

	const TBlobDim concatDim = output[0].Dim[axes[0]];

	// TODO: add the rest dims support
	CheckNeoOnnxSupport( concatDim == BD_BatchLength, "concat along dim other than BD_BatchLength", onnxNode );
	CPtr<CSubSequenceLayer> subseq = new CSubSequenceLayer( mathEngine );
	subseq->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	if( starts[0] < 0 ) {
		starts[0] += InputTensor( 0 ).Shape[axes[0]];
	}

	if( ends[0] < 0 ) {
		ends[0] += InputTensor( 0 ).Shape[axes[0]];
		if( starts[0] == ends[0] ) {
			++ends[0];
		}
	}

	subseq->SetStartPos( starts[0] );
	subseq->SetLength( ends[0] - starts[0] );

	subseq->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );

	dnn.AddLayer( *subseq );

	neoMLInputInfo.Add( CNeoMLInputInfo( subseq, 0 ) );
}

} // namespace NeoOnnx
