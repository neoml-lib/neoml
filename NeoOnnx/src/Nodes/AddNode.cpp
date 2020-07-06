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

#include "AddNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CAddNode::CAddNode( const onnx::NodeProto& add, int opsetVersion, IMathEngine& /*mathEngine*/ ) :
	CNode( add, opsetVersion )
{
	// The differences between versions are in broadcasting flags and support
	// NeoOnnx doesn't support tensor broadcast anyway
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", add );

	CheckOnnxProtocol( input.Size() == 2, "node must have 2 inputs", add );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", add );
}

void CAddNode::CalcOutputShape()
{
	bool canBeCalculated = true;

	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
		canBeCalculated = canBeCalculated && ( InputTensor( inputIndex ).Data != nullptr );
	}

	if( canBeCalculated ) {
		output[0].Data = InputTensor( 0 ).Data->GetCopy();
		output[0].Data->Add( InputTensor( 1 ).Data );
	}
}

void CAddNode::CalcOutputData()
{
	CTensorShape& outputShape = output[0].Shape;

	for( int inputIndex = 0; inputIndex < input.Size(); ++inputIndex ) {
		const CTensorShape& inputShape = InputTensor( inputIndex ).Shape;

		if( outputShape.IsEmpty() ) {
			inputShape.CopyTo( outputShape );
		} else {
			// NeoML doesn't support numpy-style tensor broadcasting...
			CheckNeoOnnxSupport( outputShape.Size() == inputShape.Size(), "tensor broadcasting", onnxNode );
			for( int i = 0; i < inputShape.Size(); ++i ) {
				CheckNeoOnnxSupport( outputShape[i] == inputShape[i], "tensor broadcasting", onnxNode );
			}
		}
	}
}

void CAddNode::MarkTensorDims()
{
	if( !InputTensor( 0 ).Dim.IsEmpty() ) {
		CheckNeoOnnxInternal( output[0].SetTensorDim( InputTensor( 0 ).Dim ),
			"marking output dimensions failed", onnxNode );
	}

	if( !output[0].Dim.IsEmpty() ) {
		CheckNeoOnnxInternal( InputTensor( 0 ).SetTensorDim( output[0].Dim ), 
			"marking input dimensions failed", onnxNode );
	}
}

void CAddNode::AddLayers( CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CEltwiseSumLayer> addLayer = new CEltwiseSumLayer( mathEngine );
	addLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	addLayer->Connect( 0, InputLayer( 0 ), InputLayerIndex( 0 ) );
	addLayer->Connect( 1, InputLayer( 1 ), InputLayerIndex( 1 ) );

	dnn.AddLayer( *addLayer );

	neoMLInputInfo.Add( CNeoMLInputInfo( addLayer, 0 ) );
}

} // namespace NeoOnnx
