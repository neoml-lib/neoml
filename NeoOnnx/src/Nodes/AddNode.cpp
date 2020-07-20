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

CAddNode::CAddNode( int nodeIndex, const onnx::NodeProto& add, int opsetVersion ) :
	COpNode( nodeIndex, add, opsetVersion )
{
	// The differences between versions are in broadcasting flags and support
	// NeoOnnx doesn't support tensor broadcast anyway
	CheckNeoOnnxSupport( opsetVersion >= 1 && opsetVersion <= MaxOpsetVersion, "opset version", add );

	CheckOnnxProtocol( InputCount() == 2, "node must have 2 inputs", add );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", add );
}

void CAddNode::CalcOutputTensors( CGraphTensors& tensors, IMathEngine& mathEngine )
{
	bool canBeCalculated = true;

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		canBeCalculated = canBeCalculated && ( InputTensor( tensors, inputIndex ).Data != nullptr );
	}

	if( canBeCalculated ) {
		OutputTensor( tensors, 0 ).Data = InputTensor( tensors, 0 ).Data->GetCopy();
		OutputTensor( tensors, 0 ).Data->Add( InputTensor( tensors, 1 ).Data );
	}

	CTensorShape& outputShape = OutputTensor( tensors, 0 ).Shape;
	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		const CTensorShape& inputShape = InputTensor( tensors, inputIndex ).Shape;

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

void CAddNode::MarkTensorDims( const CGraphTensors& tensors, CGraphDims& dims )
{
	if( !InputDim( dims, 0 ).IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( OutputTensor( tensors, 0 ).Shape, InputDim( dims, 0 ), OutputDim( dims, 0 ) ),
			"marking output dimensions failed", onnxNode );
	}

	if( !OutputDim( dims, 0 ).IsEmpty() ) {
		CheckNeoOnnxInternal( SetTensorDim( InputTensor( tensors, 0 ).Shape, OutputDim( dims, 0 ), InputDim( dims, 0 ) ), 
			"marking input dimensions failed", onnxNode );
	}
}

void CAddNode::AddLayers( const CGraph& graph, const CGraphTensors& tensors, const CGraphDims& dims, CGraphMappings& mappings, CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CEltwiseSumLayer> addLayer = new CEltwiseSumLayer( mathEngine );
	addLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	addLayer->Connect( 0, *InputMapping( mappings, 0 ).Layer, InputMapping( mappings, 0 ).OutputIndex );
	addLayer->Connect( 1, *InputMapping( mappings, 1 ).Layer, InputMapping( mappings, 1 ).OutputIndex );

	dnn.AddLayer( *addLayer );

	OutputMapping( mappings, 0 ) = CNeoMLMapping( addLayer, 0 );
}

} // namespace NeoOnnx
