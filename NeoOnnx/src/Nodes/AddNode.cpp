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
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", add );

	CheckOnnxProtocol( InputCount() == 2, "node must have 2 inputs", add );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", add );
}

void CAddNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	bool canBeCalculated = true;

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		canBeCalculated = canBeCalculated && ( tensors[Input[inputIndex]].Data != nullptr );
	}

	if( canBeCalculated ) {
		tensors[Output[0]].Data = tensors[Input[0]].Data->GetCopy();
		tensors[Output[0]].Data->Add( tensors[Input[1]].Data );
	}

	CTensorShape& outputShape = tensors[Output[0]].Shape;
	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		const CTensorShape& inputShape = tensors[Input[inputIndex]].Shape;

		if( outputShape.IsEmpty() ) {
			inputShape.CopyTo( outputShape );
		} else {
			// NeoML doesn't support numpy-style tensor broadcasting
			CheckNeoOnnxSupport( outputShape.Size() == inputShape.Size(), "tensor broadcasting", OnnxNode );
			for( int i = 0; i < inputShape.Size(); ++i ) {
				CheckNeoOnnxSupport( outputShape[i] == inputShape[i], "tensor broadcasting", OnnxNode );
			}
		}
	}
}

void CAddNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
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

void CAddNode::AddLayers( const CGraph& /* graph */, const CTensorCache& /* tensors */, const CDimCache& /* dims */,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CEltwiseSumLayer> addLayer = new CEltwiseSumLayer( mathEngine );
	addLayer->SetName( "NeoMLLayer" + Str( dnn.GetLayerCount() ) );

	addLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	addLayer->Connect( 1, *neoMLLinks[Input[1]].Layer, neoMLLinks[Input[1]].OutputIndex );

	dnn.AddLayer( *addLayer );

	neoMLLinks[Output[0]] = CNeoMLLink( addLayer, 0 );
}

} // namespace NeoOnnx
