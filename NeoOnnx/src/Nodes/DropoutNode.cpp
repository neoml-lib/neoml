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

#include "DropoutNode.h"
#include "GraphCache.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CDropoutNode::CDropoutNode( int nodeIndex, const onnx::NodeProto& dropout, int opsetVersion ) :
	COpNode( nodeIndex, dropout, opsetVersion ),
	ratio( Attributes.GetOptionalFloat( "ratio", 0.5f ) )
{
	// The differences between versions are in legacy optimization flags and is_test flags
	// is_test flag doesn't have any sense in NeoML anyway
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", dropout );

	if( OpsetVersion < 12 ) {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", dropout );
	} else {
		CheckOnnxProtocol( InputCount() >= 1 || InputCount() <= 3, "node must have from 1 up to 3 inputs", dropout );
	}
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "node must have 1 output", dropout );

	// NeoML doesn't support mask output (second output)
	// But we don't restrict dropout to only have one output
	// because there are examples where dropout has 2 outputs and only one of them is used
}

void CDropoutNode::CalcOutputTensors( CTensorCache& tensors, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxSupport( tensors[Input[0]].Data == nullptr, "output pre-calculation", OnnxNode );
	tensors[Input[0]].Shape.CopyTo( tensors[Output[0]].Shape );
}

void CDropoutNode::LabelTensorDims( const CTensorCache& tensors, CDimCache& dims )
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

void CDropoutNode::AddLayers( const CGraph& /* graph */, const CTensorCache& /* tensors */, const CDimCache& /* dims */,
	CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CDropoutLayer> dropout = new CDropoutLayer( dnn.GetMathEngine() );
	dropout->SetName( Name );
	dropout->SetDropoutRate( ratio );

	dropout->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	
	dnn.AddLayer( *dropout );

	neoMLLinks[Output[0]] = CNeoMLLink( dropout, 0 );
}

} // namespace NeoOnnx
