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

#include "TransposeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CTransposeNode::CTransposeNode( const onnx::NodeProto& transpose, int opsetVersion ) :
	CLayerOpNode( transpose, opsetVersion )
{
	// The differences between versions are in supported data types and legacy optimization attributes
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", *this );
}

void CTransposeNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	NeoAssert( inputs[0] != nullptr && !inputs[0]->IsCalculated() );

	const CTensorShape& inputShape = inputs[0]->Shape();
	const int dimCount = inputShape.Size();

	CFastArray<int, 8> perm;
	Attributes.GetOptionalIntArray( "perm", perm );
	if( perm.IsEmpty() ) {
		// Default value is reverse order
		perm.SetBufferSize( dimCount );
		for( int i = 0; i < dimCount; ++i ) {
			perm.Add( dimCount - 1 - i );
		}
	}

	// Working only with layout (converters will be added by next layers when needed)
	const CTensorLayout& inputLayout = inputs[0]->Layout();
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( dimCount );
	CTensorShape outputShape;
	outputShape.SetBufferSize( dimCount );

	for( int i = 0; i < dimCount; ++i ) {
		outputLayout.Add( inputLayout[perm[i]] );
		outputShape.Add( inputShape[perm[i]] );
	}

	outputs[0] = new CUserTensor( outputShape, outputLayout,
		dynamic_cast<const CUserTensor*>( inputs[0].Ptr() )->LayerOutput() );
}

} // namespace NeoOnnx
