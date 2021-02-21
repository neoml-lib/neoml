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

#include "ReshapeNode.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CReshapeNode::CReshapeNode( const onnx::NodeProto& reshape, int opsetVersion ) :
	COpNode( reshape, opsetVersion )
{
	// v1 - original
	// v5 - removed legacy optimization attribute, "shape" moved from attributes to inputs, supported new data types
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", reshape );

	if( OpsetVersion < 5 ) {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", reshape );
	} else {
		CheckOnnxProtocol( InputCount() == 2, "node must have 2 inputs", reshape );
	}
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", reshape );
}

void CReshapeNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CheckNeoOnnxInternal( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "unknown input", OnnxNode );

	CTensorShape outputShape;
	getShape( inputs, outputShape );

	// In order to process tensors correctly reshape is allowed only in DT_Onnx
	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], CTensorLayout() ).Ptr() );
	const CTensorShape& inputShape = input->Shape();

	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( Name() );

	int tensorSize = 1;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		tensorSize *= inputs[0]->Shape()[i];
	}

	int remainder = tensorSize;
	int remainderIndex = NotFound;
	for( int dim = 0; dim < outputShape.Size(); ++dim ) {
		CTransformLayer::CDimensionRule rule;
		if( outputShape[dim] > 0 ) {
			rule = CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, outputShape[dim] );
			remainder /= outputShape[dim];
		} else if( outputShape[dim] == 0 ) {
			rule = CTransformLayer::CDimensionRule( CTransformLayer::O_Multiply, 1 );
			outputShape[dim] = inputShape[dim];
			remainder /= outputShape[dim];
		} else if( outputShape[dim] == -1 ) {
			rule = CTransformLayer::CDimensionRule( CTransformLayer::O_Remainder, outputShape[dim] );
			remainderIndex = dim;
		} else {
			CheckOnnxProtocol( false, "Wrong shape value", OnnxNode );
		}
		transform->SetDimensionRule( static_cast<TBlobDim>( dim ), rule );
	}
	if( remainderIndex != NotFound ) {
		outputShape[remainderIndex] = remainder;
	}

	for( int dim = outputShape.Size(); dim < BD_Count; ++dim ) {
		transform->SetDimensionRule( static_cast<TBlobDim>( dim ),
			CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
	}

	transform->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *transform );

	outputs[0] = new CUserTensor( outputShape, CTensorLayout(), CLayerOutput( transform, 0 ) );
}

// Gets output shape
void CReshapeNode::getShape( const CObjectArray<const CTensorBase>& inputs, CTensorShape& shape )
{
	if( OpsetVersion < 5 ) {
		Attributes.GetRequiredIntArray( "shape", shape );
		return;
	}

	CheckNeoOnnxSupport( inputs[1] != nullptr && inputs[1]->IsCalculated(), "User-provided output shape", OnnxNode );
	const CDnnBlob* shapeBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
	CheckOnnxProtocol( shapeBlob->GetDataType() == CT_Int, "Non-integer shape", OnnxNode );
	shape.SetSize( shapeBlob->GetDataSize() );
	shapeBlob->CopyTo( shape.GetPtr() );
}

} // namespace NeoOnnx
