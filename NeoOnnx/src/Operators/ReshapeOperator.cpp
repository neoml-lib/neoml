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

#include "ReshapeOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CReshapeOperator::CReshapeOperator( const onnx::NodeProto& reshape, int opsetVersion ) :
	CLayerOperator( reshape, opsetVersion )
{
	// v1 - original
	// v5 - removed legacy optimization attribute, "shape" moved from attributes to inputs, supported new data types
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 5 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CReshapeOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	NeoAssert( inputs[0] != nullptr && !inputs[0]->IsCalculated() );

	CTensorShape outputShape;
	getShape( inputs, outputShape );

	// In order to process tensors correctly reshape is not allowed in transposed layouts
	CPtr<const CUserTensor> input;
	if( IsTransposedLayout( inputs[0]->Layout() ) ) {
		input = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0],
			CTensorLayout( inputs[0]->DimCount() ) ).Ptr() );
	} else {
		input = dynamic_cast<const CUserTensor*>( inputs[0].Ptr() );
	}
	const CTensorShape& inputShape = input->Shape();

	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( Name() );

	int tensorSize = 1;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		tensorSize *= inputs[0]->Shape()[i];
	}

	int remainder = tensorSize;
	int remainderIndex = NotFound;
	CTensorLayout outputLayout( outputShape.Size() );
	for( int dimIndex = 0; dimIndex < outputShape.Size(); ++dimIndex ) {
		CTransformLayer::CDimensionRule rule;
		if( outputShape[dimIndex] > 0 ) {
			rule = CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, outputShape[dimIndex] );
			remainder /= outputShape[dimIndex];
		} else if( outputShape[dimIndex] == 0 ) {
			rule = CTransformLayer::CDimensionRule( CTransformLayer::O_Multiply, 1 );
			outputShape[dimIndex] = inputShape[dimIndex];
			remainder /= outputShape[dimIndex];
		} else if( outputShape[dimIndex] == -1 ) {
			rule = CTransformLayer::CDimensionRule( CTransformLayer::O_Remainder, outputShape[dimIndex] );
			remainderIndex = dimIndex;
		} else {
			CheckOnnxProtocol( false, "Wrong shape value", *this );
		}
		transform->SetDimensionRule( outputLayout[dimIndex], rule );
	}

	if( remainderIndex != NotFound ) {
		outputShape[remainderIndex] = remainder;
	}

	for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
		if( outputLayout.Find( dim ) == NotFound ) {
			transform->SetDimensionRule( dim, CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
		}
	}

	transform->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *transform );

	outputs[0] = new CUserTensor( outputShape, outputLayout, CLayerOutput( transform, 0 ) );
}

// Gets output shape
void CReshapeOperator::getShape( const CTensorArray& inputs, CTensorShape& shape ) const
{
	if( OpsetVersion < 5 ) {
		CheckOnnxProtocol( GetAttribute( "shape", shape ), "'shape' attribute is missing", *this );
		return;
	}

	CheckNeoOnnxSupport( inputs[1] != nullptr && inputs[1]->IsCalculated(), "User-provided output shape", *this );
	const CDnnBlob* shapeBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
	CheckOnnxProtocol( shapeBlob->GetDataType() == CT_Int, "Non-integer shape", *this );
	shape.SetSize( shapeBlob->GetDataSize() );
	shapeBlob->CopyTo( shape.GetPtr() );
}

} // namespace NeoOnnx
