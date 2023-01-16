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
#include <NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h>

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
	CheckNoNullInputs( inputs );

	// In order to process tensors correctly reshape is not allowed in transposed layouts
	CPtr<const CUserTensor> input = AsUserTensor( *inputs[0], Name() + "_InputSource", dnn );
	if( IsTransposedLayout( input->Layout() ) ) {
		input = ConvertTensor( *input, CTensorLayout( input->DimCount() ) );
	}

	CPtr<const CShapeTensor> newShape = getShape( inputs, dnn );
	CheckNeoOnnxSupport( newShape->DimCount() == 1, "shape must have 1 dimension", *this );
	
	CTensorLayout outputLayout( newShape->Shape()[0] );
	CPtr<COnnxReshapeLayer> reshapeLayer = new COnnxReshapeLayer( dnn.GetMathEngine() );
	reshapeLayer->SetName( Name() );
	input->Layout().CopyTo( reshapeLayer->InputLayout() );
	outputLayout.CopyTo( reshapeLayer->OutputLayout() );
	reshapeLayer->Connect( 0, *input->Layer(), input->OutputIndex() );
	reshapeLayer->Connect( 1, *newShape->Layer(), newShape->OutputIndex() );
	dnn.AddLayer( *reshapeLayer );

	outputs.Add( new CUserTensor( outputLayout, CLayerOutput( reshapeLayer.Ptr(), 0 ) ) );
}

// Gets output shape
CPtr<const CShapeTensor> CReshapeOperator::getShape( const CTensorArray& inputs, CDnn& dnn ) const
{
	if( OpsetVersion < 5 ) {
		CTensorShape shapeArray;
		CheckOnnxProtocol( GetAttribute( "shape", shapeArray ), "'shape' attribute is missing", *this );
		return AsShapeTensor( shapeArray, Name() + "_ShapeSource", dnn );
	}

	CheckNeoOnnxSupport( inputs[1] != nullptr && inputs[1]->Type() != TTensorType::User,
		"User-provided output shape", *this );

	return AsShapeTensor( *inputs[1], Name() + "_ShapeSource", dnn );
}

} // namespace NeoOnnx
