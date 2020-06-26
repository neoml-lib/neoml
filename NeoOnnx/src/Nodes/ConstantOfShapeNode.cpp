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

#include "ConstantOfShapeNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConstantOfShapeNode::CConstantOfShapeNode( const onnx::NodeProto& constantOfShape, CMap<CString, CInputInfo>& nodeOutputs ) :
	CNode( constantOfShape, nodeOutputs )
{
	CheckOnnxProtocol( input.Size() == 1, "node must have 1 input", constantOfShape );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", constantOfShape );
}

void CConstantOfShapeNode::OnnxReshape()
{
	CheckNeoOnnxSupport( InputTensor( 0 ).GetType() == TT_ConstantTensor,
		"non-constant input tensor", onnxNode );
	CheckNeoOnnxSupport( InputTensor( 0 ).GetData()->GetDataType() == CT_Int,
		"non-integer input tensor", onnxNode );

	IMathEngine& mathEngine = InputTensor( 0 ).GetData()->GetMathEngine();
	CPtr<CDnnBlob> value = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
	value->Clear();
	attributes.GetOptionalTensor( "value", *value );

	CBlobDesc outputBlobDesc( value->GetDataType() );
	CTensorShape outputShape;
	outputShape.SetSize( InputTensor( 0 ).GetData()->GetDataSize() );
	InputTensor( 0 ).GetData()->CopyTo( outputShape.GetPtr() );

	for( int dimIndex = 0; dimIndex < outputShape.Size(); ++dimIndex ) {
		outputBlobDesc.SetDimSize( dimIndex, outputShape[dimIndex] );
	}

	CPtr<CDnnBlob> outputBlob = CDnnBlob::CreateBlob( mathEngine, value->GetDataType(), outputBlobDesc );
	if( outputBlob->GetDataType() == CT_Float ) {
		outputBlob->Fill( value->GetData().GetValue() );
	} else {
		outputBlob->Fill<int>( value->GetData<int>().GetValue() );
	}

	outputData.Add( CTensor( TT_ConstantTensor, outputShape, outputBlob.Ptr() ) );
}

} // namespace NeoOnnx
