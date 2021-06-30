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

#include "ConstantOfShapeOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CConstantOfShapeOperator::CConstantOfShapeOperator( const onnx::NodeProto& constantOfShape, int opsetVersion ) :
	COperator( constantOfShape, opsetVersion )
{
	// v9 - original
	CheckOnnxProtocol( OpsetVersion >= 9, "wrong opset version", *this );
	CheckNeoOnnxSupport( OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CConstantOfShapeOperator::ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNeoOnnxSupport( inputs[0] != nullptr && inputs[0]->IsCalculated(), "user-provided input", *this );
	const CDnnBlob* inputShapeBlob = dynamic_cast<const CDataTensor*>( inputs[0].Ptr() )->Data();
	CheckNeoOnnxSupport( inputShapeBlob->GetDataType() == CT_Int, "non-integer input tensor", *this );
	IMathEngine& mathEngine = dnn.GetMathEngine();

	// If "value" attribute is not set then float 0.f is assumed
	CPtr<const CDnnBlob> valueBlob;
	if( HasAttribute( "value" ) ) {
		CPtr<CDataTensor> dataTensor( new CDataTensor( mathEngine ) );
		CheckOnnxProtocol( GetAttribute( "value", dataTensor ), "'value' attribute is missing" );
		valueBlob = dataTensor->Data();
	} else {
		CPtr<CDnnBlob> zero = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
		zero->Clear();
		valueBlob = zero;
	}

	// Getting output shape from blob
	CTensorShape outputShape;
	outputShape.SetSize( inputShapeBlob->GetDataSize() );
	inputShapeBlob->CopyTo( outputShape.GetPtr() );

	// Generating output blob
	CTensorLayout outputLayout( outputShape.Size() );
	CBlobDesc outputBlobDesc( valueBlob->GetDataType() );
	for( int i = 0; i < outputShape.Size(); ++i ) {
		outputBlobDesc.SetDimSize( outputLayout[i], outputShape[i] );
	}
	CPtr<CDnnBlob> outputBlob = CDnnBlob::CreateBlob( mathEngine, valueBlob->GetDataType(), outputBlobDesc );
	if( outputBlob->GetDataType() == CT_Float ) {
		outputBlob->Fill( valueBlob->GetData().GetValue() );
	} else {
		outputBlob->Fill<int>( valueBlob->GetData<int>().GetValue() );
	}
	
	outputs[0] = new CDataTensor( outputShape, outputLayout, *outputBlob );
}

} // namespace NeoOnnx
