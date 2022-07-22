/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "CumSumOperator.h"

namespace NeoOnnx {

CCumSumOperator::CCumSumOperator( const onnx::NodeProto& cumSum, int opsetVersion ) :
	CLayerOperator( cumSum, opsetVersion )
{
	// v11 - original
	// v14 - float16 and bfloat16 data types are supported
	CheckNeoOnnxSupport( OpsetVersion >= 11 && OpsetVersion <= MaxOpsetVersion, "Opset version" );

	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	int exclusive = 0;
	GetAttribute( "exclusive", exclusive );
	CheckNeoOnnxSupport( exclusive == 0, "exclusive mode in CumSum", *this );
}

void CCumSumOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr && inputs[1] != nullptr, "inputs can't be optional", *this );
	CPtr<const CUserTensor> input = AsUserTensor( *inputs[0], Name() + "_Source", dnn );

	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided axis", *this );
	const CDnnBlob* axisBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
	CheckOnnxProtocol( axisBlob->GetDataSize() == 1, "wrong size of axis tensor", *this );
	CheckOnnxProtocol( axisBlob->GetDataType() == CT_Int, "wrong data type of axis tensor", *this );
	int axis = axisBlob->GetData<int>().GetValue();
	if( axis < 0 ) {
		axis += inputs[0]->DimCount();
	}
	CheckOnnxProtocol( axis >= 0 && axis < inputs[0]->DimCount(), "wrong axis value", *this );

	int reverse = 0;
	GetAttribute( "reverse", reverse );

	CCumSumLayer* cumSum = CumSum( inputs[0]->Layout()[axis], reverse != 0 )
		( Name(), CDnnLayerLink( input->Layer(), input->OutputIndex() ) );
	outputs.Add( new CUserTensor( input->Shape(), input->Layout(), CLayerOutput( cumSum, 0 ) ) );
}

} // namespace NeoOnnx