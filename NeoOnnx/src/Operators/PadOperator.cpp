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

#include "onnx.pb.h"

#include "PadOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

namespace NeoOnnx {

CPadOperator::CPadOperator( const onnx::NodeProto& pad, int opsetVersion ) :
	CLayerOperator( pad, opsetVersion ),
	mode( Attributes.GetOptionalString( "mode", "constant" ) ),
	value( 0.f )
{
	// In v1 pads are provided by 'paddings' attribute and pad value is provided by 'value' attribute 
	// In v2 pads are provided by 'pads' attribute and pad value is provided by 'value' attribute 
	// In v11 pads and pad value are provided by additional inputs
	if( opsetVersion < 11 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
		Attributes.GetRequiredIntArray( opsetVersion == 1 ? "paddings" : "pads", pads );
		value = Attributes.GetOptionalFloat( "value", 0.f );
	} else {
		CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "operator must have 2 or 3 inputs", *this );
	}

	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
	CheckNeoOnnxSupport( mode == "constant", "Pad with non-constant mode", *this );
}

void CPadOperator::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CDnn& /* dnn */, CObjectArray<const CTensorBase>& outputs )
{
	if( OpsetVersion >= 11 ) {
		CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided pad sizes", *this );
		const CDnnBlob& padsBlob = *( dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data() );
		CheckOnnxProtocol( padsBlob.GetDataType() == CT_Int, "non-integer pad sizes", *this );
		pads.SetSize( padsBlob.GetDataSize() );
		padsBlob.CopyTo( pads.GetPtr() );
		if( InputCount() == 3 ) {
			CheckNeoOnnxSupport( inputs[2]->IsCalculated(), "user-provided pad value", *this );
			const CDnnBlob& valueBlob = *( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data() );
			if( valueBlob.GetDataType() == CT_Float ) {
				value = valueBlob.GetData<float>().GetValue();
			} else {
				value = static_cast<float>( valueBlob.GetData<int>().GetValue() );
			}
		}
	}

	outputs[0] = PadUserTensor( dynamic_cast<const CUserTensor&>( *inputs[0] ), pads, value ).Ptr();
}

} // namespace NeoOnnx
