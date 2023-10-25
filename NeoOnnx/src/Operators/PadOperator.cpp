/* Copyright © 2017-2023 ABBYY Production LLC

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
	mode( "constant" )
{
	// In v1 pads are provided by 'paddings' attribute and pad value is provided by 'value' attribute 
	// In v2 pads are provided by 'pads' attribute and pad value is provided by 'value' attribute 
	// In v11 pads and pad value are provided by additional inputs instead of node attributes
	// In v13 bfloat16 is supported
	// In v18 axes input is added
	// In v19 wrap mode is added
	if( opsetVersion < 11 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else if( opsetVersion < 18 ) {
		CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "operator must have 2 or 3 inputs", *this );
	} else {
		CheckOnnxProtocol( InputCount() >= 2 && InputCount() <= 4, "operator must have from 2 up to 4 inputs", *this );
	}

	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	GetAttribute( "mode", mode );
}

void CPadOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoShapeInputs( inputs );
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CFastArray<int, 8> pads;
	getPads( inputs, pads );
	const float value = getPadValue( inputs );
	TBlobResizePadding padding = TBlobResizePadding::Constant;
	if( mode == "edge" ) {
		padding = TBlobResizePadding::Edge;
	} else if( mode == "reflect" ) {
		padding = TBlobResizePadding::Reflect;
	} else {
		CheckOnnxProtocol( mode == "constant", "Unknown padding mode", *this );
	}
	outputs.Add( PadUserTensor( *AsUserTensor( *inputs[0], Name() + "_Source", dnn ), pads, padding, value ).Ptr() );
}

// Gets pads sizes
void CPadOperator::getPads( const CTensorArray& inputs, CFastArray<int, 8>& pads ) const
{
	if( OpsetVersion < 11 ) {
		const CString padAttributeName = OpsetVersion == 1 ? "paddings" : "pads";
		CheckOnnxProtocol( GetAttribute( padAttributeName, pads ),
			"'pads' attribute is missing", *this );
	} else {
		CheckNeoOnnxSupport( inputs[1]->Type() == TTensorType::Data, "user-provided pad sizes", *this );
		const CDnnBlob& padsBlob = *( dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data() );
		CheckOnnxProtocol( padsBlob.GetDataType() == CT_Int, "non-integer pad sizes", *this );
		pads.SetSize( padsBlob.GetDataSize() );
		padsBlob.CopyTo( pads.GetPtr() );

		// Since v18 there's an optional 4th input which provides axes indices affected by padding
		if( OpsetVersion >= 18 && inputs.Size() >= 4 && inputs[3] != nullptr ) {
			const int inputRank = inputs[0]->DimCount();

			CFastArray<int, 8> axes;
			CheckNeoOnnxSupport( inputs[3]->Type() == TTensorType::Data, "user-provided axes", *this );
			const CDnnBlob& axesBlob = *( dynamic_cast<const CDataTensor*>( inputs[3].Ptr() )->Data() );
			axes.SetSize( axesBlob.GetDataSize() );
			axesBlob.CopyTo( axes.GetPtr() );
			CheckOnnxProtocol( axes.Size() * 2 == pads.Size(), "pads must contain 2 * axes elements", *this );
			// Now we emulate old behavior by distributing pads along given dimensions
			// As a result we'll get an array of size 2 * inputRank where pads[i] and pads[i + inputRank]
			// are paddings in the beginning and in the end of i'th axis of tensor
			CFastArray<int, 8> unsortedPads;
			pads.MoveTo( unsortedPads );
			pads.Add( 0, 2 * inputRank );
			for( int i = 0; i < axes.Size(); ++i ) {
				const int axis = axes[i] >= 0 ? axes[i] : axes[i] + inputRank;
				CheckOnnxProtocol( axis >= 0 && axis < inputRank, "axes must be in [-inputRank;inputRank-1]", *this );
				pads[axis] = unsortedPads[i];
				pads[axis + inputRank] = unsortedPads[i + axes.Size()];
			}
		}
	}
}

// Gets value which is used to fill paddings
float CPadOperator::getPadValue( const CTensorArray& inputs ) const
{
	float padValue = 0.f;
	if( OpsetVersion < 11 ) {
		GetAttribute( "value", padValue );
	} else if( InputCount() == 3 ) {
		CheckNeoOnnxSupport( inputs[2]->Type() == TTensorType::Data, "user-provided pad value", *this );
		const CDnnBlob& valueBlob = *( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data() );
		if( valueBlob.GetDataType() == CT_Float ) {
			padValue = valueBlob.GetData<float>().GetValue();
		} else {
			padValue = static_cast<float>( valueBlob.GetData<int>().GetValue() );
		}
	}
	return padValue;
}

} // namespace NeoOnnx
