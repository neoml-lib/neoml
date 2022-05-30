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

#include "onnx.pb.h"

#include "OneHotOperator.h"
#include "NeoOnnxCheck.h"

namespace NeoOnnx {

// Multiplies and shifts data from input (if needed)
template<class T>
static CBaseLayer* multiplyAndShift( CBaseLayer* input, const CDnnBlob& values, const CString& linearName )
{
	const float offValue = static_cast<float>( values.GetData<T>().GetValue() );
	const float onValue = static_cast<float>( values.GetData<T>().GetValueAt( 1 ) );
	if( offValue == 0.f && onValue == 1.f ) {
		// No shift needed
		return input;
	}

	return Linear( onValue - offValue, offValue )( linearName, input );
}

// --------------------------------------------------------------------------------------------------------------------

COneHotOperator::COneHotOperator( const onnx::NodeProto& oneHot, int opsetVersion ) :
	CLayerOperator( oneHot, opsetVersion )
{
	// v9 - original
	// v11 - negative axes supported
	CheckOnnxProtocol( InputCount() == 3, "operator must have 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void COneHotOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr,
		"inputs can't be optional", *this );
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided depth", *this );
	CheckNeoOnnxSupport( inputs[2]->IsCalculated(), "user-provided values", *this );

	CPtr<const CUserTensor> indices = AsUserTensor( *inputs[0], Name() + "_indices", dnn );

	// The CEnumBinarizationLayer always puts its enum into BD_Channels
	// Replace BD_Channels with anything else
	CheckNeoOnnxSupport( indices->DimCount() < 7, "OneHot with 7-dimensional input", *this );
	CTensorLayout outputLayout = indices->Layout();
	const int channelIndex = indices->Layout().Find( BD_Channels );
	if( channelIndex != NotFound ) {
		for( int dim = static_cast<int>( BD_Channels ) - 1; dim >= 0; --dim ) {
			if( outputLayout.Find( static_cast<TBlobDim>( dim ) ) == NotFound ) {
				outputLayout[channelIndex] = static_cast<TBlobDim>( dim );
				indices = ConvertTensor( *indices, outputLayout );
				break;
			}
		}
	}

	// Extracting depth value
	int depth = 0;
	const CDnnBlob* depthBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
	CheckOnnxProtocol( depthBlob->GetDataSize() == 1, "size of depth isn't equal to 1", *this );
	if( depthBlob->GetDataType() == CT_Float ) {
		depth = static_cast<int>( depthBlob->GetData().GetValue() );
	} else {
		depth = depthBlob->GetData<int>().GetValue();
	}
	CheckOnnxProtocol( depth > 0, "non-positive depth", *this );

	// The ONNX protocol says that if indices are not integer they should be casted to integer
	// But CEnumBinarizationLayer handles float indices in the same way
	// That's why we can omit the cast here
	// TODO: figure out how to support [-depth;depth-1] indices (at the moment only [0;depth-1] are supported)
	CBaseLayer* outputLayer = EnumBinarization( depth )( Name(), CDnnLayerLink( indices->Layer(), indices->OutputIndex() ) );

	// Add multiplication and shift (if onn/off values are not 1 and 0)
	const CDnnBlob* valuesBlob = dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data();
	CheckOnnxProtocol( valuesBlob->GetDataSize() == 2, "size of values isn't equal to 2", *this );
	if( valuesBlob->GetDataType() == CT_Float ) {
		outputLayer = multiplyAndShift<float>( outputLayer, *valuesBlob, Name() + "_linear" );
	} else {
		outputLayer = multiplyAndShift<int>( outputLayer, *valuesBlob, Name() + "_linear" );
	}

	// The output type of ONNX OneHot is equal to the type of values
	// The output type of CEnumBinarizationLayer is always float
	if( valuesBlob->GetDataType() == CT_Int ) {
		outputLayer = Cast( CT_Int )( Name() + "_output_cast", outputLayer );
	}

	int axis = -1;
	GetAttribute( "axis", axis );
	if( axis < 0 ) {
		axis += indices->DimCount() + 1;
	}
	outputLayout.InsertAt( BD_Channels, axis );
	CTensorShape outputShape;
	indices->Shape().CopyTo( outputShape );
	outputShape.InsertAt( depth, axis );
	outputs.Add( new CUserTensor( outputShape, outputLayout, CLayerOutput( outputLayer, 0 ) ) );
}

} // namespace NeoOnnx
