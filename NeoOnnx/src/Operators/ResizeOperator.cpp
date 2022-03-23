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

#include <cfloat>
#include <cmath>

#include "onnx.pb.h"

#include "ResizeOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

namespace NeoOnnx {

CResizeOperator::CResizeOperator( const onnx::NodeProto& resize, int opsetVersion ) :
	CLayerOperator( resize, opsetVersion )
{
	// In v10 it is an equivalent to Upsample-v9
	// In v11 it drastically increases the capability of this operand (and completely changes the attributes)
	// It supports roi's, configurable extrapolation, different rounding modes and others
	// In v13 'roi' input becomes optional, 'tf_half_pixel_for_nn' is not supported as 'coordinate_transformation_mode'
	// and bfloat16 is supported as data type

	CheckOnnxProtocol( opsetVersion >= 10, "Resize operator is available since opset v10", *this );
	const int minInputCount = opsetVersion == 10 ? 2 : ( opsetVersion < 13 ? 3 : 1 );
	const int maxInputCount = opsetVersion == 10 ? 2 : 4;
	CheckOnnxProtocol( InputCount() >= minInputCount && InputCount() <= maxInputCount, "Wrong number of inputs", *this );
}

// Checks the fact that this resize operator is an equivalent of upsample operator
void CResizeOperator::checkIfUpsample() const
{
	CString mode = "nearest";
	GetAttribute( "mode", mode );
	CheckNeoOnnxSupport( mode == "nearest", "Not 'nearest' mode", *this );
	if( OpsetVersion >= 11 ) {
		CString coordinateTransformationMode = "half_pixel";
		GetAttribute( "coordinate_transformation_mode", coordinateTransformationMode );
		CheckNeoOnnxSupport( coordinateTransformationMode == "asymmetric",
			"Not 'assymetric' coordinate transformation", *this );
	}
}

void CResizeOperator::AddLayers( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	// In NeoOnnx this operator is supported only as equivalent to Upsample
	// Other scenraios are not supported
	checkIfUpsample();

	CFastArray<int, 8> scales;
	getScales( inputs, scales );

	CTensorShape outputShape;
	inputs[0]->Shape().CopyTo( outputShape );
	CheckNeoOnnxSupport( outputShape.Size() == scales.Size(), "Size of 'scales' != number of input dims", *this );
	for( int i = 0; i < outputShape.Size(); ++i ) {
		outputShape[i] *= scales[i];
	}
	outputs.Add( BroadcastTensor( *inputs[0], CBroadcast( BT_Upsample ), outputShape ) );
}

void CResizeOperator::getScales( const CTensorArray& inputs, CFastArray<int, 8>& scales ) const
{
	const int scalesInputIndex = OpsetVersion == 10 ? 1 : 2;
	CheckNeoOnnxSupport( inputs.Size() > scalesInputIndex && inputs[scalesInputIndex] != nullptr,
		"Resize without scales", *this );
	CheckNeoOnnxSupport( inputs[scalesInputIndex]->IsCalculated(), "User-provided scales", *this );
	CFastArray<float, 8> floatScales;
	const CDnnBlob& scalesBlob = *( dynamic_cast<const CDataTensor*>( inputs[scalesInputIndex].Ptr() )->Data() );
	floatScales.SetSize( scalesBlob.GetDataSize() );
	scalesBlob.CopyTo( floatScales.GetPtr() );

	scales.SetSize( floatScales.Size() );
	for( int i = 0; i < floatScales.Size(); ++i ) {
		CheckNeoOnnxSupport( IsInteger( floatScales[i] ), "non-integer scale", *this );
		scales[i] = static_cast<int>( floatScales[i] );
		CheckOnnxProtocol( scales[i] >= 1, "scale < 1", *this );
	}
}

} // namespace NeoOnnx
