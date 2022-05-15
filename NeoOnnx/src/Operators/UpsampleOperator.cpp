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

#include "UpsampleOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

namespace NeoOnnx {

CUpsampleOperator::CUpsampleOperator( const onnx::NodeProto& upsample, int opsetVersion ) :
	CLayerOperator( upsample, opsetVersion ),
	mode( "nearest" )
{
	// In v1 it works only with 4-dimensional input and upsamples only 3rd and 4th dimensions
	// In v7 it supports N-dimensional tensors and upsamples any of the dimensions
	// In v9 scales are provided by additional input instead of the attribute
	// Deprecated since v10
	if( opsetVersion < 9 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inptus", *this );
	}

	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
	GetAttribute( "mode", mode );
	CheckNeoOnnxSupport( mode == "nearest" || mode == "linear", "Upsample with non-nearest and non-linear mode", *this);
}

void CUpsampleOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CFastArray<float, 8> scales;
	getScales( inputs, scales );
	CTensorShape outputShape;
	inputs[0]->Shape().CopyTo( outputShape );
	CheckOnnxProtocol( outputShape.Size() == scales.Size(), "number of scales must be equal to the input dimensions", *this );
	for( int i = 0; i < outputShape.Size(); ++i ) {
		outputShape[i] = static_cast<int>( outputShape[i] * scales[i] );
		CheckOnnxProtocol( outputShape[i] >= 1, "empty shape after scaling", *this );
	}

	CPtr<CInterpolationLayer> interpolation = new CInterpolationLayer( dnn.GetMathEngine() );
	interpolation->SetName( Name() );
	interpolation->SetCoords( TInterpolationCoords::Asymmetric );
	if( mode == "nearest" ) {
		interpolation->SetRound( TInterpolationRound::Floor );
	}

	CPtr<const CUserTensor> input = AsUserTensor( *inputs[0], Name() + "_Source", dnn );
	for( int i = 0; i < scales.Size(); ++i ) {
		interpolation->SetRule( input->Layout()[i], CInterpolationLayer::CRule::Scale( scales[i] ) );
	}
	interpolation->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *interpolation );
	outputs.Add( new CUserTensor( outputShape, input->Layout(), CLayerOutput( interpolation.Ptr(), 0 ) ) );
}

// Gets scales
void CUpsampleOperator::getScales( const CTensorArray& inputs, CFastArray<float, 8>& scales ) const
{
	if( OpsetVersion < 7 ) {
		float heightScale = 1.f;
		CheckOnnxProtocol( GetAttribute( "height_scale", heightScale ), "height_scale attribute is missing", *this );
		float widthScale = 1.f;
		CheckOnnxProtocol( GetAttribute( "width_scale", widthScale ), "width_scale attribute is missing", *this );
		scales = { 1.f, 1.f, heightScale, widthScale };
	} else if( OpsetVersion < 9 ) {
		CheckOnnxProtocol( GetAttribute( "scales", scales ), "scales attribute is missing", *this );
	} else {
		CheckOnnxProtocol( inputs[1] != nullptr, "scales can't be optional", *this );
		CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided scales", *this );
		const CDnnBlob& scalesBlob = *( dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data() );
		CheckOnnxProtocol( scalesBlob.GetDataType() == CT_Float, "non-float scales", *this );
		scales.SetSize( scalesBlob.GetDataSize() );
		scalesBlob.CopyTo( scales.GetPtr() );
	}
}

} // namespace NeoOnnx
