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

static bool isInputPresent( const CTensorArray& inputs, int index )
{
	return inputs.Size() > index && inputs[index] != nullptr && !inputs[index]->IsEmpty();
}

// --------------------------------------------------------------------------------------------------------------------

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

void CResizeOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CString mode = "nearest";
	GetAttribute( "mode", mode );
	CheckNeoOnnxSupport( mode == "nearest" || mode == "linear", "mode is not 'nearest' nor 'linear'", *this );

	CPtr<CInterpolationLayer> interpolation( new CInterpolationLayer( dnn.GetMathEngine() ) );
	interpolation->SetName( Name() );

	interpolation->SetCoords( getInterpolationCoords() );
	if( mode == "nearest" ) {
		interpolation->SetRound( getInterpolationRound() );
	}

	CPtr<const CUserTensor> x = AsUserTensor( *inputs[0], Name() + "_source", dnn );
	CTensorShape outputShape;
	x->Shape().CopyTo( outputShape );

	const int scalesInputIndex = OpsetVersion == 10 ? 1 : 2;
	const int sizesInputIndex = OpsetVersion == 10 ? INT_MAX : 3;

	if( isInputPresent( inputs, scalesInputIndex ) ) {
		CheckOnnxProtocol( !isInputPresent( inputs, sizesInputIndex ), "Both 'sizes' and 'scales' are provided", *this );
		CFastArray<float, 8> scales;
		getScales( inputs, scales );
		CheckOnnxProtocol( scales.Size() == x->DimCount(), "size(scales) != rank(X)", *this );
		for( int dimIndex = 0; dimIndex < scales.Size(); ++dimIndex ) {
			interpolation->SetRule( x->Layout()[dimIndex], CInterpolationLayer::CRule::Scale( scales[dimIndex] ) );
			outputShape[dimIndex] = static_cast<int>( x->Shape()[dimIndex] * scales[dimIndex] );
		}
	} else if( isInputPresent( inputs, sizesInputIndex ) ) {
		CFastArray<int, 8> sizes;
		getSizes( inputs, sizes );
		CheckOnnxProtocol( sizes.Size() == x->DimCount(), "size(sizes) != rank(X)", *this );
		for( int dimIndex = 0; dimIndex < sizes.Size(); ++dimIndex ) {
			interpolation->SetRule( x->Layout()[dimIndex], CInterpolationLayer::CRule::Resize( sizes[dimIndex] ) );
			outputShape[dimIndex] = sizes[dimIndex];
		}
	} else {
		CheckOnnxProtocol( false, "'sizes' or 'scales' must be present", *this );
	}

	interpolation->Connect( 0, *x->Layer(), x->OutputIndex() );
	dnn.AddLayer( *interpolation );
	outputs.Add( new CUserTensor( outputShape, x->Layout(), CLayerOutput( interpolation.Ptr(), 0 ) ) );
}

TInterpolationCoords CResizeOperator::getInterpolationCoords() const
{
	if( OpsetVersion == 10 ) {
		return TInterpolationCoords::Asymmetric;
	}

	static_assert( static_cast<int>( TInterpolationCoords::Count ) == 4, "TInterpolationCoords::Count != 4" );
	CString coordMode = "half_pixel";
	GetAttribute( "coordinate_transformation_mode", coordMode );
	if( coordMode == "half_pixel" ) {
		return TInterpolationCoords::HalfPixel;
	} else if( coordMode == "pytorch_half_pixel" ) {
		return TInterpolationCoords::PytorchHalfPixel;
	} else if( coordMode == "align_corners" ) {
		return TInterpolationCoords::AlignCorners;
	} else if( coordMode == "asymmetric" ) {
		return TInterpolationCoords::Asymmetric;
	} else if( coordMode == "tf_half_pixel_for_nn" || coordMode == "tf_crop_and_resize" ) {
		CheckNeoOnnxSupport( false, "unsupported 'coordinate_transformation_mode'", *this );
	}
	CheckOnnxProtocol( false, "unknown 'coordinate_transformation_mode'", *this );
	return TInterpolationCoords::Count;
}

TInterpolationRound CResizeOperator::getInterpolationRound() const
{
	if( OpsetVersion == 10 ) {
		return TInterpolationRound::Floor;
	}

	static_assert( static_cast<int>( TInterpolationRound::Count ) == 5, "TInterpolationRound::Count != 5" );
	CString nearestMode = "round_prefer_floor";
	GetAttribute( "nearest_mode", nearestMode );
	if( nearestMode == "round_prefer_floor" ) {
		return TInterpolationRound::RoundPreferFloor;
	} else if( nearestMode == "round_prefer_ceil" ) {
		return TInterpolationRound::RoundPreferCeil;
	} else if( nearestMode == "floor" ) {
		return TInterpolationRound::Floor;
	} else if( nearestMode == "ceil" ) {
		return TInterpolationRound::Ceil;
	}
	CheckOnnxProtocol( false, "unknown 'nearest_mode'", *this );
	return TInterpolationRound::Count;
}

void CResizeOperator::getScales( const CTensorArray& inputs, CFastArray<float, 8>& scales ) const
{
	const int scalesInputIndex = OpsetVersion == 10 ? 1 : 2;
	CheckNeoOnnxSupport( inputs[scalesInputIndex]->IsCalculated(), "User-provided scales", *this );
	const CDnnBlob& scalesBlob = *( dynamic_cast<const CDataTensor*>( inputs[scalesInputIndex].Ptr() )->Data() );
	CheckOnnxProtocol( scalesBlob.GetDataType() == CT_Float, "scales are not float", *this );
	scales.SetSize( scalesBlob.GetDataSize() );
	scalesBlob.CopyTo( scales.GetPtr() );
}

void CResizeOperator::getSizes( const CTensorArray& inputs, CFastArray<int, 8>& sizes ) const
{
	NeoAssert( OpsetVersion > 10 );
	const int sizesInputIndex = 3;
	CheckNeoOnnxSupport( inputs[sizesInputIndex]->IsCalculated(), "User-provided sizes", *this );
	const CDnnBlob& sizesBlob = *( dynamic_cast<const CDataTensor*>( inputs[sizesInputIndex].Ptr() )->Data() );
	CheckOnnxProtocol( sizesBlob.GetDataType() == CT_Int, "sizes are not integer", *this );
	sizes.SetSize( sizesBlob.GetDataSize() );
	sizesBlob.CopyTo( sizes.GetPtr() );
}

} // namespace NeoOnnx
