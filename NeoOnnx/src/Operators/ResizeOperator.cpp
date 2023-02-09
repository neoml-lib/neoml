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

#include <NeoML/Dnn/Layers/Onnx/OnnxResizeLayer.h>

namespace NeoOnnx {

static bool isInputPresent( const CTensorArray& inputs, int index )
{
	const bool initialCheck = inputs.Size() > index && inputs[index] != nullptr;
	if( !initialCheck ) {
		return false;
	}
	if( inputs[index]->Type() == TTensorType::User ) {
		return true;
	} else if( inputs[index]->Type() == TTensorType::Data ) {
		const CDataTensor& dataTensor = dynamic_cast<const CDataTensor&>( *inputs[index] );
		for( int dimIndex = 0; dimIndex < dataTensor.DimCount(); ++dimIndex ) {
			if( dataTensor.DimSize( dimIndex ) == 0 ) {
				return false;
			}
		}
		return true;
	}

	const CShapeTensor& shapeTensor = dynamic_cast<const CShapeTensor&>( *inputs[index] );
	for( int dimIndex = 0; dimIndex < shapeTensor.DimCount(); ++dimIndex ) {
		if( shapeTensor.Shape()[dimIndex] == 0 ) {
			return false;
		}
	}
	return true;
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

	CPtr<COnnxResizeLayer> resizeLayer( new COnnxResizeLayer( dnn.GetMathEngine() ) );
	resizeLayer->SetName( Name() );

	resizeLayer->SetCoords( getInterpolationCoords() );
	if( mode == "nearest" ) {
		resizeLayer->SetRound( getInterpolationRound() );
	}

	CPtr<const CUserTensor> x = AsUserTensor( *inputs[0], Name() + "_source", dnn );
	resizeLayer->Connect( 0, *x->Layer(), x->OutputIndex() );
	x->Layout().CopyTo( resizeLayer->TensorLayout() );

	const int scalesInputIndex = OpsetVersion == 10 ? 1 : 2;
	const int sizesInputIndex = OpsetVersion == 10 ? INT_MAX : 3;

	if( isInputPresent( inputs, scalesInputIndex ) ) {
		CPtr<const CShapeTensor> scales = AsShapeTensor( *inputs[scalesInputIndex], Name() + "_scales", dnn );
		resizeLayer->Connect( 1, *scales->Layer(), scales->OutputIndex() );
	} else if( isInputPresent( inputs, sizesInputIndex ) ) {
		CPtr<const CShapeTensor> sizes = AsShapeTensor( *inputs[sizesInputIndex], Name() + "_sizes", dnn );
		resizeLayer->Connect( 1, *sizes->Layer(), sizes->OutputIndex() );
	} else {
		CheckOnnxProtocol( false, "'sizes' or 'scales' must be present", *this );
	}

	dnn.AddLayer( *resizeLayer );
	outputs.Add( new CUserTensor( x->Layout(), CLayerOutput( resizeLayer, 0 ) ) );
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

} // namespace NeoOnnx
