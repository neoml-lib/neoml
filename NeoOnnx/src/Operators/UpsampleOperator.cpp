/* Copyright © 2017-2024 ABBYY

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

#include <NeoML/Dnn/Layers/Onnx/OnnxResizeLayer.h>

using namespace NeoML;

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
	CheckNoNullInputs( inputs );

	CPtr<COnnxResizeLayer> onnxResize = new COnnxResizeLayer( dnn.GetMathEngine() );
	onnxResize->SetName( Name() );
	onnxResize->SetCoords( TInterpolationCoords::Asymmetric );
	if( mode == "nearest" ) {
		onnxResize->SetRound( TInterpolationRound::Floor );
	}
	inputs[0]->Layout().CopyTo( onnxResize->TensorLayout() );

	CPtr<const CUserTensor> input = AsUserTensor( *inputs[0], Name() + "_Source", dnn );
	onnxResize->Connect( 0, *input->Layer(), input->OutputIndex() );

	CPtr<const CShapeTensor> scales = getScales( inputs, dnn );
	onnxResize->Connect( 1, *scales->Layer(), scales->OutputIndex() );

	dnn.AddLayer( *onnxResize );
	outputs.Add( new CUserTensor( input->Layout(), CLayerOutput( onnxResize.Ptr(), 0 ) ) );
}

// Gets scales
CPtr<const CShapeTensor> CUpsampleOperator::getScales( const CTensorArray& inputs, CDnn& dnn ) const
{
	if( OpsetVersion < 9 ) {
		CFastArray<float, 8> scalesArray;
		if( OpsetVersion < 7 ) {
			float heightScale = 1.f;
			CheckOnnxProtocol( GetAttribute( "height_scale", heightScale ), "height_scale attribute is missing", *this );
			float widthScale = 1.f;
			CheckOnnxProtocol( GetAttribute( "width_scale", widthScale ), "width_scale attribute is missing", *this );
			scalesArray = { 1.f, 1.f, heightScale, widthScale };
		} else {
			CheckOnnxProtocol( GetAttribute( "scales", scalesArray ), "scales attribute is missing", *this );
		}
		return AsShapeTensor( scalesArray, Name() + "_Scales", dnn );
	}

	CheckNeoOnnxSupport( inputs[1]->Type() != TTensorType::User, "user-provided scales", *this );
	return AsShapeTensor( *inputs[1], Name() + "_Scales", dnn );
}

} // namespace NeoOnnx
