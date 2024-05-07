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

#include "LrnOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

using namespace NeoML;

namespace NeoOnnx {

CLrnOperator::CLrnOperator( const onnx::NodeProto& lrn, int opsetVersion ) :
	CLayerOperator( lrn, opsetVersion )
{
	// v1 - original
	// v13 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CLrnOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );
	CheckNeoOnnxSupport( inputs[0]->DimCount() <= 5, "6+ dimensional input", *this );

	CPtr<const CUserTensor> input = AsUserTensor( *ConvertTensor( *inputs[0], CNeoMLImageLayoutValidator() ),
		Name() + "_Source", dnn );

	CPtr<CLrnLayer> lrn = new CLrnLayer( dnn.GetMathEngine() );
	lrn->SetName( Name() );
	int size = -1;
	CheckOnnxProtocol( GetAttribute( "size", size ), "'size' attribute is missing", *this );
	lrn->SetWindowSize( size );

	float bias = 1.f;
	GetAttribute( "bias", bias );
	lrn->SetBias( bias );

	float alpha = 1e-4f;
	GetAttribute( "alpha", alpha );
	lrn->SetAlpha( alpha );

	float beta = 0.75f;
	GetAttribute( "beta", beta );
	lrn->SetBeta( beta );

	lrn->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *lrn );

	outputs.Add( new CUserTensor( input->Layout(), CLayerOutput( lrn, 0 ) ) );
}

} // namespace NeoOnnx