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

#include "LrnOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CLrnOperator::CLrnOperator( const onnx::NodeProto& lrn, int opsetVersion ) :
	CLayerOperator( lrn, opsetVersion )
{
	// Introduce in opset v1
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CLrnOperator::ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CUserInputMask inputMask;
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		inputMask |= inputIndex;
	}
	CLayerOperator::ProcessTensors( inputMask, inputs, dnn, outputs );
}

void CLrnOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	NeoAssert( inputs[0] != nullptr && !inputs[0]->IsCalculated() );
	CheckNeoOnnxSupport( inputs[0]->DimCount() <= 5, "6+ dimensional input", *this );

	CTensorLayout outputLayout( { BD_BatchWidth, BD_Channels, BD_Height, BD_Width, BD_Depth } );
	outputLayout.SetSize( inputs[0]->DimCount() );

	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], outputLayout ).Ptr() );

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

	outputs[0] = new CUserTensor( input->Shape(), input->Layout(), CLayerOutput( lrn, 0 ) );
}

} // namespace NeoOnnx