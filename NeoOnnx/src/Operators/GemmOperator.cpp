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

#include "GemmOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGemmOperator::CGemmOperator( const onnx::NodeProto& gemm, int opsetVersion ) :
	CLayerOperator( gemm, opsetVersion ),
	alpha( 1.f ),
	beta( 1.f ),
	transA( 0 ),
	transB( 0 )
{
	// Older versions have broadcast support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "operator must have 2 or 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	GetAttribute( "alpha", alpha );
	CheckNeoOnnxSupport( alpha == 1.0f, "alpha != 1", *this );

	GetAttribute( "beta", beta );
	CheckNeoOnnxSupport( beta == 1.0f, "beta != 1", *this );

	GetAttribute( "transA", transA );
	CheckNeoOnnxSupport( transA == 0, "transA != 0", *this );

	GetAttribute( "transB", transB );
	CheckNeoOnnxSupport( transB != 0, "transB == 0", *this );

	if( OpsetVersion < 7 ) {
		int broadcast = 0;
		GetAttribute( "broadcast", broadcast );
		CheckNeoOnnxSupport( broadcast != 0, "broadcast == 0", *this );
	}
}

void CGemmOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr && inputs[1] != nullptr, "input can't be optional", *this );

	const CTensorShape& inputShape = inputs[0]->Shape();
	CheckNeoOnnxSupport( transA == 0, "transA != 0", *this );
	// Some models from the model zoo has this op with 4-dimensional input
	// e.g. 1 x 512 x 7 x 7 and this input is interpreted as matrix 1 x 25088
	// The documentation does mention 'input matrix' but doesn't clarify what needs to be done when input is N-dimensional
	// Thats why we're heuristically trying to process this input as matrix dim[0] x (dim[1] * dim[2] * ...)
	int inputObjectSize = inputShape[1];
	for( int i = 2; i < inputShape.Size(); ++i ) {
		inputObjectSize *= inputShape[i];
	}

	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided weights", *this );
	const CTensorShape& matrixShape = inputs[1]->Shape();
	CheckOnnxProtocol( matrixShape.Size() == 2, "weights must be 2-dimensional", *this );
	CheckOnnxProtocol( matrixShape[transB == 0 ? 0 : 1] == inputObjectSize, "wrong weight size", *this );
	const int numberOfElements = matrixShape[transB == 0 ? 1 : 0];

	if( InputCount() == 3 ) {
		CheckNeoOnnxSupport( inputs[2]->IsCalculated(), "user-provided bias", *this );
		const CTensorShape& biasShape = inputs[2]->Shape();
		CheckOnnxProtocol( biasShape.Size() == 1, "bias must be 1-dimensional", *this );
		CheckOnnxProtocol( biasShape[0] == numberOfElements, "wrong bias size", *this );
	}

	CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( dnn.GetMathEngine() );
	fc->SetName( Name() );

	fc->SetNumberOfElements( numberOfElements );

	const CTensorLayout fcLayout( { BD_BatchWidth, BD_Channels } );

	CPtr<const CTensorBase> matrixTensor = ConvertTensor( *inputs[1], fcLayout );
	fc->SetWeightsData( dynamic_cast<const CDataTensor*>( matrixTensor.Ptr() )->Data()->GetCopy() );

	if( InputCount() > 2 ) {
		fc->SetFreeTermData( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data()->GetCopy() );
	} else {
		fc->SetZeroFreeTerm( true );
	}

	CTensorLayout inputLayout = inputShape.Size() == 2 ? fcLayout : inputs[0]->Layout();
	if( inputShape.Size() > 2 ) {
		// Build input layout for N-dimensional input

		// We need this in order to guarantee that output will be in { BD_BatchWidth, BD_Channels } layout
		inputLayout[0] = BD_BatchWidth;
		for( int i = 1; i < inputLayout.Size(); ++i ) {
			// BD_Height, BD_Width etc...
			inputLayout[i] = BD_ListSize + i;
		}
	}
	CPtr<const CUserTensor> userInput = AsUserTensor( *ConvertTensor( *inputs[0], inputLayout ), Name() + "_Source", dnn );
	fc->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
	dnn.AddLayer( *fc );

	outputs.Add( new CUserTensor( { inputShape[0], numberOfElements }, fcLayout, CLayerOutput( fc, 0 ) ) );
}

} // namespace NeoOnnx
