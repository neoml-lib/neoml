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
	// v13 - bfloat16 is supported
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

	if( OpsetVersion < 7 ) {
		int broadcast = 0;
		GetAttribute( "broadcast", broadcast );
		CheckNeoOnnxSupport( broadcast != 0, "broadcast == 0", *this );
	}
}

void CGemmOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );

	CheckNeoOnnxSupport( inputs[1]->Type() == TTensorType::Data, "user-provided weights", *this );

	const CTensorLayout weightLayout( { transB == 0 ? BD_Channels: BD_BatchWidth, transB == 0 ? BD_BatchWidth : BD_Channels } );
	CPtr<const CDataTensor> matrixTensor = CheckCast<const CDataTensor>( ConvertTensor( *inputs[1], weightLayout ) );
	CheckOnnxProtocol( matrixTensor->DimCount() == 2, "weights must be 2-dimensional", *this );
	const int numberOfElements = matrixTensor->DimSize( transB == 0 ? 1 : 0 );

	CPtr<const CDataTensor> freeTermTensor = nullptr;
	if( InputCount() == 3 ) {
		CheckNeoOnnxSupport( inputs[2]->Type() == TTensorType::Data, "user-provided bias", *this );
		freeTermTensor = CheckCast<const CDataTensor>( inputs[2] );
		CheckOnnxProtocol( freeTermTensor->DimCount() == 1, "bias must be 1-dimensional", *this );
		CheckOnnxProtocol( freeTermTensor->DimSize( 0 ) == numberOfElements, "wrong bias size", *this );
	}

	CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( dnn.GetMathEngine() );
	fc->SetName( Name() );

	fc->SetNumberOfElements( numberOfElements );

	fc->SetWeightsData( matrixTensor->Data()->GetCopy() );

	if( freeTermTensor != nullptr ) {
		fc->SetFreeTermData( freeTermTensor->Data()->GetCopy() );
	} else {
		fc->SetZeroFreeTerm( true );
	}

	CTensorLayout inputLayout = inputs[0]->DimCount() == 2 ? CTensorLayout{BD_BatchWidth, BD_Channels} : inputs[0]->Layout();
	if( inputs[0]->DimCount() > 2 ) {
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

	outputs.Add( new CUserTensor( CTensorLayout{ BD_BatchWidth, BD_Channels }, CLayerOutput( fc, 0 ) ) );
}

} // namespace NeoOnnx
