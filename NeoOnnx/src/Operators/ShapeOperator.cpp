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

#include "ShapeOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CShapeOperator::CShapeOperator( const onnx::NodeProto& shape, int opsetVersion ) :
	COperator( shape, opsetVersion )
{
	// v1 - original
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CShapeOperator::ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	// This operator returns input's shape as an 1-dimensional tensor of integers
	// Due to the fact that tensor's shape doesn't depend on the actual values of tensor elements
	// this operator always returns CDataTensor
	const CTensorShape& inputShape = inputs[0]->Shape();
	CTensorLayout outputLayout( 1 );
	CBlobDesc outputBlobDesc( CT_Int );
	outputBlobDesc.SetDimSize( outputLayout[0], inputShape.Size() );
	CPtr<CDnnBlob> outputBlob = CDnnBlob::CreateBlob( dnn.GetMathEngine(), CT_Int, outputBlobDesc );
	outputBlob->CopyFrom( inputShape.GetPtr() );
	outputs.Add( new CDataTensor( { inputShape.Size() }, outputLayout, *outputBlob ) );

	if( !inputs[0]->IsCalculated() ) {
		// If input is a CUserTensor then there is a chance that this CUserTensor will lead to hanging layer output
		// Connect CSinkLayer to avoid this problem
		// TODO: find a way to detect such cases and remove all the unnecessary preceding layers instead of adding sink
		CPtr<CSinkLayer> safeSink = new CSinkLayer( dnn.GetMathEngine() );
		safeSink->SetName( Name() + "_Sink" );
		const CUserTensor* userInput = dynamic_cast<const CUserTensor*>( inputs[0].Ptr() );
		safeSink->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
		dnn.AddLayer( *safeSink );
	}
}

} // namespace NeoOnnx

