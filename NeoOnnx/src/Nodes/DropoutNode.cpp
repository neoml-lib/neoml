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

#include "DropoutNode.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CDropoutNode::CDropoutNode( const onnx::NodeProto& dropout, int opsetVersion ) :
	CLayerOpNode( dropout, opsetVersion )
{
	// v1 - original
	// v6 - removed legacy optimization attribute
	// v7 - removed "is_test" attribute
	// v10 - changed second output data type
	// v12 - added "seed" attribute, "ratio" moved from attributes to inputs, "training_mode" added
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 12 ) {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() >= 1 || InputCount() <= 3, "node must have from 1 up to 3 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "node must have 1 output", *this );
}

void CDropoutNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CheckNeoOnnxInternal( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "Input must be provided by user", *this );
	const CUserTensor* userInput = dynamic_cast<const CUserTensor*>( inputs[0].Ptr() );

	CPtr<CDropoutLayer> dropout = new CDropoutLayer( dnn.GetMathEngine() );
	dropout->SetName( Name() );
	dropout->SetDropoutRate( getRatio( inputs ) );
	dropout->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
	dnn.AddLayer( *dropout );

	outputs[0] = new CUserTensor( userInput->Shape(), userInput->Layout(), CLayerOutput( dropout, 0 ) );
}

// Gets dropout rate
float CDropoutNode::getRatio( const CObjectArray<const CTensorBase>& inputs ) const
{
	if( OpsetVersion < 12 ) {
		// Before opset 12 ratio is stored as optional attribute with default value 0.5f
		return Attributes.GetOptionalFloat( "ratio", 0.5f );
	} else if( inputs.Size() < 2 || inputs[1] == nullptr ) {
		// If "ratio" input is omitted, default value is 0.5f
		return 0.5f;
	}

	// Extracting data from input
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "User-provided ratio", *this );
	return dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data()->GetData().GetValue();
}

} // namespace NeoOnnx
