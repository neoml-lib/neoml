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

#include "DropoutOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CDropoutOperator::CDropoutOperator( const onnx::NodeProto& dropout, int opsetVersion ) :
	CLayerOperator( dropout, opsetVersion )
{
	// v1 - original
	// v6 - removed legacy optimization attribute
	// v7 - removed "is_test" attribute
	// v10 - changed second output data type
	// v12 - added "seed" attribute, "ratio" moved from attributes to inputs, "training_mode" added
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 12 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() >= 1 || InputCount() <= 3, "operator must have from 1 up to 3 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1 || OutputCount() == 2, "operator must have 1 or 2 outputs", *this );
}

void CDropoutOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	NeoAssert( inputs[0] != nullptr && !inputs[0]->IsCalculated() );
	const CUserTensor* userInput = dynamic_cast<const CUserTensor*>( inputs[0].Ptr() );

	CPtr<CDropoutLayer> dropout = new CDropoutLayer( dnn.GetMathEngine() );
	dropout->SetName( Name() );
	dropout->SetDropoutRate( getRatio( inputs ) );
	dropout->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
	dnn.AddLayer( *dropout );

	outputs.Add( new CUserTensor( userInput->Shape(), userInput->Layout(), CLayerOutput( dropout, 0 ) ) );
	if( OutputCount() == 2 ) {
		// neoml::CDropoutLayer doesn't support mask as output
		outputs.Add( nullptr );
	}
}

// Gets dropout rate
float CDropoutOperator::getRatio( const CTensorArray& inputs ) const
{
	if( OpsetVersion < 12 ) {
		// Before opset 12 ratio is stored as optional attribute with default value 0.5f
		float ratio = 0.5f;
		GetAttribute( "ratio", ratio );
		return ratio;
	} else if( inputs.Size() < 2 || inputs[1] == nullptr ) {
		// If "ratio" input is omitted, default value is 0.5f
		return 0.5f;
	}

	// Extracting data from input
	CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "User-provided ratio", *this );
	return dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data()->GetData().GetValue();
}

} // namespace NeoOnnx
