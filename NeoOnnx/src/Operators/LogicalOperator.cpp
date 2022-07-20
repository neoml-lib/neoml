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

#include "LogicalOperator.h"

namespace NeoOnnx {

CNotOperator::CNotOperator( const onnx::NodeProto& not, int opsetVersion ) :
	CLayerOperator( not, opsetVersion )
{
	// v1 - original
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CNotOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CPtr<const CUserTensor> input = AsUserTensor( *inputs[0], Name() + "_Source", dnn );

	CNotLayer* not = Not()( Name(), CDnnLayerLink( input->Layer(), input->OutputIndex() ) );
	outputs.Add( new CUserTensor( input->Shape(), input->Layout(), CLayerOutput( not, 0 ) ) );
}

} // namespace NeoOnnx