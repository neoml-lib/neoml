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

#include "onnx.pb.h"

#include "NonZeroOperator.h"
#include "NeoOnnxCheck.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxNonZeroLayer.h>

namespace NeoOnnx {

CNonZeroOperator::CNonZeroOperator( const onnx::NodeProto& nonZero, int opsetVersion ) :
	CLayerOperator( nonZero, opsetVersion )
{
	// v9 - original
	// v13 - new data types are supported
	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CNonZeroOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNeoOnnxSupport( inputs[0]->Type() != TTensorType::User, "user-provided data", *this );

	CPtr<COnnxNonZeroLayer> nonZeroLayer = new COnnxNonZeroLayer( dnn.GetMathEngine() );
	nonZeroLayer->SetName( Name() );
	inputs[0]->Layout().CopyTo( nonZeroLayer->InputLayout() );
	dnn.AddLayer( *nonZeroLayer );

	CPtr<const CShapeTensor> input = AsShapeTensor( *inputs[0], Name() + "_Source", dnn );
	nonZeroLayer->Connect( 0, *input->Layer(), input->OutputIndex() );
	outputs.Add( new CUserTensor( input->Layout(), CLayerOutput( nonZeroLayer, 0 ) ) );
}

} // namespace NeoOnnx
