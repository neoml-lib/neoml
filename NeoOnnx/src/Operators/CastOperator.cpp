/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "CastOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxCastLayer.h>

namespace NeoOnnx {

CCastOperator::CCastOperator( const onnx::NodeProto& cast, int opsetVersion ) :
	CLayerOperator( cast, opsetVersion ),
	outputType( 0 )
{
	// v1 - original
	// v6 - to attrbiute converted to integer instead of string
	// v9 - string type support is added
	// v13 - bloaf16 support is added
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
	CheckOnnxProtocol( GetAttribute( "to", outputType ), "'to 'attribute is missing", *this );
}

void CCastOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	CLayerOutput layerOutput;
	CPtr<const CShapeTensor> inputShapeTensor = nullptr;
	if( HasUserInput( inputs ) ) {
		layerOutput = AsUserTensor( *inputs[0], Name() + "_Source", dnn )->LayerOutput();
	} else {
		inputShapeTensor = AsShapeTensor( *inputs[0], Name() + "_Source", dnn );
		layerOutput = inputShapeTensor->LayerOutput();
	}

	CPtr<COnnxCastLayer> cast = new COnnxCastLayer( dnn.GetMathEngine() );
	cast->SetName( Name() );
	cast->SetOutputType( GetBlobType( static_cast<onnx::TensorProto_DataType>( outputType ) ) );
	cast->Connect( 0, *layerOutput.Layer, layerOutput.OutputIndex );
	dnn.AddLayer( *cast );

	if( HasUserInput( inputs ) ) {
		outputs.Add( new CUserTensor( inputs[0]->Layout(), CLayerOutput( cast, 0 ) ) );
	} else {
		outputs.Add( new CShapeTensor( inputs[0]->Layout(), inputShapeTensor->Shape(), CLayerOutput( cast, 0 ) ) );
	}
}

} // namespace NeoOnnx
