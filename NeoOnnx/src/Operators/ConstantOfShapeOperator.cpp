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

#include "ConstantOfShapeOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxConstantOfShapeLayer.h>

namespace NeoOnnx {

CConstantOfShapeOperator::CConstantOfShapeOperator( const onnx::NodeProto& constantOfShape, int opsetVersion ) :
	CLayerOperator( constantOfShape, opsetVersion )
{
	// v9 - original
	CheckOnnxProtocol( OpsetVersion >= 9, "wrong opset version", *this );
	CheckNeoOnnxSupport( OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CConstantOfShapeOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	CheckNeoOnnxSupport( inputs[0]->Type() != TTensorType::User, "user-provided input", *this );

	IMathEngine& mathEngine = dnn.GetMathEngine();

	// If "value" attribute is not set then float 0.f is assumed
	CPtr<const CDnnBlob> valueBlob;
	CPtr<CDataTensor> valueTensor( new CDataTensor( mathEngine ) );
	if( GetAttribute( "value", valueTensor ) ) {
		valueBlob = valueTensor->Data();
	} else {
		CPtr<CDnnBlob> zero = CDnnBlob::CreateVector( mathEngine, CT_Float, 1 );
		zero->Clear();
		valueBlob = zero;
	}
	valueTensor = nullptr;

	CPtr<const CShapeTensor> inputShapeTensor = AsShapeTensor( *inputs[0], Name() + "_ShapeSource", dnn );

	CPtr<COnnxConstantOfShapeLayer> layer = new COnnxConstantOfShapeLayer( mathEngine );
	layer->SetName( Name() );
	layer->SetValue( *valueBlob );
	layer->Connect( 0, *inputShapeTensor->Layer(), inputShapeTensor->OutputIndex() );
	dnn.AddLayer( *layer );

	const CTensorLayout outputLayout( inputShapeTensor->DimCount() == 0 ? 0 : inputShapeTensor->Shape()[0] );
	outputs.Add( new CUserTensor( outputLayout, CLayerOutput( layer, 0 ) ) );
}

} // namespace NeoOnnx
