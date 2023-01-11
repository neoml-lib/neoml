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
#include <NeoML/Dnn/Layers/Onnx/OnnxShapeLayer.h>

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
	CheckNoNullInputs( inputs );
	// Shape of shape is not supported
	CheckNoShapeInputs( inputs );

	if( inputs[0]->Type() == TTensorType::Data ) {
		// Lets calculate the shape as CDataTensor
		// If needed it could be converted to CShapeTensor at any time
		CPtr<const CDataTensor> data = CheckCast<const CDataTensor>( inputs[0] );
		CPtr<CDnnBlob> shapeBlob = CDnnBlob::CreateVector( dnn.GetMathEngine(), CT_Int, data->DimCount() );
		{
			CDnnBlobBuffer<int> buffer( *shapeBlob, 0, shapeBlob->GetDataSize(), TDnnBlobBufferAccess::Write );
			for( int i = 0; i < data->DimCount(); ++i ) {
				buffer[i] = data->DimSize( i );
			}
		}
		outputs.Add( new CDataTensor( CTensorLayout( { BD_BatchLength } ), *shapeBlob ) );
		return;
	}

	NeoAssert( inputs[0]->Type() == TTensorType::User );
	CPtr<const CUserTensor> userInput = CheckCast<const CUserTensor>( inputs[0] );
	CPtr<COnnxShapeLayer> shapeLayer = new COnnxShapeLayer( dnn.GetMathEngine() );
	shapeLayer->SetName( Name() );
	userInput->Layout().CopyTo( shapeLayer->TensorLayout() );
	shapeLayer->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
	dnn.AddLayer( *shapeLayer );
	outputs.Add( new CShapeTensor( CTensorLayout::IOLayout( 1 ), { inputs[0]->DimCount() },
		CLayerOutput( shapeLayer.Ptr(), 0 ) ) );
}

} // namespace NeoOnnx

