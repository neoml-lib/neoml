/* Copyright © 2017-2024 ABBYY

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

#include <limits.h>
#include <algorithm>

#include "ShapeOperator.h"
#include "NeoOnnxCheck.h"
#include <NeoML/Dnn/Layers/Onnx/OnnxShapeLayer.h>

#include "onnx.pb.h"

using namespace NeoML;

namespace NeoOnnx {

CShapeOperator::CShapeOperator( const onnx::NodeProto& shape, int opsetVersion ) :
	COperator( shape, opsetVersion )
{
	// v1 - original
	// v13 - bfloat16 is supported
	// v15 - start and end attributes are added
	// v19 - float-8 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CShapeOperator::ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	int startAttr = 0;
	int endAttr = INT_MAX;
	if( OpsetVersion >= 15 ) {
		GetAttribute( "start", startAttr );
		GetAttribute( "end", endAttr );
	}
	const int start = std::max<int>( 0, startAttr >= 0 ? startAttr : startAttr + inputs[0]->DimCount() );
	const int end = std::min<int>( inputs[0]->DimCount(), endAttr >= 0 ? endAttr : endAttr + inputs[0]->DimCount() );
	CheckNeoOnnxSupport( end > start, "end <= start", *this );

	if( inputs[0]->Type() != TTensorType::User ) {
		// Lets calculate the shape as CDataTensor (if we can)
		// If needed it could be converted to CShapeTensor at any time
		CTensorShape shapeArray;
		GetTensorShape( *inputs[0], shapeArray );
		CPtr<CDnnBlob> shapeBlob = CDnnBlob::CreateVector( dnn.GetMathEngine(), CT_Int, end - start );
		shapeBlob->CopyFrom( shapeArray.GetPtr() + start );
		outputs.Add( new CDataTensor( CTensorLayout( { BD_BatchLength } ), *shapeBlob ) );
		return;
	}

	NeoAssert( inputs[0]->Type() == TTensorType::User );
	CPtr<const CUserTensor> userInput = CheckCast<const CUserTensor>( inputs[0] );
	CPtr<COnnxShapeLayer> shapeLayer = new COnnxShapeLayer( dnn.GetMathEngine() );
	shapeLayer->SetName( Name() );
	userInput->Layout().CopyTo( shapeLayer->TensorLayout() );
	shapeLayer->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
	shapeLayer->StartAttr() = startAttr;
	shapeLayer->EndAttr() = endAttr;
	dnn.AddLayer( *shapeLayer );
	outputs.Add( new CShapeTensor( CTensorLayout::IOLayout( 1 ), { end - start },
		CLayerOutput( shapeLayer.Ptr(), 0 ) ) );
}

} // namespace NeoOnnx

