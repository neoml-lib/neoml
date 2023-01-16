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

#include "ReshapeOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"
#include <NeoML/Dnn/Layers/Onnx/OnnxReshapeLayer.h>

#include "onnx.pb.h"

namespace NeoOnnx {

static void calcReshapeOperatorOutputShape( const CShapeTensor& input, const CDnnBlob& newShape,
	CTensorShape& outputShape )
{
	outputShape.SetSize( newShape.GetDataSize() );
	int remSize = 1;
	for( int i = 0; i < input.DimCount(); ++i ) {
		remSize *= input.Shape()[i];
	}

	int remIndex = -1;
	for( int i = 0; i < outputShape.Size(); ++i ) {
		int dimSize = newShape.GetData<int>().GetValueAt( i );
		if( dimSize == -1 ) {
			remIndex = i;
		} else if( dimSize == 0 ) {
			remSize /= input.Shape()[i];
			outputShape[i] = input.Shape()[i];
		} else {
			remSize /= dimSize;
			outputShape[i] = dimSize;
		}
	}

	if( remIndex != -1 ) {
		outputShape[remIndex] = remSize;
	}
}

//---------------------------------------------------------------------------------------------------------------------

CReshapeOperator::CReshapeOperator( const onnx::NodeProto& reshape, int opsetVersion ) :
	CLayerOperator( reshape, opsetVersion )
{
	// v1 - original
	// v5 - removed legacy optimization attribute, "shape" moved from attributes to inputs, supported new data types
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 5 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CReshapeOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	CPtr<const CTensorBase> newShapeTensor = getShape( inputs, dnn );
	CheckNeoOnnxSupport( newShapeTensor->DimCount() == 1, "shape must have 1 dimension", *this );

	const bool hasShapeOutput = inputs[0]->Type() != TTensorType::User && newShapeTensor->Type() == TTensorType::Data;

	// In order to process tensors correctly reshape is not allowed in transposed layouts
	CPtr<const CTensorBase> inputBaseTensor = inputs[0];
	if( IsTransposedLayout( inputBaseTensor->Layout() ) ) {
		inputBaseTensor = ConvertTensor( *inputBaseTensor, CTensorLayout( inputBaseTensor->DimCount() ) );
	}
	
	CPtr<COnnxReshapeLayer> reshapeLayer = new COnnxReshapeLayer( dnn.GetMathEngine() );
	reshapeLayer->SetName( Name() );
	inputBaseTensor->Layout().CopyTo( reshapeLayer->InputLayout() );
	dnn.AddLayer( *reshapeLayer );

	CPtr<const CShapeTensor> secondInput = AsShapeTensor( *newShapeTensor, Name() + "_NewShapeSource", dnn );
	CTensorLayout outputLayout( secondInput->Shape()[0] );
	reshapeLayer->Connect( 1, *secondInput->Layer(), secondInput->OutputIndex() );
	outputLayout.CopyTo( reshapeLayer->OutputLayout() );

	if( hasShapeOutput ) {
		CPtr<const CShapeTensor> inputShapeTensor = AsShapeTensor( *inputBaseTensor, Name() + "_InputSource", dnn );
		reshapeLayer->Connect( 0, *inputShapeTensor->Layer(), inputShapeTensor->OutputIndex() );

		const CDataTensor& newShapeData = dynamic_cast<const CDataTensor&>( *newShapeTensor );
		const CDnnBlob& newShapeBlob = *newShapeData.Data();
		CTensorShape outputShape;
		calcReshapeOperatorOutputShape( *inputShapeTensor, newShapeBlob, outputShape );

		outputs.Add( new CShapeTensor( outputLayout, outputShape, CLayerOutput( reshapeLayer, 0 ) ) );
	} else {
		CPtr<const CUserTensor> inputUserTensor = AsUserTensor( *inputBaseTensor, Name() + "_InputSource", dnn );
		reshapeLayer->Connect( 0, *inputUserTensor->Layer(), inputUserTensor->OutputIndex() );

		outputs.Add( new CUserTensor( outputLayout, CLayerOutput( reshapeLayer, 0 ) ) );
	}
}

// Gets output shape
CPtr<const CTensorBase> CReshapeOperator::getShape( const CTensorArray& inputs, CDnn& dnn ) const
{
	if( OpsetVersion < 5 ) {
		CTensorShape shapeArray;
		CheckOnnxProtocol( GetAttribute( "shape", shapeArray ), "'shape' attribute is missing", *this );
		CPtr<CDnnBlob> shapeBlob = CDnnBlob::CreateVector( GetSingleThreadCpuMathEngine(), CT_Int, shapeArray.Size() );
		shapeBlob->CopyFrom( shapeArray.GetPtr() );
		return new CDataTensor( CTensorLayout( { BD_BatchLength } ), *shapeBlob );
	}

	CheckNeoOnnxSupport( inputs[1] != nullptr && inputs[1]->Type() != TTensorType::User,
		"User-provided output shape", *this );
	return inputs[1];
}

} // namespace NeoOnnx
