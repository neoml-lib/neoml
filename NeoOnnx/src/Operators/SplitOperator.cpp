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

#include "SplitOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxSplitLayer.h>

namespace NeoOnnx {

CSplitOperator::CSplitOperator( const onnx::NodeProto& split, int opsetVersion ) :
	CLayerOperator( split, opsetVersion )
{
	// v1 - original, 'split' can be obtained via attribute or input
	// v2 - default 'axis' value added + 'split' can be obtained only via attribute
	// v11 - negative axis supported
	// v13 - 'split' can be obtained only via input + bfloat16 supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion == 1 ) {
		CheckOnnxProtocol( InputCount() >= 1 && InputCount() <= 2, "operator must have 1 or 2 inputs", *this );
	} else if( OpsetVersion < 13 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} // OpsetVersion > 12 not supported by NeoOnnx
}

void CSplitOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	const int axis = getAxis( *inputs[0] );

	CPtr<const CTensorBase> splitsInput = getSplits( inputs, dnn.GetMathEngine() );

	// If we want to provide output of this operator as CShapeTensor then we need to calculate its shape
	// We can calculate its shape only when both the shape of first input and the data of second input are known
	const bool hasShapeOutput = inputs[0]->Type() != TTensorType::User
		&& ( splitsInput == nullptr || splitsInput->Type() == TTensorType::Data );

	CPtr<COnnxSplitLayer> splitLayer = new COnnxSplitLayer( dnn.GetMathEngine() );
	splitLayer->SetName( Name() );
	splitLayer->SetSplitDim( inputs[0]->Layout()[axis] );
	dnn.AddLayer( *splitLayer );
	outputs.SetBufferSize( OutputCount() );

	if( hasShapeOutput ) {
		CPtr<const CShapeTensor> shapeInput = AsShapeTensor( *inputs[0], Name() + "_Source", dnn );
		splitLayer->Connect( 0, *shapeInput->Layer(), shapeInput->OutputIndex() );

		CTensorShape outputShape;
		shapeInput->Shape().CopyTo( outputShape );
		for( int i = 0; i < OutputCount(); ++i ) {
			// If no 'split' provided, then split evenly
			outputShape[axis] = splitsInput == nullptr ? shapeInput->Shape()[axis] / OutputCount()
				: dynamic_cast<const CDataTensor&>( *splitsInput ).Data()->GetData<int>().GetValueAt( i );
			outputs.Add( new CShapeTensor( inputs[0]->Layout(), outputShape, CLayerOutput( splitLayer, i ) ) );
		}
	} else {
		CPtr<const CUserTensor> shapeInput = AsUserTensor( *inputs[0], Name() + "_Source", dnn );
		splitLayer->Connect( 0, *shapeInput->Layer(), shapeInput->OutputIndex() );

		for( int i = 0; i < OutputCount(); ++i ) {
			outputs.Add( new CUserTensor( inputs[0]->Layout(), CLayerOutput( splitLayer, i ) ) );
		}
	}

	if( splitsInput != nullptr ) {
		CPtr<const CShapeTensor> splitsShape = AsShapeTensor( *splitsInput, Name() + "_Splits", dnn );
		splitLayer->Connect( 1, *splitsShape->Layer(), splitsShape->OutputIndex() );
	}
}

// Gets the axis index for any of the opset version
int CSplitOperator::getAxis( const CTensorBase& firstInput ) const
{
	int axis = 0;
	// Axis attribute must be present only in opsetV1
	const bool hasAttribute = GetAttribute( "axis", axis );
	CheckOnnxProtocol( OpsetVersion != 1 || hasAttribute, "'axis' attribute missing", *this );

	if( axis < 0 ) {
		axis += firstInput.DimCount();
	}
	CheckOnnxProtocol( axis >= 0 && axis < firstInput.DimCount(), "invalid 'axis' value", *this );
	return axis;
}

// Gets the splits sizes for the current configuration of OpsetVersion, inputs and attributes
CPtr<const CTensorBase> CSplitOperator::getSplits( const CTensorArray& inputs, IMathEngine& mathEngine ) const
{
	if( inputs.Size() > 1 ) {
		CheckNeoOnnxSupport( inputs[1] == nullptr || inputs[1]->Type() != TTensorType::User,
			"User-provided 'split'", *this );
		return inputs[1];
	}

	CFastArray<int, 8> splits;
	if( !GetAttribute( "split", splits ) ) {
		return nullptr;
	}

	CPtr<CDnnBlob> blob = CDnnBlob::CreateVector( mathEngine, CT_Int, splits.Size() );
	blob->CopyFrom( splits.GetPtr() );
	return new CDataTensor( CTensorLayout( { BD_BatchLength } ), *blob );
}

} // namespace NeoOnnx
