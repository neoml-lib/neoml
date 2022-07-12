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
#include "LayerUtils.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

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

	CArray<int> splits;
	getSplits( inputs, axis, splits );

	CPtr<const CUserTensor> currInput = AsUserTensor( *inputs[0], Name() + "_Source", dnn );
	CPtr<CBaseSplitLayer> splitLayer = CreateSplitLayer( dnn.GetMathEngine(), currInput->Layout()[axis] );
	splitLayer->SetName( Name() );
	splitLayer->SetOutputCounts( splits );
	splitLayer->Connect( 0, *currInput->Layer(), currInput->OutputIndex() );
	dnn.AddLayer( *splitLayer );

	CTensorShape outputShape;
	currInput->Shape().CopyTo( outputShape );

	outputs.SetBufferSize( OutputCount() );
	for( int i = 0; i < OutputCount(); ++i ) {
		CheckNeoOnnxSupport( splits[i] > 0, "Non-positive split size", *this );
		outputShape[axis] = splits[i];
		outputs.Add( new CUserTensor( outputShape, currInput->Layout(), CLayerOutput( splitLayer.Ptr(), i ) ) );
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
void CSplitOperator::getSplits( const CTensorArray& inputs, int axis, CArray<int>& splits ) const
{
	if( inputs.Size() > 1 && inputs[1] != nullptr ) {
		CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "User-provided 'split'", *this );
		const CDnnBlob* blob = dynamic_cast<const CDataTensor&>( *inputs[1] ).Data();
		CheckNeoOnnxSupport( blob->GetDataType() == CT_Int, "Non-integer 'split' values", *this );
		splits.SetSize( blob->GetDataSize() );
		blob->CopyTo( splits.GetPtr() );
		return;
	}

	if( !GetAttribute( "split", splits ) ) {
		// If no 'split' provided, then split evenly
		CheckNeoOnnxSupport( inputs[0]->Shape()[axis] % OutputCount() == 0, "Shape can't be split evenly", *this );
		splits.Add( inputs[0]->Shape()[axis] / OutputCount(), OutputCount() );
	}
}

} // namespace NeoOnnx
