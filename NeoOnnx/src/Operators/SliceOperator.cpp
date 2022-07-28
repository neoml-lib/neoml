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

#include "SliceOperator.h"
#include "LayerUtils.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSliceOperator::CSliceOperator( const onnx::NodeProto& slice, int opsetVersion ) :
	CLayerOperator( slice, opsetVersion )
{
	// v1 - original
	// v10 - attributes are replaced with additional inputs + 'step' support
	// v11 - backward slicing support is added
	// v13 - bloaf16 support is added
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 10 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() >= 3 && InputCount() <= 5, "operator must have from 3 up to 5 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CSliceOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );

	CFastArray<int, 8> axes;
	getAxes( inputs, axes );

	CFastArray<int, 8> starts;
	getStarts( inputs, starts );

	CFastArray<int, 8> ends;
	getEnds( inputs, ends );

	CFastArray<int, 8> steps;
	getSteps( inputs, steps );

	CPtr<const CTensorBase> currInput = inputs[0].Ptr();
	for( int i = 0; i < axes.Size(); ++i ) {
		currInput = sliceAxis( *currInput, axes[i], starts[i], ends[i], steps[i], dnn );
	}
	outputs.Add( currInput.Ptr() );
}

// Fills array with axes, affected by slice
void CSliceOperator::getAxes( const CTensorArray& inputs, CFastArray<int, 8>& axes ) const
{
	const CTensorShape& inputShape = inputs[0]->Shape();

	if( OpsetVersion < 10 && GetAttribute( "axes", axes ) ) {
		// Successfully extracted from the attribute
		return;
	} else if( OpsetVersion >= 10 && inputs.Size() >= 4 && inputs[3] != nullptr ) {
		CheckNeoOnnxSupport( inputs[3]->IsCalculated(), "User-provided axes", *this );
		const CDnnBlob* axesBlob = dynamic_cast<const CDataTensor*>( inputs[3].Ptr() )->Data();
		CheckOnnxProtocol( axesBlob->GetDataType() == CT_Int, "Non-integer axes", *this );
		axes.SetSize( axesBlob->GetDataSize() );
		axesBlob->CopyTo( axes.GetPtr() );
	} else {
		// Fill with default value
		axes.SetBufferSize( inputShape.Size() );
		for( int i = 0; i < inputShape.Size(); ++i ) {
			axes.Add( i );
		}
	}
}

// Fills array with slice start indices
void CSliceOperator::getStarts( const CTensorArray& inputs, CFastArray<int, 8>& starts ) const
{
	if( OpsetVersion < 10 ) {
		// Extracting from attributes
		CheckOnnxProtocol( GetAttribute( "starts", starts ), "'starts' attribute is missing", *this );
	} else {
		CheckNeoOnnxSupport( inputs[1] != nullptr && inputs[1]->IsCalculated(), "User-provided starts", *this );
		const CDnnBlob* startsBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
		CheckOnnxProtocol( startsBlob->GetDataType() == CT_Int, "Non-integer starts", *this );
		starts.SetSize( startsBlob->GetDataSize() );
		startsBlob->CopyTo( starts.GetPtr() );
	}
}

// Fills array with slice end indices
void CSliceOperator::getEnds( const CTensorArray& inputs, CFastArray<int, 8>& ends ) const
{
	if( OpsetVersion < 10 ) {
		// Extracting from attributes
		CheckOnnxProtocol( GetAttribute( "ends", ends ), "'ends' attribute is missing", *this );
	} else {
		CheckNeoOnnxSupport( inputs[2] != nullptr && inputs[2]->IsCalculated(), "User-provided ends", *this );
		const CDnnBlob* endsBlob = dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data();
		CheckOnnxProtocol( endsBlob->GetDataType() == CT_Int, "Non-integer ends", *this );
		ends.SetSize( endsBlob->GetDataSize() );
		endsBlob->CopyTo( ends.GetPtr() );
	}
}

// Fills array with slice steps
void CSliceOperator::getSteps( const CTensorArray& inputs, CFastArray<int, 8>& steps ) const
{
	const CTensorShape& inputShape = inputs[0]->Shape();

	if( OpsetVersion < 10 && GetAttribute( "steps", steps ) ) {
		// Successfully extracted from the attribute
		return;
	} else if( OpsetVersion >= 10 && inputs.Size() >= 5 && inputs[4] != nullptr ) {
		// Extracting from the input
		CheckNeoOnnxSupport( inputs[4]->IsCalculated(), "User-provided steps", *this );
		const CDnnBlob* stepsBlob = dynamic_cast<const CDataTensor*>( inputs[4].Ptr() )->Data();
		CheckOnnxProtocol( stepsBlob->GetDataType() == CT_Int, "Non-integer steps", *this );
		steps.SetSize( stepsBlob->GetDataSize() );
		stepsBlob->CopyTo( steps.GetPtr() );
	} else {
		// Fill with default value
		steps.SetBufferSize( inputShape.Size() );
		steps.Add( 1, inputShape.Size() );
	}
}

// Adds slice along one axis
CPtr<const CTensorBase> CSliceOperator::sliceAxis( const CTensorBase& input, int axis, int start, int end, int step,
	CDnn& dnn ) const
{
	CheckNeoOnnxSupport( step == 1, "Slice with step", *this );

	if( axis < 0 ) {
		axis += input.DimCount();
	}
	CheckOnnxProtocol( axis >= 0 && axis < input.DimCount(), "invalid axis index", *this );
	const int dimSize = input.Shape()[axis];
	if( start < 0 ) {
		start += dimSize;
	}
	if( end < 0 ) {
		end += dimSize;
	} else if( end > dimSize ) {
		end = dimSize;
	}

	NeoAssert( start <= end );

	if( start == end ) {
		// The slice results with a tensor of 0 elements
		// NeoML doesn't support that
		return nullptr;
	}

	if( start == 0 && end == dimSize) {
		// No need to split
		return &input;
	}

	CPtr<const CUserTensor> userInput = AsUserTensor( input, Name() + "_Source", dnn );
	CPtr<CBaseSplitLayer> split = CreateSplitLayer( dnn.GetMathEngine(), userInput->Layout()[axis] );
	split->SetName( Name() + "_" + Str( axis ) );
	int outputIndex = 0;

	if( start == 0 ) {
		NeoAssert( end != dimSize );
		split->SetOutputCounts2( end );
	} else if( end == dimSize ) {
		split->SetOutputCounts2( start );
	} else {
		split->SetOutputCounts3( start, end - start );
	}

	if( start != 0 ) {
		outputIndex = 1;
		// We have to add sink for the first part
		CPtr<CSinkLayer> sink = new CSinkLayer( dnn.GetMathEngine() );
		sink->SetName( Name() + "_sink" + Str( dnn.GetLayerCount() ) );
		sink->Connect( *split );
		dnn.AddLayer( *sink );
	}

	if( end != dimSize ) {
		// We have to add sink for the remaining part
		CPtr<CSinkLayer> sink = new CSinkLayer( dnn.GetMathEngine() );
		sink->SetName( Name() + "_sink" + Str( dnn.GetLayerCount() ) );
		sink->Connect( 0, *split, outputIndex + 1 );
		dnn.AddLayer( *sink );
	}

	split->Connect( 0, *userInput->Layer(), userInput->OutputIndex() );
	dnn.AddLayer( *split );
	
	CTensorShape outputShape;
	userInput->Shape().CopyTo( outputShape );
	outputShape[axis] = end - start;
	return new CUserTensor( outputShape, userInput->Layout(), CLayerOutput( split, outputIndex ) );
}

} // namespace NeoOnnx
