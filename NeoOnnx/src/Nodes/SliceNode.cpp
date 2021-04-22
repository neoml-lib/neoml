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

#include "SliceNode.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSliceNode::CSliceNode( const onnx::NodeProto& slice, int opsetVersion ) :
	CLayerOpNode( slice, opsetVersion )
{
	// v1 - original
	// v10 - attributes replaced with additional inputs + 'step' support
	// v11 - added backward slicing support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 10 ) {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", *this );
}

void CSliceNode::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& /* dnn */ )
{
	NeoAssert( inputs[0] != nullptr && !inputs[0]->IsCalculated() );

	CFastArray<int, 8> axes;
	getAxes( inputs, axes );

	CFastArray<int, 8> starts;
	getStarts( inputs, starts );

	CFastArray<int, 8> ends;
	getEnds( inputs, ends );

	CFastArray<int, 8> steps;
	getSteps( inputs, steps );

	CPtr<const CUserTensor> currInput = dynamic_cast<const CUserTensor*>( inputs[0].Ptr() );
	for( int i = 0; i < axes.Size(); ++i ) {
		currInput = sliceAxis( *currInput, axes[i], starts[i], ends[i], steps[i] );
	}
	outputs[0] = currInput;
}

// Fills array with axes, affected by slice
void CSliceNode::getAxes( const CObjectArray<const CTensorBase>& inputs, CFastArray<int, 8>& axes ) const
{
	const CTensorShape& inputShape = inputs[0]->Shape();
	// Fill with default value
	axes.SetBufferSize( inputShape.Size() );
	for( int i = 0; i < inputShape.Size(); ++i ) {
		axes.Add( i );
	}

	if( OpsetVersion < 10 ) {
		// Extracting from attributes
		if( Attributes.Has( "axes" ) ) {
			axes.Empty();
			Attributes.GetRequiredIntArray( "axes", axes );
		}
	} else {
		if( inputs.Size() >= 4 && inputs[3] != nullptr ) {
			CheckNeoOnnxSupport( inputs[3]->IsCalculated(), "User-provided axes", *this );
			const CDnnBlob* axesBlob = dynamic_cast<const CDataTensor*>( inputs[3].Ptr() )->Data();
			CheckOnnxProtocol( axesBlob->GetDataType() == CT_Int, "Non-integer axes", *this );
			axes.SetSize( axesBlob->GetDataSize() );
			axesBlob->CopyTo( axes.GetPtr() );
		}
	}
}

// Fills array with slice start indices
void CSliceNode::getStarts( const CObjectArray<const CTensorBase>& inputs, CFastArray<int, 8>& starts ) const
{
	if( OpsetVersion < 10 ) {
		// Extracting from attributes
		Attributes.GetRequiredIntArray( "starts", starts );
	} else {
		CheckNeoOnnxSupport( inputs[2] != nullptr && inputs[2]->IsCalculated(), "User-provided starts", *this );
		const CDnnBlob* startsBlob = dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data();
		CheckOnnxProtocol( startsBlob->GetDataType() == CT_Int, "Non-integer starts", *this );
		starts.SetSize( startsBlob->GetDataSize() );
		startsBlob->CopyTo( starts.GetPtr() );
	}
}

// Fills array with slice end indices
void CSliceNode::getEnds( const CObjectArray<const CTensorBase>& inputs, CFastArray<int, 8>& ends ) const
{
	if( OpsetVersion < 10 ) {
		// Extracting from attributes
		Attributes.GetRequiredIntArray( "ends", ends );
	} else {
		CheckNeoOnnxSupport( inputs[1] != nullptr && inputs[1]->IsCalculated(), "User-provided ends", *this );
		const CDnnBlob* endsBlob = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data();
		CheckOnnxProtocol( endsBlob->GetDataType() == CT_Int, "Non-integer ends", *this );
		ends.SetSize( endsBlob->GetDataSize() );
		endsBlob->CopyTo( ends.GetPtr() );
	}
}

// Fills array with slice steps
void CSliceNode::getSteps( const CObjectArray<const CTensorBase>& inputs, CFastArray<int, 8>& steps ) const
{
	const CTensorShape& inputShape = inputs[0]->Shape();
	// Fill with default value
	steps.SetBufferSize( inputShape.Size() );
	steps.Add( 1, inputShape.Size() );

	if( OpsetVersion < 10 ) {
		// Extracting from attributes
		if( Attributes.Has( "steps" ) ) {
			steps.Empty();
			Attributes.GetRequiredIntArray( "steps", steps );
		}
	} else {
		if( inputs.Size() >= 5 && inputs[4] != nullptr ) {
			CheckNeoOnnxSupport( inputs[4]->IsCalculated(), "User-provided steps", *this );
			const CDnnBlob* stepsBlob = dynamic_cast<const CDataTensor*>( inputs[4].Ptr() )->Data();
			CheckOnnxProtocol( stepsBlob->GetDataType() == CT_Int, "Non-integer steps", *this );
			steps.SetSize( stepsBlob->GetDataSize() );
			stepsBlob->CopyTo( steps.GetPtr() );
		}
	}
}

// Adds slice along one axis
CPtr<const CUserTensor> CSliceNode::sliceAxis( const CUserTensor& input, int axis, int start, int end, int step ) const
{
	CheckNeoOnnxSupport( step == 1 || step == -1, "Slice with step", *this );

	const CTensorShape& inputShape = input.Shape();
	if( axis < 0 ) {
		axis += inputShape.Size();
	}
	if( start < 0 ) {
		start += inputShape[axis];
	}
	if( end < 0 ) {
		end += inputShape[axis];
	} else if( end > inputShape[axis] ) {
		end = inputShape[axis];
	}

	NeoAssert( start < end );

	CPtr<const CUserTensor> convertedInput = prepareInputForSlice( input, axis );
	CTensorShape outputShape;
	inputShape.CopyTo( outputShape );
	outputShape[axis] = end - start;

	CDnn& dnn = *convertedInput->Layer()->GetDnn();
	CPtr<CSubSequenceLayer> subseq = new CSubSequenceLayer( dnn.GetMathEngine() );
	subseq->SetName( Name() + "_" + Str( axis ) );
	subseq->SetStartPos( start );
	subseq->SetLength( end - start );
	if( step == -1 ) {
		subseq->SetReverse();
	}
	subseq->Connect( 0, *convertedInput->Layer(), convertedInput->OutputIndex() );
	dnn.AddLayer( *subseq );

	return new CUserTensor( outputShape, convertedInput->Layout(), CLayerOutput( subseq, 0 ) );
}

// Prepares input tensor for slice
CPtr<const CUserTensor> CSliceNode::prepareInputForSlice( const CUserTensor& input, int axis ) const
{
	TBlobDim currDim = input.Layout()[axis];

	if( currDim == BD_BatchLength ) {
		return &input;
	}

	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( input.DimCount() );
	for( int i = 0; i < input.DimCount(); ++i ) {
		outputLayout.Add( static_cast<TBlobDim>( i ) );
	}
	swap( outputLayout[axis], outputLayout[0] );

	return dynamic_cast<const CUserTensor*>( ConvertTensor( input, outputLayout ).Ptr() );
}

} // namespace NeoOnnx
