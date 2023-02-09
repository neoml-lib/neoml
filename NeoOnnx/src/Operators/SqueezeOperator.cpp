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

#include "SqueezeOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSqueezeOperator::CSqueezeOperator( const onnx::NodeProto& squeeze, int opsetVersion ) :
	CLayerOperator( squeeze, opsetVersion )
{
	// v1 - original
	// v11 - added negative axes index support
	// v13 - axes values are moved from attributes to inputs
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 13 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() == 1 || InputCount() == 2, "operator must have 1 or 2 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CSqueezeOperator::AddLayers( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	CFastArray<int, 8> axes;
	getAxes( inputs[0]->DimCount(), inputs.Size() > 1 ? inputs[1].Ptr() : nullptr, axes);

	const CTensorLayout outputLayout = calcOutputLayout( *inputs[0], axes );
	static_assert( static_cast<int>( TTensorType::Count ) == 3, "TTensorType::Count != 3" );
	if( inputs[0]->Type() == TTensorType::Data ) {
		outputs.Add( new CDataTensor( outputLayout,
			*dynamic_cast<const CDataTensor*>( inputs[0].Ptr() )->Data() ) );
	} else if( inputs[0]->Type() == TTensorType::Shape ) {
		const CShapeTensor& input = dynamic_cast<const CShapeTensor&>( *inputs[0] );
		CTensorShape outputShape;
		calcOutputShape( input.Shape(), axes, outputShape );
		outputs.Add( new CShapeTensor( outputLayout, outputShape, input.LayerOutput() ) );
	} else {
		outputs.Add( new CUserTensor( outputLayout,
			dynamic_cast<const CUserTensor*>( inputs[0].Ptr() )->LayerOutput() ) );
	}
}

// Calculates output tensor's shape
void CSqueezeOperator::calcOutputShape( const CTensorShape& inputShape, const CFastArray<int, 8>& axes, CTensorShape& outputShape ) const
{
	outputShape.Empty();
	if( axes.IsEmpty() ) {
		for( int i = 0; i < inputShape.Size(); ++i ) {
			if( inputShape[i] != 1 ) {
				outputShape.Add( inputShape[i] );
			}
		}
		return;
	}

	outputShape.SetBufferSize( inputShape.Size() - axes.Size() );

	int axeIndex = 0;
	for( int i = 0; i < inputShape.Size(); ++i ) {
		if( axeIndex < axes.Size() && i == axes[axeIndex] ) {
			++axeIndex;
		} else {
			outputShape.Add( inputShape[i] );
		}
	}
}

// Fills array with axes indices to be squeezed
// Returns array of positive indices in sorted order
void CSqueezeOperator::getAxes( int inputDimCount, const CTensorBase* axesInput, CFastArray<int, 8>& axes ) const
{
	axes.Empty();
	if( OpsetVersion >= 13 ) {
		if( axesInput == nullptr ) {
			return;
		}

		// If axes are provided by input, we need to know exact values during conversion
		// otherwise we won't be able to calculate output layout
		CheckNeoOnnxSupport( axesInput->Type() == TTensorType::Data, "'axesInput' with tensor without shape", *this );
		const CDataTensor& axesData = dynamic_cast<const CDataTensor&>( *axesInput );
		axes.SetSize( axesData.Data()->GetDataSize() );
		axesData.Data()->CopyTo( axes.GetPtr() );
		return;
	}

	CheckOnnxProtocol( GetAttribute( "axes", axes ), "'axes' attribute is missing", *this );
	for( int i = 0; i < axes.Size(); ++i ) {
		if( axes[i] < 0 ) {
			CheckOnnxProtocol( OpsetVersion >= 11, "negative axes indices are supported since v11", *this );
			axes[i] += inputDimCount;
		}
	}
	axes.QuickSort<Ascending<int>>();
}

// Calculates output tensor's layout
CTensorLayout CSqueezeOperator::calcOutputLayout( const CTensorBase& input, const CFastArray<int, 8>& axes ) const
{
	CTensorLayout outputLayout;
	if( axes.IsEmpty() ) {
		// When 'axes' are empty squeeze removes all dims of size 1
		// In NeoOnnx we have to remove them from the layout as well
		CheckNeoOnnxSupport( input.Type() != TTensorType::User,
			"when 'axes' is empty input tensor must have shape", *this );
		CTensorShape inputShape;
		GetTensorShape( input, inputShape );
		for( int i = 0; i < inputShape.Size(); ++i ) {
			if( inputShape[i] != 1 ) {
				outputLayout.Add( input.Layout()[i] );
			}
		}
		return outputLayout;
	}

	int axeIndex = 0;
	outputLayout.SetBufferSize( input.DimCount() - axes.Size());

	// Distribute unused blob dimensions among new axes
	for( int i = 0; i < input.DimCount(); ++i ) {
		if( axeIndex < axes.Size() && i == axes[axeIndex] ) {
			++axeIndex;
		} else {
			outputLayout.Add( input.Layout()[i] );
		}
	}

	return outputLayout;
}

} // namespace NeoOnnx
