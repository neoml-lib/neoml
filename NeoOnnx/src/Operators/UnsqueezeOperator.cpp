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

#include "UnsqueezeOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CUnsqueezeOperator::CUnsqueezeOperator( const onnx::NodeProto& unsqueeze, int opsetVersion ) :
	CLayerOperator( unsqueeze, opsetVersion )
{
	// v1 - original
	// v11 - supported negative axes values
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CUnsqueezeOperator::AddLayers( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	NeoAssert( inputs[0] != nullptr && !inputs[0]->IsCalculated() );

	CFastArray<int, 8> axes;
	getAxes( inputs[0]->Shape(), axes );

	CTensorShape outputShape;
	calcOutputShape( inputs[0]->Shape(), axes, outputShape );

	CTensorLayout outputLayout = calcOutputLayout( inputs[0]->Layout(), axes );

	outputs.Add( new CUserTensor( outputShape, outputLayout,
		dynamic_cast<const CUserTensor*>( inputs[0].Ptr() )->LayerOutput() ) );
}

// Fills array with axes indices to be squeezed
// Returns array of positive indices in sorted order
void CUnsqueezeOperator::getAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const
{
	axes.Empty();
	GetAttribute( "axes", axes );
	for( int i = 0; i < axes.Size(); ++i ) {
		if( axes[i] < 0 ) {
			CheckOnnxProtocol( OpsetVersion >= 11, "negative axes indices are supported since v11", *this );
			axes[i] += inputShape.Size();
		}
	}
	axes.QuickSort<Ascending<int>>();
}

// Calculates output tensor's shape
void CUnsqueezeOperator::calcOutputShape( const CTensorShape& inputShape, const CFastArray<int, 8>& axes, CTensorShape& outputShape ) const
{
	outputShape.Empty();
	outputShape.SetBufferSize( inputShape.Size() + axes.Size() );

	int axeIndex = 0;
	int inputDimIndex = 0;
	outputShape.SetBufferSize( axes.Size() + inputShape.Size() );

	for( int i = 0; i < axes.Size() + inputShape.Size(); ++i ) {
		if( axeIndex < axes.Size() && i == axes[axeIndex] ) {
			outputShape.Add( 1 );
			++axeIndex;
		} else {
			NeoAssert( inputDimIndex < inputShape.Size() );
			outputShape.Add( inputShape[inputDimIndex] );
			++inputDimIndex;
		}
	}
}

// Calculates output tensor's layout
CTensorLayout CUnsqueezeOperator::calcOutputLayout( const CTensorLayout& inputLayout, const CFastArray<int, 8>& axes ) const
{
	// NeoML layout
	TBlobDim currDim = BD_BatchLength;
	int axeIndex = 0;
	int inputDimIndex = 0;
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( axes.Size() + inputLayout.Size() );

	// Distribute unused blob dimensions among new axes
	for( int i = 0; i < axes.Size() + inputLayout.Size(); ++i ) {
		if( axeIndex < axes.Size() && i == axes[axeIndex] ) {
			// Looking for unused blob dim
			while( currDim < BD_Count && inputLayout.Find( currDim ) != NotFound ) {
				++currDim;
			}
			NeoAssert( currDim != BD_Count );
			outputLayout.Add( currDim );
			++currDim;
			++axeIndex;
		} else {
			NeoAssert( inputDimIndex < inputLayout.Size() );
			outputLayout.Add( inputLayout[inputDimIndex] );
			++inputDimIndex;
		}
	}

	return outputLayout;
}

} // namespace NeoOnnx
