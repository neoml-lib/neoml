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
	// v13 - axes are moved from attributes to inputs
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	if( OpsetVersion < 13 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CUnsqueezeOperator::AddLayers( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	CFastArray<int, 8> axes;
	getAxes( inputs, axes );

	CTensorLayout outputLayout = calcOutputLayout( inputs[0]->Layout(), axes );

	if( inputs[0]->Type() == TTensorType::Data ) {
		outputs.Add( new CDataTensor( outputLayout,
			*dynamic_cast<const CDataTensor*>( inputs[0].Ptr() )->Data() ) );
	} else if( inputs[0]->Type() == TTensorType::Shape ) {
		const CShapeTensor& inputShapeTensor = dynamic_cast<const CShapeTensor&>( *inputs[0] );
		CTensorShape outputShape;
		calcOutputShape( inputShapeTensor.Shape(), axes, outputShape );
		outputs.Add( new CShapeTensor( outputLayout, outputShape, inputShapeTensor.LayerOutput() ) );
	} else {
		outputs.Add( new CUserTensor( outputLayout,
			dynamic_cast<const CUserTensor*>( inputs[0].Ptr() )->LayerOutput() ) );
	}
}

// Fills array with axes indices to be squeezed
// Returns array of positive indices in sorted order
void CUnsqueezeOperator::getAxes( const CTensorArray& inputs, CFastArray<int, 8>& axes ) const
{
	axes.Empty();
	if( OpsetVersion >= 13 ) {
		CheckNeoOnnxSupport( inputs.Size() == 2 && inputs[1] != nullptr && inputs[1]->Type() == TTensorType::Data,
			"axes input must be constant", *this );
		const CDataTensor& axesData = dynamic_cast<const CDataTensor&>( *inputs[1] );
		axes.SetSize( axesData.Data()->GetDataSize() );
		axesData.Data()->CopyTo( axes.GetPtr() );
	} else {
		GetAttribute( "axes", axes );
	}

	const int inputDimCount = inputs[0]->DimCount();
	for( int i = 0; i < axes.Size(); ++i ) {
		if( axes[i] < 0 ) {
			axes[i] += inputDimCount + axes.Size();
		}
	}
	axes.QuickSort<Ascending<int>>();
}

// Calculates output tensor's shape
void CUnsqueezeOperator::calcOutputShape( const CTensorShape& inputShape, const CFastArray<int, 8>& axes,
	CTensorShape& outputShape ) const
{
	outputShape.DeleteAll();
	int axeIndex = 0;
	int inputDimIndex = 0;
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
