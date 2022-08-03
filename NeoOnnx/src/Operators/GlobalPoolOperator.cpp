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

#include "common.h"
#pragma hdrstop

#include "GlobalPoolOperator.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGlobalPoolOperatorBase::CGlobalPoolOperatorBase( TPoolType _poolType, const onnx::NodeProto& onnxNode, int opsetVersion ) :
	CLayerOperator( onnxNode, opsetVersion ),
	poolType( _poolType )
{
	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CGlobalPoolOperatorBase::PoolAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const
{
	// Global pooling interpret tensor as (B x C x D0 x D1 x .. x DN)
	// Where D0 ... DN are pooled dimensions
	CheckOnnxProtocol( inputShape.Size() >= 2, "Global pool input must be at least 2-dimensional", *this );
	axes.SetBufferSize( inputShape.Size() - 2 );
	for( int i = 2; i < inputShape.Size(); ++i ) {
		axes.Add( i );
	}
}

void CGlobalPoolOperatorBase::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );

	CFastArray<int, 8> axes;
	PoolAxes( inputs[0]->Shape(), axes );

	CPtr<const CUserTensor> curr = AsUserTensor( *inputs[0], Name() + "_Source", dnn );
	curr = prepareInput( *curr, axes, dnn );
	curr = addPoolingLayer( *curr, axes, dnn );
	curr = addPostProcessing( *curr, dnn );

	outputs.Add( curr.Ptr() );
}

// Prepares input for NeoML's CGlobal*PoolingLayer
CPtr<const CUserTensor> CGlobalPoolOperatorBase::prepareInput( const CUserTensor& input, const CFastArray<int, 8>& axes, CDnn& dnn ) const
{
	// Step 1: converting to proper layout (if needed)
	CPtr<const CUserTensor> result = convertInputLayout( input, axes );

	// Step 2: add pre-processing layers if needed
	static_assert( PT_Count == 5, "PT_Count != 5" );
	if( poolType == PT_Min ) {
		// MinPool( x ) == -1 * MaxPool( -1 * x )
		CPtr<CLinearLayer> linear = new CLinearLayer( dnn.GetMathEngine() );
		linear->SetName( Name() + "_preProcess" );
		linear->SetMultiplier( -1 );
		linear->Connect( 0, *result->Layer(), result->OutputIndex() );
		dnn.AddLayer( *linear );
		result = new CUserTensor( result->Shape(), result->Layout(), CLayerOutput( linear, 0 ) );
	} else if( poolType == PT_L2 ) {
		// L2Pool = Sqrt(SumPool(x^2))
		CPtr<CPowerLayer> sqare = new CPowerLayer( dnn.GetMathEngine() );
		sqare->SetName( Name() + "_preProcess" );
		sqare->SetExponent( 2.f );
		sqare->Connect( 0, *result->Layer(), result->OutputIndex() );
		dnn.AddLayer( *sqare );
		result = new CUserTensor( result->Shape(), result->Layout(), CLayerOutput( sqare, 0 ) );
	}

	return result;
}

// Check if current layout is compatible with Global*PoolingLayer
static bool isCompatibleLayout( const CTensorLayout& layout, const CFastArray<int, 8>& pooledDims,
	const CFastArray<int, 8>& remainingDims )
{
	// Check that pooled axes are BD_Height, BD_Width or BD_Depth
	for( int i = 0; i < pooledDims.Size(); ++i ) {
		if( layout[pooledDims[i]] < BD_Height || layout[pooledDims[i]] == BD_Channels ) {
			return false;
		}
	}

	// Check that remaining axes are BD_BatchLength, BD_BatchWidth, BD_ListSize or BD_Channels
	for( int i = 0; i < remainingDims.Size(); ++i ) {
		if( layout[remainingDims[i]] >= BD_Height && layout[remainingDims[i]] <= BD_Depth ) {
			return false;
		}
	}

	return true;
}

// Convert input's layout into compatible with NeoML's CGlobal*PoolingLayer
CPtr<const CUserTensor> CGlobalPoolOperatorBase::convertInputLayout( const CUserTensor& input,
	const CFastArray<int, 8>& axes ) const
{
	const CTensorShape& inputShape = input.Shape();
	const CTensorLayout& inputLayout = input.Layout();

	// Non-trivial dims to be pooled
	CFastArray<int, 8> pooledDims;
	// Non-trivial dims not to be pooled
	CFastArray<int, 8> remainingDims;
	int axeIndex = 0;
	for( int dimIndex = 0; dimIndex < inputShape.Size(); ++dimIndex ) {
		if( axeIndex < axes.Size() && dimIndex == axes[axeIndex] ) {
			if( inputShape[dimIndex] > 1 ) {
				pooledDims.Add( dimIndex );
			}
			axeIndex++;
		} else if( inputShape[dimIndex] > 1 ) {
			remainingDims.Add( dimIndex );
		}
	}
	CheckNeoOnnxSupport( pooledDims.Size() <= 3 && remainingDims.Size() <= 4,
		"Global pooling which can't be emulated by NeoML", *this );

	if( isCompatibleLayout( inputLayout, pooledDims, remainingDims ) ) {
		return &input;
	}

	CTensorLayout convertedLayout;
	convertedLayout.Add( BD_Count, inputShape.Size() );

	// Distribute dimensions which can be pooled between non-trivial pooled dims
	for( int i = 0; i < pooledDims.Size(); ++i ) {
		convertedLayout[pooledDims[i]] = BD_Height + i;
	}

	{
		// Distribute dimensions which can't be pooled between non-trivial remaining dims
		CTensorLayout possibleRemainingDims( { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Channels } );
		for( int i = 0; i < remainingDims.Size(); ++i ) {
			convertedLayout[remainingDims[i]] = possibleRemainingDims[i];
		}
	}

	// Distribute unused dimensions between the rest (trivial) of tensor dimensions
	TBlobDim currDim = BD_BatchLength;
	for( int i = 0; i < convertedLayout.Size(); ++i ) {
		if( convertedLayout[i] == BD_Count ) {
			while( convertedLayout.Find( currDim ) != NotFound && currDim != BD_Count ) {
				++currDim;
			}
			// Double-check
			NeoAssert( currDim != BD_Count );
			convertedLayout[i] = currDim;
		}
	}

	return ConvertTensor( input, convertedLayout );
}

// Adds CGlobal*Pooling layer
CPtr<const CUserTensor> CGlobalPoolOperatorBase::addPoolingLayer( const CUserTensor& preparedInput,
	const CFastArray<int, 8>& axes, CDnn& dnn ) const
{
	static_assert( PT_Count == 5, "PT_Count != 5" );
	CPtr<CBaseLayer> pooling;
	switch( poolType ) {
		case PT_Max:
		case PT_Min:
			pooling = new CGlobalMaxPoolingLayer( dnn.GetMathEngine() );
			break;
		case PT_Mean:
			pooling = new CGlobalMeanPoolingLayer( dnn.GetMathEngine() );
			break;
		case PT_Sum:
		case PT_L2:
			pooling = new CGlobalSumPoolingLayer( dnn.GetMathEngine() );
			break;
		default:
			NeoAssert( false );
	}

	pooling->SetName( Name() );
	pooling->Connect( 0, *preparedInput.Layer(), preparedInput.OutputIndex() );
	dnn.AddLayer( *pooling );

	CTensorShape outputShape;
	calcOutputShape( preparedInput.Shape(), axes, outputShape );

	return new CUserTensor( outputShape, calcOutputLayout( preparedInput.Layout(), axes ), CLayerOutput( pooling, 0 ) );
}

// Calculate this operator's output shape
void CGlobalPoolOperatorBase::calcOutputShape( const CTensorShape& inputShape, const CFastArray<int, 8>& axes, CTensorShape& outputShape ) const
{
	const bool keepDims = KeepDims();

	outputShape.DeleteAll();
	outputShape.SetBufferSize( keepDims ? inputShape.Size() : inputShape.Size() - axes.Size() );

	int axeIndex = 0;
	for( int dimIndex = 0; dimIndex < inputShape.Size(); ++dimIndex ) {
		if( axeIndex < axes.Size() && dimIndex == axes[axeIndex] ) {
			if( keepDims ) {
				outputShape.Add( 1 );
			}
			axeIndex++;
		} else {
			outputShape.Add( inputShape[dimIndex] );
		}
	}
}

// Calculate this operator's output shape
CTensorLayout CGlobalPoolOperatorBase::calcOutputLayout( const CTensorLayout& inputLayout, const CFastArray<int, 8>& axes ) const
{
	const bool keepDims = KeepDims();
	
	if( keepDims ) {
		return inputLayout;
	}

	int axeIndex = 0;
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( inputLayout.Size() - axes.Size() );

	for( int dimIndex = 0; dimIndex < inputLayout.Size(); ++dimIndex ) {
		if( axeIndex < axes.Size() && dimIndex == axes[axeIndex] ) {
			axeIndex++;
		} else {
			outputLayout.Add( inputLayout[dimIndex] );
		}
	}
	return outputLayout;
}

// Adds additional layers after pooling if needed.
CPtr<const CUserTensor> CGlobalPoolOperatorBase::addPostProcessing( const CUserTensor& layerOutput, CDnn& dnn ) const
{
	static_assert( PT_Count == 5, "PT_Count != 5" );
	if( poolType == PT_Min ) {
		// MinPool( x ) == -1 * MaxPool( -1 * x )
		CPtr<CLinearLayer> linear = new CLinearLayer( dnn.GetMathEngine() );
		linear->SetName( Name() + "_postProcess" );
		linear->SetMultiplier( -1 );
		linear->Connect( 0, *layerOutput.Layer(), layerOutput.OutputIndex() );
		dnn.AddLayer( *linear );
		return new CUserTensor( layerOutput.Shape(), layerOutput.Layout(), CLayerOutput( linear, 0 ) );
	} else if( poolType == PT_L2 ) {
		// L2Pool( x ) = Sqrt( SumPool( x^2 ) )
		CPtr<CPowerLayer> power = new CPowerLayer( dnn.GetMathEngine() );
		power->SetName( Name() + "_postProcess" );
		power->SetExponent( 0.5f );
		power->Connect( 0, *layerOutput.Layer(), layerOutput.OutputIndex() );
		dnn.AddLayer( *power );
		return new CUserTensor( layerOutput.Shape(), layerOutput.Layout(), CLayerOutput( power, 0 ) );
	}

	return &layerOutput;
}

// --------------------------------------------------------------------------------------------------------------------
// Reduce operators which can be emulated via glob pooling

bool CReducePoolOperatorBase::KeepDims() const
{
	int keepDims = 1;
	GetAttribute( "keepdims", keepDims );
	return keepDims != 0;
}

void CReducePoolOperatorBase::PoolAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const
{
	GetAttribute( "axes", axes );

	// If axes attribute is missing then all dimensions must be pooled
	if( axes.IsEmpty() ) {
		axes.SetBufferSize( inputShape.Size() );
		for( int i = 0; i < inputShape.Size(); ++i ) {
			axes.Add( i );
		}
		return;
	}

	for( int i = 0; i < axes.Size(); ++i ) {
		if( axes[i] < 0 ) {
			axes[i] += inputShape.Size();
		}
	}

	axes.QuickSort<Ascending<int>>();
}

} // namespace NeoOnnx
