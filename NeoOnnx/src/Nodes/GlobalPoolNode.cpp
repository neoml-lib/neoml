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

#include "GlobalPoolNode.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGlobalPoolNodeBase::CGlobalPoolNodeBase( TPoolType _poolType, const onnx::NodeProto& onnxNode, int opsetVersion ) :
	COpNode( onnxNode, opsetVersion ),
	poolType( _poolType )
{
	CheckOnnxProtocol( InputCount() == 1, "node must have 1 input", OnnxNode );
	CheckOnnxProtocol( OutputCount() == 1, "node must have 1 output", OnnxNode );
}

void CGlobalPoolNodeBase::PoolAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const
{
	// Global pooling interpret tensor as (B x C x D0 x D1 x .. x DN)
	// Where D0 ... DN are pooled dimensions
	CheckOnnxProtocol( inputShape.Size() >= 2, "Global pool input must be at least 2-dimensional", OnnxNode );
	axes.SetBufferSize( inputShape.Size() - 2 );
	for( int i = 2; i < inputShape.Size(); ++i ) {
		axes.Add( i );
	}
}

void CGlobalPoolNodeBase::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& outputs, CDnn& dnn )
{
	CheckNeoOnnxInternal( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "Input must be provided by user", OnnxNode );
	CheckNeoOnnxSupport( inputs[0]->Shape().Size() <= 7, "Tensor with 8+ dimensions", OnnxNode );

	CFastArray<int, 8> axes;
	PoolAxes( inputs[0]->Shape(), axes );

	CPtr<const CUserTensor> curr = dynamic_cast<const CUserTensor*>( inputs[0].Ptr() );
	curr = prepareInput( *curr, axes, dnn );
	curr = addPoolingLayer( *curr, axes, dnn );
	curr = addPostProcessing( *curr, dnn );

	outputs[0] = curr;
}

// Prepares input for NeoML's CGlobal*PoolingLayer
CPtr<const CUserTensor> CGlobalPoolNodeBase::prepareInput( const CUserTensor& input, const CFastArray<int, 8>& axes, CDnn& dnn ) const
{
	// Step 1: converting to proper dimension order
	CDimOrder inputOrder;
	calcInputDimOrder( input.Shape(), input.Layout(), axes, inputOrder );
	CPtr<const CUserTensor> result = dynamic_cast<const CUserTensor*>( ConvertTensor( input, CTensorLayout( inputOrder ) ).Ptr() );

	// Step 2: add pre-processing layers if needed
	if( poolType == PT_Min ) {
		// MinPool( x ) == -1 * MaxPool( -1 * x )
		CPtr<CLinearLayer> linear = new CLinearLayer( dnn.GetMathEngine() );
		linear->SetName( Name() + "_preProcess" );
		linear->SetMultiplier( -1 );
		linear->Connect( 0, *result->Layer(), result->OutputIndex() );
		dnn.AddLayer( *linear );
		result = new CUserTensor( result->Shape(), result->Layout(), CLayerOutput( linear, 0 ) );
	}

	return result;
}

// Gets tensor's blob dimensions (even if it isn't represented explicitly in layout.OnnxOrder)
static void getDimOrder( const CTensorShape& shape, const CTensorLayout& layout, CDimOrder& dimOrder )
{
	if( layout.DimType == DT_NeoML ) {
		layout.OnnxOrder.CopyTo( dimOrder );
		return;
	}

	dimOrder.SetBufferSize( shape.Size() );
	for( int dim = 0; dim < shape.Size(); ++dim ) {
		dimOrder.Add( static_cast<TBlobDim>( dim ) );
	}
}

// Check if current layout is compatible with Global*PoolingLayer
static bool isCompatibleLayout( const CTensorShape& shape, const CTensorLayout& layout, const CFastArray<int, 8>& pooledDims,
	const CFastArray<int, 8>& remainingDims )
{
	CDimOrder inputDimOrder;
	getDimOrder( shape, layout, inputDimOrder );

	// Check that every pooled axe is BD_Height, BD_Width or BD_Depth
	for( int i = 0; i < pooledDims.Size(); ++i ) {
		if( inputDimOrder[pooledDims[i]] < BD_Height || inputDimOrder[pooledDims[i]] == BD_Channels ) {
			return false;
		}
	}

	// Check that every remaining axe is BD_BatchLength, BD_BatchWidth, BD_ListSize or BD_Channels
	for( int i = 0; i < remainingDims.Size(); ++i ) {
		if( inputDimOrder[remainingDims[i]] >= BD_Height && inputDimOrder[remainingDims[i]] <= BD_Depth ) {
			return false;
		}
	}

	return true;
}

// Generates dimOrder which is compatible with NeoML's CGlobal*PoolingLayer
void CGlobalPoolNodeBase::calcInputDimOrder( const CTensorShape& inputShape, const CTensorLayout& inputLayout,
	const CFastArray<int, 8>& axes, CDimOrder& order ) const
{
	order.DeleteAll();
	order.Add( BD_Count, inputShape.Size() );

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
		"Global pooling which can't be emulated by NeoML", OnnxNode );

	if( isCompatibleLayout( inputShape, inputLayout, pooledDims, remainingDims ) ) {
		inputLayout.OnnxOrder.CopyTo( order );
		return;
	}

	// Distribute dimensions which can be pooled between non-trivial pooled dims
	for( int i = 0; i < pooledDims.Size(); ++i ) {
		order[pooledDims[i]] = BD_Height + i;
	}

	{
		// Distribute dimensions which can't be pooled between non-trivial remaining dims
		CDimOrder possibleRemainingDims( { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Channels } );
		for( int i = 0; i < remainingDims.Size(); ++i ) {
			order[remainingDims[i]] = possibleRemainingDims[i];
		}
	}

	// Distribute unused dimensions between the rest (trivial) of tensor dimensions
	TBlobDim currDim = BD_BatchLength;
	for( int i = 0; i < order.Size(); ++i ) {
		if( order[i] == -BD_Count ) {
			while( order.Find( currDim ) != NotFound && currDim != BD_Count ) {
				++currDim;
			}
			// Double-check
			CheckNeoOnnxInternal( currDim != BD_Count, "Can't distribute blob dimensions between tensor dims", OnnxNode );
			order[i] = currDim;
		}
	}
}

// Adds CGlobal*Pooling layer
CPtr<const CUserTensor> CGlobalPoolNodeBase::addPoolingLayer( const CUserTensor& preparedInput,
	const CFastArray<int, 8>& axes, CDnn& dnn ) const
{
	static_assert( PT_Count == 3, "PT_Count != 3" );
	CPtr<CBaseLayer> pooling;
	switch( poolType ) {
		case PT_Max:
		case PT_Min:
			pooling = new CGlobalMaxPoolingLayer( dnn.GetMathEngine() );
			break;
		case PT_Mean:
			pooling = new CGlobalMeanPoolingLayer( dnn.GetMathEngine() );
			break;
		default:
			CheckNeoOnnxInternal( false, "Unknown pooling type", OnnxNode );
	}

	pooling->SetName( Name() );
	pooling->Connect( 0, *preparedInput.Layer(), preparedInput.OutputIndex() );
	dnn.AddLayer( *pooling );

	CTensorShape outputShape;
	calcOutputShape( preparedInput.Shape(), axes, outputShape );

	CDimOrder outputDimOrder;
	calcOutputDimOrder( preparedInput.Layout().OnnxOrder, axes, outputDimOrder );

	return new CUserTensor( outputShape, CTensorLayout( outputDimOrder ), CLayerOutput( pooling, 0 ) );
}

// Calculate this node's output shape
void CGlobalPoolNodeBase::calcOutputShape( const CTensorShape& inputShape, const CFastArray<int, 8>& axes, CTensorShape& outputShape ) const
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

// Calculate this node's output shape
void CGlobalPoolNodeBase::calcOutputDimOrder( const CDimOrder& inputDimOrder,
	const CFastArray<int, 8>& axes, CDimOrder& outputDimOrder ) const
{
	outputDimOrder.DeleteAll();

	const bool keepDims = KeepDims();
	
	if( keepDims ) {
		inputDimOrder.CopyTo( outputDimOrder );
		return;
	}

	int axeIndex = 0;
	outputDimOrder.SetBufferSize( inputDimOrder.Size() - axes.Size() );

	for( int dimIndex = 0; dimIndex < inputDimOrder.Size(); ++dimIndex ) {
		if( axeIndex < axes.Size() && dimIndex == axes[axeIndex] ) {
			axeIndex++;
		} else {
			outputDimOrder.Add( inputDimOrder[dimIndex] );
		}
	}
}

// Adds additional layers after pooling if needed.
CPtr<const CUserTensor> CGlobalPoolNodeBase::addPostProcessing( const CUserTensor& layerOutput, CDnn& dnn ) const
{
	if( poolType != PT_Min ) {
		// Post-processing is needed only when MinPooling
		return &layerOutput;
	}

	// MinPool( x ) == -1 * MaxPool( -1 * x )
	CPtr<CLinearLayer> linear = new CLinearLayer( dnn.GetMathEngine() );
	linear->SetName( Name() + "_postProcess" );
	linear->SetMultiplier( -1 );
	linear->Connect( 0, *layerOutput.Layer(), layerOutput.OutputIndex() );
	dnn.AddLayer( *linear );
	return new CUserTensor( layerOutput.Shape(), layerOutput.Layout(), CLayerOutput( linear, 0 ) );
}

// --------------------------------------------------------------------------------------------------------------------
// Reduce operators which can be emulated via glob pooling

bool CReducePoolNodeBase::KeepDims() const
{
	return Attributes.GetOptionalInt( "keepdims", 1 ) != 0;
}

void CReducePoolNodeBase::PoolAxes( const CTensorShape& inputShape, CFastArray<int, 8>& axes ) const
{
	Attributes.GetOptionalIntArray( "axes", axes );

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
			CheckOnnxProtocol( OpsetVersion >= 11, "negative axes indices are supported since v11", OnnxNode );
			axes[i] += inputShape.Size();
		}
	}

	axes.QuickSort<Ascending<int>>();
}

} // namespace NeoOnnx
