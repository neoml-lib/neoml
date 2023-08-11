/* Copyright © 2017-2023 ABBYY

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

// Validator for global pool operators
class CGlobalPoolLayoutValidator : public ITensorLayoutValidator {
public:
	explicit CGlobalPoolLayoutValidator( const CFastArray<int, 8>& _pooledAxes );

	bool operator()( const CTensorLayout& layout ) const override;

private:
	CFastArray<int, 8> pooledAxes;
};

CGlobalPoolLayoutValidator::CGlobalPoolLayoutValidator( const CFastArray<int, 8>& _pooledAxes )
{
	_pooledAxes.CopyTo( pooledAxes );
	CheckNeoOnnxSupport( pooledAxes.Size() <= 3, "Global pooling may have up to 3 pooled dims" );
	for( int i = 0; i < pooledAxes.Size(); ++i ) {
		NeoAssert( pooledAxes[i] >= 0 );
		NeoAssert( i == 0 || pooledAxes[i - 1] < pooledAxes[i] );
	}
}

bool CGlobalPoolLayoutValidator::operator()( const CTensorLayout& layout ) const
{
	NeoAssert( layout.Size() >= pooledAxes.Size() );
	CheckNeoOnnxSupport( layout.Size() - pooledAxes.Size() <= 4, "Global pooling may have up to 4 unpooled dims" );
	for( int i = 0; i < layout.Size(); ++i ) {
		if( pooledAxes.Find( i ) == NotFound && layout[i] >= BD_Height && layout[i] <= BD_Depth ) {
			return false; // non-pooled axes must be in { BD_BatchLength, BD_BatchWidth, BD_ListSize, BD_Channels }
		} else if( pooledAxes.Find( i ) != NotFound && ( layout[i] < BD_Height || layout[i] == BD_Channels ) ) {
			return false; // pooled axes must be in { BD_Height, BD_Width, BD_Depth }
		}
	}
	return true;
}

//---------------------------------------------------------------------------------------------------------------------

CGlobalPoolOperatorBase::CGlobalPoolOperatorBase( TPoolType _poolType, const onnx::NodeProto& onnxNode, int opsetVersion ) :
	CLayerOperator( onnxNode, opsetVersion ),
	poolType( _poolType )
{
	if( OpsetVersion < 13 || Type() != "ReduceSum" ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	} else {
		CheckOnnxProtocol( InputCount() == 1 || InputCount() == 2, "operator must have 1 or 2 inputs", *this );
	}
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CGlobalPoolOperatorBase::PoolAxes( const CTensorArray& inputs, CFastArray<int, 8>& axes ) const
{
	// Global pooling interpret tensor as (B x C x D0 x D1 x .. x DN)
	// Where D0 ... DN are pooled dimensions
	const int inputDimCount = inputs[0]->DimCount();
	CheckOnnxProtocol( inputDimCount >= 2, "Global pool input must be at least 2-dimensional", *this );
	axes.SetBufferSize( inputDimCount - 2 );
	for( int i = 2; i < inputDimCount; ++i ) {
		axes.Add( i );
	}
}

void CGlobalPoolOperatorBase::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNeoOnnxSupport( inputs[0] != nullptr, "data input must be present", *this );
	CheckNoShapeInputs( inputs );

	CFastArray<int, 8> axes;
	PoolAxes( inputs, axes );

	if( axes.IsEmpty() ) {
		outputs.Add( inputs[0] );
		return;
	}

	CPtr<const CUserTensor> curr = AsUserTensor( *ConvertTensor( *inputs[0], CGlobalPoolLayoutValidator( axes ) ),
		Name() + "_Source", dnn );
	curr = prepareInput( *curr, dnn );
	curr = addPoolingLayer( *curr, axes, dnn );
	curr = addPostProcessing( *curr, dnn );

	outputs.Add( curr.Ptr() );
}

// Prepares input for NeoML's CGlobal*PoolingLayer
CPtr<const CUserTensor> CGlobalPoolOperatorBase::prepareInput( const CUserTensor& input, CDnn& dnn ) const
{
	// Add pre-processing layers if needed
	static_assert( PT_Count == 5, "PT_Count != 5" );
	if( poolType == PT_Min ) {
		// MinPool( x ) == -1 * MaxPool( -1 * x )
		CPtr<CLinearLayer> linear = new CLinearLayer( dnn.GetMathEngine() );
		linear->SetName( Name() + "_preProcess" );
		linear->SetMultiplier( -1 );
		linear->Connect( 0, *input.Layer(), input.OutputIndex() );
		dnn.AddLayer( *linear );
		return new CUserTensor( input.Layout(), CLayerOutput( linear, 0 ) );
	} else if( poolType == PT_L2 ) {
		// L2Pool = Sqrt(SumPool(x^2))
		CPtr<CPowerLayer> sqare = new CPowerLayer( dnn.GetMathEngine() );
		sqare->SetName( Name() + "_preProcess" );
		sqare->SetExponent( 2.f );
		sqare->Connect( 0, *input.Layer(), input.OutputIndex() );
		dnn.AddLayer( *sqare );
		return new CUserTensor( input.Layout(), CLayerOutput( sqare, 0 ) );
	}

	return &input;
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

	return new CUserTensor( calcOutputLayout( preparedInput.Layout(), axes ), CLayerOutput( pooling, 0 ) );
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
		return new CUserTensor( layerOutput.Layout(), CLayerOutput( linear, 0 ) );
	} else if( poolType == PT_L2 ) {
		// L2Pool( x ) = Sqrt( SumPool( x^2 ) )
		CPtr<CPowerLayer> power = new CPowerLayer( dnn.GetMathEngine() );
		power->SetName( Name() + "_postProcess" );
		power->SetExponent( 0.5f );
		power->Connect( 0, *layerOutput.Layer(), layerOutput.OutputIndex() );
		dnn.AddLayer( *power );
		return new CUserTensor( layerOutput.Layout(), CLayerOutput( power, 0 ) );
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

void CReducePoolOperatorBase::PoolAxes( const CTensorArray& inputs, CFastArray<int, 8>& axes ) const
{
	const int inputDimCount = inputs[0]->DimCount();
	GetAttribute( "axes", axes );

	// If axes attribute is missing then all dimensions must be pooled
	if( axes.IsEmpty() ) {
		axes.SetBufferSize( inputDimCount );
		for( int i = 0; i < inputDimCount; ++i ) {
			axes.Add( i );
		}
		return;
	}

	for( int i = 0; i < axes.Size(); ++i ) {
		if( axes[i] < 0 ) {
			axes[i] += inputDimCount;
		}
	}

	axes.QuickSort<Ascending<int>>();
}

// --------------------------------------------------------------------------------------------------------------------

void CReduceSumOperator::PoolAxes( const CTensorArray& inputs, CFastArray<int, 8>& axes ) const
{
	// Since v13 ReduceSum axes moved from attributes to inputs
	// But the rest of Reduce* operators will be changed in this way only in v18
	if( OpsetVersion < 13 ) {
		// Before v13 the axis rules were the same for all the operators
		CReducePoolOperatorBase::PoolAxes( inputs, axes );
		return;
	}

	const int inputDimCount = inputs[0]->DimCount();

	// Process case when axes input is missing
	if( inputs.Size() == 1 || inputs[1] == nullptr ) {
		axes.Empty();

		int noOpWithEmptyAxes = 0;
		GetAttribute( "noop_with_empty_axes", noOpWithEmptyAxes );
		if( noOpWithEmptyAxes == 0 ) {
			// Reduce all axes when special flag is not set
			axes.SetSize( inputDimCount );
			for( int i = 0; i < inputDimCount; ++i ) {
				axes[i] = i;
			}
		}
	}

	// Copy axes from the second input
	CheckNeoOnnxSupport( inputs[1]->Type() == TTensorType::Data, "non-constant axes", *this );
	const CDnnBlob& axesBlob = *dynamic_cast<const CDataTensor&>( *inputs[1] ).Data();
	axes.SetSize( axesBlob.GetDataSize() );
	axesBlob.CopyTo( axes.GetPtr() );

	for( int i = 0; i < axes.Size(); ++i ) {
		if( axes[i] < 0 ) {
			axes[i] += inputDimCount;
		}
	}
	axes.QuickSort<Ascending<int>>();
}

} // namespace NeoOnnx
