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

#include <common.h>
#pragma hdrstop

#include <NeoOnnx/NeoOnnxImport.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransposeHelper.h>
#include "Optimization/LayerNormFusionOptimizer.h"

namespace NeoOnnx {

class CObjectNormLayoutValidator : public ITensorLayoutValidator {
public:
	explicit CObjectNormLayoutValidator( const CFastArray<int, 8>& _axes ) { _axes.CopyTo( axes ); }

	bool operator()( const CTensorLayout& layout ) const override;
	void Print() const override { std::cout << "ObjectNorm layout"; }

private:
	CFastArray<int, 8> axes; // axes which will be normalized
};

bool CObjectNormLayoutValidator::operator()( const CTensorLayout& layout ) const
{
	for( int i = 0; i < layout.Size(); ++i ) {
		if( axes.Find( i ) == NotFound && layout[i] >= BD_Height ) {
			return false; // CObjectNormalizationLayer normalizes every object dimension
		} else if( axes.Find( i ) != NotFound && layout[i] < BD_Height ) {
			return false; // CObjectNormalizationLayer preserves every batch dimension
		}
	}
	return true;
}

namespace optimization {

static constexpr const char* const fusionNamePrefix{ "NormFusion_" };

// Information about layout change in ONNX
struct CLayerNormFusionOptimizer::CLayoutChange {
	CTensorLayout From; // Layout before this change
	CTensorLayout To; // Layout after this change

	bool operator==( const CLayoutChange& other ) const
	{ return From == other.From && To == other.To; }
	bool operator!=( const CLayoutChange& other ) const { return !( *this == other ); }
	// Checks that this layout switches back layouts from other change
	bool IsInversionOf( const CLayoutChange& other ) const
	{ return From == other.To && To == other.From; }
};

// Prepares object norm param blob
static CPtr<CDnnBlob> prepareObjNormParamBlob( const CDataLayer& dataLayer, const CTensorLayout& currentLayout,
	const CTensorLayout& objNormLayout )
{
	// Convert it via ONNX tensor utilities
	CPtr<CDataTensor> dataTensor = new CDataTensor( currentLayout, *( dataLayer.GetBlob() ) );
	return ConvertTensor( *dataTensor, objNormLayout )->Data()->GetCopy();
}

// Adds rename layer to the given input data
// Returns the output of this rename
static CLayerOutput<> addTensorLayoutRename( const CLayerOutput<> inputData,
	const CTensorLayoutRename& rename, CGraph& graph )
{
	NeoAssert( inputData.Layer != nullptr );
	NeoAssert( inputData.Index >= 0 );
	NeoAssert( rename.From.Size() == rename.To.Size() );

	if( rename.From == rename.To ) {
		return inputData; // No renaming needed
	}

	CPtr<COnnxTransformHelper> transformLayer = new COnnxTransformHelper(graph.MathEngine(),
		rename.From, rename.To );
	transformLayer->SetName( graph.GetUniqueName( CString( fusionNamePrefix ) + "_Transform" ) );
	for( int i = 0; i < rename.From.Size(); ++i ) {
		transformLayer->SetRule( rename.From[i], rename.To[i] );
	}
	graph.AddLayer( *transformLayer );
	graph.Connect( *transformLayer, 0, *inputData.Layer, inputData.Index );
	return CLayerOutput<>( transformLayer, 0 );
}

// Adds transpose layer to the given input data
// Returns the output of this transpose
static CLayerOutput<> addTensorLayoutTranspose( const CLayerOutput<> inputData,
	const CTensorLayout& inputLayout, CTensorLayout& outputLayout,
	const CTensorLayoutTranspose& transpose, CGraph& graph )
{
	NeoAssert( inputData.Layer != nullptr );
	NeoAssert( inputData.Index >= 0 );
	NeoAssert( transpose.First != transpose.Second );

	outputLayout = inputLayout;
	const int firstIndex = inputLayout.Find( transpose.First );
	const int secondIndex = inputLayout.Find( transpose.Second );
	NeoAssert( firstIndex != NotFound || secondIndex != NotFound );

	if( firstIndex != NotFound && secondIndex != NotFound ) {
		std::swap( outputLayout[firstIndex], outputLayout[secondIndex] );
	} else if( firstIndex == NotFound ) {
		outputLayout[secondIndex] = transpose.First;
	} else {
		outputLayout[firstIndex] = transpose.Second;
	}

	CPtr<COnnxTransposeHelper> transposeLayer = new COnnxTransposeHelper( graph.MathEngine(),
		inputLayout, outputLayout );
	transposeLayer->SetDims( transpose.First, transpose.Second );
	transposeLayer->SetName( graph.GetUniqueName( CString( fusionNamePrefix ) + "_Transpose" ) );
	graph.AddLayer( *transposeLayer );
	graph.Connect( *transposeLayer, 0, *inputData.Layer, inputData.Index );
	return CLayerOutput<>( transposeLayer, 0 );
}

// Selects the following chain of ONNX layers
//     -> [Transform] -> [Transpose ->]* -> [Transform] ->
// connected to inputIndex'th input of inputLayer
// Chains like these are generated by NeoOnnx in order to change tensor layout
// Must select at least one layer (otherwise it's a failure)
// If succeeded returns output connected to Transform's input and fills information about layouts before/after
// If failed returns empty output: CLayerOutput( nullptr, NotFound )
CLayerOutput<> CLayerNormFusionOptimizer::selectLayoutChange( CBaseLayer& inputLayer, int inputIndex,
	CLayoutChange& change ) const
{
	change.From.Empty();
	change.To.Empty();

	CLayerOutput<> currentOutput = graph.GetConnectedOutput<>( inputLayer, inputIndex );

	auto trySetOutputLayout = [&change] ( const CFastArray<TBlobDim, 8>& layout ) -> void
	{
		if( change.To.IsEmpty() ) {
			layout.CopyTo( change.To );
		}
	};

	auto processOptionalTransform = [&] () -> bool
	{
		COnnxTransformHelper* transform = dynamic_cast<COnnxTransformHelper*>( currentOutput.Layer );
		if( transform != nullptr ) {
			if( graph.GetConnectedInputsCount( *transform, 0 ) != 1 ) {
				return false;
			}
			trySetOutputLayout( transform->OutputLayout() );
			transform->InputLayout().CopyTo( change.From );
			graph.SelectLayer( *transform );
			currentOutput = graph.GetConnectedOutput( *transform, 0 );
		}
		return true;
	};

	// Process possible COnnxTransformLayer after transposes
	if( !processOptionalTransform() ) {
		return CLayerOutput<>();
	}

	// Process transposes
	while( dynamic_cast<COnnxTransposeHelper*>( currentOutput.Layer ) != nullptr ) {
		COnnxTransposeHelper* transpose = dynamic_cast<COnnxTransposeHelper*>( currentOutput.Layer );
		if( graph.GetConnectedInputsCount( *transpose, 0 ) != 1 ) {
			return CLayerOutput<>();
		}
		trySetOutputLayout( transpose->OutputLayout() );
		transpose->InputLayout().CopyTo( change.From );
		graph.SelectLayer( *currentOutput.Layer );
		currentOutput = graph.GetConnectedOutput( *transpose, 0 );
	}

	// Process possible COnnxTransformLayer before transposes
	if( !processOptionalTransform() ) {
		return CLayerOutput<>();
	}

	if( change.From.IsEmpty() ) {
		return CLayerOutput<>(); // No layout conversion detected
	}

	return currentOutput;
}

// Checks if DataLayer is valid for CLayerNormFusionOptimizer conversion
bool CLayerNormFusionOptimizer::isValidDataLayer( const CDataLayer& dataLayer, TBlobType blobType, int blobSize ) const
{
	NeoAssert( graph.GetInputCount( dataLayer ) == 0 );
	NeoAssert( graph.GetOutputCount( dataLayer ) == 1 );

	if( graph.GetConnectedInputsCount( dataLayer, /*outputIndex*/0 ) != 1 ) {
		return false;
	}

	CPtr<CDnnBlob> blob = dataLayer.GetBlob();
	return ( blob->GetDataType() == blobType
		&& ( blobSize == NotFound || blob->GetDataSize() == blobSize ) );
}

// Checks if CastLayer is valid for CLayerNormFusionOptimizer conversion
bool CLayerNormFusionOptimizer::isValidCastLayer( const CCastLayer& castLayer ) const
{
	NeoAssert( graph.GetInputCount( castLayer ) == 1 );
	NeoAssert( graph.GetOutputCount( castLayer ) == 1 );

	return castLayer.GetOutputType() == CT_Float;
}

// Adds tensor layout change to the given input data
// Returns the output of this change
CLayerOutput<> CLayerNormFusionOptimizer::changeLayout( const CLayerOutput<>& inputData,
	const CTensorLayout& inputLayout, const ITensorLayoutValidator& validator,
	CTensorLayout& outputLayout )
{
	CTensorLayoutRename preTransposeRename;
	CFastArray<CTensorLayoutTranspose, 2> transposes;
	CTensorLayoutRename postTransposeRename;
	outputLayout = FindOptimalConversion( inputLayout, validator,
		preTransposeRename, transposes, postTransposeRename );

	CLayerOutput<> currentOutput = addTensorLayoutRename( inputData, preTransposeRename, graph );

	CTensorLayout transposeInputLayout = preTransposeRename.To.IsEmpty() ? inputLayout : preTransposeRename.To;
	CTensorLayout transposeOutputLayout = transposeInputLayout;
	for( const CTensorLayoutTranspose& transpose : transposes ) {
		currentOutput = addTensorLayoutTranspose( currentOutput, transposeInputLayout, transposeOutputLayout,
			transpose, graph );
		transposeOutputLayout.CopyTo( transposeInputLayout );
	}

	return addTensorLayoutRename( currentOutput, postTransposeRename, graph );
}

int CLayerNormFusionOptimizer::Apply()
{
	int optimizedLayers = 0;

	NeoAssert( graph.SelectionSize() == 0 );

	CArray<CBaseLayer*> layers{};
	graph.GetLayers( layers );
	for( auto& layer : layers ) {
		graph.ClearSelection();

		if( !graph.HasLayer( layer ) ) { // Skip already replaced layers
			continue;
		}

		// Searching for a group of layers to replace by an object normalization layer in the backward direction through the graph
		// From bottom to upside for graph
		auto* addLayerLast = dynamic_cast<COnnxEltwiseLayer*>( layer );
		if( addLayerLast == nullptr
			|| !isValidArithmeticLayer( *addLayerLast, COnnxEltwiseLayer::TOperation::Add )
			|| graph.IsLayerSelected( *addLayerLast ) )
		{
			continue; // fail this Fusion
		}
		graph.SelectLayer( *addLayerLast );

		CLayerOutput<CDataLayer> bias{};
		CLayerOutput<COnnxEltwiseLayer> mul{};
		if( !graph.SelectBothConnectedOutputs<COnnxEltwiseLayer, CDataLayer>( *addLayerLast, mul, bias, /*checkOutOfSelectionLinks*/false )
			|| !isValidArithmeticLayer( *mul.Layer, COnnxEltwiseLayer::TOperation::Mul )
			|| !isValidDataLayer( *bias.Layer, CT_Float, /*size*/NotFound ) )
		{
			continue; // fail this Fusion
		}

		CLayerOutput<CDataLayer> scale{};
		CLayerOutput<COnnxEltwiseLayer> div{};
		CLayerOutput<CCastLayer> uselessCast{}; // try to skip CAST layer as operand (1)
		if( !graph.SelectBothConnectedOutputs<CCastLayer, CDataLayer>( *mul.Layer, uselessCast, scale, /*checkOutOfSelectionLinks*/false )
			|| !isValidCastLayer( *uselessCast.Layer )
			|| !isValidDataLayer( *scale.Layer, CT_Float, /*size*/NotFound ) )
		{
			scale.Clear();
			uselessCast.Clear(); // try to skip CAST layer as operand (2)
			if( !graph.SelectBothConnectedOutputs<COnnxEltwiseLayer, CCastLayer>( *mul.Layer, div, uselessCast, /*checkOutOfSelectionLinks*/false )
				|| !isValidArithmeticLayer( *div.Layer, COnnxEltwiseLayer::TOperation::Div )
				|| !isValidCastLayer( *uselessCast.Layer ) )
			{
				div.Clear();
				uselessCast.Clear(); // no CAST layer as both of operands (3)
				if( !graph.SelectBothConnectedOutputs<COnnxEltwiseLayer, CDataLayer>( *mul.Layer, div, scale, /*checkOutOfSelectionLinks*/false )
					|| !isValidArithmeticLayer( *div.Layer, COnnxEltwiseLayer::TOperation::Div )
					|| !isValidDataLayer( *scale.Layer, CT_Float, /*size*/NotFound ) )
				{
					continue; // fail this Fusion
				}
			} else { // success to find the CAST layer as operand (2)
				scale.Layer = graph.SelectTheOnlyConnectedOutput<CDataLayer>( *uselessCast.Layer, /*checkOutOfSelectionLinks*/false );
				if( scale.Layer == nullptr
					|| !isValidDataLayer( *scale.Layer, CT_Float, /*size*/NotFound ) )
				{
					continue; // fail this Fusion
				}
			}
		} else { // success to find the CAST layer as operand (1)
			div.Layer = graph.SelectTheOnlyConnectedOutput<COnnxEltwiseLayer>( *uselessCast.Layer, /*checkOutOfSelectionLinks*/false );
			if( div.Layer == nullptr
				|| !isValidArithmeticLayer( *div.Layer, COnnxEltwiseLayer::TOperation::Div ) )
			{
				continue; // fail this Fusion
			}
		}

		CLayerOutput<> sqrtOutput{};
		CLayerOutput<COnnxEltwiseLayer> sub2{};
		CLayoutChange layoutChangeAfterSecondPooling;
		for( int subIndex = 0; subIndex < 2; ++subIndex ) {
			sub2 = graph.SelectConnectedOutput<COnnxEltwiseLayer>( *div.Layer, subIndex, false );
			if( sub2.Layer != nullptr ) {
				sqrtOutput = selectLayoutChange( *div.Layer, 1 - subIndex, layoutChangeAfterSecondPooling );
				break;
			}
		}

		auto* sqrtLayer = dynamic_cast<CPowerLayer*>( sqrtOutput.Layer );
		if( sqrtLayer == nullptr || !isValidPowerLayer( *sqrtLayer, 0.5f ) || sub2.Layer == nullptr
			|| !isValidArithmeticLayer( *sub2.Layer, COnnxEltwiseLayer::TOperation::Sub ) )
		{
			continue; // fail this Fusion
		}
		graph.SelectLayer( *sqrtLayer );

		COnnxEltwiseLayer* addLayer = graph.SelectTheOnlyConnectedOutput<COnnxEltwiseLayer>( *sqrtLayer, false );
		if( addLayer == nullptr || !isValidArithmeticLayer( *addLayer, COnnxEltwiseLayer::TOperation::Add ) ) {
			continue; // fail this Fusion
		}

		CLayerOutput<CGlobalMeanPoolingLayer> reduceMean2{};
		CLayerOutput<CDataLayer> eps{};
		if( !graph.SelectBothConnectedOutputs<CGlobalMeanPoolingLayer, CDataLayer>( *addLayer, reduceMean2, eps, false )
			|| graph.GetInputCount( *reduceMean2.Layer ) != 1 || !isValidDataLayer( *eps.Layer, CT_Float, /*size*/1 ) )
		{
			continue; // fail this Fusion
		}

		CLayoutChange layoutChangeBeforeSecondPooling;
		auto* sqrLayer = dynamic_cast<CPowerLayer*>(
			selectLayoutChange( *reduceMean2.Layer, 0, layoutChangeBeforeSecondPooling ).Layer );
		if( sqrLayer == nullptr || !isValidPowerLayer( *sqrLayer, 2.f ) ||
			!layoutChangeBeforeSecondPooling.IsInversionOf( layoutChangeAfterSecondPooling ) )
		{
			continue; // fail this Fusion
		}
		graph.SelectLayer( *sqrLayer );

		COnnxEltwiseLayer* subLayer = nullptr;
		// try to skip CAST layer in operand (1)
		CCastLayer* unusedCastLayer = graph.SelectTheOnlyConnectedOutput<CCastLayer>( *sqrLayer, false );
		if( unusedCastLayer != nullptr ) { // success to find the CAST layer as operand (1)
			if( !isValidCastLayer( *unusedCastLayer ) ) {
				continue; // fail this Fusion
			}
			subLayer = graph.GetConnectedOutput<COnnxEltwiseLayer>( *unusedCastLayer, /*inputIndex*/0 ).Layer;
		} else { // fail to find the CAST layer as operand (1)
			subLayer = graph.GetConnectedOutput<COnnxEltwiseLayer>( *sqrLayer, /*inputIndex*/0 ).Layer;
		}

		if( subLayer == nullptr || !isValidArithmeticLayer( *subLayer, COnnxEltwiseLayer::TOperation::Sub ) ) {
			continue; // fail this Fusion
		}

		CLayoutChange layoutChangeAfterFirstPooling;
		CGlobalMeanPoolingLayer* reduceMeanLayer = nullptr;
		CBaseLayer* inputNormLayerX = nullptr;
		for( int layoutChangeIndex = 0; layoutChangeIndex < 2; ++layoutChangeIndex ) {
			CLayerOutput<> reduceOutput = selectLayoutChange( *subLayer, layoutChangeIndex,
				layoutChangeAfterFirstPooling );
			if( reduceOutput.Layer != nullptr ) {
				inputNormLayerX = graph.GetConnectedOutput<CBaseLayer>( *subLayer, 1 - layoutChangeIndex ).Layer;
				reduceMeanLayer = dynamic_cast<CGlobalMeanPoolingLayer*>( reduceOutput.Layer );
				break;
			}
		}

		if( reduceMeanLayer == nullptr || graph.GetInputCount( *reduceMeanLayer ) != 1
			|| layoutChangeAfterFirstPooling != layoutChangeAfterSecondPooling )
		{
			continue; // fail this Fusion
		}
		graph.SelectLayer( *reduceMeanLayer );

		CLayoutChange layoutChangeBeforeFirstPooling;
		CLayerOutput<> blockData = selectLayoutChange( *reduceMeanLayer, 0, layoutChangeBeforeFirstPooling );
		if( blockData.Layer == nullptr || layoutChangeBeforeFirstPooling != layoutChangeBeforeSecondPooling ) {
			continue; // fail this Fusion
		}

		// Handle cyclic edges check (1)
		if( sub2.Layer != subLayer ) { // Duplicated sub-layers exported from older version of PyTorch
			NeoAssert( graph.IsLayerSelected( *subLayer ) == false );
			graph.SelectLayer( *subLayer );
			auto* in1 = graph.GetConnectedOutput<CGlobalMeanPoolingLayer>( *sub2.Layer, /*inputIndex*/0 ).Layer;
			auto* in2 = graph.GetConnectedOutput<CBaseLayer>( *sub2.Layer, /*inputIndex*/1 ).Layer;
			if( in1 != reduceMeanLayer || in2 != inputNormLayerX ) {
				continue; // fail this Fusion
			}
		}

		// Handle cyclic edges check (2)
		if( blockData.Layer != inputNormLayerX ) {
			continue; // fail this Fusion
		}
		// Current Fusion succeed!

		// ObjectNorm and GlobalMeanPooling are working with different layouts
		// (the difference is in BD_Channels, ObjectNorm affects data along it while GlobalMeanPooling doesn't)
		// Let's find a conversion to layout which will good for CObjectNormalizationLayer

		// Find axes indices which must be normalized
		// GlobalMeanPooling affects BD_Height, BD_Width and BD_Depth
		const CTensorLayout& ioLayout = layoutChangeBeforeFirstPooling.From;
		const CTensorLayout& globalPoolLayout = layoutChangeBeforeFirstPooling.To;
		CFastArray<int, 8> axes;
		for( int i = 0; i < globalPoolLayout.Size(); ++i ) {
			if( globalPoolLayout[i] >= BD_Height && globalPoolLayout[i] <= BD_Depth ) {
				axes.Add( i );
			}
		}

		// Convert data to layout valid for CObjectNormalizationLayer
		CTensorLayout objNormLayout;
		CLayerOutput<> objNormInput = changeLayout( blockData, ioLayout, CObjectNormLayoutValidator( axes ),
			objNormLayout );

		CPtr<CObjectNormalizationLayer> normLayer{ new CObjectNormalizationLayer( graph.MathEngine() ) };
		normLayer->SetName( graph.GetUniqueName( CString( fusionNamePrefix ) + "ObjNorm_" ) );
		normLayer->SetEpsilon( eps.Layer->GetBlob()->GetData().GetValue() );
		graph.AddLayer( *normLayer );
		graph.Connect( *normLayer, 0, *objNormInput.Layer, objNormInput.Index );

		// Prepare parameters for CObjectNormalizationLayer before setting them
		normLayer->SetBias( prepareObjNormParamBlob( *bias.Layer, ioLayout, objNormLayout ) );
		normLayer->SetScale( prepareObjNormParamBlob( *scale.Layer, ioLayout, objNormLayout ) );

		// Now change back the layout after the CObjectNormalizationLayer
		CTensorLayout afterObjNormLayout;
		CLayerOutput<> newBlockOutput = changeLayout( CLayerOutput<>( normLayer, 0 ), objNormLayout,
			CTensorLayoutMatchValidator( ioLayout ), afterObjNormLayout );
		NeoAssert( ioLayout == afterObjNormLayout );

		// And switch everythin that was connected with old huge block to new one
		graph.SwitchOutputs( *addLayerLast, 0, *newBlockOutput.Layer, newBlockOutput.Index );

		graph.DeleteSelectedLayers();

		++optimizedLayers;
	} //for layers

	return optimizedLayers;
}

} // namespace optimization

} // namespace NeoOnnx

