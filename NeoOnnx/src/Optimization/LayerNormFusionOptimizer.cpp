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

#pragma once

#include <NeoOnnx/NeoOnnxImport.h>
#include "Optimization/LayerNormFusionOptimizer.h"

namespace NeoOnnx {

namespace optimization {

bool CLayerNormFusionOptimizer::isValidTransformLayer( const COnnxTransformHelper& transformHelperLayer ) const
{
	NeoAssert( graph.GetInputCount( transformHelperLayer ) == 1 );
	NeoAssert( graph.GetOutputCount( transformHelperLayer ) == 1 );

	auto checkTransformRule = [&transformHelperLayer]( const int rule[BD_Count] ) -> bool
	{
		auto isEmpty = []( int index, int rule ) -> bool { return rule == index || ( /*invalid*/ rule < 0 || rule >= BD_Count ); };

		for( int index = 0; index < BD_Count; ++index ) {
			const TBlobDim value = transformHelperLayer.GetRule( TBlobDim( index ) );
			if( value != rule[index] && ( !isEmpty( index, value ) || !isEmpty( index, rule[index] ) ) ) {
				return false;
			}
		}
		return true;
	};

	constexpr int Ls_to_H[BD_Count]{ -1, -1, /*ListSize*/ BD_Height, -1, -1, -1, -1 };
	constexpr int H_to_Ls[BD_Count]{ -1,  -1,  -1, /*Height*/ BD_ListSize,  -1,  -1,  -1 };

	constexpr int BlBwH_to_LsHC[BD_Count]{ /*BatchLength*/ BD_ListSize, /*BatchWidth*/ BD_Height, -1, /*Height*/ BD_Channels, -1, -1, -1 };
	constexpr int LsHC_to_BlBwH[BD_Count]{ -1, -1, /*ListSize*/ BD_BatchLength, /*Height*/ BD_BatchWidth, -1, -1, /*Channels*/ BD_Height };

	return checkTransformRule( Ls_to_H )
		|| checkTransformRule( H_to_Ls )
		|| checkTransformRule( BlBwH_to_LsHC )
		|| checkTransformRule( LsHC_to_BlBwH );
}

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

bool CLayerNormFusionOptimizer::isValidCastLayer( const CCastLayer& castLayer ) const
{
	NeoAssert( graph.GetInputCount( castLayer ) == 1 );
	NeoAssert( graph.GetOutputCount( castLayer ) == 1 );

	return castLayer.GetOutputType() == CT_Float;
}

//--------------------------------------------------------------------------------------------------------------
void CLayerNormFusionOptimizer::Apply()
{
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

		CLayerOutput<COnnxTransformHelper> transform4{};
		CLayerOutput<COnnxEltwiseLayer> sub2{};
		if( !graph.SelectBothConnectedOutputs<COnnxEltwiseLayer, COnnxTransformHelper>( *div.Layer, sub2, transform4, /*checkOutOfSelectionLinks*/false )
			|| !isValidArithmeticLayer( *sub2.Layer, COnnxEltwiseLayer::TOperation::Sub )
			|| !isValidTransformLayer( *transform4.Layer ) )
		{
			continue; // fail this Fusion
		}

		auto* sqrtLayer = graph.SelectTheOnlyConnectedOutput<CPowerLayer>( *transform4.Layer, /*checkOutOfSelectionLinks*/false );
		if( sqrtLayer == nullptr
			|| !isValidPowerLayer( *sqrtLayer, 0.5f ) )
		{
			continue; // fail this Fusion
		}

		auto* addLayer = graph.SelectTheOnlyConnectedOutput<COnnxEltwiseLayer>( *sqrtLayer, /*checkOutOfSelectionLinks*/false );
		if( addLayer == nullptr
			|| !isValidArithmeticLayer( *addLayer, COnnxEltwiseLayer::TOperation::Add ) )
		{
			continue; // fail this Fusion
		}

		CLayerOutput<CGlobalMeanPoolingLayer> reduceMean2{};
		CLayerOutput<CDataLayer> epsilont{};
		if( !graph.SelectBothConnectedOutputs<CGlobalMeanPoolingLayer, CDataLayer>( *addLayer, reduceMean2, epsilont, /*checkOutOfSelectionLinks*/false )
			|| graph.GetInputCount( *reduceMean2.Layer ) != 1
			|| !isValidDataLayer( *epsilont.Layer, CT_Float, /*size*/1 ) )
		{
			continue; // fail this Fusion
		}

		auto* transform3Layer = graph.SelectTheOnlyConnectedOutput<COnnxTransformHelper>( *reduceMean2.Layer, /*checkOutOfSelectionLinks*/false );
		if( transform3Layer == nullptr
			|| !isValidTransformLayer( *transform3Layer ) )
		{
			continue; // fail this Fusion
		}

		auto* powLayer = graph.SelectTheOnlyConnectedOutput<CPowerLayer>( *transform3Layer, /*checkOutOfSelectionLinks*/false );
		if( powLayer == nullptr
			|| !isValidPowerLayer( *powLayer, 2.f ) )
		{
			continue; // fail this Fusion
		}

		COnnxEltwiseLayer* subLayer = nullptr;
		CCastLayer* unusedCastLayer = graph.SelectTheOnlyConnectedOutput<CCastLayer>( *powLayer, /*checkOutOfSelectionLinks*/false ); // try to skip CAST layer in operand (1)
		if( unusedCastLayer != nullptr ) { // success to find the CAST layer as operand (1)
			if( !isValidCastLayer( *unusedCastLayer ) ) {
				continue; // fail this Fusion
			}
			subLayer = graph.GetConnectedOutput<COnnxEltwiseLayer>( *unusedCastLayer, /*inputIndex*/0 ).Layer;
		} else { // fail to find the CAST layer as operand (1)
			subLayer = graph.GetConnectedOutput<COnnxEltwiseLayer>( *powLayer, /*inputIndex*/0 ).Layer;
		}
		if( subLayer == nullptr
			|| !isValidArithmeticLayer( *subLayer, COnnxEltwiseLayer::TOperation::Sub ) )
		{
			continue; // fail this Fusion
		}

		CBaseLayer* inputNormLayerX = nullptr;
		auto* transform2Layer = graph.GetConnectedOutput<COnnxTransformHelper>( *subLayer, /*inputIndex*/0 ).Layer;
		if( transform2Layer == nullptr
			|| isValidTransformLayer( *transform2Layer ) )
		{
			transform2Layer = graph.GetConnectedOutput<COnnxTransformHelper>( *subLayer, /*inputIndex*/1 ).Layer;
			inputNormLayerX = graph.GetConnectedOutput<CBaseLayer>( *subLayer, /*inputIndex*/0 ).Layer;
			if( transform2Layer == nullptr
				|| !isValidTransformLayer( *transform2Layer ) )
			{
				continue; // fail this Fusion
			}
		} else {
			inputNormLayerX = graph.GetConnectedOutput<CBaseLayer>( *subLayer, /*inputIndex*/1 ).Layer;
		}
		graph.SelectLayer( *transform2Layer );

		CGlobalMeanPoolingLayer* reduceMeanLayer = graph.SelectTheOnlyConnectedOutput<CGlobalMeanPoolingLayer>( *transform2Layer, /*checkOutOfSelectionLinks*/false );
		if( reduceMeanLayer == nullptr
			|| graph.GetInputCount( *reduceMeanLayer ) != 1 )
		{
			continue; // fail this Fusion
		}

		auto* transform1Layer = graph.SelectTheOnlyConnectedOutput<COnnxTransformHelper>( *reduceMeanLayer, /*checkOutOfSelectionLinks*/false );
		if( transform1Layer == nullptr
			|| !isValidTransformLayer( *transform1Layer ) )
		{
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
		auto* inputReduceMeanLayer = graph.GetConnectedOutput( *transform1Layer, /*inputIndex*/0 ).Layer;
		if( inputReduceMeanLayer != inputNormLayerX ) {
			CBaseLayer* transformLayer = transform1Layer;
			while( ( transformLayer = graph.SelectTheOnlyConnectedOutput<COnnxTransformHelper>( *transformLayer ) ) != nullptr ) {
				inputReduceMeanLayer = graph.GetConnectedOutput( *transformLayer, /*inputIndex*/0 ).Layer;
			}
			if ( inputReduceMeanLayer != inputNormLayerX ) {
				continue; // fail this Fusion
			}
		}
		// Current Fusion succeed!

		const auto& biasBlob = bias.Layer->GetBlob();
		const auto& scaleBlob = scale.Layer->GetBlob();
		const auto& epsBlob = epsilont.Layer->GetBlob();

		CPtr<CObjectNormalizationLayer> normLayer{ new CObjectNormalizationLayer( graph.MathEngine() ) };
		normLayer->SetName( graph.GetUniqueName( CString( FusionNamePrefix ) + "ObjNorm_" ) );
		normLayer->SetBias( biasBlob->GetTransposed( BD_ListSize, BD_Height ) );
		normLayer->SetScale( scaleBlob->GetTransposed( BD_ListSize, BD_Height ) );
		normLayer->SetEpsilon( epsBlob->GetData().GetValue() );
		graph.AddLayer( *normLayer );

		// Layer ObjectNormalization should have some number of objects to reduction ( ObjectCount > 1 && ObjectSize >= 1 ).
		// Mostly in given ANNs X layer (input of ObjNorm) has Height > 1, so it used to increase the ObjectCount.

		CPtr<CTransposeLayer> layerTransposeBefore{ new CTransposeLayer( graph.MathEngine() ) };
		layerTransposeBefore->SetName( graph.GetUniqueName( CString( FusionNamePrefix ) + "TransposeBefore_" ) );
		layerTransposeBefore->SetTransposedDimensions( BD_ListSize, BD_Height );
		graph.AddLayer( *layerTransposeBefore );

		CPtr<CTransposeLayer> layerTransposeAfter{ new CTransposeLayer( graph.MathEngine() ) };
		layerTransposeAfter->SetName( graph.GetUniqueName( CString( FusionNamePrefix ) + "TransposeAfter_" ) );
		layerTransposeAfter->SetTransposedDimensions( BD_ListSize, BD_Height );
		graph.AddLayer( *layerTransposeAfter );

		graph.Connect( *layerTransposeBefore,/*inputIndex*/0, *inputNormLayerX, /*outputIndex*/0 );
		graph.Connect( *normLayer, /*inputIndex*/0, *layerTransposeBefore, /*outputIndex*/0 );
		graph.Connect( *layerTransposeAfter,/*inputIndex*/0, *normLayer, /*outputIndex*/0 );

		// Search for the output-layers of addLayerLast for new added ObjNorm layer
		graph.SwitchOutputs( *addLayerLast, /*outputIndex*/0, *layerTransposeAfter, /*outputIndex*/0 );

		// All selected layers would be removed from the dnn
		graph.DeleteSelectedLayers();
	} //for layers
}

} // namespace optimization

} // namespace NeoOnnx

