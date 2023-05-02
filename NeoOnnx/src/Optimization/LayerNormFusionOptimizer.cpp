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
#include "Optimization/LayerNormFusionOptimizer.h"

namespace NeoOnnx {

namespace optimization {

void CLayerNormFusionOptimizer::getTransformRule( const COnnxTransformHelper& transformLayer, const bool opposite, int rule[BD_Count] ) const
{
	for( int index = 0; index < BD_Count; ++index ) {
		const int dim = transformLayer.GetRule( TBlobDim( index ) );
		if( !opposite ) {
			rule[index] = dim;
		} else if( isValidBlobDim( dim ) ) {
			rule[dim] = index;
		}
	}
}

//--------------------------------------------------------------------------------------------------------------
bool CLayerNormFusionOptimizer::isValidTransformLayer( const COnnxTransformHelper& transformLayer,
	const COnnxTransformHelper* transformLayerPrevious,
	bool opposite,
	bool& objTransform ) const
{
	NeoAssert( graph.GetInputCount( transformLayer ) == 1 );
	NeoAssert( graph.GetOutputCount( transformLayer ) == 1 );

	bool result = true;
	if( transformLayerPrevious ) {
		int rule[BD_Count]{ -1, -1, -1, -1, -1, -1, -1 };
		getTransformRule( *transformLayerPrevious, opposite, rule );

		for( int index = 0; index < BD_Count; ++index ) {
			const TBlobDim dim = transformLayer.GetRule( TBlobDim( index ) );
			if( dim != rule[index] && ( !isEmptyBlobDim( index, dim ) || !isEmptyBlobDim( index, rule[index] ) ) ) {
				result = false;
				break;
			}
		}
	} else {
		bool objChange = false;
		for( int index = 0; index < BD_Count; ++index ) {
			const TBlobDim dim = transformLayer.GetRule( TBlobDim( index ) );
			if( !isEmptyBlobDim( index, dim )
				&& ( ( dim > BD_ListSize && index <= BD_ListSize )
				|| ( dim <= BD_ListSize && index > BD_ListSize ) ) )
			{
				objChange = true;
				break;
			}
		}
		if( objTransform || objChange ) {
			result = ( objTransform = objChange );
		} 
		// Or allow any transform, which does not change the object size
	}
	return result;
}

//--------------------------------------------------------------------------------------------------------------
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

//--------------------------------------------------------------------------------------------------------------
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

		bool objTransform = false;
		CLayerOutput<COnnxTransformHelper> transform4{};
		CLayerOutput<COnnxEltwiseLayer> sub2{};
		if( !graph.SelectBothConnectedOutputs<COnnxEltwiseLayer, COnnxTransformHelper>( *div.Layer, sub2, transform4, /*checkOutOfSelectionLinks*/false )
			|| !isValidArithmeticLayer( *sub2.Layer, COnnxEltwiseLayer::TOperation::Sub )
			|| !isValidTransformLayer( *transform4.Layer, /*prevTransform*/nullptr, /*opposite*/false, objTransform ) )
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
			|| !isValidTransformLayer( *transform3Layer, /*prevTransform*/transform4.Layer, /*opposite*/true, objTransform ) )
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
			|| isValidTransformLayer( *transform2Layer, /*prevTransform*/transform4.Layer, /*opposite*/false, objTransform ) )
		{
			transform2Layer = graph.GetConnectedOutput<COnnxTransformHelper>( *subLayer, /*inputIndex*/1 ).Layer;
			inputNormLayerX = graph.GetConnectedOutput<CBaseLayer>( *subLayer, /*inputIndex*/0 ).Layer;
			if( transform2Layer == nullptr
				|| !isValidTransformLayer( *transform2Layer, /*prevTransform*/transform4.Layer, /*opposite*/false, objTransform ) )
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
			|| !isValidTransformLayer( *transform1Layer, /*prevTransform*/transform4.Layer, /*opposite*/true, objTransform ) )
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
		normLayer->SetEpsilon( epsBlob->GetData().GetValue() );
		graph.AddLayer( *normLayer );

		CBaseLayer* newNormLayer = nullptr;
		if( objTransform ) {
			int rule[BD_Count]{ -1, -1, -1, -1, -1, -1, -1 };
			getTransformRule( *transform4.Layer, /*opposite*/false, rule );

			for( int index = 0; index < BD_Count; ++index ) {
				if( !isEmptyBlobDim( index, rule[index] )  ) {
					biasBlob->GetTransposed( index, rule[index] );
					scaleBlob->GetTransposed( index, rule[index] );
				}
			}
			normLayer->SetBias( biasBlob );
			normLayer->SetScale( scaleBlob );

			// Layer ObjectNormalization should have some number of objects to reduction ( ObjectCount > 1 && ObjectSize >= 1 ).
			// Mostly in given ANNs X layer (input of ObjNorm) has Height > 1, so it used to increase the ObjectCount.

			CPtr<COnnxTransformHelper> layerTransformBefore{ new COnnxTransformHelper( graph.MathEngine() ) };
			layerTransformBefore->SetName( graph.GetUniqueName( CString( FusionNamePrefix ) + "TransformBefore_" ) );
			graph.AddLayer( *layerTransformBefore );

			for( int index = 0; index < BD_Count; ++index ) {
				if( isValidBlobDim( rule[index] ) ) {
					layerTransformBefore->SetRule( TBlobDim( index ), TBlobDim( rule[index] ) );
				}
			}

			CPtr<COnnxTransformHelper> layerTransformAfter{ new COnnxTransformHelper( graph.MathEngine() ) };
			layerTransformAfter->SetName( graph.GetUniqueName( CString( FusionNamePrefix ) + "TransformAfter_" ) );
			graph.AddLayer( *layerTransformAfter );

			int ruleOpposite[BD_Count]{ -1, -1, -1, -1, -1, -1, -1 };
			getTransformRule( *transform4.Layer, /*opposite*/true, ruleOpposite );
			for( int index = 0; index < BD_Count; ++index ) {
				if( isValidBlobDim( ruleOpposite[index] ) ) {
					layerTransformAfter->SetRule( TBlobDim( index ), TBlobDim( ruleOpposite[index] ) );
				}
			}

			graph.Connect( *layerTransformBefore, /*inputIndex*/0, *inputNormLayerX, /*outputIndex*/0 );
			graph.Connect( *normLayer, /*inputIndex*/0, *layerTransformBefore, /*outputIndex*/0 );
			graph.Connect( *layerTransformAfter, /*inputIndex*/0, *normLayer, /*outputIndex*/0 );
			newNormLayer = layerTransformAfter;
		} else {
			normLayer->SetBias( biasBlob );
			normLayer->SetScale( scaleBlob );

			graph.Connect( *normLayer, /*inputIndex*/0, *inputNormLayerX, /*outputIndex*/0 );
			newNormLayer = normLayer;
		};

		// Search for the output-layers of addLayerLast for new added ObjNorm layer
		graph.SwitchOutputs( *addLayerLast, /*outputIndex*/0, *newNormLayer, /*outputIndex*/0 );

		// All selected layers would be removed from the dnn
		graph.DeleteSelectedLayers();
	} //for layers
}

} // namespace optimization

} // namespace NeoOnnx

