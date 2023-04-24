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
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>
#include "Optimization/LayerNormFusionOptimizer.h"

namespace NeoOnnx {

namespace optimization {

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
bool CLayerNormFusionOptimizer::isValidTransformLayer( const COnnxTransformHelper& transformHelperLayer ) const
{
	NeoAssert( graph.GetInputCount( transformHelperLayer ) == 1 );
	NeoAssert( graph.GetOutputCount( transformHelperLayer ) == 1 );

	auto isEmptyRule = [&transformHelperLayer]( TBlobDim index ) -> bool {
		const TBlobDim value = transformHelperLayer.GetRule( index );
		return value == index || ( /* invalid-value */ value < 0 || value >= BD_Count );
	};

	const bool success =
		( isEmptyRule( BD_BatchLength )
		&& isEmptyRule( BD_BatchWidth )
		&& transformHelperLayer.GetRule( BD_ListSize ) == BD_Height
		&& isEmptyRule( BD_Height )
		&& isEmptyRule( BD_Width )
		&& isEmptyRule( BD_Depth )
		&& isEmptyRule( BD_Channels ) )
		||
		( isEmptyRule( BD_BatchLength )
		&& isEmptyRule( BD_BatchWidth )
		&& isEmptyRule( BD_ListSize )
		&& transformHelperLayer.GetRule( BD_Height ) == BD_ListSize
		&& isEmptyRule( BD_Width )
		&& isEmptyRule( BD_Depth )
		&& isEmptyRule( BD_Channels ) )
		||
		( transformHelperLayer.GetRule( BD_BatchLength ) == BD_ListSize
		&& transformHelperLayer.GetRule( BD_BatchWidth ) == BD_Height
		&& isEmptyRule( BD_ListSize )
		&& transformHelperLayer.GetRule( BD_Height ) == BD_Channels
		&& isEmptyRule( BD_Width )
		&& isEmptyRule( BD_Depth )
		&& isEmptyRule( BD_Channels ) )
		||
		( isEmptyRule( BD_BatchLength )
		&& isEmptyRule( BD_BatchWidth )
		&& transformHelperLayer.GetRule( BD_ListSize ) == BD_BatchLength
		&& transformHelperLayer.GetRule( BD_Height ) == BD_BatchWidth
		&& isEmptyRule( BD_Width )
		&& isEmptyRule( BD_Depth )
		&& transformHelperLayer.GetRule( BD_Channels ) == BD_Height );
	return success;
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
		auto* addLayerLast = getExactLayer<COnnxEltwiseLayer>( layer, /*addToSelection*/true );
		if( addLayerLast == nullptr
			|| addLayerLast->GetOperation() != COnnxEltwiseLayer::TOperation::Add
			|| graph.GetInputCount( *addLayerLast ) != 2
			|| graph.GetOutputCount( *addLayerLast ) != 1 )
		{
			continue; // fail this Fusion
		}

		CDataLayer* biasLayer = nullptr;
		COnnxEltwiseLayer* mulLayer = nullptr;
		if( !selectTwoExactInputLayersRecursive<COnnxEltwiseLayer, CDataLayer>( *addLayerLast, &mulLayer, &biasLayer )
			|| mulLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Mul
			|| graph.GetInputCount( *mulLayer ) != 2
			|| !isValidDataLayer( *biasLayer, CT_Float, /*size*/NotFound ) )
		{
			continue; // fail this Fusion
		}

		CDataLayer* scaleLayer = nullptr;
		COnnxEltwiseLayer* divLayer = nullptr;
		CCastLayer* uselessCastLayer = nullptr; // try to skip CAST layer as operand (1)
		if( !selectTwoExactInputLayersRecursive<CCastLayer, CDataLayer>( *mulLayer, &uselessCastLayer, &scaleLayer )
			|| uselessCastLayer->GetOutputType() != CT_Float
			|| graph.GetInputCount( *uselessCastLayer ) != 1
			|| graph.GetOutputCount( *uselessCastLayer ) != 1
			|| !isValidDataLayer( *scaleLayer, CT_Float, /*size*/NotFound ) )
		{
			scaleLayer = nullptr;
			uselessCastLayer = nullptr; // try to skip CAST layer as operand (2)
			if( !selectTwoExactInputLayersRecursive<COnnxEltwiseLayer, CCastLayer>( *mulLayer, &divLayer, &uselessCastLayer )
				|| divLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Div
				|| graph.GetInputCount( *divLayer ) != 2
				|| uselessCastLayer->GetOutputType() != CT_Float
				|| graph.GetInputCount( *uselessCastLayer ) != 1
				|| graph.GetOutputCount( *uselessCastLayer ) != 1 )
			{
				divLayer = nullptr;
				uselessCastLayer = nullptr; // no CAST layer as both of operands (3)
				if( !selectTwoExactInputLayersRecursive<COnnxEltwiseLayer, CDataLayer>( *mulLayer, &divLayer, &scaleLayer )
					|| divLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Div
					|| graph.GetInputCount( *divLayer ) != 2
					|| !isValidDataLayer( *scaleLayer, CT_Float, /*size*/NotFound ) )
				{
					continue; // fail this Fusion
				}
			} else { // success to find the CAST layer as operand (2)
				scaleLayer = selectOneExactInputLayerRecursive<CDataLayer>( *uselessCastLayer );
				if( scaleLayer == nullptr
					|| !isValidDataLayer( *scaleLayer, CT_Float, /*size*/NotFound ) )
				{
					continue; // fail this Fusion
				}
			}
		} else { // success to find the CAST layer as operand (1)
			divLayer = selectOneExactInputLayerRecursive<COnnxEltwiseLayer>( *uselessCastLayer );
			if( divLayer == nullptr
				|| divLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Div
				|| graph.GetInputCount( *divLayer ) != 2 )
			{
				continue; // fail this Fusion
			}
		}

		CPowerLayer* sqrtLayer = nullptr;
		COnnxEltwiseLayer* sub2Layer = nullptr;
		if( !selectTwoExactInputLayersRecursive<COnnxEltwiseLayer, CPowerLayer>( *divLayer, &sub2Layer, &sqrtLayer )
			|| sqrtLayer->GetExponent() != 0.5f
			|| graph.GetInputCount( *sqrtLayer ) != 1 )
		{
			continue; // fail this Fusion
		}

		auto* addLayer = selectOneExactInputLayerRecursive<COnnxEltwiseLayer>( *sqrtLayer );
		if( addLayer == nullptr
			|| addLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Add
			|| graph.GetInputCount( *addLayer ) != 2 )
		{
			continue; // fail this Fusion
		}

		CGlobalMeanPoolingLayer* reduceMean2Layer = nullptr;
		CDataLayer* epsilontLayer = nullptr;
		if( !selectTwoExactInputLayersRecursive<CGlobalMeanPoolingLayer, CDataLayer>( *addLayer, &reduceMean2Layer, &epsilontLayer )
			|| graph.GetInputCount( *reduceMean2Layer ) != 1
			|| !isValidDataLayer( *epsilontLayer, CT_Float, /*size*/1 ) )
		{
			continue; // fail this Fusion
		}

		auto* powLayer = selectOneExactInputLayerRecursive<CPowerLayer>( *reduceMean2Layer );
		if( powLayer == nullptr
			|| powLayer->GetExponent() != 2.f
			|| graph.GetInputCount( *powLayer ) != 1 )
		{
			continue; // fail this Fusion
		}

		COnnxEltwiseLayer* subLayer = nullptr;
		CCastLayer* unusedCastLayer = selectOneExactInputLayerRecursive<CCastLayer>( *powLayer ); // try to skip CAST layer in operand (1)
		if( unusedCastLayer != nullptr ) { // success to find the CAST layer as operand (1)
			if( unusedCastLayer->GetOutputType() != CT_Float
				|| graph.GetInputCount( *unusedCastLayer ) != 1
				|| graph.GetOutputCount( *unusedCastLayer ) != 1 )
			{
				continue; // fail this Fusion
			}
			subLayer = selectOneExactInputLayerRecursive<COnnxEltwiseLayer>( *unusedCastLayer );
		} else { // fail to find the CAST layer as operand (1)
			subLayer = selectOneExactInputLayerRecursive<COnnxEltwiseLayer>( *powLayer );
		}
		if( subLayer == nullptr
			|| subLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Sub
			|| graph.GetInputCount( *subLayer ) != 2 )
		{
			continue; // fail this Fusion
		}

		CGlobalMeanPoolingLayer* reduceMeanLayer = nullptr;
		CBaseLayer* inputNormLayerX = nullptr;
		if( !selectTwoExactInputLayersRecursive<CGlobalMeanPoolingLayer, CBaseLayer>( *subLayer, &reduceMeanLayer, &inputNormLayerX )
			|| graph.GetInputCount( *reduceMeanLayer ) != 1 )
		{
			continue; // fail this Fusion
		}

		// Handle cyclic edges check (1)
		if( sub2Layer != subLayer ) { // Duplicated sub-layers exported from older version of PyTorch
			CGlobalMeanPoolingLayer* in1 = nullptr;
			CBaseLayer *in2 = nullptr;
			if( !selectTwoExactInputLayersRecursive<CGlobalMeanPoolingLayer, CBaseLayer>( *sub2Layer, &in1, &in2 )
				|| in1 != reduceMeanLayer
				|| in2 != inputNormLayerX )
			{
				continue; // fail this Fusion
			}
		}

		// Throw away from the dnn excess transforms as many as I could
		if ( inputNormLayerX->GetInputCount() == 1 ) {
			CBaseLayer* transformLayer = inputNormLayerX;
			while( ( transformLayer = graph.SelectTheOnlyConnectedOutput<COnnxTransformHelper>( *transformLayer ) ) != nullptr ) {
				inputNormLayerX = transformLayer;
			}
		}
		// Handle cyclic edges check (2)
		auto* inputReduceMeanLayer = graph.GetConnectedOutput( *reduceMeanLayer, /*inputIndex*/0 ).Layer;
		if( inputReduceMeanLayer != inputNormLayerX ) {
			CBaseLayer* transformLayer = reduceMeanLayer;
			while( ( transformLayer = graph.SelectTheOnlyConnectedOutput<COnnxTransformHelper>( *transformLayer ) ) != nullptr ) {
				inputReduceMeanLayer = graph.GetConnectedOutput( *transformLayer, /*inputIndex*/0 ).Layer;
			}
			if ( inputReduceMeanLayer != inputNormLayerX ) {
				continue; // fail this Fusion
			}
		}
		// Current Fusion succeed!

		const auto& biasBlob = biasLayer->GetBlob();
		const auto& scaleBlob = scaleLayer->GetBlob();
		const auto& epsBlob = epsilontLayer->GetBlob();

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

