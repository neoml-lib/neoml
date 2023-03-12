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

#include <cstring>
#include "Optimization/LayerNormFusionOptimizer.h"

namespace NeoOnnx {

const char* const CLayerNormFusionOptimizer::classesOfSkipLayers[]{
	"NeoMLDnnBroadcastLayer",
	"FmlCnnTransformWithoutTransposeLayer",
	"FmlCnnTransposeLayer"
};

void CLayerNormFusionOptimizer::Apply()
{
	Graph.Reshape(); // For the all of BLOBs' sizes be initialized in the graph

	int normLayersCount = 0;
	CArray<const char*> layersList{};
	Graph.GetLayerList( layersList );

	int configureLayerConnection = 0; // Defines the way of connection once, all other connections are the same
	for( int i = 0; i < layersList.Size(); ++i ) {
		ClearSelectedLayers();

		if( layersList[i] == nullptr ) { // Skip already replaced layers
			continue;
		}

		// Searching for a group of layers to replace by an object normalization layer in the backward direction through the graph
		CPtr<CBaseLayer> addLayerLast = Graph.GetLayer( layersList[i] );
		if( !IsExactLayer( addLayerLast, "FmlCnnEltwiseSumLayer" ) ) {
			continue;
		}
		if( addLayerLast->GetInputCount() != 2 ) {
			continue;
		}

		CPtr<CBaseLayer> biasLayer = nullptr, mulLayer = nullptr;
		if( !GetExactInputLayers( addLayerLast, mulLayer, "FmlCnnEltwiseMulLayer", biasLayer, "NeoMLDnnDataLayer", /*layerSkipClass*/"" ) ) {
			continue;
		}
		if( mulLayer->GetInputCount() != 2 ) {
			continue;
		}

		CPtr<CBaseLayer> scaleLayer = nullptr, divLayer = nullptr;
		if( !GetExactInputLayers( mulLayer, divLayer, "NeoMLDnnEltwiseDivLayer", scaleLayer, "NeoMLDnnDataLayer", "NeoMLDnnCastLayer" ) ) {
			continue;
		}
		if( divLayer->GetInputCount() != 2 ) {
			continue;
		}

		CPtr<CBaseLayer> sqrtLayer = nullptr, sub2Layer = nullptr;
		if( !GetExactInputLayers( divLayer, sqrtLayer, "FmlCnnPowerLayer", sub2Layer, "NeoMLDnnEltwiseSubLayer", /*layerSkipClass*/"" ) ) {
			continue;
		}
		const auto* sqrt = dynamic_cast<CPowerLayer*>( sqrtLayer.Ptr() );
		if( !sqrt || sqrtLayer->GetInputCount() != 1 || sqrt->GetExponent() != 0.5f ) {
			continue;
		}

		CPtr<CBaseLayer> addLayer = nullptr, unusedLayer = nullptr;
		if( !GetExactInputLayers( sqrtLayer, addLayer, "FmlCnnEltwiseSumLayer", unusedLayer, "", /*layerSkipClass*/"" ) ) {
			continue;
		}
		if( addLayer->GetInputCount() != 2 ) {
			continue;
		}

		CPtr<CBaseLayer> reduceMean2Layer = nullptr, epsilontLayer = nullptr;
		if( !GetExactInputLayers( addLayer, reduceMean2Layer, "FmlCnnGlobalMainPoolingLayer", epsilontLayer, "NeoMLDnnDataLayer", /*layerSkipClass*/"" ) ) {
			continue;
		}
		if( reduceMean2Layer->GetInputCount() != 1 ) {
			continue;
		}

		CPtr<CBaseLayer> powLayer = nullptr;
		if( !GetExactInputLayers( reduceMean2Layer, powLayer, "FmlCnnPowerLayer", unusedLayer, "", /*layerSkipClass*/"" ) ) {
			continue;
		}
		const auto* pow = dynamic_cast<CPowerLayer*>( powLayer.Ptr() );
		if( !pow || pow->GetInputCount() != 1 || pow->GetExponent() != 2.f ) {
			continue;
		}

		CPtr<CBaseLayer> subLayer = nullptr;
		if( !GetExactInputLayers( powLayer, subLayer, "NeoMLDnnEltwiseSubLayer", unusedLayer, "", "NeoMLDnnCastLayer" ) ) {
			continue;
		}
		if( subLayer->GetInputCount() != 2 ) {
			continue;
		}
		if( unusedLayer ) {
			const auto* cast = dynamic_cast<CCastLayer*>( unusedLayer.Ptr() );
			if( !cast || cast->GetOutputType() != CT_Float ) {
				continue;
			}
		}

		CPtr<CBaseLayer> reduceMeanLayer = nullptr;
		if( !GetExactInputLayers( subLayer, reduceMeanLayer, "FmlCnnGlobalMainPoolingLayer", unusedLayer, "", /*layerSkipClass*/"" ) ) {
			continue;
		}
		if( reduceMeanLayer->GetInputCount() != 1 ) {
			continue;
		}

		CPtr<CBaseLayer> reduceMeanInputLayer{}, inputNormLayer{};
		for( int j = 0; ( reduceMeanInputLayer = GetAnyInputLayer( reduceMeanLayer, j, "NeoMLDnnCastLayer" ) ) != nullptr; ++j ) {
			ASSERT_EXPR( j == 0 );
			inputNormLayer = reduceMeanInputLayer;
		}

		if( sub2Layer != subLayer ) { // Duplicated sub-layers exported from older version of PyTorch
			if(    std::strcmp( subLayer->GetInputName( 0 ), inputNormLayer->GetName() ) != 0
				&& std::strcmp( subLayer->GetInputName( 1 ), inputNormLayer->GetName() ) != 0 )
			{
				continue; // If there no correct cyclic reference, skip
			}
			if(    !( std::strcmp( sub2Layer->GetInputName( 0 ), inputNormLayer->GetName()  ) == 0
				&&    std::strcmp( sub2Layer->GetInputName( 1 ), reduceMeanLayer->GetName() ) == 0 )
				&& !( std::strcmp( sub2Layer->GetInputName( 0 ), reduceMeanLayer->GetName() ) == 0
				&&    std::strcmp( sub2Layer->GetInputName( 1 ), inputNormLayer->GetName()  ) == 0 ) )
			{
				continue; // If there no correct cyclic reference, skip
			}
		}

		const auto& epsBlob = dynamic_cast<CDataLayer*>( epsilontLayer.Ptr() )->GetBlob();
		const auto& scaleBlob = dynamic_cast<CDataLayer*>( scaleLayer.Ptr() )->GetBlob();
		const auto& biasBlob = dynamic_cast<CDataLayer*>( biasLayer.Ptr() )->GetBlob();

		CPtr<CObjectNormalizationLayer> normLayer{};
		CPtr<CBaseLayer> newNormLayer{}; // Norm layer's wrapper to connect further

		if( configureLayerConnection == 1 || inputNormLayer->GetOutputBlobsDesc( 0 ).ObjectSize() == 1 ) {
			if( configureLayerConnection == 1 || inputNormLayer->GetOutputBlobsDesc( 0 ).ListSize() > 1 ) {
				ASSERT_EXPR( configureLayerConnection == 1
					|| ( configureLayerConnection == 0
					&& inputNormLayer->GetOutputBlobsDesc( 0 ).ObjectSize() == 1
					&& inputNormLayer->GetOutputBlobsDesc( 0 ).ListSize() > 1 ) );

				normLayer = new CObjectNormalizationLayer( Graph.GetMathEngine() );
				normLayer->SetName( "myObjNorm_" + Str( normLayersCount ) );
				normLayer->SetScale( scaleBlob->GetTransposed( BD_ListSize, BD_Height ) );
				normLayer->SetBias( biasBlob->GetTransposed( BD_ListSize, BD_Height ) );
				Graph.AddLayer( *normLayer );

				CPtr<CTransposeLayer> layerTransposeBefore( new CTransposeLayer( Graph.GetMathEngine() ) );
				layerTransposeBefore->SetName( "myTransposeBefore_" + Str( normLayersCount ) );
				layerTransposeBefore->SetTransposedDimensions( BD_ListSize, BD_Height );
				Graph.AddLayer( *layerTransposeBefore );
				layerTransposeBefore->Connect( *inputNormLayer );
				normLayer->Connect( /*input number*/0, *layerTransposeBefore, /*output number*/0 );

				CPtr<CTransposeLayer> layerTransposeAfter( new CTransposeLayer( Graph.GetMathEngine() ) );
				layerTransposeAfter->SetName( "myTransposeAfter_" + Str( normLayersCount ) );
				layerTransposeAfter->SetTransposedDimensions( BD_ListSize, BD_Height );
				Graph.AddLayer( *layerTransposeAfter );
				layerTransposeAfter->Connect( *normLayer );
				newNormLayer = layerTransposeAfter;
				configureLayerConnection = 1;
			} else {
				//continue;
				return; // Not fit for the whole ANN
			}
		} else {
			ASSERT_EXPR( configureLayerConnection == 2
				|| ( configureLayerConnection == 0 && inputNormLayer->GetOutputBlobsDesc( 0 ).ObjectSize() > 1 ) );
			normLayer = new CObjectNormalizationLayer( Graph.GetMathEngine() );
			normLayer->SetName( "myObjNorm_" + Str( normLayersCount ) );
			normLayer->SetScale( scaleBlob );
			normLayer->SetBias( biasBlob );
			Graph.AddLayer( *normLayer );
			normLayer->Connect( *inputNormLayer );
			newNormLayer = normLayer;
			configureLayerConnection = 2;
		}
		normLayer->SetEpsilon( epsBlob->GetData().GetValue() );

		CPtr<CTransformLayer> transformLayer( new CTransformLayer( Graph.GetMathEngine() ) );
		transformLayer->SetName( "myTransform_" + Str( normLayersCount ) );
		transformLayer->SetDimensionRule( TBlobDim::BD_Channels, CTransformLayer::O_InputDim, TBlobDim::BD_Height );
		transformLayer->SetDimensionRule( TBlobDim::BD_Height, CTransformLayer::O_InputDim, TBlobDim::BD_Channels );
		transformLayer->SetDimensionRule( TBlobDim::BD_BatchWidth, CTransformLayer::O_InputDim, TBlobDim::BD_BatchLength );
		transformLayer->SetDimensionRule( TBlobDim::BD_BatchLength, CTransformLayer::O_InputDim, TBlobDim::BD_BatchWidth );
		Graph.AddLayer( *transformLayer );
		transformLayer->Connect( /*input number*/0, *newNormLayer, /*output number*/0 );
		const CBaseLayer& newLayer = static_cast<const CBaseLayer&>( *transformLayer );

		for( int ii = i; ii < layersList.Size(); ++ii ) {
			const char* const nameOutputLayer = layersList[ii];
			auto layerDnn = Graph.GetLayer( nameOutputLayer );
			for( int inputNum = 0; inputNum < layerDnn->GetInputCount(); ++inputNum ) {
				if( std::strcmp( layerDnn->GetInputName( inputNum ), addLayerLast->GetName() ) == 0 ) {
					layerDnn->Connect( inputNum, newLayer, /*output number*/0 );
				}
			}
			if( HasSelectedLayer( layerDnn ) ) {
				layersList[ii] = nullptr;
			}
		}

		for( int ii = 0; ii < GetSelectedLayersSize(); ++ii ) {
			auto layer = GetSelectedLayer( ii );
			if( Graph.HasLayer( layer->GetName() ) ) {
				Graph.DeleteLayer( *layer );
			}
		}
		++normLayersCount;

		// Use 'configureLayerConnection' once to find out the ANN architecture and speed-up the procedure, instead of 'Reshape' each time
		//Graph.Reshape(); // Check for architecture correctness
	}
}

} // namespace NeoOnnx

