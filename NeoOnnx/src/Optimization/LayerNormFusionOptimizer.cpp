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
	Graph.Reshape(); // be sure, the all BLOBs' sizes are inited in the graph

	int normLayersCount = 0;
	CArray<const char*> layersList{};
	Graph.GetLayerList( layersList );

	for( int i = 0; i < layersList.Size(); ++i ) {
		ClearSelectedLayers();

		if( layersList[i] == nullptr )
			continue;

		CPtr<CBaseLayer> addLayerLast = Graph.GetLayer( layersList[i] );
		if( !IsExactLayer( addLayerLast, "FmlCnnEltwiseSumLayer" ) )
			continue;

		CPtr<CBaseLayer> biasLayer = nullptr, mulLayer = nullptr;
		if( !GetExactInputLayers( addLayerLast, mulLayer, "FmlCnnEltwiseMulLayer", biasLayer, "NeoMLDnnDataLayer", /*layerSkipClass*/"" ) )
			continue;

		CPtr<CBaseLayer> scaleLayer = nullptr, divLayer = nullptr;
		if( !GetExactInputLayers( mulLayer, divLayer, "NeoMLDnnEltwiseDivLayer", scaleLayer, "NeoMLDnnDataLayer", "NeoMLDnnCastLayer" ) )
			continue;

		CPtr<CBaseLayer> sqrtLayer = nullptr, sub2Layer = nullptr;
		if( !GetExactInputLayers( divLayer, sqrtLayer, "FmlCnnPowerLayer", sub2Layer, "NeoMLDnnEltwiseSubLayer", /*layerSkipClass*/"" ) )
			continue;

		CPtr<CBaseLayer> addLayer = nullptr, unusedLayer = nullptr;
		if( !GetExactInputLayers( sqrtLayer, addLayer, "FmlCnnEltwiseSumLayer", unusedLayer, "", /*layerSkipClass*/"" ) )
			continue;

		CPtr<CBaseLayer> reduceMean2Layer = nullptr, epsilontLayer = nullptr;
		if( !GetExactInputLayers( addLayer, reduceMean2Layer, "FmlCnnGlobalMainPoolingLayer", epsilontLayer, "NeoMLDnnDataLayer", /*layerSkipClass*/"" ) )
			continue;

		CPtr<CBaseLayer> powLayer = nullptr;
		if( !GetExactInputLayers( reduceMean2Layer, powLayer, "FmlCnnPowerLayer", unusedLayer, "", /*layerSkipClass*/"" ) )
			continue;

		CPtr<CBaseLayer> subLayer = nullptr;
		if( !GetExactInputLayers( powLayer, subLayer, "NeoMLDnnEltwiseSubLayer", unusedLayer, "", "NeoMLDnnCastLayer" ) )
			continue;

		CPtr<CBaseLayer> reduceMeanLayer = nullptr;
		if( !GetExactInputLayers( subLayer, reduceMeanLayer, "FmlCnnGlobalMainPoolingLayer", unusedLayer, "", /*layerSkipClass*/"" ) )
			continue;

		CPtr<CBaseLayer> reduceMeanInputLayer{}, inputNormLayer{};
		for( int j = 0; ( reduceMeanInputLayer = GetAnyInputLayer( reduceMeanLayer, j, "NeoMLDnnCastLayer" ) ) != nullptr; ++j ) {
			ASSERT_EXPR( j == 0 );
			inputNormLayer = reduceMeanInputLayer;
		}

		const auto& epsBlob = dynamic_cast<CDataLayer*>( epsilontLayer.Ptr() )->GetBlob();
		const auto& scaleBlob = dynamic_cast<CDataLayer*>( scaleLayer.Ptr() )->GetBlob();
		const auto& biasBlob = dynamic_cast<CDataLayer*>( biasLayer.Ptr() )->GetBlob();

		CPtr<CObjectNormalizationLayer> normLayer{};
		CPtr<CBaseLayer> newNormLayer{}; //layer wrapper to connect further
		if( inputNormLayer->GetOutputBlobsDesc( 0 ).ObjectSize() == 1 ) {
			if( inputNormLayer->GetOutputBlobsDesc( 0 ).ListSize() > 1 ) {
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
			} else {
				continue;
			}
		} else {
			normLayer = new CObjectNormalizationLayer( Graph.GetMathEngine() );
			normLayer->SetName( "myObjNorm_" + Str( normLayersCount ) );
			normLayer->SetScale( scaleBlob );
			normLayer->SetBias( biasBlob );
			Graph.AddLayer( *normLayer );
			normLayer->Connect( *inputNormLayer );
			newNormLayer = normLayer;
		}
		normLayer->SetEpsilon( epsBlob->GetData().GetValue() );

		CPtr<CTransformLayer> transformLayer( new CTransformLayer( Graph.GetMathEngine() ) );
		transformLayer->SetName( "myTransform_" + Str( normLayersCount ) );
		transformLayer->SetDimensionRule( TBlobDim::BD_Channels, CTransformLayer::O_InputDim, TBlobDim::BD_Height );
		transformLayer->SetDimensionRule( TBlobDim::BD_Height, CTransformLayer::O_InputDim, TBlobDim::BD_BatchLength );
		transformLayer->SetDimensionRule( TBlobDim::BD_BatchLength, CTransformLayer::O_InputDim, TBlobDim::BD_Channels );
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

		Graph.Reshape(); // check for architecture correctness
	}
}

} // namespace NeoOnnx

