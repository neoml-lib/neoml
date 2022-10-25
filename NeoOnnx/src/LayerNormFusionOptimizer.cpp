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

#include "LayerNormFusionOptimizer.h"

namespace NeoOnnx {

void CLayerNormFusionOptimizer::Apply()
{
	//graph.Reshape(); // be sure, the all BLOBs' sizes are inited in the graph

	int normLayersCount = 0;
	CArray<const char*> layersList{};
	graph.GetLayerList( layersList );
	for( int i = 0; i < layersList.Size(); ++i ) {
		layersToRemove.DeleteAll();

		CPtr<CBaseLayer> addLayerLast = graph.GetLayer( layersList[i] );
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

		CPtr<CObjectNormalizationLayer> normLayer = new CObjectNormalizationLayer( graph.GetMathEngine() );
		normLayer->SetName( ( "myObjNorm" + std::to_string( normLayersCount ) ).c_str() );
		graph.AddLayer( *normLayer );

		const auto& epsBlob = dynamic_cast<CDataLayer*>( epsilontLayer.Ptr() )->GetBlob();
		const auto& scaleBlob = dynamic_cast<CDataLayer*>( scaleLayer.Ptr() )->GetBlob();
		const auto& biasBlob = dynamic_cast<CDataLayer*>( biasLayer.Ptr() )->GetBlob();

		normLayer->SetEpsilon( epsBlob->GetData().GetValue() );
		normLayer->SetScale( scaleBlob );
		normLayer->SetBias( biasBlob );

		CPtr<CBaseLayer> subCheckLayer = ( sub2Layer == subLayer ) ? subLayer : sub2Layer;

		CPtr<CBaseLayer> reduceMeanInputLayer;
		for( int j = 0; ( reduceMeanInputLayer = GetAnyInputLayer( reduceMeanLayer, j, "NeoMLDnnCastLayer" ) ) != nullptr; ++j ) {
			normLayer->Connect( *reduceMeanInputLayer );
		}

		CPtr<CTransposeLayer> transpose1Layer = new CTransposeLayer( graph.GetMathEngine() );
		transpose1Layer->SetName( ( "myTranspose1_" + std::to_string( normLayersCount ) ).c_str() );
		transpose1Layer->SetTransposedDimensions( TBlobDim::BD_Height, TBlobDim::BD_Channels );
		graph.AddLayer( *transpose1Layer );
		transpose1Layer->Connect(/*input number*/0, *normLayer, /*output number*/0 );

		CPtr<CTransposeLayer> transpose2Layer = new CTransposeLayer( graph.GetMathEngine() );
		transpose2Layer->SetName( ( "myTranspose2_" + std::to_string( normLayersCount ) ).c_str() );
		transpose2Layer->SetTransposedDimensions( TBlobDim::BD_Height, TBlobDim::BD_BatchLength );
		graph.AddLayer( *transpose2Layer );
		transpose2Layer->Connect(/*input number*/0, *transpose1Layer, /*output number*/0 );

		const CBaseLayer& newLayer = (const CBaseLayer&)( *transpose2Layer.Ptr() );

		for( int ii = i; ii < layersList.Size(); ++ii ) {
			const char* const nameOutputLayer = layersList[ii];
			auto layerDnn = graph.GetLayer( nameOutputLayer );
			for( int j = 0; j < layerDnn->GetInputCount(); ++j ) {
				if( layerDnn->GetInputName( j ) == CString( addLayerLast->GetName() ) ) {
					layerDnn->Connect(/*input number*/j, newLayer, /*output number*/0 );
				}
			}
		}

		for( int ii = 0; ii < layersToRemove.Size(); ++ii ) {
			auto& layer = layersToRemove[ii];
			if( graph.HasLayer( layer->GetName() ) ) {
				graph.DeleteLayer( *layer );
			}
		}
		++normLayersCount;

		//graph.Reshape(); // check for architecture correctness
	}
}

} // namespace NeoOnnx

