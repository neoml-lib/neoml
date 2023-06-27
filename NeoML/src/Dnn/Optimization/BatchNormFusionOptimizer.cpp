/* Copyright © 2017-2023 ABBYY Production LLC

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

#include "BatchNormFusionOptimizer.h"
#include <NeoML/Dnn/DnnOptimization.h>
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Optimization/Graph.h>

namespace NeoML {

namespace optimization {

void CBatchNormFusionOptimizer::Apply( CDnnOptimizationReport& report )
{
	report.FusedBatchNormalizations = 0;

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );
	for( CBaseLayer* layer: layers ) {
		CBatchNormalizationLayer* bn = dynamic_cast<CBatchNormalizationLayer*>( layer );
		if( bn == nullptr ) {
			continue;
		}

		CBaseLayer* nextLayer = graph.GetConnectedOutput<>( *bn, 0 ).Layer;
		if( graph.GetOutputCount( *nextLayer ) != 1 || graph.GetConnectedInputsCount( *nextLayer, 0 ) != 1 ) {
			continue;
		}
		
		bool fused = false;

		CBaseConvLayer* conv = dynamic_cast<CBaseConvLayer*>( nextLayer );
		CFullyConnectedLayer* fc = dynamic_cast<CFullyConnectedLayer*>( nextLayer );
		if( conv != nullptr ) {
			conv->ApplyBatchNormalization( *bn );
			fused = true;
		} else if( fc != nullptr ) {
			fc->ApplyBatchNormalization( *bn );
			fused = true;
		}

		if( fused ) {
			graph.SwitchOutputs( *bn, 0, *nextLayer, 0 );
			graph.DeleteLayer( *bn );
			report.FusedBatchNormalizations++;
		}
	}
}

} // namespace optimization

} // namespace NeoML
