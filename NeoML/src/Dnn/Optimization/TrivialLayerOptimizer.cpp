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

#include "TrivialLayerOptimizer.h"
#include <NeoML/Dnn/Optimization/Graph.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>

namespace NeoML {

namespace optimization {

int RemoveTrivialLayers( CGraph& graph )
{
	int trivialLayersRemoved = 0;

	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	for( CBaseLayer* layer: layers ) {
		CDropoutLayer* dropout = dynamic_cast<CDropoutLayer*>( layer );
		CLinearLayer* linear = dynamic_cast<CLinearLayer*>( layer );
		if( ( linear != nullptr && linear->GetMultiplier() == 1.f && linear->GetFreeTerm() == 0.f ) 
			|| dropout != nullptr )
		{
			NeoAssert( graph.GetInputCount( *layer ) == 1 );
			NeoAssert( graph.GetOutputCount( *layer ) == 1 );

			CLayerOutput<> newOutput = graph.GetConnectedOutput( *layer, 0 );
			graph.SwitchOutputs( *layer, 0, *newOutput.Layer, newOutput.Index );
			graph.DeleteLayer( *layer );
			++trivialLayersRemoved;
		}
	}

	return trivialLayersRemoved;
}

}

}
