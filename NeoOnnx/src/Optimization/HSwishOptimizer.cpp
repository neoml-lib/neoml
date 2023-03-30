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

#include "common.h"
#pragma hdrstop

#include <cmath>

#include "Graph.h"
#include "HSwishOptimizer.h"
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoOnnx {

namespace optimization {

void CHSwishOptimizer::Apply()
{
	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	for( CBaseLayer* layer : layers ) {
		if( !graph.HasLayer( layer ) ) {
			// There is a risk that layer has already been deleted
			// and 'layer' points to an invalid object
			continue;
		}

		// Find the last mul layer
		COnnxEltwiseLayer* mulLayer = dynamic_cast<COnnxEltwiseLayer*>( layer );
		if( mulLayer == nullptr || graph.GetInputCount( *mulLayer ) != 2
			|| mulLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Mul )
		{
			continue;
		}

		// Check mulLayer's inputs
		for( int i = 0; i < 2; ++i ) {
			CHardSigmoidLayer* hardSigmoid = graph.GetConnectedOutput<CHardSigmoidLayer>( *mulLayer, i ).Layer;
			if( hardSigmoid == nullptr ) {
				continue;
			}
			CLayerOutput<> hSwishInputData = graph.GetConnectedOutput<>( *mulLayer, 1 - i );

			if( isValidHardSigmoidLayer( *hardSigmoid, hSwishInputData ) ) {
				CPtr<CHSwishLayer> hSwishLayer = new CHSwishLayer( graph.MathEngine() );
				hSwishLayer->SetName( graph.GetUniqueName( "HSwish" ) );
				graph.AddLayer( *hSwishLayer );

				graph.Connect( *hSwishLayer, 0, *hSwishInputData.Layer, hSwishInputData.Index );
				graph.SwitchOutputs( *mulLayer, 0, *hSwishLayer, 0 );

				graph.DeleteLayer( *mulLayer );
				graph.DeleteLayer( *hardSigmoid );
				break;
			}
		}
	}
}

// Checks if CHardSigmoidLayer is valid for CHSwishLayer conversion
bool CHSwishOptimizer::isValidHardSigmoidLayer( CHardSigmoidLayer& hardSigmoidLayer,
	const CLayerOutput<>& hSwishInputData ) const
{
	// HardSigmoid layer always has 1 input and 1 output
	NeoAssert( graph.GetInputCount( hardSigmoidLayer ) == 1 );
	NeoAssert( graph.GetOutputCount( hardSigmoidLayer ) == 1 );

	// If HardSigmoid is used by some other layer then we can't replace it with CHSwishLayer
	if( graph.GetConnectedInputsCount( hardSigmoidLayer, 0 ) != 1 ) {
		return false;
	}

	if( std::abs( hardSigmoidLayer.GetSlope() - 1.f / 6 ) > 1e-4f
		|| std::abs( hardSigmoidLayer.GetBias() - 0.5f ) > 1e-4f )
	{
		return false;
	}

	// Check that hard sigmoid is connected to the same input, as other connection of mulLayer
	return graph.GetConnectedOutput<>( hardSigmoidLayer, 0 ) == hSwishInputData;
}

} // namespace optimization

} // namespace NeoOnnx
