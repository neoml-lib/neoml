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

#include "HardSwishOptimizer.h"

namespace NeoOnnx {

void CHardSwishOptimizer::Apply()
{
	graph.Build();

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
		if( mulLayer == nullptr ) {
			continue;
		}

		// Check mulLayer's inputs
		CHardSigmoidLayer* hardSigmoidLayer = nullptr;
		CDnnGraphLink hardSwishInput;
		if( !isValidHardSwish( *mulLayer, hardSigmoidLayer, hardSwishInput ) ) {
			continue;
		}

		CPtr<CHSwishLayer> hardSwishLayer = new CHSwishLayer( graph.MathEngine() );
		hardSwishLayer->SetName( graph.GetUniqueName( "HardSwish" ) );
		graph.AddLayer( *hardSwishLayer );

		graph.Connect( { hardSwishLayer, 0 }, hardSwishInput );
		graph.SwitchOutputs( { mulLayer, 0 }, { hardSwishLayer, 0 } );

		graph.DeleteLayer( *mulLayer );
		graph.DeleteLayer( *hardSigmoidLayer );
	}
}

// Checks if mul layer is valid for CHSwishLayer conversion
bool CHardSwishOptimizer::isValidHardSwish( const COnnxEltwiseLayer& mulLayer,
	CHardSigmoidLayer*& hardSigmoidLayer, CDnnGraphLink& hardSwishInput ) const
{
	// Eltwise layer always has 1 output
	NeoAssert( graph.GetOutputCount( mulLayer ) == 1 );

	if( mulLayer.GetOperation() != COnnxEltwiseLayer::TOperation::Mul ) {
		return false;
	}

	if( graph.GetInputCount( mulLayer ) != 2 ) {
		return false;
	}

	for( int i = 0; i < 2; ++i ) {
		const CDnnGraphLink& hardSigmoidOutput = graph.GetInputLink( mulLayer, i );
		hardSwishInput = graph.GetInputLink( mulLayer, 1 - i );
		hardSigmoidLayer = dynamic_cast<CHardSigmoidLayer*>( hardSigmoidOutput.Layer );
		if( hardSigmoidLayer == nullptr ) {
			continue;
		}
		if( isValidHardSigmoidLayer( *hardSigmoidLayer, hardSwishInput ) ) {
			return true;
		}
	}

	return false;
}

// Checks if CHardSigmoidLayer is valid for CHSwishLayer conversion
bool CHardSwishOptimizer::isValidHardSigmoidLayer( const CHardSigmoidLayer& hardSigmoidLayer,
	const CDnnGraphLink& hardSwishInput ) const
{
	// HardSigmoid layer always has 1 input and 1 output
	NeoAssert( graph.GetInputCount( hardSigmoidLayer ) == 1 );
	NeoAssert( graph.GetOutputCount( hardSigmoidLayer ) == 1 );

	// If HardSigmoid is used by some other layer then we can't replace it with CHSwishLayer
	if( graph.GetOutputLinkCount( hardSigmoidLayer, 0 ) != 1 ) {
		return false;
	}

	if( std::fabsf( hardSigmoidLayer.GetSlope() - 1.f / 6 ) > 1e-4f
		|| std::fabsf( hardSigmoidLayer.GetBias() - 0.5f ) > 1e-4f )
	{
		return false;
	}

	// Check that hard sigmoid is connected to the same input, as other connection of mulLayer
	const CDnnGraphLink& currInput = graph.GetInputLink( hardSigmoidLayer, 0 );
	return currInput.Layer == hardSwishInput.Layer && currInput.Index == hardSwishInput.Index;
}

} // namespace NeoOnnx