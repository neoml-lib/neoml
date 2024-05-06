/* Copyright © 2017-2024 ABBYY

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

#include "OptimizerFunctions.h"
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/DnnHeadAdapterLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/RecurrentLayer.h>
#include <NeoML/Dnn/Optimization/Graph.h>

namespace NeoML {

namespace optimization {

// Returns copy of an original layer
static CPtr<CBaseLayer> copyLayer( CBaseLayer& original )
{
	CMemoryFile file;
	{
		CArchive archive( &file, CArchive::SD_Storing );
		CPtr<CBaseLayer> originalPtr( &original );
		SerializeLayer( archive, original.MathEngine(), originalPtr );
	}
	file.SeekToBegin();
	CPtr<CBaseLayer> result;
	CArchive archive( &file, CArchive::SD_Loading );
	SerializeLayer( archive, original.MathEngine(), result );
	return result;
}

// Returns true if the given composite is a recurrent
static bool isRecurrent( const CCompositeLayer& composite )
{
	auto recurrent = dynamic_cast<const CRecurrentLayer*>( &composite );
	return recurrent != nullptr
		&& ( recurrent->GetBackLinkCount() != 0 || recurrent->GetRepeatCount() != 1 );
}

// Return the index of the given composite source or sink name
static int getCompositeIOIndex( const CString& name ) {
	// HACK: the only place where CCompositeSource/SinkLayer contains its index is its name
	auto getIndex = [&name] ( const char* prefix ) -> int {
		const int prefixLen = static_cast<int>( strlen( prefix ) );
		NeoAssert( name.CompareSubstr( 0, prefix, prefixLen ) == 0 );
		int index = 0;
		NeoAssert( Value( name.Mid( prefixLen, name.Length() - prefixLen ), index ) );
		return index;
	};

	NeoAssert( name.Length() > 14 );
	if( name[10] == 'o' ) {
		// CompositeSource
		return getIndex( "CompositeSource." );
	}

	return getIndex( "CompositeSink." );
}

// Moves internal layers of composite to the root graph
static void unpackComposite( CGraph& graph, CCompositeLayer& composite )
{
	// Naming:
	// subLayer - layers inside of composite
	// newLayer - the copy of subLayer in the graph (outside of composite)
	CArray<const char*> subLayerNames;
	// GetLayerList doesn't return CCompositeSource and CCompositeSink
	composite.GetLayerList( subLayerNames );

	// Copy internal layers from composite to graph (without connections)
	for( const char* subLayerName : subLayerNames ) {
		CBaseLayer* subLayer = composite.GetLayer( subLayerName );
		CPtr<CBaseLayer> newLayer = copyLayer( *subLayer );
		newLayer->SetName( subLayer->GetPath() ); // Avoid conflict of names
		graph.AddLayer( *newLayer );
	}

	// Restore the connections between layers
	for( const char* subLayerName : subLayerNames ) {
		CBaseLayer* subLayer = composite.GetLayer( subLayerName );
		CBaseLayer* newLayer = graph.GetLayer( subLayer->GetPath() );

		for( int inputIndex = 0; inputIndex < subLayer->GetInputCount(); ++inputIndex ) {
			const CString subLayerName = subLayer->GetInputName( inputIndex );
			if( !composite.HasLayer( subLayerName ) ) {
				// subLayerName is a name of composite source
				const int compositeInputIndex = getCompositeIOIndex( subLayerName );
				CLayerOutput<> compositeInput = graph.GetConnectedOutput( composite, compositeInputIndex );
				graph.Connect( *newLayer, inputIndex, *compositeInput.Layer, compositeInput.Index );
				continue;
			}

			CBaseLayer* inputSubLayer = composite.GetLayer( subLayerName );
			NeoAssert( graph.HasLayer( inputSubLayer->GetPath() ) );
			CBaseLayer* newInputLayer = graph.GetLayer( inputSubLayer->GetPath() );
			const int outputIndex = subLayer->GetInputOutputNumber( inputIndex );
			graph.Connect( *newLayer, inputIndex, *newInputLayer, outputIndex );
		}
	}

	for( int outputIndex = 0; outputIndex < composite.GetOutputMappingCount(); ++outputIndex ) {
		// Connect everythin that previously was connected to the composite
		// to a new layer which is a copy of smth that was connected to this sink
		const CBaseLayer* subLayer = composite.GetLayer( composite.GetOutputMappingLayer( outputIndex ) );
		const CString newLayerName = subLayer->GetPath();
		NeoAssert( graph.HasLayer( newLayerName ) );
		const int newOutputIndex = composite.GetOutputMappingIndex( outputIndex );
		graph.SwitchOutputs( composite, outputIndex, *graph.GetLayer( newLayerName ), newOutputIndex );
	}

	graph.DeleteLayer( composite );
}

int UnpackComposites( CGraph& graph )
{
	int removedComposites = 0;
	int removedCompositesThisIteration = 0;

	do {
		removedCompositesThisIteration = 0;
		CArray<CBaseLayer*> layers;
		graph.GetLayers( layers );
		for( CBaseLayer* layer : layers ) {
			auto composite = dynamic_cast<CCompositeLayer*>( layer );
			if( composite == nullptr || isRecurrent( *composite ) ) {
				continue;
			}
			unpackComposite( graph, *composite );
			++removedCompositesThisIteration;
		}
		removedComposites += removedCompositesThisIteration;
	} while( removedCompositesThisIteration > 0 );

	return removedComposites;
}

//---------------------------------------------------------------------------------------------------------------------

int OptimizeDnnHeadAdapters( CGraph& graph )
{
	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	int result = 0;
	for( CBaseLayer* layer : layers ) {
		CDnnHeadAdapterLayer* head = dynamic_cast<CDnnHeadAdapterLayer*>( layer );
		if( head != nullptr && OptimizeDnn( *( head->GetDnnHead()->dnn ) ).IsOptimized() ) {
			++result;
		}
	}
	return result;
}

//---------------------------------------------------------------------------------------------------------------------

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

} // namespace optimization

} // namespace NeoML
