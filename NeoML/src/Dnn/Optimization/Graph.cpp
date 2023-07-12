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

#include "common.h"
#pragma once

#include <NeoML/Dnn/Optimization/Graph.h>

namespace NeoML {

namespace optimization {

CGraph::CGraph( CDnn& dnn ) :
	dnn( dnn )
{
	CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );

	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CBaseLayer& layer = *dnn.GetLayer( layerNames[layerIndex] );
		CLayerLinks& inputLayerLinks = graphLinks.GetOrCreateValue( &layer );
		inputLayerLinks.Inputs.SetSize( layer.GetInputCount() );

		CLayerInput<> input;
		input.Layer = &layer;
		for( input.Index = 0; input.Index < layer.GetInputCount(); ++input.Index ) {
			CLayerOutput<> output;
			output.Layer = dnn.GetLayer( layer.GetInputName( input.Index ) );
			output.Index = layer.GetInputOutputNumber( input.Index );
			inputLayerLinks.Inputs[input.Index] = output;

			CLayerLinks& outputLayerLinks = graphLinks.GetOrCreateValue( output.Layer );
			if( outputLayerLinks.Outputs.Size() <= output.Index ) {
				outputLayerLinks.Outputs.SetSize( output.Index + 1 );
			}
			outputLayerLinks.Outputs[output.Index].Add( input );
		}
	}
}

void CGraph::GetLayers( CArray<CBaseLayer*>& layers ) const
{
	CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );

	layers.Empty();
	layers.SetBufferSize( layerNames.Size() );

	CHashTable<CBaseLayer*> visited;
	auto dfs = [&visited, &layers] ( CDnn& dnn, CBaseLayer* layer, auto&& dfs ) -> void {
		if( visited.Has( layer ) ) {
			return;
		}

		visited.Add( layer );
		for( int i = 0; i < layer->GetInputCount(); ++i ) {
			dfs( dnn, dnn.GetLayer( layer->GetInputName( i ) ).Ptr(), dfs );
		}
		layers.Add( layer );
	};

	for( const char* layerName : layerNames ) {
		dfs( dnn, dnn.GetLayer( layerName ).Ptr(), dfs );
	}
}

CBaseLayer* CGraph::GetLayer( const char* name ) const
{
	NeoAssert( dnn.HasLayer( name ) );
	return dnn.GetLayer( name ).Ptr();
}

int CGraph::GetInputCount( const CBaseLayer& layer ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );
	return graphLinks.GetValue( layerLinksPos ).Inputs.Size();
}

int CGraph::GetOutputCount( const CBaseLayer& layer ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );

	return graphLinks.GetValue( layerLinksPos ).Outputs.Size();
}

int CGraph::GetConnectedInputsCount( const CBaseLayer& outputLayer, int outputIndex ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &outputLayer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &outputLayer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	NeoAssert( outputIndex < layerLinks.Outputs.Size() );
	return layerLinks.Outputs[outputIndex].Size();
}

void CGraph::AddLayer( CBaseLayer& layer )
{
	NeoAssert( !dnn.HasLayer( layer.GetName() ) );
	NeoAssert( !graphLinks.Has( &layer ) );
	dnn.AddLayer( layer );
	graphLinks.AddValue( &layer );
}

void CGraph::DeleteLayer( CBaseLayer& layer )
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );
	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );

	// Disconnect all the inputs of deleted layer
	CArray<CLayerOutput<>> usedOutputs;
	layerLinks.Inputs.CopyTo( usedOutputs );
	CLayerInput<> input;
	input.Layer = &layer;
	for( input.Index = 0; input.Index < usedOutputs.Size(); ++input.Index ) {
		if( usedOutputs[input.Index].Layer != nullptr ) {
			Disconnect( *input.Layer, input.Index, *usedOutputs[input.Index].Layer,
				usedOutputs[input.Index].Index );
		}
	}

	const int outputCount = GetOutputCount( layer );
	CLayerOutput<> output;
	output.Layer = &layer;
	for( output.Index = 0; output.Index < outputCount; ++output.Index ) {
		CArray<CLayerInput<>> usedInputs;
		layerLinks.Outputs[output.Index].CopyTo( usedInputs );
		for( const CLayerInput<>& usedInput : usedInputs ) {
			Disconnect( *usedInput.Layer, usedInput.Index, *output.Layer, output.Index );
		}
	}

	// Delete layer from CDnn and graphLinks
	dnn.DeleteLayer( layer );
	graphLinks.DeleteAt( layerLinksPos );
}

void CGraph::Connect( CBaseLayer& inputLayer, int inputIndex, CBaseLayer& outputLayer, int outputIndex )
{
	NeoAssert( HasLayer( &inputLayer ) );
	NeoAssert( inputIndex >= 0 );
	NeoAssert( HasLayer( &outputLayer ) );
	NeoAssert( outputIndex >= 0 );

	// Update input link info
	const int inputLayerLinksPos = graphLinks.GetFirstPosition( &inputLayer );
	NeoAssert( inputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &inputLayer, inputLayerLinksPos ) == NotFound );
	CLayerLinks& inputLayerLinks = graphLinks.GetValue( inputLayerLinksPos );
	if( inputLayerLinks.Inputs.Size() <= inputIndex ) {
		// Allocate inputs which have never been used before
		inputLayerLinks.Inputs.SetSize( inputIndex + 1 );
	} else if( inputLayerLinks.Inputs[inputIndex].Layer != nullptr ) {
		// Disconnect previous link
		Disconnect( inputLayer, inputIndex, *inputLayerLinks.Inputs[inputIndex].Layer,
			inputLayerLinks.Inputs[inputIndex].Index );
	}
	inputLayerLinks.Inputs[inputIndex].Layer = &outputLayer;
	inputLayerLinks.Inputs[inputIndex].Index = outputIndex;

	// Update output link info
	const int outputLayerLinksPos = graphLinks.GetFirstPosition( &outputLayer );
	NeoAssert( outputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &outputLayer, outputLayerLinksPos ) == NotFound );
	CLayerLinks& outputLayerLinks = graphLinks.GetValue( outputLayerLinksPos );
	if( outputLayerLinks.Outputs.Size() <= outputIndex ) {
		// Allocate outputs which have never been used before
		outputLayerLinks.Outputs.SetSize( outputIndex + 1 );
	}
	outputLayerLinks.Outputs[outputIndex].Add( CLayerInput<>( &inputLayer, inputIndex ) );

	// Connect inside CDnn
	inputLayer.Connect( inputIndex, outputLayer, outputIndex );
}

void CGraph::Disconnect( CBaseLayer& inputLayer, int inputIndex, CBaseLayer& outputLayer, int outputIndex )
{
	NeoAssert( HasLayer( &inputLayer ) );
	NeoAssert( inputIndex >= 0 );
	NeoAssert( HasLayer( &outputLayer ) );
	NeoAssert( outputIndex >= 0 );
	CLayerInput<> input( &inputLayer, inputIndex );
	CLayerOutput<> output( &outputLayer, outputIndex );

	// Update input link info
	const int inputLayerLinksPos = graphLinks.GetFirstPosition( input.Layer );
	NeoAssert( inputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( input.Layer, inputLayerLinksPos ) == NotFound );
	CLayerLinks& inputLayerLinks = graphLinks.GetValue( inputLayerLinksPos );
	NeoAssert( input.Index < inputLayerLinks.Inputs.Size() );
	NeoAssert( inputLayerLinks.Inputs[input.Index] == output );
	inputLayerLinks.Inputs[input.Index].Layer = nullptr;
	inputLayerLinks.Inputs[input.Index].Index = NotFound;

	// Update output link info
	const int outputLayerLinksPos = graphLinks.GetFirstPosition( output.Layer );
	NeoAssert( outputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( output.Layer, outputLayerLinksPos ) == NotFound );
	CLayerLinks& outputLayerLinks = graphLinks.GetValue( outputLayerLinksPos );
	NeoAssert( output.Index < outputLayerLinks.Outputs.Size() );
	const int inputLinkPos = outputLayerLinks.Outputs[output.Index].Find( input );
	NeoAssert( inputLinkPos != NotFound );
	NeoAssert( outputLayerLinks.Outputs[output.Index].Find( input, inputLinkPos + 1 ) == NotFound );
	outputLayerLinks.Outputs[output.Index].DeleteAt( inputLinkPos );

	// TODO: implement proper disconnect for CDnn and CDnnBaseLayer
}

void CGraph::SelectLayer( CBaseLayer& layer )
{
	NeoAssert( HasLayer( &layer ) );
	NeoAssert( !IsLayerSelected( layer ) );
	selection.Add( &layer );
}

void CGraph::DeleteSelectedLayers()
{
	for( int pos = selection.GetFirstPosition(); pos != NotFound; pos = selection.GetNextPosition( pos ) ) {
		DeleteLayer( *selection.GetValue( pos ) );
	}

	ClearSelection();
}

CString CGraph::GetUniqueName( const CString& prefix ) const
{
	if( !dnn.HasLayer( prefix ) ) {
		// If layer doesn't have such layer, no need to add any suffix
		return prefix;
	}

	int suffix = dnn.GetLayerCount();
	CString result = prefix + Str( suffix );
	while( dnn.HasLayer( result ) ) {
		result = prefix + Str( ++suffix );
	}
	return result;
}

void CGraph::SwitchOutputs( CBaseLayer& oldOutputLayer, int oldOutputIndex,
	CBaseLayer& newOutputLayer, int newOutputIndex )
{
	const int oldLayerPos = graphLinks.GetFirstPosition( &oldOutputLayer );
	NeoAssert( oldLayerPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &oldOutputLayer, oldLayerPos ) == NotFound );
	CLayerLinks& oldLayerLinks = graphLinks.GetValue( oldLayerPos );
	NeoAssert( oldOutputIndex < oldLayerLinks.Outputs.Size() );

	// Making local copy of inputs because of graph modifications during Disconnect/Connect calls
	CArray<CLayerInput<>> inputsToSwitch;
	oldLayerLinks.Outputs[oldOutputIndex].CopyTo( inputsToSwitch );
	for( const CLayerInput<>& inputToSwitch : inputsToSwitch ) {
		Disconnect( *inputToSwitch.Layer, inputToSwitch.Index, oldOutputLayer, oldOutputIndex );
		Connect( *inputToSwitch.Layer, inputToSwitch.Index, newOutputLayer, newOutputIndex );
	}
}

// Checks that every input connected to any outputs of the layer
// has already been selected
bool CGraph::checkOutOfSelectionConnectedInputs( const CBaseLayer& layer ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	for( const CArray<CLayerInput<>>& connectedInputs : layerLinks.Outputs ) {
		for( const CLayerInput<>& connectedInput : connectedInputs ) {
			if( !IsLayerSelected( *connectedInput.Layer ) ) {
				return false;
			}
		}
	}

	return true;
}

} // namespace optimization

} // namespace NeoML
