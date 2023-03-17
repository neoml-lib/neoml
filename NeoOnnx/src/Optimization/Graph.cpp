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
#pragma once

#include "Graph.h"

namespace NeoOnnx {

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
	for( const char* layerName : layerNames ) {
		layers.Add( dnn.GetLayer( layerName ) );
	}
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
			Disconnect( input, usedOutputs[input.Index] );
		}
	}

	const int outputCount = GetOutputCount( layer );
	CLayerOutput<> output;
	output.Layer = &layer;
	for( output.Index = 0; output.Index < outputCount; ++output.Index ) {
		CArray<CLayerInput<>> usedInputs;
		layerLinks.Outputs[output.Index].CopyTo( usedInputs );
		for( const CLayerInput<>& usedInput : usedInputs ) {
			Disconnect( usedInput, output );
		}
	}

	// Delete layer from CDnn and graphLinks
	dnn.DeleteLayer( layer );
	graphLinks.DeleteAt( layerLinksPos );
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

} // namespace optimization

} // namespace NeoOnnx
