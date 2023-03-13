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

#include "DnnGraphWrapper.h"

namespace NeoOnnx {

CDnnGraphWrapper::CDnnGraphWrapper( CDnn& _dnn ) :
	dnn( _dnn )
{
}

void CDnnGraphWrapper::Build()
{
	CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );

	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CBaseLayer& layer = *dnn.GetLayer( layerNames[layerIndex] );
		CLayerLinks& inputLayerLinks = graphLinks.GetOrCreateValue( &layer );
		inputLayerLinks.Inputs.SetSize( layer.GetInputCount() );

		CDnnGraphLink inputLink;
		inputLink.Layer = &layer;
		for( inputLink.Index = 0; inputLink.Index < layer.GetInputCount(); ++inputLink.Index ) {
			CDnnGraphLink outputLink;
			outputLink.Layer = dnn.GetLayer( layer.GetInputName( inputLink.Index ) );
			outputLink.Index = layer.GetInputOutputNumber( inputLink.Index );
			inputLayerLinks.Inputs[inputLink.Index] = outputLink;

			CLayerLinks& outputLayerLinks = graphLinks.GetOrCreateValue( outputLink.Layer );
			if( outputLayerLinks.Outputs.Size() <= outputLink.Index ) {
				outputLayerLinks.Outputs.SetSize( outputLink.Index + 1 );
			}
			outputLayerLinks.Outputs[outputLink.Index].Add( inputLink );
		}
	}
}

void CDnnGraphWrapper::GetLayers( CArray<CBaseLayer*>& layers ) const
{
	CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );

	layers.Empty();
	layers.SetBufferSize( layerNames.Size() );
	for( const char* layerName : layerNames ) {
		layers.Add( dnn.GetLayer( layerName ) );
	}
}

int CDnnGraphWrapper::GetInputCount( const CBaseLayer& layer ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );
	return graphLinks.GetValue( layerLinksPos ).Inputs.Size();
}

const CDnnGraphLink& CDnnGraphWrapper::GetInputLink( const CBaseLayer& layer, int inputIndex ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	NeoAssert( inputIndex < layerLinks.Inputs.Size() );
	return layerLinks.Inputs[inputIndex];
}

int CDnnGraphWrapper::GetOutputCount( const CBaseLayer& layer ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );
	
	return graphLinks.GetValue( layerLinksPos ).Outputs.Size();
}

int CDnnGraphWrapper::GetOutputLinkCount( const CBaseLayer& layer, int outputIndex ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	NeoAssert( outputIndex < layerLinks.Outputs.Size() );
	return layerLinks.Outputs[outputIndex].Size();
}

const CDnnGraphLink& CDnnGraphWrapper::GetOutputLink( const CBaseLayer& layer, int outputIndex,
	int linkIndex ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	NeoAssert( outputIndex < layerLinks.Outputs.Size() );
	NeoAssert( linkIndex < layerLinks.Outputs[outputIndex].Size() );
	return layerLinks.Outputs[outputIndex][linkIndex];
}

CString CDnnGraphWrapper::GetUniqueName( const CString& prefix ) const
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

void CDnnGraphWrapper::AddLayer( CBaseLayer& layer )
{
	NeoAssert( !dnn.HasLayer( layer.GetName() ) );
	NeoAssert( !graphLinks.Has( &layer ) );
	dnn.AddLayer( layer );
	graphLinks.AddValue( &layer );
}

void CDnnGraphWrapper::DeleteLayer( CBaseLayer& layer )
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &layer, layerLinksPos ) == NotFound );
	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );

	// Disconnect all the inputs of deleted layer
	CArray<CDnnGraphLink> usedOutputs;
	layerLinks.Inputs.CopyTo( usedOutputs );
	CDnnGraphLink inputLink;
	inputLink.Layer = &layer;
	for( inputLink.Index = 0; inputLink.Index < usedOutputs.Size(); ++inputLink.Index ) {
		Disconnect( inputLink, usedOutputs[inputLink.Index] );
	}

	const int outputCount = GetOutputCount( layer );
	CDnnGraphLink outputLink;
	outputLink.Layer = &layer;
	for( outputLink.Index = 0; outputLink.Index < outputCount; ++outputLink.Index ) {
		CArray<CDnnGraphLink> usedInputs;
		layerLinks.Outputs[outputLink.Index].CopyTo( usedInputs );
		for( const CDnnGraphLink& usedInput : usedInputs ) {
			Disconnect( usedInput, outputLink );
		}
	}

	// Delete layer from CDnn and graphLinks
	dnn.DeleteLayer( layer );
	graphLinks.DeleteAt( layerLinksPos );
}

void CDnnGraphWrapper::Connect( const CDnnGraphLink& inputLink, const CDnnGraphLink& outputLink )
{
	NeoAssert( inputLink.Layer != nullptr );
	NeoAssert( inputLink.Index >= 0 );
	NeoAssert( outputLink.Layer != nullptr );
	NeoAssert( outputLink.Index >= 0 );

	// Update input link info
	const int inputLayerLinksPos = graphLinks.GetFirstPosition( inputLink.Layer );
	NeoAssert( inputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( inputLink.Layer, inputLayerLinksPos ) == NotFound );
	CLayerLinks& inputLayerLinks = graphLinks.GetValue( inputLayerLinksPos );
	if( inputLayerLinks.Inputs.Size() <= inputLink.Index ) {
		// Allocate inputs which have never been used before
		inputLayerLinks.Inputs.SetSize( inputLink.Index + 1 );
	} else if( inputLayerLinks.Inputs[inputLink.Index].Layer != nullptr ) {
		// Disconnect previous link
		Disconnect( inputLink, inputLayerLinks.Inputs[inputLink.Index] );
	}
	inputLayerLinks.Inputs[inputLink.Index] = outputLink;

	// Update output link info
	const int outputLayerLinksPos = graphLinks.GetFirstPosition( outputLink.Layer );
	NeoAssert( outputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( outputLink.Layer, outputLayerLinksPos ) == NotFound );
	CLayerLinks& outputLayerLinks = graphLinks.GetValue( outputLayerLinksPos );
	if( outputLayerLinks.Outputs.Size() <= outputLink.Index ) {
		// Allocate outputs which have never been used before
		outputLayerLinks.Outputs.SetSize( outputLink.Index + 1 );
	}
	outputLayerLinks.Outputs[outputLink.Index].Add( inputLink );

	// Connect inside CDnn
	inputLink.Layer->Connect( inputLink.Index, *outputLink.Layer, outputLink.Index );
}

void CDnnGraphWrapper::Disconnect( const CDnnGraphLink& inputLink, const CDnnGraphLink& outputLink )
{
	NeoAssert( inputLink.Layer != nullptr );
	NeoAssert( inputLink.Index >= 0 );
	NeoAssert( outputLink.Layer != nullptr );
	NeoAssert( outputLink.Index >= 0 );

	// Update input link info
	const int inputLayerLinksPos = graphLinks.GetFirstPosition( inputLink.Layer );
	NeoAssert( inputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( inputLink.Layer, inputLayerLinksPos ) == NotFound );
	CLayerLinks& inputLayerLinks = graphLinks.GetValue( inputLayerLinksPos );
	NeoAssert( inputLink.Index < inputLayerLinks.Inputs.Size() );
	NeoAssert( inputLayerLinks.Inputs[inputLink.Index] == outputLink );
	inputLayerLinks.Inputs[inputLink.Index].Layer = nullptr;
	inputLayerLinks.Inputs[inputLink.Index].Index = NotFound;

	// Update output link info
	const int outputLayerLinksPos = graphLinks.GetFirstPosition( outputLink.Layer );
	NeoAssert( outputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( outputLink.Layer, outputLayerLinksPos ) == NotFound );
	CLayerLinks& outputLayerLinks = graphLinks.GetValue( outputLayerLinksPos );
	NeoAssert( outputLink.Index < outputLayerLinks.Outputs.Size() );
	const int inputLinkPos = outputLayerLinks.Outputs[outputLink.Index].Find( inputLink );
	NeoAssert( inputLinkPos != NotFound );
	NeoAssert( outputLayerLinks.Outputs[outputLink.Index].Find( inputLink, inputLinkPos + 1 )
		== NotFound );
	outputLayerLinks.Outputs[outputLink.Index].DeleteAt( inputLinkPos );

	// TODO: implement proper disconnect for CDnn and CDnnBaseLayer
}

void CDnnGraphWrapper::SwitchOutputs( const CDnnGraphLink& oldOutput, const CDnnGraphLink& newOutput )
{
	const int oldLayerPos = graphLinks.GetFirstPosition( oldOutput.Layer );
	NeoAssert( oldLayerPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( oldOutput.Layer, oldLayerPos ) == NotFound );
	CLayerLinks& oldLayerLinks = graphLinks.GetValue( oldLayerPos );
	NeoAssert( oldOutput.Index < oldLayerLinks.Outputs.Size() );

	// Making local copy of inputs because of graph modifications during Disconnect/Connect calls
	CArray<CDnnGraphLink> inputsToSwitch;
	oldLayerLinks.Outputs[oldOutput.Index].CopyTo( inputsToSwitch );
	for( const CDnnGraphLink& inputToSwitch : inputsToSwitch ) {
		Disconnect( inputToSwitch, oldOutput );
		Connect( inputToSwitch, newOutput );
	}
}

} // namespace NeoOnnx
