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

#pragma once

#include <NeoOnnx/NeoOnnxDefs.h>
#include <NeoML/NeoML.h>

#include <type_traits>

namespace NeoOnnx {

namespace optimization {

// Input of a layer
template<typename TLayer = CBaseLayer,
	std::enable_if_t<std::is_base_of<CBaseLayer, TLayer>::value, int> = 0>
struct CLayerInput
{
	TLayer* Layer = nullptr; // layer which this input belongs to
	int Index = NotFound; // index of this input

	template<typename TOtherLayer>
	bool operator==( const CLayerInput<TOtherLayer>& other ) const
		{ return static_cast<CBaseLayer*>( Layer ) == static_cast<CBaseLayer*>( other.Layer ) && Index == other.Index; }
};

// Output of a layer
template<typename TLayer = CBaseLayer,
	std::enable_if_t<std::is_base_of<CBaseLayer, TLayer>::value, int> = 0>
struct CLayerOutput
{
	TLayer* Layer = nullptr; // layer which this output belongs to
	int Index = NotFound; // index of this output

	template<typename TOtherLayer>
	bool operator==( const CLayerOutput<TOtherLayer>& other ) const
		{ return static_cast<CBaseLayer*>( Layer ) == static_cast<CBaseLayer*>( other.Layer ) && Index == other.Index; }
};

// This representation of CDnn as a graph
// Provides better interface for traversing over the graph and graph modifications
class CGraph
{
public:
	explicit CGraph( CDnn& dnn );
	CGraph( const CGraph& ) = delete;
	CGraph& operator=( const CGraph& ) = delete;

	// Layers in CDnn

	// Checks whether 'layer' points to an existing layer in the graph
	// Works correctly even if 'layer' is invalid
	bool HasLayer( const CBaseLayer* layer ) const { return graphLinks.Has( layer ); }

	// Writes pointers of all of the layers of this graph
	// Be cautious that after graph modification pointers may lead to deleted objects
	void GetLayers( CArray<CBaseLayer*>& layers ) const;

	// Properties of a layer

	// Number of inputs of this layer
	int GetInputCount( const CBaseLayer& layer ) const;

	// Number of outputs of this layer
	int GetOutputCount( const CBaseLayer& layer ) const;

	// Connections between layers

	// Returns the output to which the input is connected
	// If output layer can't be casted to TOutputLayer then CLayerOutput::Layer == nullptr
	template<typename TOutputLayer, typename TInputLayer>
	CLayerOutput<TOutputLayer> GetConnectedOutput( const CLayerInput<TInputLayer>& input ) const;

	// Number of different inputs connected to the output
	template<typename TOutputLayer>
	int GetConnectedInputsCount( const CLayerOutput<TOutputLayer>& output ) const;

	// Returns index'th input connected to the output
	// index must be between 0 and GetConnectedInputsCount(output)-1
	template<typename TInputLayer, typename TOutputLayer>
	CLayerInput<TInputLayer> GetConnectedInput( const CLayerOutput<TOutputLayer>& output, int index ) const;

	// Addition/removal of the layers

	// Adds layer to the Graph
	void AddLayer( CBaseLayer& layer );

	// Deletes layer from the Graph
	// Destroys all the connections to and from this layer
	void DeleteLayer( CBaseLayer& layer );

	// Addition/removal of connections

	// Connects the given input to the given output
	template<typename TInputLayer, typename TOutputLayer>
	void Connect( const CLayerInput<TInputLayer>& input, const CLayerOutput<TOutputLayer>& output );

	// Destroys connection between input.Index'th input of input.Layer to the output.Index'th output of output.Layer
	template<typename TInputLayer, typename TOutputLayer>
	void Disconnect( const CLayerInput<TInputLayer>& input, const CLayerOutput<TOutputLayer>& output );

	// Layer selection mechanism
	
	// This mechanism is used during detecting specific constructions in the graph
	// and remove them swiftly from the graph
	
	// The idea is to iterate over the graph layer-by-layer detecting the whole construction
	// If somewhere during this iteration construction was not found then ClearSelection without any modifications
	// If the whole construction has been found then add the replacement, reconnect everything and DeleteSelectedLayers

	// Checks whether the layer has already been selected
	bool IsLayerSelected( CBaseLayer& layer ) const { return selection.Has( &layer ); }

	// Adds the layer to selection
	void SelectLayer( CBaseLayer& layer );

	// Checks that the layer connected to the given input can be casted to TOutputLayer
	// Then if checkOutOfSelectionLinks == true it performs additional check
	// that every layer connected to !any! of CLayerOutput<>.Layer outputs has already been selected
	// If checkOutOfSelectionLinks == false then the additional check is skipped
	// If all of the performed checks are succeeded then it adds CLayerOutput<>.Layer to selection
	// Otherwise CLayerOutput<>.Layer is set to nullptr
	template<typename TOutputLayer, typename TInputLayer>
	CLayerOutput<TOutputLayer> SelectConnectedOutput( const CLayerInput<TInputLayer>& input,
		bool checkOutOfSelectionLinks );

	// Clears the current selection
	void ClearSelection() { selection.DeleteAll(); }

	// Deletes selected layers from the graph
	void DeleteSelectedLayers();

	// Auxiliary functions

	// MathEngine used in CDnn
	IMathEngine& MathEngine() { return dnn.GetMathEngine(); }

	// Gets a layer name with the given prefix which isn't use in the net
	// May return prefix itself (if such name is not used in graph)
	CString GetUniqueName( const CString& prefix ) const;

	// Switches all the inputs which are connected to the oldOutput
	// to the newOutput
	template<typename TOldLayer, typename TNewLayer>
	void SwitchOutputs( const CLayerOutput<TOldLayer>& oldOutput, const CLayerOutput<TNewLayer>& newOutput );

private:
	// Information about all connections of one layer
	struct CLayerLinks
	{
		// Inputs[i] contains CLayerOutput to which i'th input is connected to
		CArray<CLayerOutput<>> Inputs;
		// Outputs[i] contains array of CLayerInput connected to i'th output
		CArray<CArray<CLayerInput<>>> Outputs;
	};

	// Dnn graph to be modified
	CDnn& dnn;
	// All connections in the graph
	CMap<const CBaseLayer*, CLayerLinks> graphLinks;
	// Currently selected layers
	CHashTable<CBaseLayer*> selection;
};

//---------------------------------------------------------------------------------------------------------------------

template<typename TOutputLayer, typename TInputLayer>
inline CLayerOutput<TOutputLayer> CGraph::GetConnectedOutput( const CLayerInput<TInputLayer>& input ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( input.Layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( input.Layer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	NeoAssert( input.Index < layerLinks.Inputs.Size() );
	CLayerOutput<TOutputLayer> result;
	result.Layer = dynamic_cast<TOutputLayer*>( layerLinks.Inputs[input.Index].Layer );
	result.Index = layerLinks.Inputs[input.Index].Index;
	return result;
}

template<typename TOutputLayer>
inline int CGraph::GetConnectedInputsCount( const CLayerOutput<TOutputLayer>& output ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( output.Layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( output.Layer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	NeoAssert( output.Index < layerLinks.Outputs.Size() );
	return layerLinks.Outputs[output.Index].Size();
}

template<typename TInputLayer, typename TOutputLayer>
inline CLayerInput<TInputLayer> CGraph::GetConnectedInput( const CLayerOutput<TOutputLayer>& output, int index ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( output.Layer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( output.Layer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	NeoAssert( output.Index < layerLinks.Outputs.Size() );
	NeoAssert( index < layerLinks.Outputs[output.Index].Size() );
	CLayerInput<TInputLayer> result;
	result.Layer = dynamic_cast<TInputLayer*>( layerLinks.Outputs[output.Index][index].Layer );
	result.Index = layerLinks.Outputs[output.Index][index].Index;
	return result;
}

template<typename TInputLayer, typename TOutputLayer>
inline void CGraph::Connect( const CLayerInput<TInputLayer>& input, const CLayerOutput<TOutputLayer>& output )
{
	NeoAssert( input.Layer != nullptr );
	NeoAssert( input.Index >= 0 );
	NeoAssert( output.Layer != nullptr );
	NeoAssert( output.Index >= 0 );

	// Update input link info
	const int inputLayerLinksPos = graphLinks.GetFirstPosition( input.Layer );
	NeoAssert( inputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( input.Layer, inputLayerLinksPos ) == NotFound );
	CLayerLinks& inputLayerLinks = graphLinks.GetValue( inputLayerLinksPos );
	if( inputLayerLinks.Inputs.Size() <= input.Index ) {
		// Allocate inputs which have never been used before
		inputLayerLinks.Inputs.SetSize( input.Index + 1 );
	} else if( inputLayerLinks.Inputs[input.Index].Layer != nullptr ) {
		// Disconnect previous link
		Disconnect( input, inputLayerLinks.Inputs[input.Index] );
	}
	inputLayerLinks.Inputs[input.Index] = output;

	// Update output link info
	const int outputLayerLinksPos = graphLinks.GetFirstPosition( output.Layer );
	NeoAssert( outputLayerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( output.Layer, outputLayerLinksPos ) == NotFound );
	CLayerLinks& outputLayerLinks = graphLinks.GetValue( outputLayerLinksPos );
	if( outputLayerLinks.Outputs.Size() <= output.Index ) {
		// Allocate outputs which have never been used before
		outputLayerLinks.Outputs.SetSize( output.Index + 1 );
	}
	outputLayerLinks.Outputs[output.Index].Add( input );

	// Connect inside CDnn
	input.Layer->Connect( input.Index, *output.Layer, output.Index );
}

template<typename TInputLayer, typename TOutputLayer>
inline void CGraph::Disconnect( const CLayerInput<TInputLayer>& input, const CLayerOutput<TOutputLayer>& output )
{
	NeoAssert( input.Layer != nullptr );
	NeoAssert( input.Index >= 0 );
	NeoAssert( output.Layer != nullptr );
	NeoAssert( output.Index >= 0 );

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

template<typename TOldLayer, typename TNewLayer>
inline void CGraph::SwitchOutputs( const CLayerOutput<TOldLayer>& oldOutput, const CLayerOutput<TNewLayer>& newOutput )
{
	const int oldLayerPos = graphLinks.GetFirstPosition( oldOutput.Layer );
	NeoAssert( oldLayerPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( oldOutput.Layer, oldLayerPos ) == NotFound );
	CLayerLinks& oldLayerLinks = graphLinks.GetValue( oldLayerPos );
	NeoAssert( oldOutput.Index < oldLayerLinks.Outputs.Size() );

	// Making local copy of inputs because of graph modifications during Disconnect/Connect calls
	CArray<CLayerInput<>> inputsToSwitch;
	oldLayerLinks.Outputs[oldOutput.Index].CopyTo( inputsToSwitch );
	for( const CLayerInput<>& inputToSwitch : inputsToSwitch ) {
		Disconnect( inputToSwitch, oldOutput );
		Connect( inputToSwitch, newOutput );
	}
}

template<typename TOutputLayer, typename TInputLayer>
inline CLayerOutput<TOutputLayer> CGraph::SelectConnectedOutput( const CLayerInput<TInputLayer>& input,
	bool checkOutOfSelectionLinks )
{
	CLayerOutput<TOutputLayer> result = GetConnectedOutput<TOutputLayer>( input );
	if( result.Layer == nullptr ) {
		return result;
	}

	if( checkOutOfSelectionLinks ) {
		const int layerPos = graphLinks.GetFirstPosition( result.Layer );
		NeoAssert( layerPos != NotFound );
		NeoAssert( graphLinks.GetNextPosition( result.Layer, layerPos ) == NotFound );

		// Check that every input connected to any of the outputs is selected
		for( const CArray<CLayerInput<>>& connectedInputs : graphLinks.GetValue( layerPos ).Outputs ) {
			for( const CLayerInput<>& connectedInput : connectedInputs ) {
				if( !IsLayerSelected( *connectedInput.Layer ) ) {
					result.Layer = nullptr;
					return result;
				}
			}
		}
	}

	SelectLayer( *result.Layer );
	return result;
}

} // namespace optimization

} // namespace NeoOnnx
