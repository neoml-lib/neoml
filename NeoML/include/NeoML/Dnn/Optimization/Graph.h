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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

#include <type_traits>

namespace NeoML {

namespace optimization {

// Input of a layer
template<typename TLayer = CBaseLayer,
	std::enable_if_t<std::is_base_of<CBaseLayer, TLayer>::value, int> = 0>
struct CLayerInput final {
	TLayer* Layer = nullptr; // layer which this input belongs to
	int Index = NotFound; // index of this input

	CLayerInput( TLayer* layer, int index ) : Layer( layer ), Index( index ) {}
	CLayerInput() = default;

	template<typename TOtherLayer>
	bool operator==( const CLayerInput<TOtherLayer>& other ) const
	{ return static_cast<CBaseLayer*>( Layer ) == static_cast<CBaseLayer*>( other.Layer ) && Index == other.Index; }
	template<typename TOtherLayer>
	bool operator!=( const CLayerInput<TOtherLayer>& other ) const
	{ return !( *this == other ); }
};

//---------------------------------------------------------------------------------------------------------------------

// Output of a layer
template<typename TLayer = CBaseLayer,
	std::enable_if_t<std::is_base_of<CBaseLayer, TLayer>::value, int> = 0>
struct CLayerOutput final {
	TLayer* Layer = nullptr; // layer which this output belongs to
	int Index = NotFound; // index of this output

	CLayerOutput( TLayer* layer, int index ) : Layer( layer ), Index( index ) {}
	CLayerOutput() = default;

	void Clear();

	template<typename TOtherLayer>
	bool operator==( const CLayerOutput<TOtherLayer>& other ) const
	{ return static_cast<CBaseLayer*>( Layer ) == static_cast<CBaseLayer*>( other.Layer ) && Index == other.Index; }
	template<typename TOtherLayer>
	bool operator!=( const CLayerOutput<TOtherLayer>& other ) const
	{ return !( *this == other ); }
};

template<typename TLayer,
	std::enable_if_t<std::is_base_of<CBaseLayer, TLayer>::value, int> Enabler>
inline void CLayerOutput<TLayer, Enabler>::Clear()
{
	Layer = nullptr;
	Index = NotFound;
}

//---------------------------------------------------------------------------------------------------------------------

// This representation of CDnn as a graph
// Provides better interface for traversing over the graph and graph modifications
class NEOML_API CGraph final {
public:
	explicit CGraph( CDnn& dnn );
	CGraph( const CGraph& ) = delete;
	CGraph& operator=( const CGraph& ) = delete;

	// Layers in CDnn

	// Checks whether 'layer' points to an existing layer in the graph
	// Works correctly even if 'layer' is invalid or nullptr
	bool HasLayer( const CBaseLayer* layer ) const { return graphLinks.Has( layer ); }

	// Writes pointers of all of the layers of this graph
	// NOTE: Be cautious that after graph modification pointers may lead to deleted objects
	void GetLayers( CArray<CBaseLayer*>& layers ) const;

	// Properties of a layer

	// Number of inputs of this layer
	int GetInputCount( const CBaseLayer& layer ) const;

	// Number of outputs of this layer
	int GetOutputCount( const CBaseLayer& layer ) const;

	// Connections between layers

	// Returns the output to which the inputLayer's inputIndex'th input is connected.
	// If output layer can't be casted to TOutputLayer, then CLayerOutput::Layer == nullptr.
	template<typename TOutputLayer = CBaseLayer>
	CLayerOutput<TOutputLayer> GetConnectedOutput( const CBaseLayer& inputLayer, int inputIndex ) const;

	// Number of different inputs connected to the outputLayer's outputIndex'th output
	int GetConnectedInputsCount( const CBaseLayer& outputLayer, int outputIndex ) const;

	// Addition/removal of the layers

	// Adds layer to the Graph
	void AddLayer( CBaseLayer& layer );

	// Deletes layer from the Graph
	// Destroys all the connections to and from this layer
	void DeleteLayer( CBaseLayer& layer );

	// Addition/removal of connections

	// Connects the given inputLayer's inputIndex'th input to the outputLayer's outputIndex'th output
	void Connect( CBaseLayer& inputLayer, int inputIndex, CBaseLayer& outputLayer, int outputIndex );

	// Destroys connection between inputIndex'th input of inputLayer and the outputIndex'th output of outputLayer
	void Disconnect( CBaseLayer& inputLayer, int inputIndex, CBaseLayer& outputLayer, int outputIndex );

	// Layer selection mechanism
	
	// This mechanism is used during detecting specific constructions in the graph
	// and remove them swiftly from the graph
	
	// The idea is to iterate over the graph layer-by-layer detecting the whole construction
	// If somewhere during this iteration construction was not found then ClearSelection without any modifications
	// If the whole construction has been found then add the replacement, reconnect everything and DeleteSelectedLayers

	// Number of layers in selection
	int SelectionSize() const { return selection.Size(); }

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
	template<typename TOutputLayer = CBaseLayer>
	CLayerOutput<TOutputLayer> SelectConnectedOutput( CBaseLayer& inputLayer, int inputIndex,
		bool checkOutOfSelectionLinks );

	template<typename TOutLayer>
	TOutLayer* SelectTheOnlyConnectedOutput( const CBaseLayer& layer, bool checkOutOfSelectionLinks = false );

	// Checks that the layer has 2 inputs, and that those inputs are connected
	// to outputs of TFirstLayer and TSecondLayer (in any order)
	// If checkOutOfSelectionLinks == true then it performs additional check
	// (see SelectConnectedOutput)
	// If all checks are passed then adds both CLayerOutput<>.Layer to selection
	// and returns true
	// Otherwise returns false and doesn't add any layers to selection
	// It's recommended to use this method when some layer in the selected construction
	// must have 2 inputs of the specific type
	template<typename TFirstType, typename TSecondType>
	bool SelectBothConnectedOutputs( CBaseLayer& layer, CLayerOutput<TFirstType>& firstConnectedOutput,
		CLayerOutput<TSecondType>& secondConnectedOutput, bool checkOutOfSelectionLinks );

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

	// Switches all the inputs which are connected to the oldOutput to the newOutput
	void SwitchOutputs( CBaseLayer& oldOutputLayer, int oldOutputIndex,
		CBaseLayer& newOutputLayer, int newOutputIndex );

private:
	// Information about all connections of one layer
	struct CLayerLinks final
	{
		// Inputs[i] contains CLayerOutput to which i'th input is connected to
		CArray<CLayerOutput<>> Inputs{};
		// Outputs[i] contains array of CLayerInput connected to i'th output
		CArray<CArray<CLayerInput<>>> Outputs{};
	};

	// Dnn graph to be modified
	CDnn& dnn;
	// All connections in the graph
	CMap<const CBaseLayer*, CLayerLinks> graphLinks;
	// Currently selected layers
	CHashTable<CBaseLayer*> selection;

	bool checkOutOfSelectionConnectedInputs( const CBaseLayer& layer ) const;
};

//---------------------------------------------------------------------------------------------------------------------

template<typename TOutputLayer>
inline CLayerOutput<TOutputLayer> CGraph::GetConnectedOutput( const CBaseLayer& inputLayer, int inputIndex ) const
{
	const int layerLinksPos = graphLinks.GetFirstPosition( &inputLayer );
	NeoAssert( layerLinksPos != NotFound );
	NeoAssert( graphLinks.GetNextPosition( &inputLayer, layerLinksPos ) == NotFound );

	const CLayerLinks& layerLinks = graphLinks.GetValue( layerLinksPos );
	NeoAssert( inputIndex < layerLinks.Inputs.Size() );
	CLayerOutput<TOutputLayer> result;
	result.Layer = dynamic_cast<TOutputLayer*>( layerLinks.Inputs[inputIndex].Layer );
	result.Index = result.Layer ? layerLinks.Inputs[inputIndex].Index : NotFound;
	return result;
}

template<typename TOutputLayer>
inline CLayerOutput<TOutputLayer> CGraph::SelectConnectedOutput( CBaseLayer& inputLayer, int inputIndex,
	bool checkOutOfSelectionLinks )
{
	CLayerOutput<TOutputLayer> result = GetConnectedOutput<TOutputLayer>( inputLayer, inputIndex );
	if( result.Layer == nullptr ) {
		return result;
	}
	if( checkOutOfSelectionLinks && !checkOutOfSelectionConnectedInputs( *result.Layer ) ) {
		result.Layer = nullptr;
		return result;
	}
	SelectLayer( *result.Layer );
	return result;
}

template<typename TOutLayer>
inline TOutLayer* CGraph::SelectTheOnlyConnectedOutput( const CBaseLayer& layer, bool checkOutOfSelectionLinks )
{
	if( GetInputCount( layer ) != 1 ) {
		return nullptr;
	}
	CLayerOutput<TOutLayer> connectedOutput = GetConnectedOutput<TOutLayer>( layer, /*inputIndex*/0 );
	if( connectedOutput.Layer == nullptr || IsLayerSelected( *connectedOutput.Layer ) ) {
		return nullptr;
	}
	if( checkOutOfSelectionLinks && !checkOutOfSelectionConnectedInputs( *connectedOutput.Layer ) ) {
		return nullptr;
	}
	SelectLayer( *connectedOutput.Layer );
	return connectedOutput.Layer;
}

template<typename TFirstType, typename TSecondType>
inline bool CGraph::SelectBothConnectedOutputs( CBaseLayer& layer, CLayerOutput<TFirstType>& firstConnectedOutput,
	CLayerOutput<TSecondType>& secondConnectedOutput, bool checkOutOfSelectionLinks )
{
	if( GetInputCount( layer ) != 2 ) {
		return false;
	}

	for( int i = 0; i < 2; ++i ) {
		firstConnectedOutput = GetConnectedOutput<TFirstType>( layer, i );
		secondConnectedOutput = GetConnectedOutput<TSecondType>( layer, 1 - i );
		if( firstConnectedOutput.Layer == nullptr || secondConnectedOutput.Layer == nullptr
			|| IsLayerSelected( *firstConnectedOutput.Layer ) || IsLayerSelected( *secondConnectedOutput.Layer ) )
		{
			continue;
		}

		if( checkOutOfSelectionLinks
			&& ( !checkOutOfSelectionConnectedInputs( *firstConnectedOutput.Layer )
				|| !checkOutOfSelectionConnectedInputs( *secondConnectedOutput.Layer ) ) )
		{
			continue;
		}

		SelectLayer( *firstConnectedOutput.Layer );
		SelectLayer( *secondConnectedOutput.Layer );
		return true;
	}

	return false;
}

} // namespace optimization

} // namespace NeoML
