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

namespace NeoOnnx {

// Information about connection
struct CDnnGraphLink {
	CBaseLayer* Layer = nullptr; // The other side of connection
	int Index = NotFound; // Number of layer's input or output used in connection

	bool operator==( const CDnnGraphLink& other ) const { return Layer == other.Layer && Index == other.Index; }
};

// Wrapper which provides better interface for modifying CDnn graph than the CDnn itself
class CDnnGraphWrapper {
public:
	explicit CDnnGraphWrapper( CDnn& dnn );
	CDnnGraphWrapper( const CDnnGraphWrapper& ) = delete;
	CDnnGraphWrapper& operator=( const CDnnGraphWrapper& ) = delete;

	void Build();

	void GetLayers( CArray<CBaseLayer*>& layers ) const;

	bool HasLayer( const CBaseLayer* layer ) const { return graphLinks.Has( layer ); }

	// Information about incoming connections to the given layer
	// TODO: iterator?
	int GetInputCount( const CBaseLayer& layer ) const;
	const CDnnGraphLink& GetInputLink( const CBaseLayer& layer, int inputIndex ) const;

	// Information about layers connected to the outputs of the given layer
	// TODO: iterator?
	int GetOutputCount( const CBaseLayer& layer ) const;
	int GetOutputLinkCount( const CBaseLayer& layer, int outputIndex ) const;
	const CDnnGraphLink& GetOutputLink( const CBaseLayer& layer, int outputIndex, int linkIndex ) const;

	// Gets a layer name with the given prefix which isn't use in the net
	CString GetUniqueName( const CString& prefix ) const;

	// Adds layer to Dnn
	void AddLayer( CBaseLayer& layer );

	// Deletes layer from Dnn
	void DeleteLayer( CBaseLayer& layer );

	// Connects input.Index'th input of input.Layer to the output.Index'th output of output.Layer
	void Connect( const CDnnGraphLink& inputLink, const CDnnGraphLink& outputLink );

	// Destroys connection between input.Index'th input of input.Layer to the output.Index'th output of output.Layer
	void Disconnect( const CDnnGraphLink& inputLink, const CDnnGraphLink& outputLink );

	// Switches all layers connected to oldOutput.Layer's oldOutput.Index'th output
	// to the newOutput.Layer's newOutput.Index'th output
	void SwitchOutputs( const CDnnGraphLink& oldOutput, const CDnnGraphLink& newOutput );

	IMathEngine& MathEngine() { return dnn.GetMathEngine(); }

private:
	// Information about all the connections of one layer
	struct CLayerLinks {
		CArray<CDnnGraphLink> Inputs;
		CArray<CArray<CDnnGraphLink>> Outputs;		
	};

	// Dnn graph to be modified
	CDnn& dnn;
	// Flat array which contains actual information about connections in graph
	CMap<const CBaseLayer*, CLayerLinks> graphLinks;
};

} // namespace NeoOnnx
