/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoOnnx/NeoOnnx.h>

#include "NeoOnnxCheck.h"
#include "Node.h"
#include "Nodes/GraphInitializer.h"
#include "Nodes/GraphInput.h"
#include "Nodes/GraphOutput.h"
#include "Graph.h"
#include "GraphCache.h"

#include <onnx.pb.h>

#include <fstream>
#include <iostream>

namespace NeoOnnx {

// Gets opset version from ModelProto
static int getOpsetVersion( const onnx::ModelProto& model )
{
	for( const auto& opset : model.opset_import() ) {
		if( opset.domain().empty() ) { // Default onnx opset
			return static_cast<int>( opset.version() );
		}
	}

	CheckOnnxProtocol( false, "Can't determine opset version for a model" );

	return -1;
}

// Builds array of CNode's based on onnxGraph
static void buildGraph( const onnx::GraphProto& onnxGraph, int opsetVersion, CGraph& graph )
{
	graph.SetBufferSize( onnxGraph.input_size() + onnxGraph.initializer_size() + onnxGraph.node_size()
		+ onnxGraph.output_size() );
	CMap<CString, CLink> nodeOutputs;

	// Add graph initializers
	CHashTable<CString> initializers;
	for( const onnx::TensorProto& onnxInitializer : onnxGraph.initializer() ) {
		if( onnxInitializer.dims_size() > 0 ) {
			graph.Add( new CGraphInitializer( graph.NodeCount(), onnxInitializer ) );
			nodeOutputs.Add( onnxInitializer.name().c_str(), CLink( graph.NodeCount() - 1, 0 ) );
			initializers.Add( onnxInitializer.name().c_str() );
		}
	}

	// Add graph inputs
	for( const onnx::ValueInfoProto& onnxInput : onnxGraph.input() ) {
		if( initializers.Has( onnxInput.name().c_str() ) ) {
			// Networks from PyTorch can have separate inputs for every initializer (every weight/filter etc.)
			// In case of NeoML inputs like these won't be needed (all of weights must be calculated from initializers)
			continue;
		}
		graph.Add( new CGraphInput( graph.NodeCount(), onnxInput ) );
		nodeOutputs.Add( onnxInput.name().c_str(), CLink( graph.NodeCount() - 1, 0 ) );
	}

	const int firstOpNodeIndex = graph.NodeCount();

	// Add onnx graph's nodes
	for( const onnx::NodeProto& onnxNode : onnxGraph.node() ) {
		graph.Add( COpNode::CreateOpNode( graph.NodeCount(), onnxNode, opsetVersion ) );

		// Add this onnxNode's outputs to the map of onnxNode outputs
		for( int outputIndex = 0; outputIndex < onnxNode.output_size(); ++outputIndex ) {
			nodeOutputs.Add( onnxNode.output( outputIndex ).c_str(), CLink( graph.NodeCount() - 1, outputIndex ) );
		}
	}

	// Connect nodes
	for( int opNodeIndex = 0; opNodeIndex < onnxGraph.node_size(); ++opNodeIndex ) {
		const onnx::NodeProto& onnxNode = onnxGraph.node( opNodeIndex );
		for( int inputIndex = 0; inputIndex < onnxNode.input_size(); ++inputIndex ) {
			const std::string& inputName = onnxNode.input( inputIndex );
			if( inputName.size() > 0 ) {
				graph[firstOpNodeIndex + opNodeIndex]->Connect( inputIndex, nodeOutputs.Get( inputName.data() ) );
			}
		}
	}

	// Add graph outputs
	for( const onnx::ValueInfoProto& onnxOutput : onnxGraph.output() ) {
		graph.Add( new CGraphOutput( graph.NodeCount(), onnxOutput ) );
		graph[graph.NodeCount() - 1]->Connect( 0, nodeOutputs.Get( onnxOutput.name().c_str() ) );
	}
}

// Calculates tensor shape and (if possible) data for every node output
static void calcGraphTensors( CGraph& graph, IMathEngine& mathEngine, CTensorCache& tensors )
{
	// Iterate over graph in top sorted order
	for( int nodeIndex = 0; nodeIndex < graph.NodeCount(); ++nodeIndex ) {
		graph[nodeIndex]->CalcOutputTensors( tensors, mathEngine );
	}
}

// Labels tensor dimensions (which are unnamed) with NeoML blob dimensions
static void labelTensorsDimensions( CGraph& graph, const CTensorCache& tensors, CDimCache& dims )
{
	for( int nodeIndex = 0; nodeIndex < graph.NodeCount(); ++nodeIndex ) {
		graph[nodeIndex]->LabelTensorDims( tensors, dims );
	}

	// Sometimes there are additional operators between graph inputs and
	// nodes, whose operations can interpret tensor deimensions
	// E.g. input -> transpose -> conv
	// In that case input's dims will be still unmarked
	// That's why we call labeling method one more time in reversed order
	for( int nodeIndex = graph.NodeCount() - 1; nodeIndex >= 0; --nodeIndex ) {
		graph[nodeIndex]->LabelTensorDims( tensors, dims );
	}
}

// Adds layers to dnn based on graph, tensors and marked dimensions
static void addLayersToDnn( CGraph& graph, const CTensorCache& tensors, const CDimCache& dims, CDnn& dnn )
{
	CNeoMLLinkCache neoMLLinks( graph );
	for( int nodeIndex = 0; nodeIndex < graph.NodeCount(); ++nodeIndex ) {
		graph[nodeIndex]->AddLayers( graph, tensors, dims, neoMLLinks, dnn );
	}
}

// Builds dnn based on GraphProto
static void buildDnnFromGraphProto( const onnx::GraphProto& onnxGraph, int opsetVersion, CDnn& dnn )
{
	CheckOnnxProtocol( opsetVersion > 0, "Wrong onnx version: " + Str( opsetVersion ) );
	CheckNeoOnnxSupport( opsetVersion <= MaxOpsetVersion, "Unsupported opset version: " + Str( opsetVersion ) );
	CheckNeoOnnxInternal( dnn.GetLayerCount() == 0, "dnn must be empty" );

	// Step 1: create graph nodes and connect them
	CGraph graph;
	buildGraph( onnxGraph, opsetVersion, graph );

	// Step 2: calculate tensor shape and data for every node output in graph
	CTensorCache tensors( graph );
	calcGraphTensors( graph, dnn.GetMathEngine(), tensors );

	// Step 3: label onnx tensors dimensions with NeoML blob dimensions
	CDimCache dims( graph );
	labelTensorsDimensions( graph, tensors, dims );

	// Step 4: add layers to dnn
	addLayersToDnn( graph, tensors, dims, dnn );
}

void LoadFromOnnx( const char* fileName, CDnn& dnn )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	onnx::ModelProto model;
	
	std::ifstream input(fileName, std::ios::binary);
	if( !input ) {
		NeoOnnxCheck( false, CString( "Failed to open file " ) + fileName );
	}

	try {
		if( !model.ParseFromIstream( &input ) ) {
			NeoOnnxCheck( false, CString( "Failed to parse model from file " ) + fileName );
		}

		buildDnnFromGraphProto( model.graph(), getOpsetVersion( model ), dnn );
	} catch( ... ) {
		input.close();
		google::protobuf::ShutdownProtobufLibrary();
		throw;
	}

	input.close();
	google::protobuf::ShutdownProtobufLibrary();
}

void LoadFromOnnx( const void* buffer, int bufferSize, CDnn& dnn )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	onnx::ModelProto model;

	std::string strBuffer( static_cast<const char*>( buffer ), bufferSize );

	try {
		if( !model.ParseFromString( strBuffer ) ) {
			NeoOnnxCheck( false, "Failed to parse model from buffer" );
		}

		buildDnnFromGraphProto( model.graph(), getOpsetVersion( model ), dnn );
	} catch( ... ) {
		google::protobuf::ShutdownProtobufLibrary();
		throw;
	}

	google::protobuf::ShutdownProtobufLibrary();
}

}
