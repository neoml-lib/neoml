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

#include "DnnBuilder.h"

#include "Node.h"
#include "Nodes/GraphInitializer.h"
#include "Nodes/GraphInput.h"
#include "Nodes/GraphOutput.h"
#include "NeoOnnxCheck.h"

namespace NeoOnnx {

// Checks if onnx graph's nodes are already in topological sorted order.
static bool isTopSorted( const onnx::GraphProto& onnxGraph )
{
	std::unordered_set<std::string> visited;

	for( const onnx::ValueInfoProto& input : onnxGraph.input() ) {
		visited.insert( input.name() );
	}

	for( const onnx::TensorProto& initializer : onnxGraph.initializer() ) {
		visited.insert( initializer.name() );
	}

	for( const onnx::NodeProto& node : onnxGraph.node() ) {
		for( const std::string& nodeInput : node.input() ) {
			if( nodeInput.size() > 0 && visited.find( nodeInput ) == visited.end() ) {
				return false;
			}
		}

		for( const std::string& nodeOutput : node.output() ) {
			visited.insert( nodeOutput );
		}
	}

	return true;
}

// Build array of CNode's based on onnxGraph.
static void buildGraph( const onnx::GraphProto& onnxGraph, int opsetVersion, IMathEngine& mathEngine, CGraph& graph )
{
	graph.Empty();
	graph.SetBufferSize( onnxGraph.input_size() + onnxGraph.initializer_size() + onnxGraph.node_size()
		+ onnxGraph.output_size() );
	CMap<CString, CNode::CInputInfo> nodeOutputs;

	// Add graph initializers.
	CHashTable<CString> initializers;
	for( const onnx::TensorProto& onnxInitializer : onnxGraph.initializer() ) {
		if( onnxInitializer.dims_size() > 0 ) {
			graph.Add( new CGraphInitializer( graph.Size(), onnxInitializer, mathEngine ) );
			nodeOutputs.Add( onnxInitializer.name().c_str(), CNode::CInputInfo( graph.Size() - 1, 0 ) );
			initializers.Add( onnxInitializer.name().c_str() );
		}
	}

	// Add graph inputs.
	for( const onnx::ValueInfoProto& onnxInput : onnxGraph.input() ) {
		if( initializers.Has( onnxInput.name().c_str() ) ) {
			// Networks from PyTorch can have separate inputs for every initializer (every weight/filter etc.).
			// In case of NeoML inputs like these won't be needed (all of weights must be calculated from initializers).
			continue;
		}
		graph.Add( new CGraphInput( graph.Size(), onnxInput ) );
		nodeOutputs.Add( onnxInput.name().c_str(), CNode::CInputInfo( graph.Size() - 1, 0 ) );
	}

	// Add graph graph.
	for( const onnx::NodeProto& onnxNode : onnxGraph.node() ) {
		graph.Add( COpNode::CreateOpNode( graph.Size(), onnxNode, opsetVersion ) );
		for( int inputIndex = 0; inputIndex < onnxNode.input_size(); ++inputIndex ) {
			const std::string& inputName = onnxNode.input( inputIndex );
			if( inputName.size() > 0 ) {
				graph.Last()->SetInput( inputIndex, nodeOutputs.Get( inputName.data() ) );
			}
		}

		// Adding this onnxNode's outputs to the map of onnxNode outputs.
		for( int outputIndex = 0; outputIndex < onnxNode.output_size(); ++outputIndex ) {
			nodeOutputs.Add( onnxNode.output( outputIndex ).c_str(), CNode::CInputInfo( graph.Size() - 1, outputIndex ) );
		}
	}

	// Add graph outputs.
	for( const onnx::ValueInfoProto& onnxOutput : onnxGraph.output() ) {
		graph.Add( new CGraphOutput( graph.Size(), onnxOutput ) );
		graph.Last()->SetInput( 0, nodeOutputs.Get( onnxOutput.name().c_str() ) );
	}
}

void CDnnBuilder::BuildDnn( const onnx::GraphProto& onnxGraph, int opsetVersion, CDnn& dnn )
{
	CheckOnnxProtocol( opsetVersion > 0, "Wrong onnx version: " + Str( opsetVersion ) );
	CheckNeoOnnxSupport( opsetVersion <= MaxOpsetVersion, "Unsupported opset version: " + Str( opsetVersion ) );

	CheckNeoOnnxInternal( dnn.GetLayerCount() == 0, "dnn must be empty" );
	CheckNeoOnnxSupport( isTopSorted( onnxGraph ), "onnxGraph is not topologically sorted" );

	// Step 1: creating nodes of the graph and connections between them.
	CGraph graph;
	buildGraph( onnxGraph, opsetVersion, dnn.GetMathEngine(), graph );

	// Iterate over graph in top sorted order.
	CGraphTensors tensors;
	tensors.SetSize( graph.Size() );
	for( int nodeIndex = 0; nodeIndex < graph.Size(); ++nodeIndex ) {
		// Step 2: Calculate output tensors' shapes and data.
		tensors[nodeIndex].SetSize( graph[nodeIndex]->OutputCount() );
		graph[nodeIndex]->CalcOutputTensors( tensors, dnn.GetMathEngine() );
	}

	// Step 3: Mark onnx tensors' dimensions with NeoML blob dimensions.
	CGraphDims dims;
	dims.SetSize( graph.Size() );
	for( int nodeIndex = 0; nodeIndex < graph.Size(); ++nodeIndex ) {
		// Matching onnx tensors' dimensions with NeoML blob dimensions.
		dims[nodeIndex].SetSize( graph[nodeIndex]->OutputCount() );
		graph[nodeIndex]->MarkTensorDims( tensors, dims );
	}

	// Still step 3.
	// Sometimes there are additional operations between graph inputs and
	// nodes, whose operations can interpret tensor deimensions.
	// E.g. input -> transpose -> conv.
	// In that case input's dims will be still unmarked.
	// That's why we call marking method one more time in reversed order.
	for( int nodeIndex = graph.Size() - 1; nodeIndex >= 0; --nodeIndex ) {
		graph[nodeIndex]->MarkTensorDims( tensors, dims );
	}

	// Step 4: Adding layers to dnn.
	CGraphMappings mappings;
	mappings.SetSize( graph.Size() );
	for( int nodeIndex = 0; nodeIndex < graph.Size(); ++nodeIndex ) {
		mappings[nodeIndex].SetSize( graph[nodeIndex]->OutputCount() );
		graph[nodeIndex]->AddLayers( graph, tensors, dims, mappings, dnn );
	}
}

} // namespace NeoOnnx
