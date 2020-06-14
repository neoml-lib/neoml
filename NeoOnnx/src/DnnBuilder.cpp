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

// Checks if ONNX graph nodes are already sorted in topological order
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

// Builds the array of CNode's based on onnxGraph
static void buildNodes( const onnx::GraphProto& onnxGraph, IMathEngine& mathEngine, CPointerArray<CNode>& nodes )
{
	nodes.Empty();
	nodes.SetBufferSize( onnxGraph.input_size() + onnxGraph.initializer_size() + onnxGraph.node_size()
		+ onnxGraph.output_size() );
	CMap<CString, CNode::CInputInfo> nodeOutputs;

	// Add graph initializers.
	CHashTable<CString> initializers;
	for( const onnx::TensorProto& onnxInitializer : onnxGraph.initializer() ) {
		if( onnxInitializer.dims_size() > 0 ) {
			nodes.Add( new CGraphInitializer( onnxInitializer, nodeOutputs, mathEngine ) );
			initializers.Add( onnxInitializer.name().c_str() );
		}
	}

	// Add graph inputs.
	for( const onnx::ValueInfoProto& onnxInput : onnxGraph.input() ) {
		if( initializers.Has( onnxInput.name().c_str() ) ) {
			// Networks from PyTorch can have separate inputs for every initializer (every weight/filter etc.)
			// In case of NeoML these inputs won't be needed (all weights must be calculated from initializers)
			continue;
		}
		nodes.Add( new CGraphInput( onnxInput, nodeOutputs ) );
	}

	// Add graph nodes.
	for( const onnx::NodeProto& onnxNode : onnxGraph.node() ) {
		nodes.Add( CNode::CreateNode( onnxNode, nodeOutputs, mathEngine ) );
	}

	// Add graph outputs.
	for( const onnx::ValueInfoProto& onnxOutput : onnxGraph.output() ) {
		nodes.Add( new CGraphOutput( onnxOutput, nodeOutputs ) );
	}
}

void CDnnBuilder::BuildDnn( const onnx::GraphProto& onnxGraph, CDnn& dnn )
{
	CheckNeoOnnxInternal( dnn.GetLayerCount() == 0, "dnn must be empty" );
	CheckNeoOnnxSupport( isTopSorted( onnxGraph ), "onnxGraph is not top sorted" );

	CPointerArray<CNode> nodes;
	buildNodes( onnxGraph, dnn.GetMathEngine(), nodes );

	// Iterate over the graph in top sorted order
	for( int nodeIndex = 0; nodeIndex < nodes.Size(); ++nodeIndex ) {
		// Calculate output tensors size and (if possible) value.
		nodes[nodeIndex]->OnnxReshape();
		// Matching ONNX tensors dimensions with NeoML blob dimensions.
		nodes[nodeIndex]->MarkTensorDims();
	}

	// Sometimes there are additional operations between graph inputs and
	// nodes that can interpret tensor dimensions.
	// E.g. input -> transpose -> conv.
	// In that case input's dimensions will be still unmarked.
	// That's why we call the marking method again in reverse order.
	for( int nodeIndex = nodes.Size() - 1; nodeIndex >= 0; --nodeIndex ) {
		nodes[nodeIndex]->MarkTensorDims();
	}

	// Adding layers to dnn.
	for( int nodeIndex = 0; nodeIndex < nodes.Size(); ++nodeIndex ) {
		nodes[nodeIndex]->AddLayers( dnn );
	}
}

} // namespace NeoOnnx