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

#include <NeoOnnx/NeoOnnxImport.h>

#include "NeoOnnxCheck.h"
#include "Node.h"
#include "Nodes/GraphInitializer.h"
#include "Nodes/GraphInput.h"
#include "Nodes/GraphOutput.h"

#include "onnx.pb.h"

#include <fstream>
#include <iostream>

namespace NeoOnnx {

// Checks if all the operators are supported by NeoOnnx
// Throws exception if some op operators are not supoorted
static void checkOperatorSupport( const onnx::GraphProto& onnxGraph )
{
	CHashTable<CString> notSupportedOps;
	for( const onnx::NodeProto& onnxNode : onnxGraph.node() ) {
		if( !COpNode::IsSupportedOperator( onnxNode.op_type() ) && !notSupportedOps.Has( onnxNode.op_type() ) ) {
			notSupportedOps.Add( onnxNode.op_type() );
		}
	}

	if( !notSupportedOps.IsEmpty() ) {
		CString message = "Operators:";
		for( int i = notSupportedOps.GetFirstPosition(); i != NotFound; i = notSupportedOps.GetNextPosition( i ) ) {
			message += CString( "\n\t" ) + notSupportedOps.GetValue( i );
		}
		CheckNeoOnnxSupport( false, message );
	}
}

// Gets opset version from ModelProto
static int getOpsetVersion( const onnx::ModelProto& model )
{
	for( const auto& opset : model.opset_import() ) {
		if( opset.domain().empty() ) { // Default onnx opset
			return static_cast<int>( opset.version() );
		}
	}

	CheckOnnxProtocol( false, "Default operator set is missing" );

	return -1;
}

typedef CMap<CString, CPtr<const CTensorBase>> CTensorCache;

static void addNode( CNode& node, CTensorCache& tensors, CDnn& dnn )
{
	CObjectArray<const CTensorBase> inputs;
	inputs.Add( nullptr, node.InputCount() );
	for( int inputIndex = 0; inputIndex < node.InputCount(); ++inputIndex ) {
		const CString& inputName = node.InputName( inputIndex );
		if( inputName != "" ) {
			CheckOnnxProtocol( tensors.Has( inputName ), "Unknown input: " + inputName );
			inputs[inputIndex] = tensors[inputName];
		}
	}

	CObjectArray<const CTensorBase> outputs;
	outputs.Add( nullptr, node.OutputCount() );

	if( node.CanCalculateOutput( inputs ) ) {
		node.CalculateOutput( inputs, dnn.GetMathEngine(), outputs );
	} else {
		node.AddLayers( inputs, outputs, dnn );
	}

	for( int outputIndex = 0; outputIndex < node.OutputCount(); ++outputIndex ) {
		const CString& outputName = node.OutputName( outputIndex );
		CheckOnnxProtocol( !tensors.Has( outputName ), "Output already exist: " + outputName );
		tensors.Add( outputName, outputs[outputIndex] );
	}
}

// Builds dnn based on GraphProto
static void buildDnnFromGraphProto( const onnx::GraphProto& onnxGraph, int opsetVersion, CDnn& dnn, CArray<const char*>& inputs, CArray<const char*>& outputs )
{
	CheckOnnxProtocol( opsetVersion > 0, "Wrong onnx version: " + Str( opsetVersion ) );
	CheckNeoOnnxSupport( opsetVersion <= MaxOpsetVersion, "Unsupported opset version: " + Str( opsetVersion ) );

	// Prepare: check if every operator is supported by NeOnnx
	checkOperatorSupport( onnxGraph );

	dnn.DeleteAllLayers();
	CTensorCache tensors;

	// Add graph initializers
	CHashTable<CString> initializers;
	for( const onnx::TensorProto& onnxInitializer : onnxGraph.initializer() ) {
		CGraphInitializer initializer( onnxInitializer );
		addNode( initializer, tensors, dnn );
		initializers.Add( onnxInitializer.name().c_str() );
	}

	// Add graph inputs
	for( const onnx::ValueInfoProto& onnxInput : onnxGraph.input() ) {
		if( initializers.Has( onnxInput.name().c_str() ) ) {
			// Networks from PyTorch can have separate inputs for every initializer (every weight/filter etc.)
			// In case of NeoML inputs like these won't be needed because all the weights must be calculated from initializers
			continue;
		}
		CGraphInput input( onnxInput );
		addNode( input, tensors, dnn );
		inputs.Add( dnn.GetLayer( onnxInput.name().c_str() )->GetName() );
	}

	// Add onnx graph's nodes and connect them
	for( const onnx::NodeProto& onnxNode : onnxGraph.node() ) {
		std::unique_ptr<CNode> node( COpNode::CreateOpNode( onnxNode, opsetVersion ) );
		addNode( *node, tensors, dnn );
	}

	// Add graph outputs
	for( const onnx::ValueInfoProto& onnxOutput : onnxGraph.output() ) {
		CGraphOutput output( onnxOutput );
		addNode( output, tensors, dnn );
		outputs.Add( dnn.GetLayer( onnxOutput.name().c_str() )->GetName() );
	}
}

void LoadFromOnnx( const char* fileName, CDnn& dnn, CArray<const char*>& inputs, CArray<const char*>& outputs )
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

		buildDnnFromGraphProto( model.graph(), getOpsetVersion( model ), dnn, inputs, outputs );
	} catch( ... ) {
		input.close();
		google::protobuf::ShutdownProtobufLibrary();
		throw;
	}

	input.close();
	google::protobuf::ShutdownProtobufLibrary();
}

void LoadFromOnnx( const void* buffer, int bufferSize, CDnn& dnn, CArray<const char*>& inputs, CArray<const char*>& outputs )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	onnx::ModelProto model;

	std::string strBuffer( static_cast<const char*>( buffer ), bufferSize );

	try {
		if( !model.ParseFromString( strBuffer ) ) {
			NeoOnnxCheck( false, "Failed to parse model from buffer" );
		}

		buildDnnFromGraphProto( model.graph(), getOpsetVersion( model ), dnn, inputs, outputs );
	} catch( ... ) {
		google::protobuf::ShutdownProtobufLibrary();
		throw;
	}

	google::protobuf::ShutdownProtobufLibrary();
}

}
