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
#pragma hdrstop

#include <fstream>
#include <iostream>

#include <NeoOnnx/NeoOnnxImport.h>

#include "onnx.pb.h"

#include "NeoOnnxCheck.h"
#include "Operator.h"
#include "GraphInitializer.h"
#include "GraphInput.h"
#include "GraphOutput.h"
#include "Optimization/DnnOptimizer.h"

namespace NeoOnnx {

// Checks if all of the operators are supported by NeoOnnx
// Throws an exception if some of the operators are not supoorted
static void checkOperatorSupport( const onnx::GraphProto& onnxGraph )
{
	CHashTable<CString> notSupportedOps;
	for( const onnx::NodeProto& onnxNode : onnxGraph.node() ) {
		const CString opType( onnxNode.op_type().data() );
		if( !COperator::IsSupportedOperator( opType ) && !notSupportedOps.Has( opType ) ) {
			notSupportedOps.Add( opType );
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

// Gets opset version from the ModelProto
static int getOpsetVersion( const onnx::ModelProto& model )
{
	for( const auto& opset : model.opset_import() ) {
		if( opset.domain().empty() ) {
			// Default onnx opset
			return static_cast<int>( opset.version() );
		}
	}

	CheckOnnxProtocol( false, "Default operator set is missing" );

	return -1;
}

// Tensor cache used to store all named tensors during graph building
typedef CMap<CString, CPtr<const CTensorBase>> CTensorCache;

// Processes given operator
// The operator processing includes the following steps:
// - Get operator's input tensors from the tensor cache
// - Acquire operator's output tensors and add required layers to the dnn
// - Add output tensors to the tensor cache
static void processOperator( const COperator& op, CTensorCache& tensors, CDnn& dnn )
{
	CTensorArray inputs;
	inputs.Add( nullptr, op.InputCount() );
	for( int inputIndex = 0; inputIndex < op.InputCount(); ++inputIndex ) {
		const CString& inputName = op.InputName( inputIndex );
		// In onnx operator may have 3 inputs (e.g. Conv where 1 and 2 inputs are required and 3 is optional)
		// In that case operator may have 3 inputs where the third input's name is empty
		if( inputName != "" ) {
			CheckOnnxProtocol( tensors.Has( inputName ), "Unknown input: " + inputName );
			inputs[inputIndex] = tensors[inputName];
		}
	}

	CTensorArray outputs;
	outputs.SetBufferSize( op.OutputCount() );
	op.ProcessTensors( inputs, dnn, outputs );
	NeoAssert( outputs.Size() == op.OutputCount() );

	for( int outputIndex = 0; outputIndex < op.OutputCount(); ++outputIndex ) {
		const CString& outputName = op.OutputName( outputIndex );
		CheckOnnxProtocol( !tensors.Has( outputName ), "Output already exist: " + outputName );
		tensors.Add( outputName, outputs[outputIndex] );
	}
}

// Builds dnn based on GraphProto
static void buildDnnFromGraphProto( const onnx::GraphProto& onnxGraph, int opsetVersion, const CImportSettings& settings,
	CDnn& dnn, CArray<CImportedModelInfo::CInputInfo>& inputs, CArray<CImportedModelInfo::COutputInfo>& outputs)
{
	CheckOnnxProtocol( opsetVersion > 0, "Wrong onnx version: " + Str( opsetVersion ) );
	CheckNeoOnnxSupport( opsetVersion <= MaxOpsetVersion, "Unsupported opset version: " + Str( opsetVersion ) );

	// Check if every operator is supported by NeOnnx
	checkOperatorSupport( onnxGraph );

	dnn.DeleteAllLayers();
	CTensorCache tensors;

	// Add graph initializers
	CHashTable<CString> initializers;
	for( const onnx::TensorProto& onnxInitializer : onnxGraph.initializer() ) {
		CGraphInitializer initializer( onnxInitializer );
		tensors.Add( initializer.Name(), initializer.GetDataTensor( dnn.GetMathEngine() ).Ptr() );
		initializers.Add( initializer.Name() );
	}

	// Add graph inputs
	for( const onnx::ValueInfoProto& onnxInput : onnxGraph.input() ) {
		const CString inputName = onnxInput.name().c_str();
		if( initializers.Has( inputName ) ) {
			// Networks from PyTorch can have separate inputs for every initializer (every weight/filter etc.)
			// In case of NeoML inputs like these won't be needed because all the weights must be calculated from initializers
			continue;
		}
		CGraphInput graphInput( onnxInput );
		CPtr<const CUserTensor> inputTensor;
		if( settings.InputLayouts.Has( inputName ) ) {
			inputTensor = graphInput.AddSourceLayer( dnn, &settings.InputLayouts[inputName] ).Ptr();
		} else {
			inputTensor = graphInput.AddSourceLayer( dnn, nullptr ).Ptr();
		}
		tensors.Add( graphInput.Name(), inputTensor.Ptr() );
		CImportedModelInfo::CInputInfo& inputInfo = inputs.Append();
		inputInfo.Name = CString( inputTensor->Layer()->GetName() );
	}

	// Add graph operators
	for( const onnx::NodeProto& onnxNode : onnxGraph.node() ) {
		std::unique_ptr<COperator> op( COperator::CreateOperator( onnxNode, opsetVersion ) );
		processOperator( *op, tensors, dnn );
	}

	// Add graph outputs
	for( const onnx::ValueInfoProto& onnxOutput : onnxGraph.output() ) {
		CGraphOutput output( onnxOutput );
		CheckOnnxProtocol( tensors.Has( output.Name() ), "output tensor is missing" );
		CheckNeoOnnxSupport( tensors[output.Name()] != nullptr, "output tensor can't be calculated" );

		// API obliges us to return output tensors in layout from settings
		// or (if settings don't have layout) in the IOLayout
		const CTensorBase& currTensor = *tensors[output.Name()];
		const int layoutPos = settings.OutputLayouts.GetFirstPosition( output.Name() );
		const CTensorLayout outputLayout = layoutPos == NotFound ? CTensorLayout::IOLayout( currTensor.DimCount() )
			: settings.OutputLayouts.GetValue( layoutPos );
		CPtr<const CTensorBase> convertedTensor = ConvertTensor( currTensor, outputLayout );

		// API obliges us to return CDnn outputs in the corresponding CSinkLayer
		// Which is why convert any tensor to CUserTensor and connect it to CSinkLayer
		CPtr<const CUserTensor> userTensor = AsUserTensor( *convertedTensor, output.Name() + "_UserSource", dnn );
		CPtr<const CSinkLayer> sink;
		if( settings.OutputLayouts.Has( output.Name() ) ) {
			sink = output.AddSinkLayer( *userTensor, dnn );
		} else {
			sink = output.AddSinkLayer( *userTensor, dnn );
		}

		CImportedModelInfo::COutputInfo& outputInfo = outputs.Append();
		outputInfo.Name = CString( sink->GetName() );
	}
}

// Extracts metadata from the model
static void extractMetadata( const onnx::ModelProto& model, CMap<CString, CString>& metadata )
{
	metadata.Empty();
	for( const onnx::StringStringEntryProto& entry : model.metadata_props() ) {
		metadata.Add( CString( entry.key().data() ), CString( entry.value().data() ) );
	}
}

void LoadFromOnnx( const char* fileName, const CImportSettings& importSettings,
	CDnn& dnn, CImportedModelInfo& info )
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
		buildDnnFromGraphProto( model.graph(), getOpsetVersion( model ), importSettings,
			dnn, info.Inputs, info.Outputs );
		extractMetadata( model, info.Metadata );
		NeoOnnx::optimization::CDnnOptimizer( dnn ).Optimize( info.OnnxOptimizationReport );
		info.OptimizationReport = NeoML::OptimizeDnn( dnn, importSettings.DnnOptimizationSettings );
	} catch( ... ) {
		input.close();
		google::protobuf::ShutdownProtobufLibrary();
		throw;
	}

	input.close();
	google::protobuf::ShutdownProtobufLibrary();
}

void LoadFromOnnx( const void* buffer, int bufferSize, const CImportSettings& importSettings,
	CDnn& dnn, CImportedModelInfo& info )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	onnx::ModelProto model;

	std::string strBuffer( static_cast<const char*>( buffer ), bufferSize );

	try {
		if( !model.ParseFromString( strBuffer ) ) {
			NeoOnnxCheck( false, "Failed to parse model from buffer" );
		}
		buildDnnFromGraphProto( model.graph(), getOpsetVersion( model ), importSettings,
			dnn, info.Inputs, info.Outputs );
		extractMetadata( model, info.Metadata );
		NeoOnnx::optimization::CDnnOptimizer( dnn ).Optimize( info.OnnxOptimizationReport );
		info.OptimizationReport = NeoML::OptimizeDnn( dnn, importSettings.DnnOptimizationSettings );
	} catch( ... ) {
		google::protobuf::ShutdownProtobufLibrary();
		throw;
	}

	google::protobuf::ShutdownProtobufLibrary();
}

} //NeoOnnx

