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

#include "DnnBuilder.h"
#include "NeoOnnxCheck.h"

#include <onnx.pb.h>

#include <fstream>
#include <iostream>

namespace NeoOnnx {

void LoadFromOnnx( const char* fileName, CDnn& dnn )
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	onnx::ModelProto model;
	
	std::ifstream input(fileName, std::ios::binary);
	if( !input ) {
		NeoOnnxCheck( false, CString( "Failed to open file " ) + fileName );
	}

	if( !model.ParseFromIstream( &input ) ) {
		input.close();
		NeoOnnxCheck( false, CString( "Failed to parse model from file " ) + fileName );
	}

	try {
		NeoOnnx::CDnnBuilder dnnBuilder;
		dnnBuilder.BuildDnn( model.graph(), dnn );
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
	
	if( !model.ParseFromString( strBuffer ) ) {
		NeoOnnxCheck( false, "Failed to parse model from buffer" );
	}

	try {
		NeoOnnx::CDnnBuilder dnnBuilder;
		dnnBuilder.BuildDnn( model.graph(), dnn );
	} catch( ... ) {
		google::protobuf::ShutdownProtobufLibrary();
		throw;
	}

	google::protobuf::ShutdownProtobufLibrary();
}

}
