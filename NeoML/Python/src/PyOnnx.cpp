/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <common.h>
#pragma hdrstop

#include "PyOnnx.h"
#include "PyDnn.h"

#include <NeoOnnx/NeoOnnx.h>

py::object wrapResults( const NeoOnnx::CImportedModelInfo& cInfo )
{
	py::object pyModule = py::module::import( "neoml.Onnx" );
	py::object pyInfoConstructor = pyModule.attr( "ImportedModelInfo" );
	py::object pyInputConstructor = pyModule.attr( "ImportedModelInputInfo" );
	py::object pyOutputConstructor = pyModule.attr( "ImportedModelOutputInfo" );

	py::object info = pyInfoConstructor();
	py::list inputs = info.attr( "inputs" );
	for( int i = 0; i < cInfo.Inputs.Size(); ++i ) {
		inputs.append( pyInputConstructor( py::str( cInfo.Inputs[i].Name ) ) );
	}
	py::list outputs = info.attr( "outputs" );
	for( int i = 0; i < cInfo.Outputs.Size(); ++i ) {
		outputs.append( pyOutputConstructor( py::str( cInfo.Outputs[i].Name ), py::int_( cInfo.Outputs[i].DimCount ) ) );
	}
	py::dict metadata = info.attr( "metadata" );
	const CMap<CString, CString>& cMapMetadata = cInfo.Metadata;
	for( int pos = cMapMetadata.GetFirstPosition(); pos != NotFound; pos = cMapMetadata.GetNextPosition( pos ) ) {
		metadata[py::str( cMapMetadata.GetKey( pos ) )] = py::str( cMapMetadata.GetKey( pos ) );
	}
	return info;
}

py::object loadFromFile(const std::string& fileName, CPyDnn& pyDnn)
{
	NeoOnnx::CImportedModelInfo modelInfo;
	{
		py::gil_scoped_release release;
		NeoOnnx::CImportSettings settings;
		NeoOnnx::LoadFromOnnx( fileName.data(), settings, pyDnn.Dnn(), modelInfo );
	}
	return wrapResults( modelInfo );
}

py::object loadFromBuffer(const std::string& buffer, CPyDnn& pyDnn)
{
	NeoOnnx::CImportedModelInfo modelInfo;
	{
		py::gil_scoped_release release;
		NeoOnnx::CImportSettings settings;
		NeoOnnx::LoadFromOnnx( buffer.data(), buffer.size(), settings, pyDnn.Dnn(), modelInfo );
	}
	return wrapResults( modelInfo );
}

void InitializeOnnx(py::module& m)
{
	m.def("load_onnx_from_file", &loadFromFile);
	m.def("load_onnx_from_buffer", &loadFromBuffer);
}
