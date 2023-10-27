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

#include <common.h>
#pragma hdrstop

#include "PyOnnx.h"
#include "PyDnn.h"

#include <NeoOnnx/NeoOnnx.h>

static py::dict wrapResults( const NeoOnnx::CImportedModelInfo& cInfo )
{
	py::list inputList;
	for( const NeoOnnx::CImportedModelInfo::CInputInfo& inputInfo : cInfo.Inputs ) {
		inputList.append( py::str( inputInfo.Name ) );
	}
	py::list outputList;
	for( const NeoOnnx::CImportedModelInfo::COutputInfo& outputInfo : cInfo.Outputs ) {
		outputList.append( py::str( outputInfo.Name ) );
	}
	py::dict metadata;
	const CMap<CString, CString>& cMapMetadata = cInfo.Metadata;
	for( int pos = cMapMetadata.GetFirstPosition(); pos != NotFound; pos = cMapMetadata.GetNextPosition( pos ) ) {
		metadata[py::str( cMapMetadata.GetKey( pos ) )] = py::str( cMapMetadata.GetKey( pos ) );
	}

	py::dict result;
	result[py::str( "inputs" )] = inputList;
	result[py::str( "outputs" )] = outputList;
	result[py::str( "metadata" )] = metadata;
	return result;
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
