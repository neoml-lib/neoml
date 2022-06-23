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

py::tuple wrapResults( const CArray<const char*>& cArrInputs, const CArray<NeoOnnx::COutputInfo>& cArrOutputs,
	const CMap<CString, CString>& cMapMetadata )
{
	py::list inputs;
	for( int i = 0; i < cArrInputs.Size(); ++i ) {
		inputs.append( py::str( cArrInputs[i] ) );
	}
	py::list outputs;
	for( int i = 0; i < cArrOutputs.Size(); ++i ) {
		py::tuple currOutputInfo( 2 );
		currOutputInfo[0] = py::str( static_cast<const char*>( cArrOutputs[i].Name ) );
		currOutputInfo[1] = py::int_( cArrOutputs[i].DimCount );
		outputs.append( currOutputInfo );
	}
	py::dict metadata;
	for( int pos = cMapMetadata.GetFirstPosition(); pos != NotFound; pos = cMapMetadata.GetNextPosition( pos ) ) {
		metadata[py::str(static_cast<const char*>(cMapMetadata.GetKey(pos)))] = py::str(static_cast<const char*>(cMapMetadata.GetKey(pos)));
	}
	py::tuple result( 3 );
	result[0] = inputs;
	result[1] = outputs;
	result[2] = metadata;
	return result;
}

py::tuple loadFromFile(const std::string& fileName, CPyDnn& pyDnn)
{
	CArray<const char*> cArrInputs;
	CArray<NeoOnnx::COutputInfo> cArrOutputs;
	CMap<CString, CString> cMapMetadata;
	{
		py::gil_scoped_release release;
		NeoOnnx::LoadFromOnnx( fileName.data(), pyDnn.Dnn(), cArrInputs, cArrOutputs, cMapMetadata );
	}
	return wrapResults( cArrInputs, cArrOutputs, cMapMetadata );
}

py::tuple loadFromBuffer(const std::string& buffer, CPyDnn& pyDnn)
{
	CArray<const char*> cArrInputs;
	CArray<NeoOnnx::COutputInfo> cArrOutputs;
	CMap<CString, CString> cMapMetadata;
	{
		py::gil_scoped_release release;
		NeoOnnx::LoadFromOnnx( buffer.data(), buffer.size(), pyDnn.Dnn(), cArrInputs, cArrOutputs, cMapMetadata );
	}
	return wrapResults( cArrInputs, cArrOutputs, cMapMetadata );
}

void InitializeOnnx(py::module& m)
{
	m.def("load_onnx_from_file", &loadFromFile);
	m.def("load_onnx_from_buffer", &loadFromBuffer);
}
