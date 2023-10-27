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

#include <iostream>

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

static void fillLayouts(py::object pyLayouts, CMap<CString, NeoOnnx::CTensorLayout>& layouts)
{
	if( pyLayouts == Py_None ) {
		return;
	}

	static const CArray<CString> dims = { CString( "batch_length" ), CString( "batch_width" ),
		CString( "list_size" ), CString( "height" ), CString( "width" ), CString( "depth" ), CString( "channels" ) };
	for( const CString& dim : dims ) {
		std::cerr << static_cast<std::string>( dim ) << ' ';
	}
	std::cerr << '\n';

	py::dict layoutDict = pyLayouts.cast<py::dict>();
	for( const auto& it : layoutDict ) {
		const CString name( std::string( it.first.cast<py::str>() ) );
		NeoOnnx::CTensorLayout& layout = layouts.GetOrCreateValue( name );
		py::list pyLayout = it.second.cast<py::list>();
		std::cerr << "Name: : " << name << "\tLayout: ";
		for( const auto& dimStr : pyLayout ) {
			std::cerr << "'" << std::string( dimStr.cast<py::str>() ) << "'";
			layout.Add( static_cast<TBlobDim>( dims.Find( CString( dimStr.cast<py::str>() ) ) ) );
			std::cerr << static_cast<int>( layout.Last() ) << ' ';
		}
		std::cerr << "\n";
	}
}

py::object loadFromFile(const std::string& fileName, CPyDnn& pyDnn, py::object inputLayouts, py::object outputLayouts)
{
	NeoOnnx::CImportedModelInfo modelInfo;
	{
		py::gil_scoped_release release;
		NeoOnnx::CImportSettings settings;
		fillLayouts( inputLayouts, settings.InputLayouts );
		fillLayouts( outputLayouts, settings.OutputLayouts );
		NeoOnnx::LoadFromOnnx( fileName.data(), settings, pyDnn.Dnn(), modelInfo );
	}
	return wrapResults( modelInfo );
}

py::object loadFromBuffer(const std::string& buffer, CPyDnn& pyDnn, py::object inputLayouts, py::object outputLayouts)
{
	NeoOnnx::CImportedModelInfo modelInfo;
	{
		py::gil_scoped_release release;
		NeoOnnx::CImportSettings settings;
		fillLayouts( inputLayouts, settings.InputLayouts );
		fillLayouts( outputLayouts, settings.OutputLayouts );
		NeoOnnx::LoadFromOnnx( buffer.data(), buffer.size(), settings, pyDnn.Dnn(), modelInfo );
	}
	return wrapResults( modelInfo );
}

void InitializeOnnx(py::module& m)
{
	m.def("load_onnx_from_file", &loadFromFile);
	m.def("load_onnx_from_buffer", &loadFromBuffer);
}
