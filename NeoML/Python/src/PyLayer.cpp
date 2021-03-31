/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "PyLayer.h"

std::string FindFreeLayerName( const CDnn& dnn, const std::string& layerName, const std::string& userName )
{
	if( userName != "None" ) {
		return userName;
	}
	int index = 0;
	std::string name;
	do {
		index++;
		name = layerName + std::to_string(index);
	} while( dnn.HasLayer( name.c_str() ) );
	
	return name;
}

void CPyLayer::Connect( CPyLayer &layer, int outputIndex, int inputIndex )
{
	baseLayer->Connect( inputIndex, *layer.baseLayer, outputIndex );
}

void InitializeLayer( py::module& m )
{
	py::class_<CPyLayer>(m, "Layer")
		.def( "output_count", &CPyLayer::GetOutputCount, py::return_value_policy::reference )
		.def( "get_name", &CPyLayer::GetName, py::return_value_policy::reference )
		.def( "create_python_object", &CPyLayer::CreatePythonObject, py::return_value_policy::reference )
		.def( "connect", &CPyLayer::Connect, py::return_value_policy::reference )
	;
}
