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

template <typename TValue, typename TValueGetter>
TValue safeValueGetter( int inputIdx, int inputCount, TValue invalidValue, TValueGetter valueGetter )
{
	if( inputIdx >= inputCount ) {
		// positive index is invalid when it is greater or equal to inputCount
		return invalidValue;
	} else if ( inputIdx >= 0 ) {
		return valueGetter( inputIdx );
	} else {
		if( inputIdx + inputCount >= 0 ){
		    // negative index is valid up to -inputCount
			return valueGetter( inputIdx + inputCount );
		} else {
		    // negative index is invalid when it is less than -inputCount
			return invalidValue;
		}
	};
}

std::string CPyLayer::GetInputName( int inputIdx ) const
{
	return safeValueGetter(
		inputIdx,
		GetInputCount(),
		std::string(),
		[this](int i){return baseLayer->GetInputName(i);}
	);
}

int CPyLayer::GetInputOutputIdx( int inputIdx ) const
{
	return safeValueGetter(
		inputIdx,
		GetInputCount(),
		-1,
		[this](int i){return baseLayer->GetInputOutputNumber(i);}
	);
}


void InitializeLayer( py::module& m )
{
	py::class_<CPyLayer>(m, "Layer")
		.def( "get_name", &CPyLayer::GetName, py::return_value_policy::reference )
		.def( "get_input_count", &CPyLayer::GetInputCount, py::return_value_policy::reference )
		.def( "get_input_name", &CPyLayer::GetInputName, py::return_value_policy::reference )
		.def( "get_input_output_idx", &CPyLayer::GetInputOutputIdx, py::return_value_policy::reference )
		.def( "create_python_object", &CPyLayer::CreatePythonObject, py::return_value_policy::reference )
		.def( "connect", &CPyLayer::Connect, py::return_value_policy::reference )
		.def( "is_learning_enabled", &CPyLayer::IsLearningEnabled, py::return_value_policy::reference )
		.def( "enable_learning", &CPyLayer::EnableLearning, py::return_value_policy::reference )
		.def( "disable_learning", &CPyLayer::DisableLearning, py::return_value_policy::reference )
	;
}
