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

#include "PySinkLayer.h"
#include "PyDnnBlob.h"

class CPySinkLayer : public CPyLayer {
public:
	explicit CPySinkLayer( CSinkLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	CPyBlob GetBlob() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CSinkLayer>()->GetBlob() );
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Sink" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeSinkLayer( py::module& m )
{
	py::class_<CPySinkLayer, CPyLayer>(m, "Sink")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySinkLayer( *layer.Layer<CSinkLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber ) {
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSinkLayer> sink = new CSinkLayer( mathEngine );
			sink->SetName( name == "" ? findFreeLayerName( dnn, "SinkLayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *sink );
			sink->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPySinkLayer( *sink, layer.MathEngineOwner() );
		}) )
		.def( "get_blob", &CPySinkLayer::GetBlob, py::return_value_policy::reference )
	;
}
