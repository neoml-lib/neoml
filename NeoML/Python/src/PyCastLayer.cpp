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

#include "PyCastLayer.h"

static TBlobType fromStr( const std::string& outputType )
{
	if( outputType == "float" ) {
		return CT_Float;
	} else if( outputType == "int" ) {
		return CT_Int;
	}
	return CT_Invalid;
}

static std::string toStr( TBlobType type )
{
	switch( type ) {
		case CT_Float:
			return "float";
		case CT_Int:
			return "int";
		case CT_Invalid:
		default:
			return "invalid";
	}

	return "invalid";
}

class CPyCastLayer : public CPyLayer {
public:
	explicit CPyCastLayer( CCastLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLayer( layer, mathEngineOwner ) {}

	void SetOutputType( const std::string& type ) { Layer<CCastLayer>()->SetOutputType( fromStr( type ) ); }
	std::string GetOutputType() const { return toStr( Layer<CCastLayer>()->GetOutputType() ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Cast" );
		return pyConstructor( py::cast( this ), 0 );
	}
};

void InitializeCastLayer( py::module& m )
{
	py::class_<CPyCastLayer, CPyLayer>(m, "Cast")
		.def( py::init( []( const CPyLayer& layer )
		{
			return new CPyCastLayer( *layer.Layer<CCastLayer>(), layer.MathEngineOwner() );
		} ) )
		.def( py::init( []( const std::string& name, const CPyLayer& layer, int outputNumber, const std::string& type )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CCastLayer> cast  = new CCastLayer( mathEngine );
			cast->SetOutputType( fromStr( type ) );
			cast->SetName( FindFreeLayerName( dnn, "Cast", name ).c_str() );
			dnn.AddLayer( *cast );
			cast->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyCastLayer( *cast, layer.MathEngineOwner() );
		} ) )
		.def( "get_output_type", &CPyCastLayer::GetOutputType, py::return_value_policy::reference )
		.def( "set_output_type", &CPyCastLayer::SetOutputType, py::return_value_policy::reference )
	;
}
