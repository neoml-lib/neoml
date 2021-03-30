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

#include "PyAddToObjectLayer.h"

class CPyAddToObjectLayer : public CPyLayer {
public:
	explicit CPyAddToObjectLayer( CAddToObjectLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "AddToObject" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeAddToObjectLayer( py::module& m )
{
	py::class_<CPyAddToObjectLayer, CPyLayer>(m, "AddToObject")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyAddToObjectLayer( *layer.Layer<CAddToObjectLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, const CPyLayer& layer2, int outputNumber1, int outputNumber2 ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CAddToObjectLayer> addToObject = new CAddToObjectLayer( mathEngine );
			addToObject->SetName( FindFreeLayerName( dnn, "AddToObject", name ).c_str() );
			dnn.AddLayer( *addToObject );
			addToObject->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			addToObject->Connect( 1, layer2.BaseLayer(), outputNumber2 );
			return new CPyAddToObjectLayer( *addToObject, layer1.MathEngineOwner() );
		}) )
	;
}