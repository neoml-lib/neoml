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

#include "PyDotProductLayer.h"

class CPyDotProductLayer : public CPyLayer {
public:
	explicit CPyDotProductLayer( CDotProductLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "DotProduct" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeDotProductLayer( py::module& m )
{
	py::class_<CPyDotProductLayer, CPyLayer>(m, "DotProduct")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyDotProductLayer( *layer.Layer<CDotProductLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, const CPyLayer& layer2, int outputNumber1, int outputNumber2 ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CDotProductLayer> dotProduct = new CDotProductLayer( mathEngine );
			dotProduct->SetName( FindFreeLayerName( dnn, "DotProduct", name ).c_str() );
			dnn.AddLayer( *dotProduct );
			dotProduct->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			dotProduct->Connect( 1, layer2.BaseLayer(), outputNumber2 );
			return new CPyDotProductLayer( *dotProduct, layer1.MathEngineOwner() );
		}) )
	;
}