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

#include "PySoftmaxLayer.h"

class CPySoftmaxLayer : public CPyLayer {
public:
	explicit CPySoftmaxLayer( CSoftmaxLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetArea(int value) { Layer<CSoftmaxLayer>()->SetNormalizationArea(static_cast<CSoftmaxLayer::TNormalizationArea>(value)); }
	int GetArea() const { return static_cast<int>(Layer<CSoftmaxLayer>()->GetNormalizationArea()); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Softmax" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeSoftmaxLayer( py::module& m )
{
	py::class_<CPySoftmaxLayer, CPyLayer>(m, "Softmax")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySoftmaxLayer( *layer.Layer<CSoftmaxLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, int area_index ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer( mathEngine );
			softmax->SetNormalizationArea( static_cast<CSoftmaxLayer::TNormalizationArea>(area_index) );
			softmax->SetName( FindFreeLayerName( dnn, "Softmax", name ).c_str() );
			dnn.AddLayer( *softmax );
			softmax->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPySoftmaxLayer( *softmax, layer1.MathEngineOwner() );
		}) )
		.def( "get_area", &CPySoftmaxLayer::GetArea, py::return_value_policy::reference )
		.def( "set_area", &CPySoftmaxLayer::SetArea, py::return_value_policy::reference )
	;
}
