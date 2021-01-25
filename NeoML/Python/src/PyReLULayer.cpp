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

#include "PyReLULayer.h"

class CPyReluLayer : public CPyLayer {
public:
	explicit CPyReluLayer( CReLULayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}
};

void InitializeReLULayer( py::module& m )
{
	py::class_<CPyReluLayer, CPyLayer>(m, "ReLU")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyReluLayer( *layer.Layer<CReLULayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber ) {
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CReLULayer> relu = new CReLULayer( mathEngine );
			relu->SetName( name == "" ? findFreeLayerName( dnn, "ReLULayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *relu );
			relu->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyReluLayer( *relu, layer.MathEngineOwner() );
		}) )
	;
}
