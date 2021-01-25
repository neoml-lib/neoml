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

#include "PyFullyConnectedLayer.h"

class CPyFullyConnectedLayer : public CPyLayer {
public:
	explicit CPyFullyConnectedLayer( CFullyConnectedLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}
};

void InitializeFullyConnectedLayer( py::module& m )
{
	py::class_<CPyFullyConnectedLayer, CPyLayer>(m, "FullyConnected")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyFullyConnectedLayer( *layer.Layer<CFullyConnectedLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int numberOfElements ) {
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( mathEngine );
			fc->SetName( name == "" ? findFreeLayerName( dnn, "FullyConnectedLayer" ).c_str() : name.c_str() );
			fc->SetNumberOfElements( numberOfElements );
			dnn.AddLayer( *fc );
			fc->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyFullyConnectedLayer( *fc, layer.MathEngineOwner() );
		}) )
	;
}
