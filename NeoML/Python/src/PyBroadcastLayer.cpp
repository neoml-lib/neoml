/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "PyBroadcastLayer.h"

class CPyBroadcastLayer : public CPyLayer {
public:
	explicit CPyBroadcastLayer( CBroadcastLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Broadcast" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeBroadcastLayer( py::module& m )
{
	py::class_<CPyBroadcastLayer, CPyLayer>(m, "Broadcast")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyBroadcastLayer( *layer.Layer<CBroadcastLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs )
		{
			py::gil_scoped_release release;
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CBroadcastLayer> broadcast = new CBroadcastLayer( mathEngine );
			broadcast->SetName( FindFreeLayerName( dnn, "Broadcast", name ).c_str() );
			dnn.AddLayer( *broadcast );

			for( int i = 0; i < inputs.size(); ++i ) {
				broadcast->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyBroadcastLayer( *broadcast, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}));
}
