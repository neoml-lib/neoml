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

#include "PySequenceSumLayer.h"

class CPySequenceSumLayer : public CPyLayer {
public:
	explicit CPySequenceSumLayer( CSequenceSumLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SequenceSum" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeSequenceSumLayer( py::module& m )
{
	py::class_<CPySequenceSumLayer, CPyLayer>(m, "SequenceSum")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySequenceSumLayer( *layer.Layer<CSequenceSumLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1 ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSequenceSumLayer> sequence = new CSequenceSumLayer( mathEngine );
			sequence->SetName( FindFreeLayerName( dnn, "SequenceSum", name ).c_str() );
			dnn.AddLayer( *sequence );
			sequence->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPySequenceSumLayer( *sequence, layer1.MathEngineOwner() );
		}) )
	;
}