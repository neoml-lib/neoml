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

#include "PyRepeatSequenceLayer.h"

class CPyRepeatSequenceLayer : public CPyLayer {
public:
	explicit CPyRepeatSequenceLayer( CRepeatSequenceLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetRepeatCount(int value) { Layer<CRepeatSequenceLayer>()->SetRepeatCount(value); }
	int GetRepeatCount() const { return Layer<CRepeatSequenceLayer>()->GetRepeatCount(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "RepeatSequence" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeRepeatSequenceLayer( py::module& m )
{
	py::class_<CPyRepeatSequenceLayer, CPyLayer>(m, "RepeatSequence")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyRepeatSequenceLayer( *layer.Layer<CRepeatSequenceLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, int repeatCount ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CRepeatSequenceLayer> sequence = new CRepeatSequenceLayer( mathEngine );
			sequence->SetRepeatCount(repeatCount);
			sequence->SetName( FindFreeLayerName( dnn, "RepeatSequence", name ).c_str() );
			dnn.AddLayer( *sequence );
			sequence->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPyRepeatSequenceLayer( *sequence, layer1.MathEngineOwner() );
		}) )
		.def( "get_repeat_count", &CPyRepeatSequenceLayer::GetRepeatCount, py::return_value_policy::reference )
		.def( "set_repeat_count", &CPyRepeatSequenceLayer::SetRepeatCount, py::return_value_policy::reference )
	;
}
