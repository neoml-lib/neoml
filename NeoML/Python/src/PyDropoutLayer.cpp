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

#include "PyDropoutLayer.h"

class CPyDropoutLayer : public CPyLayer {
public:
	explicit CPyDropoutLayer( CDropoutLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetDropoutRate( float value ) { Layer<CDropoutLayer>()->SetDropoutRate( value ); }
	float GetDropoutRate() const { return Layer<CDropoutLayer>()->GetDropoutRate(); }

	bool GetSpatial() const { return Layer<CDropoutLayer>()->IsSpatial(); }
	void SetSpatial( bool value ) { Layer<CDropoutLayer>()->SetSpatial( value ); }

	bool GetBatchwise() const { return Layer<CDropoutLayer>()->IsBatchwise(); }
	void SetBatchwise( bool value ) { Layer<CDropoutLayer>()->SetBatchwise( value ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Dropout" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeDropoutLayer( py::module& m )
{
	py::class_<CPyDropoutLayer, CPyLayer>(m, "Dropout")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyDropoutLayer( *layer.Layer<CDropoutLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, float dropoutRate,
			bool isSpatial, bool isBatchwise )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CDropoutLayer> dropout = new CDropoutLayer( mathEngine );
			dropout->SetDropoutRate( dropoutRate );
			dropout->SetSpatial( isSpatial );
			dropout->SetBatchwise( isBatchwise );
			dropout->SetName( FindFreeLayerName( dnn, "Dropout", name ).c_str() );
			dnn.AddLayer( *dropout );
			dropout->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyDropoutLayer( *dropout, layer.MathEngineOwner() );
		}) )
		.def( "get_rate", &CPyDropoutLayer::GetDropoutRate, py::return_value_policy::reference )
		.def( "set_rate", &CPyDropoutLayer::SetDropoutRate, py::return_value_policy::reference )
		.def( "get_spatial", &CPyDropoutLayer::GetSpatial, py::return_value_policy::reference )
		.def( "set_spatial", &CPyDropoutLayer::SetSpatial, py::return_value_policy::reference )
		.def( "get_batchwise", &CPyDropoutLayer::GetBatchwise, py::return_value_policy::reference )
		.def( "set_batchwise", &CPyDropoutLayer::SetBatchwise, py::return_value_policy::reference )
	;
}
