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

#include "PyGruLayer.h"

class CPyGruLayer : public CPyLayer {
public:
	explicit CPyGruLayer( CGruLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetHiddenSize(int reset) { Layer<CGruLayer>()->SetHiddenSize(reset); }
	int GetHiddenSize() const { return Layer<CGruLayer>()->GetHiddenSize(); }

	CPyBlob GetMainWeightsData() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CGruLayer>()->GetMainWeightsData() );
	}
	CPyBlob GetMainFreeTermData() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CGruLayer>()->GetMainFreeTermData() );
	}
	CPyBlob GetGateWeightsData() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CGruLayer>()->GetGateWeightsData() );
	}
	CPyBlob GetGateFreeTermData() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CGruLayer>()->GetGateFreeTermData() );
	}

	void SetMainWeightsData( const CPyBlob& blob )
	{
		Layer<CGruLayer>()->SetMainWeightsData( blob.Blob() );
	}
	void SetMainFreeTermData( const CPyBlob& blob )
	{
		Layer<CGruLayer>()->SetMainFreeTermData( blob.Blob() );
	}
	void SetGateWeightsData( const CPyBlob& blob )
	{
		Layer<CGruLayer>()->SetGateWeightsData( blob.Blob() );
	}
	void SetGateFreeTermData( const CPyBlob& blob )
	{
		Layer<CGruLayer>()->SetGateFreeTermData( blob.Blob() );
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Gru" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeGruLayer( py::module& m )
{
	py::class_<CPyGruLayer, CPyLayer>(m, "Gru")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyGruLayer( *layer.Layer<CGruLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int hidden_size )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CGruLayer> gru = new CGruLayer( mathEngine );
			gru->SetHiddenSize( hidden_size );
			gru->SetName( FindFreeLayerName( dnn, "Gru", name ).c_str() );
			dnn.AddLayer( *gru );
			gru->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );

			if( layers.size() == 2 ) {
				gru->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			}

			return CPyGruLayer( *gru, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_hidden_size", &CPyGruLayer::GetHiddenSize, py::return_value_policy::reference )
		.def( "set_hidden_size", &CPyGruLayer::SetHiddenSize, py::return_value_policy::reference )

		.def( "get_main_weights", &CPyGruLayer::GetMainWeightsData, py::return_value_policy::reference )
		.def( "set_main_weights", &CPyGruLayer::SetMainWeightsData, py::return_value_policy::reference )

		.def( "get_main_free_term", &CPyGruLayer::GetMainFreeTermData, py::return_value_policy::reference )
		.def( "set_main_free_term", &CPyGruLayer::SetMainFreeTermData, py::return_value_policy::reference )

		.def( "get_gate_weights", &CPyGruLayer::GetGateWeightsData, py::return_value_policy::reference )
		.def( "set_gate_weights", &CPyGruLayer::SetGateWeightsData, py::return_value_policy::reference )

		.def( "get_gate_free_term", &CPyGruLayer::GetGateFreeTermData, py::return_value_policy::reference )
		.def( "set_gate_free_term", &CPyGruLayer::SetGateFreeTermData, py::return_value_policy::reference )
	;
}
