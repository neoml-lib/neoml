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

#include "PyLstmLayer.h"

class CPyLstmLayer : public CPyLayer {
public:
	explicit CPyLstmLayer( CLstmLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetHiddenSize(int value) { Layer<CLstmLayer>()->SetHiddenSize(value); }
	int GetHiddenSize() const { return Layer<CLstmLayer>()->GetHiddenSize(); }

	void SetActivation(int value) { Layer<CLstmLayer>()->SetRecurrentActivation(static_cast<TActivationFunction>(value)); }
	int GetActivation() const { return static_cast<int>(Layer<CLstmLayer>()->GetRecurrentActivation()); }

	void SetDropout(float value) { Layer<CLstmLayer>()->SetDropoutRate(value); }
	float GetDropout() const { return Layer<CLstmLayer>()->GetDropoutRate(); }

	void SetReverseSequence(bool value) { Layer<CLstmLayer>()->SetReverseSequence(value); }
	bool GetReverseSequence() const { return Layer<CLstmLayer>()->IsReverseSequence(); }

	void SetInputWeights( const CPyBlob& blob ) { Layer<CLstmLayer>()->SetInputWeightsData( blob.Blob() ); }
	CPyBlob GetInputWeights() const { return CPyBlob( MathEngineOwner(), Layer<CLstmLayer>()->GetInputWeigthsData() ); }

	void SetInputFreeTerm( const CPyBlob& blob ) { Layer<CLstmLayer>()->SetInputFreeTermData( blob.Blob() ); }
	CPyBlob GetInputFreeTerm() const { return CPyBlob( MathEngineOwner(), Layer<CLstmLayer>()->GetInputFreeTermData() ); }

	void SetRecurrentWeights( const CPyBlob& blob ) { Layer<CLstmLayer>()->SetRecurWeightsData( blob.Blob() ); }
	CPyBlob GetRecurrentWeights() const { return CPyBlob( MathEngineOwner(), Layer<CLstmLayer>()->GetRecurWeigthsData() ); }

	void SetRecurrentFreeTerm( const CPyBlob& blob ) { Layer<CLstmLayer>()->SetRecurFreeTermData( blob.Blob() ); }
	CPyBlob GetRecurrentFreeTerm() const { return CPyBlob( MathEngineOwner(), Layer<CLstmLayer>()->GetRecurFreeTermData() ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Lstm" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeLstmLayer( py::module& m )
{
	py::class_<CPyLstmLayer, CPyLayer>(m, "Lstm")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyLstmLayer( *layer.Layer<CLstmLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs,
			int hiddent_size, float dropout_rate, int recurrent_activation_index, bool reverse_seq )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CLstmLayer> lstm = new CLstmLayer( mathEngine );
			lstm->SetName( name == "" ? findFreeLayerName( dnn, "Lstm" ).c_str() : name.c_str() );
			lstm->SetHiddenSize( hiddent_size );
			lstm->SetDropoutRate( dropout_rate );
			lstm->SetRecurrentActivation( static_cast<TActivationFunction>(recurrent_activation_index) );
			lstm->SetReverseSequence(reverse_seq);
			dnn.AddLayer( *lstm );
			for( int i = 0; i < inputs.size(); i++ ) {
				lstm->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}
			return new CPyLstmLayer( *lstm, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_hidden_size", &CPyLstmLayer::GetHiddenSize, py::return_value_policy::reference )
		.def( "set_hidden_size", &CPyLstmLayer::SetHiddenSize, py::return_value_policy::reference )
		.def( "get_activation", &CPyLstmLayer::GetActivation, py::return_value_policy::reference )
		.def( "set_activation", &CPyLstmLayer::SetActivation, py::return_value_policy::reference )
		.def( "get_dropout", &CPyLstmLayer::GetDropout, py::return_value_policy::reference )
		.def( "set_dropout", &CPyLstmLayer::SetDropout, py::return_value_policy::reference )
		.def( "get_input_weights", &CPyLstmLayer::GetInputWeights, py::return_value_policy::reference )
		.def( "set_input_weights", &CPyLstmLayer::SetInputWeights, py::return_value_policy::reference )
		.def( "get_input_free_term", &CPyLstmLayer::GetInputFreeTerm, py::return_value_policy::reference )
		.def( "set_input_free_term", &CPyLstmLayer::SetInputFreeTerm, py::return_value_policy::reference )
		.def( "get_recurrent_weights", &CPyLstmLayer::GetRecurrentWeights, py::return_value_policy::reference )
		.def( "set_recurrent_weights", &CPyLstmLayer::SetRecurrentWeights, py::return_value_policy::reference )
		.def( "get_recurrent_free_term", &CPyLstmLayer::GetRecurrentFreeTerm, py::return_value_policy::reference )
		.def( "set_recurrent_free_term", &CPyLstmLayer::SetRecurrentFreeTerm, py::return_value_policy::reference )
		.def( "get_reverse_sequence", &CPyLstmLayer::GetReverseSequence, py::return_value_policy::reference )
		.def( "set_reverse_sequence", &CPyLstmLayer::SetReverseSequence, py::return_value_policy::reference )
	;
}
