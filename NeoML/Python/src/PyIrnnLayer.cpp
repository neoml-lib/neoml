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

#include "PyIrnnLayer.h"

class CPyIrnnLayer : public CPyLayer {
public:
	explicit CPyIrnnLayer( CIrnnLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetHiddenSize(int value) { Layer<CIrnnLayer>()->SetHiddenSize(value); }
	int GetHiddenSize() const { return Layer<CIrnnLayer>()->GetHiddenSize(); }

	void SetIdentityScale(float scale) { Layer<CIrnnLayer>()->SetIdentityScale(scale); }
	float GetIdentityScale() const { return Layer<CIrnnLayer>()->GetIdentityScale(); }

	void SetInputWeightStd(float std) { Layer<CIrnnLayer>()->SetInputWeightStd(std); }
	float GetInputWeightStd() const { return Layer<CIrnnLayer>()->GetInputWeightStd(); }

	void SetInputWeights( const CPyBlob& blob ) { Layer<CIrnnLayer>()->SetInputWeightsData( blob.Blob() ); }
	CPyBlob GetInputWeights() const { return CPyBlob( MathEngineOwner(), Layer<CIrnnLayer>()->GetInputWeightsData() ); }

	void SetInputFreeTerm( const CPyBlob& blob ) { Layer<CIrnnLayer>()->SetInputFreeTermData( blob.Blob() ); }
	CPyBlob GetInputFreeTerm() const { return CPyBlob( MathEngineOwner(), Layer<CIrnnLayer>()->GetInputFreeTermData() ); }

	void SetRecurrentWeights( const CPyBlob& blob ) { Layer<CIrnnLayer>()->SetRecurWeightsData( blob.Blob() ); }
	CPyBlob GetRecurrentWeights() const { return CPyBlob( MathEngineOwner(), Layer<CIrnnLayer>()->GetRecurWeightsData() ); }

	void SetRecurrentFreeTerm( const CPyBlob& blob ) { Layer<CIrnnLayer>()->SetRecurFreeTermData( blob.Blob() ); }
	CPyBlob GetRecurrentFreeTerm() const { return CPyBlob( MathEngineOwner(), Layer<CIrnnLayer>()->GetRecurFreeTermData() ); }

	void SetReverseSequence(bool value) { Layer<CIrnnLayer>()->SetReverseSequence(value); }
	bool GetReverseSequence() const { return Layer<CIrnnLayer>()->IsReverseSequence(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Irnn" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeIrnnLayer( py::module& m )
{
	py::class_<CPyIrnnLayer, CPyLayer>(m, "Irnn")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyIrnnLayer( *layer.Layer<CIrnnLayer>(), layer.MathEngineOwner() );
		}))
		.def(py::init([](const std::string& name, const py::list& inputs, const py::list& input_outputs,
			int hidden_size, float identity_scale, float input_weight_std, bool reverse_seq)
		{
			py::gil_scoped_release release;
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CIrnnLayer> irnn = new CIrnnLayer(mathEngine);
			irnn->SetHiddenSize(hidden_size);
			irnn->SetIdentityScale(identity_scale);
			irnn->SetInputWeightStd(input_weight_std);
			irnn->SetReverseSequence(reverse_seq);
			irnn->SetName( FindFreeLayerName( dnn, "Irnn", name ).c_str() );
			dnn.AddLayer( *irnn );

			for( int i = 0; i < inputs.size(); i++ ) {
				irnn->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyIrnnLayer( *irnn, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_hidden_size", &CPyIrnnLayer::GetHiddenSize, py::return_value_policy::reference )
		.def( "set_hidden_size", &CPyIrnnLayer::SetHiddenSize, py::return_value_policy::reference )
		.def( "get_identity_scale", &CPyIrnnLayer::GetIdentityScale, py::return_value_policy::reference )
		.def( "set_identity_scale", &CPyIrnnLayer::SetIdentityScale, py::return_value_policy::reference )
		.def( "get_input_weight_std", &CPyIrnnLayer::GetInputWeightStd, py::return_value_policy::reference )
		.def( "set_input_weight_std", &CPyIrnnLayer::SetInputWeightStd, py::return_value_policy::reference )
		.def( "get_input_weights", &CPyIrnnLayer::GetInputWeights, py::return_value_policy::reference )
		.def( "set_input_weights", &CPyIrnnLayer::SetInputWeights, py::return_value_policy::reference )
		.def( "get_input_free_term", &CPyIrnnLayer::GetInputFreeTerm, py::return_value_policy::reference )
		.def( "set_input_free_term", &CPyIrnnLayer::SetInputFreeTerm, py::return_value_policy::reference )
		.def( "get_recurrent_weights", &CPyIrnnLayer::GetRecurrentWeights, py::return_value_policy::reference )
		.def( "set_recurrent_weights", &CPyIrnnLayer::SetRecurrentWeights, py::return_value_policy::reference )
		.def( "get_recurrent_free_term", &CPyIrnnLayer::GetRecurrentFreeTerm, py::return_value_policy::reference )
		.def( "set_recurrent_free_term", &CPyIrnnLayer::SetRecurrentFreeTerm, py::return_value_policy::reference )
		.def( "get_reverse_sequence", &CPyIrnnLayer::GetReverseSequence, py::return_value_policy::reference )
		.def( "set_reverse_sequence", &CPyIrnnLayer::SetReverseSequence, py::return_value_policy::reference )
	;
}
