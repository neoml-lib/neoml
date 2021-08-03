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

#include "PyIndRnnLayer.h"
#include "PyDnnBlob.h"

class CPyIndRnnLayer : public CPyLayer {
public:
	CPyIndRnnLayer( CIndRnnLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetHiddenSize( int hiddenSize ) { Layer<CIndRnnLayer>()->SetHiddenSize( hiddenSize ); }
	int GetHiddenSize() const { return Layer<CIndRnnLayer>()->GetHiddenSize(); }

	void SetDropoutRate( float dropoutRate ) { Layer<CIndRnnLayer>()->SetDropoutRate( dropoutRate ); }
	float GetDropoutRate() const { return Layer<CIndRnnLayer>()->GetDropoutRate(); }

	void SetReverseSequence( bool reverse ) { Layer<CIndRnnLayer>()->SetReverseSequence( reverse ); }
	bool GetReverseSequence() const { return Layer<CIndRnnLayer>()->IsReverseSequence(); }

	void SetInputWeights( const CPyBlob& blob ) { Layer<CIndRnnLayer>()->SetInputWeights( blob.Blob() ); }
	CPyBlob GetInputWeights() const { return CPyBlob( MathEngineOwner(), Layer<CIndRnnLayer>()->GetInputWeights() ); }

	void SetRecurrentWeights( const CPyBlob& blob ) { Layer<CIndRnnLayer>()->SetRecurrentWeights( blob.Blob() ); }
	CPyBlob GetRecurrentWeights() const { return CPyBlob( MathEngineOwner(), Layer<CIndRnnLayer>()->GetRecurrentWeights() ); }

	void SetBias( const CPyBlob& blob ) { Layer<CIndRnnLayer>()->SetBias( blob.Blob() ); }
	CPyBlob GetBias() const { return CPyBlob( MathEngineOwner(), Layer<CIndRnnLayer>()->GetBias() ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "IndRnn" );
		return pyConstructor( py::cast( this ) );
	}
};

void InitializeIndRnnLayer( py::module& m )
{
	py::class_<CPyIndRnnLayer, CPyLayer>( m, "IndRnn" )
		.def( py::init([]( const CPyLayer& layer ) {
			return new CPyIndRnnLayer( *layer.Layer<CIndRnnLayer>(), layer.MathEngineOwner() );
		} ) )
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs,
			int hidden_size, float dropout_rate, bool reverse )
		{
			py::gil_scoped_release release;
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CIndRnnLayer> indRnn = new CIndRnnLayer( mathEngine );
			indRnn->SetHiddenSize( hidden_size );
			indRnn->SetDropoutRate( dropout_rate );
			indRnn->SetReverseSequence( reverse );
			indRnn->SetName( FindFreeLayerName( dnn, "IndRnn", name ).c_str() );
			dnn.AddLayer( *indRnn );

			for( int i = 0; i < inputs.size(); ++i ) {
				indRnn->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyIndRnnLayer( *indRnn, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		} ) )
		.def( "get_hidden_size", &CPyIndRnnLayer::GetHiddenSize, py::return_value_policy::reference )
		.def( "set_hidden_size", &CPyIndRnnLayer::SetHiddenSize, py::return_value_policy::reference )
		.def( "set_dropout_rate", &CPyIndRnnLayer::SetDropoutRate, py::return_value_policy::reference )
		.def( "get_dropout_rate", &CPyIndRnnLayer::GetDropoutRate, py::return_value_policy::reference )
		.def( "get_reverse_sequence", &CPyIndRnnLayer::GetReverseSequence, py::return_value_policy::reference )
		.def( "set_reverse_sequence", &CPyIndRnnLayer::SetReverseSequence, py::return_value_policy::reference )
		.def( "get_input_weights", &CPyIndRnnLayer::GetInputWeights, py::return_value_policy::reference )
		.def( "set_input_weights", &CPyIndRnnLayer::SetInputWeights, py::return_value_policy::reference )
		.def( "get_recurrent_weights", &CPyIndRnnLayer::GetRecurrentWeights, py::return_value_policy::reference )
		.def( "set_recurrent_weights", &CPyIndRnnLayer::SetRecurrentWeights, py::return_value_policy::reference )
		.def( "get_bias", &CPyIndRnnLayer::GetBias, py::return_value_policy::reference )
		.def( "set_bias", &CPyIndRnnLayer::SetBias, py::return_value_policy::reference );
}
