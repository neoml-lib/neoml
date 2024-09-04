/* Copyright © 2017-2024 ABBYY

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

#include "PyTransformerLayer.h"

class CPyTransformerEncoderLayer : public CPyLayer {
public:
	explicit CPyTransformerEncoderLayer( CTransformerEncoderLayer& layer, CPyMathEngineOwner& mathEngineOwner )
		: CPyLayer( layer, mathEngineOwner ) {}

	int GetHeadCount() const { return Layer<CTransformerEncoderLayer>()->GetHeadCount(); }
	void SetHeadCount( int headCount ) { Layer<CTransformerEncoderLayer>()->SetHeadCount( headCount ); }

	int GetHiddenSize() const { return Layer<CTransformerEncoderLayer>()->GetHiddenSize(); }
	void SetHiddenSize( int hiddenSize ) { Layer<CTransformerEncoderLayer>()->SetHiddenSize( hiddenSize ); }

	float GetDropoutRate() const { return Layer<CTransformerEncoderLayer>()->GetDropoutRate(); }
	void SetDropoutRate( float rate ) { Layer<CTransformerEncoderLayer>()->SetDropoutRate( rate ); }

	float GetSelfAttentionDropoutRate() const { return Layer<CTransformerEncoderLayer>()->GetSelfAttentionDropoutRate(); }
	void SetSelfAttentionDropoutRate( float rate ) { Layer<CTransformerEncoderLayer>()->SetSelfAttentionDropoutRate( rate ); }
	
	int GetFeedForwardSize() const { return Layer<CTransformerEncoderLayer>()->GetFeedForwardSize(); }
	void SetFeedForwardSize( int size ) { Layer<CTransformerEncoderLayer>()->SetFeedForwardSize( size ); }

	// Place of the normalization layer: right after input or before feedForward as usual
	bool GetPreNorm() const { return Layer<CTransformerEncoderLayer>()->GetPreNorm(); }
	void SetPreNorm( bool preNorm ) { return Layer<CTransformerEncoderLayer>()->SetPreNorm( preNorm ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "TransformerEncoder" );
		return pyConstructor( py::cast( this ) );
	}
};


void InitializeTransformerLayer( py::module& m )
{
	py::class_<CPyTransformerEncoderLayer, CPyLayer>(m, "TransformerEncoder")
		.def( py::init( []( const CPyLayer& layer )
		{
			return new CPyTransformerEncoderLayer( *layer.Layer<CTransformerEncoderLayer>(), layer.MathEngineOwner() );
		} ) )
		.def( py::init( []( const std::string& name, const py::list& inputs, const py::list& input_outputs,
			int headCount, int hiddenSize, float dropout, float sa_dropout, int feedForwardSize, int activationIndex, bool pre_norm )
		{
			py::gil_scoped_release release;
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CTransformerEncoderLayer> transformer = new CTransformerEncoderLayer( mathEngine );
			transformer->SetName( FindFreeLayerName( dnn, "TransformerEncoder", name ).c_str() );
			transformer->SetHeadCount( headCount );
			transformer->SetHiddenSize( hiddenSize );
			transformer->SetDropoutRate( dropout );
			transformer->SetSelfAttentionDropoutRate( sa_dropout );
			transformer->SetFeedForwardSize( feedForwardSize );
			transformer->SetActivation( static_cast<TActivationFunction>( activationIndex ) );
			transformer->SetPreNorm( pre_norm );
			for( int i = 0; i < inputs.size(); i++ ) {
				transformer->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}
			dnn.AddLayer( *transformer );
			return CPyTransformerEncoderLayer( *transformer, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		} ) )
		.def( "get_head_count", &CPyTransformerEncoderLayer::GetHeadCount, py::return_value_policy::reference )
		.def( "set_head_count", &CPyTransformerEncoderLayer::SetHeadCount, py::return_value_policy::reference )
		.def( "get_hidden_size", &CPyTransformerEncoderLayer::GetHiddenSize, py::return_value_policy::reference )
		.def( "set_hidden_size", &CPyTransformerEncoderLayer::SetHiddenSize, py::return_value_policy::reference )
		.def( "get_dropout", &CPyTransformerEncoderLayer::GetDropoutRate, py::return_value_policy::reference )
		.def( "set_dropout", &CPyTransformerEncoderLayer::SetDropoutRate, py::return_value_policy::reference )
		.def( "get_sa_dropout", &CPyTransformerEncoderLayer::GetSelfAttentionDropoutRate, py::return_value_policy::reference )
		.def( "set_sa_dropout", &CPyTransformerEncoderLayer::SetSelfAttentionDropoutRate, py::return_value_policy::reference )
		.def( "get_feed_forward_size", &CPyTransformerEncoderLayer::GetFeedForwardSize, py::return_value_policy::reference )
		.def( "set_feed_forward_size", &CPyTransformerEncoderLayer::SetFeedForwardSize, py::return_value_policy::reference )
		.def( "get_pre_norm", &CPyTransformerEncoderLayer::GetPreNorm, py::return_value_policy::reference )
		.def( "set_pre_norm", &CPyTransformerEncoderLayer::SetPreNorm, py::return_value_policy::reference )
	;
}
