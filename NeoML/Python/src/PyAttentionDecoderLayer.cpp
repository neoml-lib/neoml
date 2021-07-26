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

#include "PyAttentionDecoderLayer.h"

class CPyAttentionDecoderLayer : public CPyLayer {
public:
	explicit CPyAttentionDecoderLayer( CAttentionDecoderLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	int GetAttentionScore() const { return static_cast<int>( Layer<CAttentionDecoderLayer>()->GetAttentionScore() ); }
	void SetAttentionScore( int score ) { Layer<CAttentionDecoderLayer>()->SetAttentionScore( static_cast<TAttentionScore>(score) ); }

	int GetOutputObjectSize() const { return Layer<CAttentionDecoderLayer>()->GetOutputObjectSize(); }
	void SetOutputObjectSize(int outObjectSize) { Layer<CAttentionDecoderLayer>()->SetOutputObjectSize( outObjectSize ); }
	int GetOutputSequenceLen() const { return Layer<CAttentionDecoderLayer>()->GetOutputSequenceLen(); }
	void SetOutputSequenceLen(int outSeqLen) { Layer<CAttentionDecoderLayer>()->SetOutputSequenceLen( outSeqLen ); }

	int GetHiddenLayerSize() const { return Layer<CAttentionDecoderLayer>()->GetHiddenLayerSize(); }
	void SetHiddenLayerSize( int size ) { Layer<CAttentionDecoderLayer>()->SetHiddenLayerSize( size ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "AttentionDecoder" );
		return pyConstructor( py::cast(this), 0, 0, 0, 0 );
	}
};

void InitializeAttentionDecoderLayer( py::module& m )
{
	py::class_<CPyAttentionDecoderLayer, CPyLayer>(m, "AttentionDecoder")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyAttentionDecoderLayer( *layer.Layer<CAttentionDecoderLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer0, int outputNumber0, const CPyLayer& layer1, int outputNumber1,
			int score_index, int output_object_size, int output_seq_len, int hidden_size )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer0.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CAttentionDecoderLayer> decoder = new CAttentionDecoderLayer( mathEngine );
			decoder->SetName( FindFreeLayerName( dnn, "AttentionDecoder", name ).c_str() );
			dnn.AddLayer( *decoder );
			decoder->Connect( 0, layer0.BaseLayer(), outputNumber0 );
			decoder->Connect( 1, layer1.BaseLayer(), outputNumber1 );
			decoder->SetAttentionScore( static_cast<TAttentionScore>(score_index) );
			decoder->SetOutputObjectSize( output_object_size );
			decoder->SetOutputSequenceLen( output_seq_len );
			decoder->SetHiddenLayerSize( hidden_size );
			return new CPyAttentionDecoderLayer( *decoder, layer1.MathEngineOwner() );
		}) )
		.def( "set_score", &CPyAttentionDecoderLayer::SetAttentionScore, py::return_value_policy::reference )
		.def( "get_score", &CPyAttentionDecoderLayer::GetAttentionScore, py::return_value_policy::reference )
		.def( "set_output_seq_len", &CPyAttentionDecoderLayer::SetOutputSequenceLen, py::return_value_policy::reference )
		.def( "get_output_seq_len", &CPyAttentionDecoderLayer::GetOutputSequenceLen, py::return_value_policy::reference )
		.def( "set_output_object_size", &CPyAttentionDecoderLayer::SetOutputObjectSize, py::return_value_policy::reference )
		.def( "get_output_object_size", &CPyAttentionDecoderLayer::GetOutputObjectSize, py::return_value_policy::reference )
		.def( "set_hidden_layer_size", &CPyAttentionDecoderLayer::SetHiddenLayerSize, py::return_value_policy::reference )
		.def( "get_hidden_layer_size", &CPyAttentionDecoderLayer::GetHiddenLayerSize, py::return_value_policy::reference )
	;
}
