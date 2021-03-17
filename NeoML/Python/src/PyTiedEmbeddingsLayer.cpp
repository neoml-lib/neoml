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

#include "PyTiedEmbeddingsLayer.h"

class CPyTiedEmbeddingsLayer : public CPyLayer {
public:
	explicit CPyTiedEmbeddingsLayer( CTiedEmbeddingsLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	std::string GetEmbeddingsLayerName() const { return Layer<CTiedEmbeddingsLayer>()->GetEmbeddingsLayerName(); }
	void SetEmbeddingsLayerName(const std::string& name) { Layer<CTiedEmbeddingsLayer>()->SetEmbeddingsLayerName(name.c_str()); }

 	int GetChannel() const { return Layer<CTiedEmbeddingsLayer>()->GetChannelIndex(); }
	void SetChannel(int value) { Layer<CTiedEmbeddingsLayer>()->SetChannelIndex(value); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "TiedEmbeddings" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

void InitializeTiedEmbeddingsLayer( py::module& m )
{
	py::class_<CPyTiedEmbeddingsLayer, CPyLayer>(m, "TiedEmbeddings")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyTiedEmbeddingsLayer( *layer.Layer<CTiedEmbeddingsLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs, const std::string& embeddingsName, int channel )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CTiedEmbeddingsLayer> tied = new CTiedEmbeddingsLayer( mathEngine );
			tied->SetName( FindFreeLayerName( dnn, "TiedEmbeddings", name ).c_str() );
			tied->SetChannelIndex( channel );
			tied->SetEmbeddingsLayerName( embeddingsName.c_str() );

			dnn.AddLayer( *tied );

			for( int i = 0; i < inputs.size(); i++ ) {
				tied->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyTiedEmbeddingsLayer( *tied, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_channel", &CPyTiedEmbeddingsLayer::GetChannel, py::return_value_policy::reference )
		.def( "set_channel", &CPyTiedEmbeddingsLayer::SetChannel, py::return_value_policy::reference )
		.def( "get_embeddings_layer_name", &CPyTiedEmbeddingsLayer::GetEmbeddingsLayerName, py::return_value_policy::reference )
		.def( "set_embeddings_layer_name", &CPyTiedEmbeddingsLayer::SetEmbeddingsLayerName, py::return_value_policy::reference )
	;
}
