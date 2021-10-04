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

#include "PyTransformerLayer.h"

class CPyTransformerLayer : public CPyLayer {
public:
	explicit CPyTransformerLayer( CTransformerLayer& layer, CPyMathEngineOwner& mathEngineOwner )
		: CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Transformer" );
		return pyConstructor( py::cast( this ) );
	}
};

void InitializeTransformerLayer( py::module& m )
{
	py::class_<CPyTransformerLayer, CPyLayer>(m, "Transformer")
		.def( py::init( []( const CPyLayer& layer )
		{
			return new CPyTransformerLayer( *layer.Layer<CTransformerLayer>(), layer.MathEngineOwner() );
		} ) )
		.def( py::init( []( const std::string& name, const CPyLayer& layer1, int outputNumber1, int headCount, int hiddenSize,
			int outputSize, float attentionDropout, int feedForwardSize, float feedForwardDropout, int activationIndex )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CTransformerLayer> transformer = new CTransformerLayer( mathEngine );
			transformer->SetName( FindFreeLayerName( dnn, "Transformer", name ).c_str() );
			transformer->SetHeadCount( headCount );
			transformer->SetHiddenSize( hiddenSize );
			transformer->SetOutputSize( outputSize );
			transformer->SetAttentionDropout( attentionDropout );
			transformer->SetFeedForwardSize( feedForwardSize );
			transformer->SetFeedForwardDropout( feedForwardDropout );
			transformer->SetActivation( static_cast<TActivationFunction>( activationIndex ) );
			dnn.AddLayer( *transformer );
			transformer->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return CPyTransformerLayer( *transformer, layer1.MathEngineOwner() );
		} ) )
	;
}
