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
};

void InitializeLstmLayer( py::module& m )
{
	py::class_<CPyLstmLayer, CPyLayer>(m, "Lstm")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyLstmLayer( *layer.Layer<CLstmLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer0, int outputNumber0,
			int hiddent_size, float dropout_rate, int recurrent_activation_index, bool reverse_seq )
		{
			CDnn& dnn = layer0.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CLstmLayer> lstm = new CLstmLayer( mathEngine );
			lstm->SetName( name == "" ? findFreeLayerName( dnn, "LstmLayer" ).c_str() : name.c_str() );
			lstm->SetHiddenSize( hiddent_size );
			lstm->SetDropoutRate( dropout_rate );
			lstm->SetRecurrentActivation( static_cast<TActivationFunction>(recurrent_activation_index) );
			lstm->SetReverseSequence(reverse_seq);
			dnn.AddLayer( *lstm );
			lstm->Connect( 0, layer0.BaseLayer(), outputNumber0 );
			return new CPyLstmLayer( *lstm, layer0.MathEngineOwner() );
		}) )
	;
}
