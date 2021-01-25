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

#include "PyConcatChannelsLayer.h"

class CPyConcatChannelsLayer : public CPyLayer {
public:
	explicit CPyConcatChannelsLayer( CConcatChannelsLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}
};

void InitializeConcatChannelsLayer( py::module& m )
{
	py::class_<CPyConcatChannelsLayer, CPyLayer>(m, "ConcatChannels")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyConcatChannelsLayer( *layer.Layer<CConcatChannelsLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CConcatChannelsLayer> concat = new CConcatChannelsLayer( mathEngine );
			concat->SetName( name == "" ? findFreeLayerName( dnn, "ConcatChannelsLayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *concat );

			for( int i = 0; i < inputs.size(); i++ ) {
				concat->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyConcatChannelsLayer( *concat, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;
}
