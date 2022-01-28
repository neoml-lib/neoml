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

#include "PyBertConvLayer.h"

class CPyBertConvLayer : public CPyLayer {
public:
	explicit CPyBertConvLayer( CBertConvLayer& layer, CPyMathEngineOwner& mathEngineOwner )
		: CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "BertConv" );
		return pyConstructor( py::cast( this ) );
	}
};

void InitializeBertConvLayer( py::module& m )
{
	py::class_<CPyBertConvLayer, CPyLayer>(m, "BertConv")
		.def( py::init( []( const CPyLayer& layer )
		{
			return new CPyBertConvLayer( *layer.Layer<CBertConvLayer>(), layer.MathEngineOwner() );
		} ) )
		.def( py::init( []( const std::string& name, const py::list& inputs, const py::list& output_indices )
		{
			py::gil_scoped_release release;
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CBertConvLayer> bertConv = new CBertConvLayer( mathEngine );
			bertConv->SetName( FindFreeLayerName( dnn, "BertConv", name ).c_str() );
			dnn.AddLayer( *bertConv );
			for( int i = 0; i < inputs.size(); i++ ) {
				bertConv->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), output_indices[i].cast<int>() );
			}
			return new CPyBertConvLayer( *bertConv, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		} ) )
	;
}
