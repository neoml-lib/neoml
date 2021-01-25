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

#include "PyMultichannelLookupLayer.h"

class CPyMultichannelLookupLayer : public CPyLayer {
public:
	explicit CPyMultichannelLookupLayer( CMultichannelLookupLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void Initialize( const CPyInitializer& initializer )
	{
		Layer<CMultichannelLookupLayer>()->Initialize( initializer.Initializer<CDnnInitializer>() );
	}
	void Clear()
	{
		Layer<CMultichannelLookupLayer>()->Initialize( 0 );
	}
};

void InitializeMultichannelLookupLayer( py::module& m )
{
	py::class_<CPyMultichannelLookupLayer, CPyLayer>(m, "MultichannelLookup")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyMultichannelLookupLayer( *layer.Layer<CMultichannelLookupLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs, const py::list& dimensions )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMultichannelLookupLayer> lookup = new CMultichannelLookupLayer( mathEngine );
			lookup->SetName( name == "" ? findFreeLayerName( dnn, "MultichannelLookupLayer" ).c_str() : name.c_str() );
			CArray<CLookupDimension> d;
			for( int i = 0; i < dimensions.size(); i++ ) {
				py::tuple t = dimensions[i].cast<py::tuple>();
				CLookupDimension dimension;
				dimension.VectorCount = t[0].cast<int>();
				dimension.VectorSize = t[1].cast<int>();
				d.Add( dimension );
			}
			lookup->SetDimensions( d );

			dnn.AddLayer( *lookup );

			for( int i = 0; i < inputs.size(); i++ ) {
				lookup->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyMultichannelLookupLayer( *lookup, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "initialize", &CPyMultichannelLookupLayer::Initialize, py::return_value_policy::reference )
		.def( "clear", &CPyMultichannelLookupLayer::Clear, py::return_value_policy::reference )
	;
}
