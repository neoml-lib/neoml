
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

#include "PyScatterGatherLayers.h"

class CPyScatterNDLayer : public CPyLayer {
public:
	explicit CPyScatterNDLayer( CScatterNDLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ScatterND" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeScatterGatherLayers( py::module& m )
{
	py::class_<CPyScatterNDLayer, CPyLayer>(m, "ScatterND")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyScatterNDLayer( *layer.Layer<CScatterNDLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& firstLayer, int firstOutputNumber,
				const CPyLayer& secondLayer, int secondOutputNumber, const CPyLayer& thirdLayer, int thirdOutputNumber ) {
			py::gil_scoped_release release;
			CDnn& dnn = firstLayer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CScatterNDLayer> scatter = new CScatterNDLayer( mathEngine );
			scatter->SetName( FindFreeLayerName( dnn, "ScatterND", name ).c_str() );
			dnn.AddLayer( *scatter );
			scatter->Connect( 0, firstLayer.BaseLayer(), firstOutputNumber );
			scatter->Connect( 1, secondLayer.BaseLayer(), secondOutputNumber );
			scatter->Connect( 2, thirdLayer.BaseLayer(), thirdOutputNumber );
			return new CPyScatterNDLayer( *scatter, firstLayer.MathEngineOwner() );
		}) )
	;
}
