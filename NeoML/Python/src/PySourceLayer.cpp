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

#include "PySourceLayer.h"
#include "PyDnnBlob.h"
#include "PyDnn.h"
#include "PyMathEngine.h"

class CPySourceLayer : public CPyLayer {
public:
	CPySourceLayer( CSourceLayer* layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( *layer, mathEngineOwner ) {}

	void SetBlob( const CPyBlob& blob )
	{
		Layer<CSourceLayer>()->SetBlob( blob.Blob() );
	}

	CPyBlob GetBlob() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CSourceLayer>()->GetBlob() );
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Source" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeSourceLayer( py::module& m )
{
	py::class_<CPySourceLayer, CPyLayer>(m, "Source")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySourceLayer( layer.Layer<CSourceLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const CPyDnn& dnn, const std::string& name )
		{
			CPtr<CSourceLayer> sourceLayer( new CSourceLayer( dnn.MathEngine() ) );
			sourceLayer->SetName( FindFreeLayerName( dnn.Dnn(), "Source", name ).c_str() );
			dnn.Dnn().AddLayer( *sourceLayer );

			return new CPySourceLayer( sourceLayer, dnn.MathEngineOwner() );
		}))
		.def( "set_blob", &CPySourceLayer::SetBlob, py::return_value_policy::reference )
		.def( "get_blob", &CPySourceLayer::GetBlob, py::return_value_policy::reference )
	;
}
