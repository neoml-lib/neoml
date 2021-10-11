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

#include "PyDataLayer.h"
#include "PyDnnBlob.h"
#include "PyDnn.h"
#include "PyMathEngine.h"

class CPyDataLayer : public CPyLayer {
public:
	CPyDataLayer( CDataLayer* layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( *layer, mathEngineOwner ) {}

	void SetBlob( const CPyBlob& blob )
	{
		Layer<CDataLayer>()->SetBlob( blob.Blob() );
	}

	CPyBlob GetBlob() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CDataLayer>()->GetBlob() );
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Data" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeDataLayer( py::module& m )
{
	py::class_<CPyDataLayer, CPyLayer>(m, "Data")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyDataLayer( layer.Layer<CDataLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const CPyDnn& dnn, const std::string& name )
		{
			py::gil_scoped_release release;
			CPtr<CDataLayer> dataLayer( new CDataLayer( dnn.MathEngine() ) );
			dataLayer->SetName( FindFreeLayerName( dnn.Dnn(), "Data", name ).c_str() );
			dnn.Dnn().AddLayer( *dataLayer );

			return new CPyDataLayer( dataLayer, dnn.MathEngineOwner() );
		}))
		.def( "set_blob", &CPyDataLayer::SetBlob, py::return_value_policy::reference )
		.def( "get_blob", &CPyDataLayer::GetBlob, py::return_value_policy::reference )
	;
}
