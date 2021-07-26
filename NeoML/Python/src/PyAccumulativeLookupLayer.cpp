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

#include "PyAccumulativeLookupLayer.h"

class CPyAccumulativeLookupLayer : public CPyLayer {
public:
	explicit CPyAccumulativeLookupLayer( CAccumulativeLookupLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetCount(int count)
	{
		CLookupDimension d = Layer<CAccumulativeLookupLayer>()->GetDimension();
		d.VectorCount = count;
		Layer<CAccumulativeLookupLayer>()->SetDimension( d );
	}

	int GetCount() const
	{
		return Layer<CAccumulativeLookupLayer>()->GetDimension().VectorCount;
	}

	void SetSize(int size)
	{
		CLookupDimension d = Layer<NeoML::CAccumulativeLookupLayer>()->GetDimension();
		d.VectorSize = size;
		Layer<CAccumulativeLookupLayer>()->SetDimension( d );
	}
	
	int GetSize() const
	{
		return Layer<CAccumulativeLookupLayer>()->GetDimension().VectorSize;
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "AccumulativeLookup" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

void InitializeAccumulativeLookupLayer( py::module& m )
{
	py::class_<CPyAccumulativeLookupLayer, CPyLayer>(m, "AccumulativeLookup")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyAccumulativeLookupLayer( *layer.Layer<CAccumulativeLookupLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int count, int size ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CAccumulativeLookupLayer> accumulativeLookup = new CAccumulativeLookupLayer( mathEngine );
			CLookupDimension d;
			d.VectorCount = count;
			d.VectorSize = size;
			accumulativeLookup->SetDimension( d );
			accumulativeLookup->SetName( FindFreeLayerName( dnn, "AccumulativeLookup", name ).c_str() );
			dnn.AddLayer( *accumulativeLookup );
			accumulativeLookup->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyAccumulativeLookupLayer( *accumulativeLookup, layer.MathEngineOwner() );
		}) )

		.def( "set_size", &CPyAccumulativeLookupLayer::SetSize, py::return_value_policy::reference )
		.def( "get_size", &CPyAccumulativeLookupLayer::GetSize, py::return_value_policy::reference )
		.def( "get_count", &CPyAccumulativeLookupLayer::GetCount, py::return_value_policy::reference )
		.def( "set_count", &CPyAccumulativeLookupLayer::SetCount, py::return_value_policy::reference )
	;
}
