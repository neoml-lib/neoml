/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "PyCumSumLayer.h"

class CPyCumSumLayer : public CPyLayer {
public:
	explicit CPyCumSumLayer( CCumSumLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetDimension(int d) { Layer<CCumSumLayer>()->SetDimension( static_cast<TBlobDim>( d ) ); }
	int GetDimension() const { return static_cast<int>( Layer<CCumSumLayer>()->GetDimension() ); }

	void SetReverse(bool reverse) { Layer<CCumSumLayer>()->SetReverse( reverse ); }
	bool IsReverse() const { return Layer<CCumSumLayer>()->IsReverse(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "CumSum" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeCumSumLayer( py::module& m )
{
	py::class_<CPyCumSumLayer, CPyLayer>(m, "CumSum")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyCumSumLayer( *layer.Layer<CCumSumLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int dimension, bool reverse ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CCumSumLayer> cumSum = new CCumSumLayer( mathEngine );
			cumSum->SetDimension( static_cast<TBlobDim>(dimension) );
			cumSum->SetReverse( reverse );
			cumSum->SetName( FindFreeLayerName( dnn, "CumSum", name ).c_str() );
			dnn.AddLayer( *cumSum );
			cumSum->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyCumSumLayer( *cumSum, layer.MathEngineOwner() );
		}) )
		.def( "get_dimension", &CPyCumSumLayer::GetDimension, py::return_value_policy::reference )
		.def( "set_dimension", &CPyCumSumLayer::SetDimension, py::return_value_policy::reference )
		.def( "is_reverse", &CPyCumSumLayer::IsReverse, py::return_value_policy::reference )
		.def( "set_reverse", &CPyCumSumLayer::SetReverse, py::return_value_policy::reference )
	;
}
