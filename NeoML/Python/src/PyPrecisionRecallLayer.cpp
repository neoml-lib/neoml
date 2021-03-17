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

#include "PyPrecisionRecallLayer.h"

class CPyPrecisionRecallLayer : public CPyLayer {
public:
	explicit CPyPrecisionRecallLayer( CPrecisionRecallLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetReset(bool reset) { Layer<CQualityControlLayer>()->SetReset(reset); }
	bool GetReset() const { return Layer<CQualityControlLayer>()->IsResetNeeded(); }

	py::list GetResult() const
	{
		CArray<int> result;
		Layer<CPrecisionRecallLayer>()->GetLastResult(result);
		py::list list;
		list.append(result[0]);
		list.append(result[1]);
		list.append(result[2]);
		list.append(result[3]);
		return list;
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "PrecisionRecall" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializePrecisionRecallLayer( py::module& m )
{
	py::class_<CPyPrecisionRecallLayer, CPyLayer>(m, "PrecisionRecall")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyPrecisionRecallLayer( *layer.Layer<CPrecisionRecallLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, const CPyLayer& layer2, int outputNumber1, int outputNumber2, bool reset ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CPrecisionRecallLayer> recall = new CPrecisionRecallLayer( mathEngine );
			recall->SetReset(reset);
			recall->SetName( FindFreeLayerName( dnn, "PrecisionRecall", name ).c_str() );
			dnn.AddLayer( *recall );
			recall->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			recall->Connect( 1, layer2.BaseLayer(), outputNumber2 );
			return new CPyPrecisionRecallLayer( *recall, layer1.MathEngineOwner() );
		}) )
		.def( "get_reset", &CPyPrecisionRecallLayer::GetReset, py::return_value_policy::reference )
		.def( "set_reset", &CPyPrecisionRecallLayer::SetReset, py::return_value_policy::reference )
		.def( "get_result", &CPyPrecisionRecallLayer::GetResult, py::return_value_policy::reference )
	;
}