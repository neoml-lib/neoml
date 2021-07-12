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

#include "PyAccuracyLayer.h"

class CPyAccuracyLayer : public CPyLayer {
public:
	explicit CPyAccuracyLayer( CAccuracyLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetReset(bool reset) { Layer<CQualityControlLayer>()->SetReset(reset); }
	bool GetReset() const { return Layer<CQualityControlLayer>()->IsResetNeeded(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Accuracy" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyConfusionMatrixLayer : public CPyLayer {
public:
	explicit CPyConfusionMatrixLayer( CConfusionMatrixLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetReset(bool reset) { Layer<CQualityControlLayer>()->SetReset(reset); }
	bool GetReset() const { return Layer<CQualityControlLayer>()->IsResetNeeded(); }

	py::array GetMatrix() const
	{
		const CVariableMatrix<float>& matrix = Layer<CConfusionMatrixLayer>()->GetMatrix();

		py::array_t<float, py::array::c_style> totalResult( { matrix.SizeY(), matrix.SizeX() } );

		auto r = totalResult.mutable_unchecked<2>();
		for( int i = 0; i < matrix.SizeY(); i++ ) {
			for( int j = 0; j < matrix.SizeX(); j++ ) {
				r(i, j) = matrix(i, j);
			}
		}

		return totalResult;
	}

	void ResetMatrix() { Layer<CConfusionMatrixLayer>()->ResetMatrix(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ConfusionMatrix" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeAccuracyLayer( py::module& m )
{
	py::class_<CPyAccuracyLayer, CPyLayer>(m, "Accuracy")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyAccuracyLayer( *layer.Layer<CAccuracyLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, const CPyLayer& layer2, int outputNumber1, int outputNumber2, bool reset ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CAccuracyLayer> accuracy = new CAccuracyLayer( mathEngine );
			accuracy->SetReset( reset );
			accuracy->SetName( FindFreeLayerName( dnn, "Accuracy", name ).c_str() );
			dnn.AddLayer( *accuracy );
			accuracy->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			accuracy->Connect( 1, layer2.BaseLayer(), outputNumber2 );
			return new CPyAccuracyLayer( *accuracy, layer1.MathEngineOwner() );
		}) )
		.def( "get_reset", &CPyAccuracyLayer::GetReset, py::return_value_policy::reference )
		.def( "set_reset", &CPyAccuracyLayer::SetReset, py::return_value_policy::reference )
	;

	py::class_<CPyConfusionMatrixLayer, CPyLayer>(m, "ConfusionMatrix")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyConfusionMatrixLayer( *layer.Layer<CConfusionMatrixLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, const CPyLayer& layer2, int outputNumber1, int outputNumber2, bool reset ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CConfusionMatrixLayer> confusion = new CConfusionMatrixLayer( mathEngine );
			confusion->SetReset( reset );
			confusion->SetName( FindFreeLayerName( dnn, "ConfusionMatrix", name ).c_str() );
			dnn.AddLayer( *confusion );
			confusion->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			confusion->Connect( 1, layer2.BaseLayer(), outputNumber2 );
			return new CPyConfusionMatrixLayer( *confusion, layer1.MathEngineOwner() );
		}) )
		.def( "get_reset", &CPyConfusionMatrixLayer::GetReset, py::return_value_policy::reference )
		.def( "set_reset", &CPyConfusionMatrixLayer::SetReset, py::return_value_policy::reference )
		.def( "get_matrix", &CPyConfusionMatrixLayer::GetMatrix, py::return_value_policy::reference )
		.def( "reset_matrix", &CPyConfusionMatrixLayer::ResetMatrix, py::return_value_policy::reference )
	;
}
