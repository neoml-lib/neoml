/* Copyright © 2017-2024 ABBYY

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

#include "PyTransformLayer.h"

class CPyTransformLayer : public CPyLayer {
public:
	explicit CPyTransformLayer( CTransformLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetTransforms(py::array operations, py::array parameters)
	{
		const int* operationsPtr = reinterpret_cast<const int*>(operations.data());
		const int* parametersPtr = reinterpret_cast<const int*>(parameters.data());

		for( int i = 0; i < 7; i++ ) {
			CTransformLayer::TOperation operation = static_cast<CTransformLayer::TOperation>(operationsPtr[i]);
			Layer<CTransformLayer>()->SetDimensionRule(static_cast<TBlobDim>(i), operation, parametersPtr[i]);
		}
	}
	py::array GetOperations() const
	{
		py::array_t<int, py::array::c_style> result( py::ssize_t{ 7 } );
		NeoAssert( 7 == result.size() );
		auto temp = result.mutable_unchecked();
		for( int i = 0; i < 7; i++ ) {
			const CTransformLayer::CDimensionRule& rule = Layer<CTransformLayer>()->GetDimensionRule(static_cast<TBlobDim>(i));
			temp[i] = static_cast<int>(rule.Operation);
		}
		return result;
	}
	py::array GetParameters() const
	{
		py::array_t<int, py::array::c_style> result( py::ssize_t{ 7 } );
		NeoAssert( 7 == result.size() );
		auto temp = result.mutable_unchecked();
		for( int i = 0; i < 7; i++ ) {
			const CTransformLayer::CDimensionRule& rule = Layer<CTransformLayer>()->GetDimensionRule(static_cast<TBlobDim>(i));
			temp[i] = rule.Parameter;
		}
		return result;
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Transform" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeTransformLayer( py::module& m )
{
	py::class_<CPyTransformLayer, CPyLayer>(m, "Transform")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyTransformLayer( *layer.Layer<CTransformLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array operations, py::array parameters ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CTransformLayer> transform = new CTransformLayer( mathEngine );
			transform->SetName( FindFreeLayerName( dnn, "Transform", name ).c_str() );
			dnn.AddLayer( *transform );
			transform->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			CPyTransformLayer* result = new CPyTransformLayer( *transform, layer1.MathEngineOwner() );
			result->SetTransforms(operations, parameters);
			return result;
		}) )
		.def( "set_transforms", &CPyTransformLayer::SetTransforms, py::return_value_policy::reference )
		.def( "get_operations", &CPyTransformLayer::GetOperations, py::return_value_policy::reference )
		.def( "get_parameters", &CPyTransformLayer::GetParameters, py::return_value_policy::reference )
	;
}
