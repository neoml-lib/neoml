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

#include "PyOnnxLayers.h"
#include "PyLayer.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransposeHelper.h>

class CPyOnnxTransposeHelper : public CPyLayer {
public:
	explicit CPyOnnxTransposeHelper( COnnxTransposeHelper& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLayer( layer, mathEngineOwner ) {}

	int GetFirstDim() const
	{
		TBlobDim firstDim;
		TBlobDim secondDim;
		Layer<COnnxTransposeHelper>()->GetDims( firstDim, secondDim );
		return firstDim;
	}

	int GetSecondDim() const
	{
		TBlobDim firstDim;
		TBlobDim secondDim;
		Layer<COnnxTransposeHelper>()->GetDims( firstDim, secondDim );
		return secondDim;
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "OnnxTranspose" );
		return pyConstructor( py::cast( this ) );
	}
};

class CPyOnnxTransformHelper : public CPyLayer {
public:
	explicit CPyOnnxTransformHelper( COnnxTransformHelper& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLayer( layer, mathEngineOwner ) {}

	py::array GetRules() const
	{
		py::array_t<int, py::array::c_style> result( py::ssize_t{ 7 } );
		NeoAssert( 7 == result.size() );
		auto temp = result.mutable_unchecked();
		for( int i = 0; i < 7; ++i ) {
			temp[i] = static_cast<int>( Layer<COnnxTransformHelper>()->GetRule( static_cast<TBlobDim>( i ) ) ); 
		}
		return result;
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "OnnxTransform" );
		return pyConstructor( py::cast( this ) );
	}
};

void InitializeOnnxLayers( py::module& m )
{
	py::class_<CPyOnnxTransposeHelper, CPyLayer>( m, "OnnxTranspose" )
		.def( py::init( []( const CPyLayer& layer )
		{
			return new CPyOnnxTransposeHelper( *layer.Layer<COnnxTransposeHelper>(), layer.MathEngineOwner() );
		} ) )
		.def( "get_first_dim", &CPyOnnxTransposeHelper::GetFirstDim, py::return_value_policy::reference )
		.def( "get_second_dim", &CPyOnnxTransposeHelper::GetSecondDim, py::return_value_policy::reference )
	;

	py::class_<CPyOnnxTransformHelper, CPyLayer>( m, "OnnxTransform" )
		.def( py::init( []( const CPyLayer& layer )
		{
			return new CPyOnnxTransformHelper( *layer.Layer<COnnxTransformHelper>(), layer.MathEngineOwner() );
		} ) )
		.def( "get_rules", &CPyOnnxTransformHelper::GetRules, py::return_value_policy::reference )
	;
}
