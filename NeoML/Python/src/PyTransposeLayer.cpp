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

#include "PyTransposeLayer.h"

class CPyTransposeLayer : public CPyLayer {
public:
	explicit CPyTransposeLayer( CTransposeLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetFirstDim(int value)
	{
		TBlobDim dim1;
		TBlobDim dim2;
		Layer<CTransposeLayer>()->GetTransposedDimensions(dim1, dim2);
		dim1 = static_cast<TBlobDim>(value);
		Layer<CTransposeLayer>()->SetTransposedDimensions(dim1, dim2);
	}

	int GetFirstDim() const
	{
		TBlobDim dim1;
		TBlobDim dim2;
		Layer<CTransposeLayer>()->GetTransposedDimensions(dim1, dim2);
		return static_cast<int>(dim1);
	}

	void SetSecondDim(int value)
	{
		TBlobDim dim1;
		TBlobDim dim2;
		Layer<CTransposeLayer>()->GetTransposedDimensions(dim1, dim2);
		dim2 = static_cast<TBlobDim>(value);
		Layer<CTransposeLayer>()->SetTransposedDimensions(dim1, dim2);
	}

	int GetSecondDim() const
	{
		TBlobDim dim1;
		TBlobDim dim2;
		Layer<CTransposeLayer>()->GetTransposedDimensions(dim1, dim2);
		return static_cast<int>(dim2);
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Transpose" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeTransposeLayer( py::module& m )
{
	py::class_<CPyTransposeLayer, CPyLayer>(m, "Transpose")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyTransposeLayer( *layer.Layer<CTransposeLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, int firstIndex, int secondIndex ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CTransposeLayer> transpose = new CTransposeLayer( mathEngine );
			transpose->SetTransposedDimensions(static_cast<TBlobDim>(firstIndex), static_cast<TBlobDim>(secondIndex));
			transpose->SetName( name == "" ? findFreeLayerName( dnn, "Transpose" ).c_str() : name.c_str() );
			dnn.AddLayer( *transpose );
			transpose->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPyTransposeLayer( *transpose, layer1.MathEngineOwner() );
		}) )
		.def( "get_first_dim", &CPyTransposeLayer::GetFirstDim, py::return_value_policy::reference )
		.def( "set_first_dim", &CPyTransposeLayer::SetFirstDim, py::return_value_policy::reference )
		.def( "get_second_dim", &CPyTransposeLayer::GetSecondDim, py::return_value_policy::reference )
		.def( "set_second_dim", &CPyTransposeLayer::SetSecondDim, py::return_value_policy::reference )
	;
}