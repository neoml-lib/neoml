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

#include "PyPositionalEmbeddingLayer.h"

class CPyPositionalEmbeddingLayer : public CPyLayer {
public:
	explicit CPyPositionalEmbeddingLayer( CPositionalEmbeddingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetType(int value) { Layer<CPositionalEmbeddingLayer>()->SetType(static_cast<CPositionalEmbeddingLayer::TPositionalEmbeddingType>(value)); }
	int GetType() const { return static_cast<int>( Layer<CPositionalEmbeddingLayer>()->GetType() ); }

	CPyBlob GetAddends() const { return CPyBlob( MathEngineOwner(), Layer<CPositionalEmbeddingLayer>()->GetAddends() ); }
	void SetAddends( const CPyBlob& blob ) { Layer<CPositionalEmbeddingLayer>()->SetAddends( blob.Blob(), true ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "PositionalEmbedding" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializePositionalEmbeddingLayer( py::module& m )
{
	py::class_<CPyPositionalEmbeddingLayer, CPyLayer>(m, "PositionalEmbedding")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyPositionalEmbeddingLayer( *layer.Layer<CPositionalEmbeddingLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, int typeIndex ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CPositionalEmbeddingLayer> pos = new CPositionalEmbeddingLayer( mathEngine );
			pos->SetType(static_cast<CPositionalEmbeddingLayer::TPositionalEmbeddingType>(typeIndex));
			pos->SetName( FindFreeLayerName( dnn, "PositionalEmbedding", name ).c_str() );
			dnn.AddLayer( *pos );
			pos->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPyPositionalEmbeddingLayer( *pos, layer1.MathEngineOwner() );
		}) )
		.def( "get_type", &CPyPositionalEmbeddingLayer::GetType, py::return_value_policy::reference )
		.def( "set_type", &CPyPositionalEmbeddingLayer::SetType, py::return_value_policy::reference )
		.def( "get_addends", &CPyPositionalEmbeddingLayer::GetAddends, py::return_value_policy::reference )
		.def( "set_addends", &CPyPositionalEmbeddingLayer::SetAddends, py::return_value_policy::reference )
	;
}