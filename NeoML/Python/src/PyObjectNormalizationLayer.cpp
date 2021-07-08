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

#include "PyObjectNormalizationLayer.h"

class CPyObjectNormalizationLayer : public CPyLayer {
public:
	explicit CPyObjectNormalizationLayer( CObjectNormalizationLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetEpsilon(float value) { Layer<CObjectNormalizationLayer>()->SetEpsilon(value); }
	float GetEpsilon() const { return Layer<CObjectNormalizationLayer>()->GetEpsilon(); }

	CPyBlob GetScale() const { return CPyBlob( MathEngineOwner(), Layer<CObjectNormalizationLayer>()->GetScale() ); }
	CPyBlob GetBias() const { return CPyBlob( MathEngineOwner(), Layer<CObjectNormalizationLayer>()->GetBias() ); }

	void SetScale( const CPyBlob& blob ) { Layer<CObjectNormalizationLayer>()->SetScale( blob.Blob() ); }
	void SetBias( const CPyBlob& blob ) { Layer<CObjectNormalizationLayer>()->SetBias( blob.Blob() ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ObjectNormalization" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeObjectNormalizationLayer( py::module& m )
{
	py::class_<CPyObjectNormalizationLayer, CPyLayer>(m, "ObjectNormalization")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyObjectNormalizationLayer( *layer.Layer<CObjectNormalizationLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, float epsilon ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CObjectNormalizationLayer> norm = new CObjectNormalizationLayer( mathEngine );
			norm->SetEpsilon(epsilon);
			norm->SetName( FindFreeLayerName( dnn, "ObjectNormalization", name ).c_str() );
			dnn.AddLayer( *norm );
			norm->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPyObjectNormalizationLayer( *norm, layer1.MathEngineOwner() );
		}) )
		.def( "get_epsilon", &CPyObjectNormalizationLayer::GetEpsilon, py::return_value_policy::reference )
		.def( "set_epsilon", &CPyObjectNormalizationLayer::SetEpsilon, py::return_value_policy::reference )
		.def( "get_bias", &CPyObjectNormalizationLayer::GetBias, py::return_value_policy::reference )
		.def( "set_bias", &CPyObjectNormalizationLayer::SetBias, py::return_value_policy::reference )
		.def( "get_scale", &CPyObjectNormalizationLayer::GetScale, py::return_value_policy::reference )
		.def( "set_scale", &CPyObjectNormalizationLayer::SetScale, py::return_value_policy::reference )
	;
}
