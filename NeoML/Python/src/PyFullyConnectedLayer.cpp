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

#include "PyFullyConnectedLayer.h"

class CPyFullyConnectedLayer : public CPyLayer {
public:
	explicit CPyFullyConnectedLayer( CFullyConnectedLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}
	
	int GetElementCount() const { return Layer<CFullyConnectedLayer>()->GetNumberOfElements(); }
	void SetElementCount(int value) { Layer<CFullyConnectedLayer>()->SetNumberOfElements(value); }

	void ApplyBatchNormalization(const CPyLayer& layer)
	{
		Layer<CFullyConnectedLayer>()->ApplyBatchNormalization(*layer.Layer<CBatchNormalizationLayer>());
	}

	bool GetZeroFreeTerm() const { return Layer<CFullyConnectedLayer>()->IsZeroFreeTerm(); }
	void SetZeroFreeTerm(bool value) { Layer<CFullyConnectedLayer>()->SetZeroFreeTerm(value); }

	void SetWeights( const CPyBlob& blob ) { Layer<CFullyConnectedLayer>()->SetWeightsData( blob.Blob() ); }
	CPyBlob GetWeights() const { return CPyBlob( MathEngineOwner(), Layer<CFullyConnectedLayer>()->GetWeightsData() ); }

	void SetFreeTerm( const CPyBlob& blob ) { Layer<CFullyConnectedLayer>()->SetFreeTermData( blob.Blob() ); }
	CPyBlob GetFreeTerm() const { return CPyBlob( MathEngineOwner(), Layer<CFullyConnectedLayer>()->GetFreeTermData() ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "FullyConnected" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeFullyConnectedLayer( py::module& m )
{
	py::class_<CPyFullyConnectedLayer, CPyLayer>(m, "FullyConnected")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyFullyConnectedLayer( *layer.Layer<CFullyConnectedLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int numberOfElements, bool freeTerm ) {
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CFullyConnectedLayer> fc = new CFullyConnectedLayer( mathEngine );
			fc->SetName( FindFreeLayerName( dnn, "FullyConnected", name ).c_str() );
			fc->SetZeroFreeTerm( freeTerm );
			fc->SetNumberOfElements( numberOfElements );
			dnn.AddLayer( *fc );
			for( int i = 0; i < layers.size(); i++ ) {
				fc->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}
			return new CPyFullyConnectedLayer( *fc, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}))
		.def("apply_batch_normalization", &CPyFullyConnectedLayer::ApplyBatchNormalization, py::return_value_policy::reference)
		.def("get_element_count", &CPyFullyConnectedLayer::GetElementCount, py::return_value_policy::reference)
		.def("set_element_count", &CPyFullyConnectedLayer::SetElementCount, py::return_value_policy::reference)
		.def("get_zero_free_term", &CPyFullyConnectedLayer::GetZeroFreeTerm, py::return_value_policy::reference)
		.def("set_zero_free_term", &CPyFullyConnectedLayer::SetZeroFreeTerm, py::return_value_policy::reference)
		.def("get_weights", &CPyFullyConnectedLayer::GetWeights, py::return_value_policy::reference)
		.def("set_weights", &CPyFullyConnectedLayer::SetWeights, py::return_value_policy::reference)
		.def("get_free_term", &CPyFullyConnectedLayer::GetFreeTerm, py::return_value_policy::reference)
		.def("set_free_term", &CPyFullyConnectedLayer::SetFreeTerm, py::return_value_policy::reference)
	;
}
