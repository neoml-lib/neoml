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

#include "PyMultiheadAttentionLayer.h"

class CPyMultiheadAttentionLayer : public CPyLayer {
public:
	explicit CPyMultiheadAttentionLayer( CMultiheadAttentionLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetHeadCount(int value) { Layer<CMultiheadAttentionLayer>()->SetHeadCount(value); }
	int GetHeadCount() const { return Layer<CMultiheadAttentionLayer>()->GetHeadCount(); }

	void SetHiddenSize(int value) { Layer<CMultiheadAttentionLayer>()->SetHiddenSize(value); }
	int GetHiddenSize() const { return Layer<CMultiheadAttentionLayer>()->GetHiddenSize(); }

	void SetDropoutRate(float value) { Layer<CMultiheadAttentionLayer>()->SetDropoutRate(value); }
	float GetDropoutRate() const { return Layer<CMultiheadAttentionLayer>()->GetDropoutRate(); }

	void SetUseMask(bool value) { Layer<CMultiheadAttentionLayer>()->SetUseMask(value); }
	bool GetUseMask() const { return Layer<CMultiheadAttentionLayer>()->GetUseMask(); }

	void SetOutputSize(int value) { Layer<CMultiheadAttentionLayer>()->SetOutputSize(value); }
	int GetOutputSize() const { return Layer<CMultiheadAttentionLayer>()->GetOutputSize(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MultiheadAttention" );
		return pyConstructor( py::cast(this), 0, 0, 0, 0 );
	}
};

void InitializeMultiheadAttentionLayer( py::module& m )
{
	py::class_<CPyMultiheadAttentionLayer, CPyLayer>(m, "MultiheadAttention")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyMultiheadAttentionLayer( *layer.Layer<CMultiheadAttentionLayer>(), layer.MathEngineOwner() );
		}))
		.def(py::init([](const std::string& name, const py::list& layers, const py::list& outputs, int head_count, int hiddenSize,
			int outputSize, float dropoutRate)
		{
				py::gil_scoped_release release;
				CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
				IMathEngine& mathEngine = dnn.GetMathEngine();

				CPtr<CMultiheadAttentionLayer> attention = new CMultiheadAttentionLayer(mathEngine);
				attention->SetHeadCount(head_count);
				attention->SetHiddenSize(hiddenSize);
				attention->SetOutputSize(outputSize);
				attention->SetDropoutRate(dropoutRate);
				attention->SetName( FindFreeLayerName(dnn, "MultiheadAttention", name).c_str() );
				dnn.AddLayer(*attention);
				attention->Connect(0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>());
				attention->Connect(1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>());
				attention->Connect(2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>());
				if( layers.size() == 4 ) {
					attention->Connect(3, layers[3].cast<CPyLayer>().BaseLayer(), outputs[3].cast<int>());
				}

				return CPyMultiheadAttentionLayer(*attention, layers[0].cast<CPyLayer>().MathEngineOwner());
		}))
		.def("set_head_count", &CPyMultiheadAttentionLayer::SetHeadCount, py::return_value_policy::reference)
		.def("get_head_count", &CPyMultiheadAttentionLayer::GetHeadCount, py::return_value_policy::reference)
		.def("set_hidden_size", &CPyMultiheadAttentionLayer::SetHiddenSize, py::return_value_policy::reference)
		.def("get_hidden_size", &CPyMultiheadAttentionLayer::GetHiddenSize, py::return_value_policy::reference)
		.def("set_dropout_rate", &CPyMultiheadAttentionLayer::SetDropoutRate, py::return_value_policy::reference)
		.def("get_dropout_rate", &CPyMultiheadAttentionLayer::GetDropoutRate, py::return_value_policy::reference)
		.def("set_use_mask", &CPyMultiheadAttentionLayer::SetUseMask, py::return_value_policy::reference)
		.def("get_use_mask", &CPyMultiheadAttentionLayer::GetUseMask, py::return_value_policy::reference)
		.def("set_output_size", &CPyMultiheadAttentionLayer::SetOutputSize, py::return_value_policy::reference)
		.def("get_output_size", &CPyMultiheadAttentionLayer::GetOutputSize, py::return_value_policy::reference)
	;
}
