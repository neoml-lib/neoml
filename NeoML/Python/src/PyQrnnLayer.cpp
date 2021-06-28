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

#include "PyQrnnLayer.h"

class CPyQrnnLayer : public CPyLayer {
public:
	explicit CPyQrnnLayer( CQrnnLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetHiddenSize(int value) { Layer<CQrnnLayer>()->SetHiddenSize(value); }
	int GetHiddenSize() const { return Layer<CQrnnLayer>()->GetHiddenSize(); }

	void SetWindowSize(int value) { Layer<CQrnnLayer>()->SetWindowSize(value); }
	int GetWindowSize() const { return Layer<CQrnnLayer>()->GetWindowSize(); }

	void SetStride(int value) { Layer<CQrnnLayer>()->SetStride(value); }
	int GetStride() const { return Layer<CQrnnLayer>()->GetStride(); }

	int GetPaddingFront() const { return Layer<CQrnnLayer>()->GetPaddingFront(); }
	void SetPaddingFront( int value ) { Layer<CQrnnLayer>()->SetPaddingFront(value); }
	int GetPaddingBack() const { return Layer<CQrnnLayer>()->GetPaddingBack(); }
	void SetPaddingBack( int value ) { Layer<CQrnnLayer>()->SetPaddingBack(value); }

	void SetActivation(int value) { Layer<CQrnnLayer>()->SetActivation(static_cast<TActivationFunction>(value)); }
	int GetActivation() const { return static_cast<int>(Layer<CQrnnLayer>()->GetActivation()); }

	void SetDropout(float value) { Layer<CQrnnLayer>()->SetDropout(value); }
	float GetDropout() const { return Layer<CQrnnLayer>()->GetDropout(); }

	void SetFilter( const CPyBlob& blob ) { Layer<CQrnnLayer>()->SetFilterData( blob.Blob() ); }
	CPyBlob GetFilter() const { return CPyBlob( MathEngineOwner(), Layer<CQrnnLayer>()->GetFilterData() ); }

	void SetFreeTerm( const CPyBlob& blob ) { Layer<CQrnnLayer>()->SetFreeTermData( blob.Blob() ); }
	CPyBlob GetFreeTerm() const { return CPyBlob( MathEngineOwner(), Layer<CQrnnLayer>()->GetFreeTermData() ); }

	void SetRecurrentMode(int value) { Layer<CQrnnLayer>()->SetRecurrentMode(static_cast<CQrnnLayer::TRecurrentMode>(value)); }
	int GetRecurrentMode() const { return static_cast<int>(Layer<CQrnnLayer>()->GetRecurrentMode()); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Qrnn" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

void InitializeQrnnLayer( py::module& m )
{
	py::class_<CPyQrnnLayer, CPyLayer>(m, "Qrnn")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyQrnnLayer( *layer.Layer<CQrnnLayer>(), layer.MathEngineOwner() );
			}))
		.def(py::init([](const std::string& name, const py::list& inputs, int pooling, int hiddenSize, int windowSize, int stride,
			int paddingFront, int paddingBack, int activation, float dropoutRate, int mode, const py::list& input_outputs)
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CQrnnLayer> qrnn = new CQrnnLayer(mathEngine);
			qrnn->SetPoolingType(static_cast<CQrnnLayer::TPoolingType>(pooling));
			qrnn->SetHiddenSize(hiddenSize);
			qrnn->SetWindowSize(windowSize);
			qrnn->SetStride(stride);
			qrnn->SetPaddingFront(paddingFront);
			qrnn->SetPaddingBack(paddingBack);
			qrnn->SetActivation(static_cast<TActivationFunction>(activation));
			qrnn->SetDropout(dropoutRate);
			qrnn->SetRecurrentMode(static_cast<CQrnnLayer::TRecurrentMode>(mode));
			qrnn->SetName( FindFreeLayerName( dnn, "Qrnn", name ).c_str() );
			dnn.AddLayer( *qrnn );

			for( int i = 0; i < inputs.size(); i++ ) {
				qrnn->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyQrnnLayer( *qrnn, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_hidden_size", &CPyQrnnLayer::GetHiddenSize, py::return_value_policy::reference )
		.def( "set_hidden_size", &CPyQrnnLayer::SetHiddenSize, py::return_value_policy::reference )
		.def( "get_window_size", &CPyQrnnLayer::GetWindowSize, py::return_value_policy::reference )
		.def( "set_window_size", &CPyQrnnLayer::SetWindowSize, py::return_value_policy::reference )
		.def( "get_stride", &CPyQrnnLayer::GetStride, py::return_value_policy::reference )
		.def( "set_stride", &CPyQrnnLayer::SetStride, py::return_value_policy::reference )
		.def( "get_padding_front", &CPyQrnnLayer::GetPaddingFront, py::return_value_policy::reference )
		.def( "set_padding_front", &CPyQrnnLayer::SetPaddingFront, py::return_value_policy::reference )
		.def( "get_padding_back", &CPyQrnnLayer::GetPaddingBack, py::return_value_policy::reference )
		.def( "set_padding_back", &CPyQrnnLayer::SetPaddingBack, py::return_value_policy::reference )
		.def( "get_activation", &CPyQrnnLayer::GetActivation, py::return_value_policy::reference )
		.def( "set_activation", &CPyQrnnLayer::SetActivation, py::return_value_policy::reference )
		.def( "get_dropout", &CPyQrnnLayer::GetDropout, py::return_value_policy::reference )
		.def( "set_dropout", &CPyQrnnLayer::SetDropout, py::return_value_policy::reference )
		.def( "get_filter", &CPyQrnnLayer::GetFilter, py::return_value_policy::reference )
		.def( "set_filter", &CPyQrnnLayer::SetFilter, py::return_value_policy::reference )
		.def( "get_free_term", &CPyQrnnLayer::GetFreeTerm, py::return_value_policy::reference )
		.def( "set_free_term", &CPyQrnnLayer::SetFreeTerm, py::return_value_policy::reference )
		.def( "get_recurrent_mode", &CPyQrnnLayer::GetRecurrentMode, py::return_value_policy::reference )
		.def( "set_recurrent_mode", &CPyQrnnLayer::SetRecurrentMode, py::return_value_policy::reference )
	;
}