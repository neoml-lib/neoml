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

#include "PyCrfLayer.h"

class CPyCrfLayer : public CPyLayer {
public:
	explicit CPyCrfLayer( CCrfLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetNumberOfClasses(int value) { Layer<CCrfLayer>()->SetNumberOfClasses(value); }
	int GetNumberOfClasses() const { return Layer<CCrfLayer>()->GetNumberOfClasses(); }

	void SetPaddingClass(int value) { Layer<CCrfLayer>()->SetPaddingClass(value); }
	int GetPaddingClass() const { return Layer<CCrfLayer>()->GetPaddingClass(); }

	void SetDropoutRate(float value) { Layer<CCrfLayer>()->SetDropoutRate(value); }
	float GetDropoutRate() const { return Layer<CCrfLayer>()->GetDropoutRate(); }

	void SetBestPrevClassEnabled(bool value) { Layer<CCrfLayer>()->SetBestPrevClassEnabled(value); }
	bool GetBestPrevClassEnabled() const { return Layer<CCrfLayer>()->GetBestPrevClassEnabled(); }

	CPyBlob GetHiddenWeights() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CCrfLayer>()->GetHiddenWeights() );
	}
	CPyBlob GetFreeTerms() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CCrfLayer>()->GetFreeTerms() );
	}
	CPyBlob GetTransitions() const
	{
		return CPyBlob( MathEngineOwner(), Layer<CCrfLayer>()->GetTransitions() );
	}
	void SetHiddenWeights( const CPyBlob& blob )
	{
		Layer<CCrfLayer>()->SetHiddenWeights( blob.Blob() );
	}
	void SetFreeTerms( const CPyBlob& blob )
	{
		Layer<CCrfLayer>()->SetFreeTerms( blob.Blob() );
	}
	void SetTransitions( const CPyBlob& blob )
	{
		Layer<CCrfLayer>()->SetTransitions( blob.Blob() );
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Crf" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPyCrfLossLayer : public CPyLayer {
public:
	explicit CPyCrfLossLayer( CCrfLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	float GetLastLoss() const { return Layer<CCrfLossLayer>()->GetLastLoss(); }

	void SetLossWeight(float value) { Layer<CCrfLossLayer>()->SetLossWeight(value); }
	float GetLossWeight() const { return Layer<CCrfLossLayer>()->GetLossWeight(); }

	void SetMaxGradient(float value) { Layer<CCrfLossLayer>()->SetMaxGradientValue(value); }
	float GetMaxGradient() const { return Layer<CCrfLossLayer>()->GetMaxGradientValue(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "CrfLoss" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyBestSequenceLayer : public CPyLayer {
public:
	explicit CPyBestSequenceLayer(CBestSequenceLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "BestSequence" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeCrfLayer( py::module& m )
{
	py::class_<CPyCrfLayer, CPyLayer>(m, "Crf")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyCrfLayer( *layer.Layer<CCrfLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, const CPyLayer& layer2, int outputNumber1, int outputNumber2,
			int classCount, int padding, float dropoutRate )
		{
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CCrfLayer> crf = new CCrfLayer( mathEngine );
			crf->SetNumberOfClasses(classCount);
			crf->SetPaddingClass(padding);
			crf->SetDropoutRate(dropoutRate);
			crf->SetName( FindFreeLayerName( dnn, "Crf", name ).c_str() );
			dnn.AddLayer( *crf );
			crf->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			crf->Connect( 1, layer2.BaseLayer(), outputNumber2 );
			return new CPyCrfLayer( *crf, layer1.MathEngineOwner() );
		}) )
		.def( "get_class_count", &CPyCrfLayer::GetNumberOfClasses, py::return_value_policy::reference )
		.def( "set_class_count", &CPyCrfLayer::SetNumberOfClasses, py::return_value_policy::reference )
		.def( "get_padding", &CPyCrfLayer::GetPaddingClass, py::return_value_policy::reference )
		.def( "set_padding", &CPyCrfLayer::SetPaddingClass, py::return_value_policy::reference )
		.def( "get_dropout_rate", &CPyCrfLayer::GetDropoutRate, py::return_value_policy::reference )
		.def( "set_dropout_rate", &CPyCrfLayer::SetDropoutRate, py::return_value_policy::reference )
		.def( "get_calc_best_prev_class", &CPyCrfLayer::GetBestPrevClassEnabled, py::return_value_policy::reference )
		.def( "set_calc_best_prev_class", &CPyCrfLayer::SetBestPrevClassEnabled, py::return_value_policy::reference )

		.def( "get_hidden_weights", &CPyCrfLayer::GetHiddenWeights, py::return_value_policy::reference )
		.def( "set_hidden_weights", &CPyCrfLayer::SetHiddenWeights, py::return_value_policy::reference )
		.def( "get_free_terms", &CPyCrfLayer::GetFreeTerms, py::return_value_policy::reference )
		.def( "set_free_terms", &CPyCrfLayer::SetFreeTerms, py::return_value_policy::reference )
		.def( "get_transitions", &CPyCrfLayer::GetTransitions, py::return_value_policy::reference )
		.def( "set_transitions", &CPyCrfLayer::SetTransitions, py::return_value_policy::reference )
	;

	py::class_<CPyCrfLossLayer, CPyLayer>(m, "CrfLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyCrfLossLayer( *layer.Layer<CCrfLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float loss_weight )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CCrfLossLayer> loss = new CCrfLossLayer( mathEngine );
			loss->SetLossWeight( loss_weight );
			loss->SetName( FindFreeLayerName( dnn, "CrfLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );

			return CPyCrfLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_last_loss", &CPyCrfLossLayer::GetLastLoss, py::return_value_policy::reference )
		.def( "get_loss_weight", &CPyCrfLossLayer::GetLossWeight, py::return_value_policy::reference )
		.def( "set_loss_weight", &CPyCrfLossLayer::SetLossWeight, py::return_value_policy::reference )
		.def( "get_max_gradient", &CPyCrfLossLayer::GetMaxGradient, py::return_value_policy::reference )
		.def( "set_max_gradient", &CPyCrfLossLayer::SetMaxGradient, py::return_value_policy::reference )
	;

	py::class_<CPyBestSequenceLayer, CPyLayer>(m, "BestSequence")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyBestSequenceLayer( *layer.Layer<CBestSequenceLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CBestSequenceLayer> best = new CBestSequenceLayer( mathEngine );
			best->SetName( FindFreeLayerName( dnn, "BestSequence", name ).c_str() );
			dnn.AddLayer( *best );
			best->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			best->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );

			return CPyBestSequenceLayer( *best, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;
}
