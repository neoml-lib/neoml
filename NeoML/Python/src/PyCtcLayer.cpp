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

#include "PyLossLayer.h"

class CPyCtcLossLayer : public CPyLayer {
public:
	explicit CPyCtcLossLayer(CCtcLossLayer& layer, CPyMathEngineOwner& mathEngineOwner) :
		CPyLayer(layer, mathEngineOwner) {}

	int GetBlankLabel() const { return Layer<CCtcLossLayer>()->GetBlankLabel(); }
	void SetBlankLabel(int toSet) { Layer<CCtcLossLayer>()->SetBlankLabel(toSet); }

	float GetLossWeight() const { return Layer<CCtcLossLayer>()->GetLossWeight(); }
	void SetLossWeight(float lossWeight) { Layer<CCtcLossLayer>()->SetLossWeight(lossWeight); }

	float GetLastLoss() const { return Layer<CCtcLossLayer>()->GetLastLoss(); }

	float GetMaxGradientValue() const { return Layer<CCtcLossLayer>()->GetMaxGradientValue(); }
	void SetMaxGradientValue(float maxValue) { Layer<CCtcLossLayer>()->SetMaxGradientValue(maxValue); }

	bool GetAllowBlankLabelSkips() const { return Layer<CCtcLossLayer>()->GetAllowBlankLabelSkips(); }
	void SetAllowBlankLabelSkips(bool value) { Layer<CCtcLossLayer>()->SetAllowBlankLabelSkips(value); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "CtcLoss" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyCtcDecodingLayer : public CPyLayer {
public:
	explicit CPyCtcDecodingLayer(CCtcDecodingLayer& layer, CPyMathEngineOwner& mathEngineOwner) :
		CPyLayer(layer, mathEngineOwner) {}

	int GetBlankLabel() const { return Layer<CCtcDecodingLayer>()->GetBlankLabel(); }
	void SetBlankLabel(int toSet) { Layer<CCtcDecodingLayer>()->SetBlankLabel(toSet); }

	float GetBlankProbabilityThreshold() const { return Layer<CCtcDecodingLayer>()->GetBlankProbabilityThreshold(); }
	void SetBlankProbabilityThreshold(float threshold) { Layer<CCtcDecodingLayer>()->SetBlankProbabilityThreshold(threshold); }

	float GetArcProbabilityThreshold() const { return Layer<CCtcDecodingLayer>()->GetArcProbabilityThreshold(); }
	void SetArcProbabilityThreshold(float threshold) { Layer<CCtcDecodingLayer>()->SetArcProbabilityThreshold(threshold); }

	int GetSequenceLength() const { return Layer<CCtcDecodingLayer>()->GetSequenceLength(); }

	int GetBatchWidth() const { return Layer<CCtcDecodingLayer>()->GetBatchWidth(); }

	int GetLabelsCount() const { return Layer<CCtcDecodingLayer>()->GetLabelsCount(); }

	py::list GetBestSequence( int sequenceNumber ) const
	{
		CArray<int> res;
		Layer<CCtcDecodingLayer>()->GetBestSequence(sequenceNumber, res);

		py::list list;
		for(int i = 0; i < res.Size(); i++) {
			list.append(res[i]);
		}
		return list;
	}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "CtcDecoding" );
		return pyConstructor( py::cast(this), 0, 0, 0 );
	}
};

void InitializeCtcLayer( py::module& m )
{
	py::class_<CPyCtcLossLayer, CPyLayer>(m, "CtcLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyCtcLossLayer( *layer.Layer<CCtcLossLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs,
			int blank, bool skip, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CCtcLossLayer> loss = new CCtcLossLayer( mathEngine );
			loss->SetBlankLabel( blank );
			loss->SetAllowBlankLabelSkips( skip );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "CtcLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			for( int i = 0; i < layers.size(); i++ ) {
				loss->Connect(i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>());
			}
			return CPyCtcLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_blank_label", &CPyCtcLossLayer::GetBlankLabel, py::return_value_policy::reference )
		.def( "set_blank_label", &CPyCtcLossLayer::SetBlankLabel, py::return_value_policy::reference )
		.def( "get_loss_weight", &CPyCtcLossLayer::GetLossWeight, py::return_value_policy::reference )
		.def( "set_loss_weight", &CPyCtcLossLayer::SetLossWeight, py::return_value_policy::reference )
		.def( "get_last_loss", &CPyCtcLossLayer::GetLastLoss, py::return_value_policy::reference )
		.def( "get_max_gradient", &CPyCtcLossLayer::GetMaxGradientValue, py::return_value_policy::reference )
		.def( "set_max_gradient", &CPyCtcLossLayer::SetMaxGradientValue, py::return_value_policy::reference )
		.def( "get_skip", &CPyCtcLossLayer::GetAllowBlankLabelSkips, py::return_value_policy::reference )
		.def( "set_skip", &CPyCtcLossLayer::SetAllowBlankLabelSkips, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyCtcDecodingLayer, CPyLayer>(m, "CtcDecoding")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyCtcDecodingLayer( *layer.Layer<CCtcDecodingLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs,
			int blank, float blankThreshold, float arcThreshold )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CCtcDecodingLayer> loss = new CCtcDecodingLayer( mathEngine );
			loss->SetBlankLabel( blank );
			loss->SetBlankProbabilityThreshold( blankThreshold );
			loss->SetArcProbabilityThreshold( arcThreshold );
			loss->SetName( FindFreeLayerName( dnn, "CtcDecoding", name ).c_str() );
			dnn.AddLayer( *loss );
			for(int i = 0; i < layers.size(); i++) {
				loss->Connect(i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>());
			}
			return CPyCtcDecodingLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_blank_label", &CPyCtcDecodingLayer::GetBlankLabel, py::return_value_policy::reference )
		.def( "set_blank_label", &CPyCtcDecodingLayer::SetBlankLabel, py::return_value_policy::reference )
		.def( "get_blank_threshold", &CPyCtcDecodingLayer::GetBlankProbabilityThreshold, py::return_value_policy::reference )
		.def( "set_blank_threshold", &CPyCtcDecodingLayer::SetBlankProbabilityThreshold, py::return_value_policy::reference )
		.def( "get_arc_threshold", &CPyCtcDecodingLayer::GetArcProbabilityThreshold, py::return_value_policy::reference )
		.def( "set_arc_threshold", &CPyCtcDecodingLayer::SetArcProbabilityThreshold, py::return_value_policy::reference )
		.def( "get_sequence_length", &CPyCtcDecodingLayer::GetSequenceLength, py::return_value_policy::reference )
		.def( "get_batch_width", &CPyCtcDecodingLayer::GetBatchWidth, py::return_value_policy::reference )
		.def( "get_label_count", &CPyCtcDecodingLayer::GetLabelsCount, py::return_value_policy::reference )
		.def( "get_best_sequence", &CPyCtcDecodingLayer::GetBestSequence, py::return_value_policy::reference )
	;
}
