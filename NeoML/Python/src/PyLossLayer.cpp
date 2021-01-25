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

class CPyLossLayer : public CPyLayer {
public:
	explicit CPyLossLayer( CLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLayer( layer, mathEngineOwner ) {}

	float GetLastLoss() const { return Layer<CLossLayer>()->GetLastLoss(); }

	float GetLossWeight() const { return Layer<CLossLayer>()->GetLossWeight(); }
	void SetLossWeight( float lossWeight ) { Layer<CLossLayer>()->SetLossWeight(lossWeight); }

	bool GetTrainLabels() const { return Layer<CLossLayer>()->TrainLabels(); }
	void SetTrainLabels( bool toSet ) { Layer<CLossLayer>()->SetTrainLabels(toSet); }

	float GetMaxGradientValue() const { return Layer<CLossLayer>()->GetMaxGradientValue(); }
	void SetMaxGradientValue(float maxValue) { Layer<CLossLayer>()->SetMaxGradientValue(maxValue); }
};

//------------------------------------------------------------------------------------------------------------

class CPyCrossEntropyLossLayer : public CPyLossLayer {
public:
	explicit CPyCrossEntropyLossLayer( CCrossEntropyLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	void SetApplySoftmax( bool applySoftmax ) { Layer<CCrossEntropyLossLayer>()->SetApplySoftmax(applySoftmax); }
	bool GetApplySoftmax() const { return Layer<CCrossEntropyLossLayer>()->IsSoftmaxApplied(); }
};

//------------------------------------------------------------------------------------------------------------

class CPyBinaryCrossEntropyLossLayer : public CPyLossLayer {
public:
	explicit CPyBinaryCrossEntropyLossLayer( CBinaryCrossEntropyLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	void SetPositiveWeight( float value ) { Layer<CBinaryCrossEntropyLossLayer>()->SetPositiveWeight(value); }
	float GetPositiveWeight() const { return Layer<CBinaryCrossEntropyLossLayer>()->GetPositiveWeight(); }
};

void InitializeLossLayer( py::module& m )
{
	py::class_<CPyLossLayer, CPyLayer>(m, "Loss")
		.def( "get_last_loss", &CPyLossLayer::GetLastLoss, py::return_value_policy::reference )

		.def( "get_loss_weight", &CPyLossLayer::GetLossWeight, py::return_value_policy::reference )
		.def( "set_loss_weight", &CPyLossLayer::SetLossWeight, py::return_value_policy::reference )

		.def( "get_train_labels", &CPyLossLayer::GetTrainLabels, py::return_value_policy::reference )
		.def( "set_train_labels", &CPyLossLayer::SetTrainLabels, py::return_value_policy::reference )

		.def( "get_max_gradient", &CPyLossLayer::GetMaxGradientValue, py::return_value_policy::reference )
		.def( "set_max_gradient", &CPyLossLayer::SetMaxGradientValue, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyCrossEntropyLossLayer, CPyLossLayer>(m, "CrossEntropyLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyCrossEntropyLossLayer( *layer.Layer<CCrossEntropyLossLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs,
			bool softmax, float loss_weight )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CCrossEntropyLossLayer> loss = new CCrossEntropyLossLayer( mathEngine );
			loss->SetName( name == "" ? findFreeLayerName( dnn, "CrossEntropyLossLayer" ).c_str() : name.c_str() );
			loss->SetApplySoftmax( softmax );
			loss->SetLossWeight( loss_weight );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyCrossEntropyLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_apply_softmax", &CPyCrossEntropyLossLayer::GetApplySoftmax, py::return_value_policy::reference )
		.def( "set_apply_softmax", &CPyCrossEntropyLossLayer::SetApplySoftmax, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyBinaryCrossEntropyLossLayer, CPyLossLayer>(m, "BinaryCrossEntropyLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyBinaryCrossEntropyLossLayer( *layer.Layer<CBinaryCrossEntropyLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CBinaryCrossEntropyLossLayer> loss = new CBinaryCrossEntropyLossLayer( mathEngine );
			loss->SetName( name == "" ? findFreeLayerName( dnn, "BinaryCrossEntropyLossLayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyBinaryCrossEntropyLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_positive_weight", &CPyBinaryCrossEntropyLossLayer::GetPositiveWeight, py::return_value_policy::reference )
		.def( "set_positive_weight", &CPyBinaryCrossEntropyLossLayer::SetPositiveWeight, py::return_value_policy::reference )
	;
}
