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

#pragma once

#include "PyLayer.h"

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

void InitializeLossLayer( py::module& m );