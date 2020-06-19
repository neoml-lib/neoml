/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Layers/LossLayer.h>

namespace NeoML {

// The layer calculates a focal loss function for binary classification.
// Focal loss is a modification of cross-entropy that reduces penalty for easily separable objects, 
// which helps concentrate on separating similar objects from different classes
// It accepts a network output r and labels y = -1; 1. 
// Calculated according to the formula: -(1-p_t)^gamma * log(p_t), p_t=sigma(y*r)
// See the paper https://arxiv.org/pdf/1708.02002.pdf
// Parameters: gamma > 0.
class NEOML_API CBinaryFocalLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CBinaryFocalLossLayer )
public:
	explicit CBinaryFocalLossLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	static const float DefaultFocalForceValue;

	// This parameter controls the degree to which the algorithm focuses 
	// on the similar objects (>0, the greater it is the stronger the focus)
	// The paper referred to calls this parameter gamma
	float GetFocalForce() const { return focalForce->GetData().GetValue(); }
	void SetFocalForce( float value );

protected:
	virtual void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data,
		int vectorSize, CConstFloatHandle label, int labelSize, CFloatHandle lossValue,
		CFloatHandle lossGradient ) override;

private:
	// gamma parameter from the referred paper
	// Controls the degree to which the algorithm focuses on objects difficult to classify
	CPtr<CDnnBlob> focalForce;

	void calculateGradient( CFloatHandle onesVector, CFloatHandle entropyValues, CFloatHandle sigmoidVector,
		CFloatHandle sigmoidMinusOneVector, CConstFloatHandle labels, int batchSize, CFloatHandle lossGradient );
};

} // namespace NeoML
