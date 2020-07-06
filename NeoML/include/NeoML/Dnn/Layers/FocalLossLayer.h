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

// The layer calculates a focal loss function for a classification scenario with multiple classes.
// The focal loss function is a modified version of cross-entropy loss function 
// in which the objects that are easily distinguished receive smaller penalties. 
// This helps focus on learning the difference between similar-looking elements of different classes.
// The formula: -(1-p_t)^gamma * log(p_t)
// Refer also to the paper: https://arxiv.org/pdf/1708.02002.pdf
// Parameters: gamma > 0
// The input contains of a vector with probabilities (values from 0 to 1)
class NEOML_API CFocalLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CFocalLossLayer )
public:
	explicit CFocalLossLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	static const float DefaultFocalForceValue;

	// The focal force, that is, the degree to which learning will concentrate on similar objects.
	// The greater the number, the more focused the learning will become. Always > 0
	// In the paper referred to it is called gamma
	float GetFocalForce() const { return focalForce->GetData().GetValue(); }
	void SetFocalForce( float value );
	
protected:
	virtual void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient ) override;

private:
	// The gamma parameter from the paper
	// Specifies the degree to which learning will concentrate on difficult-to-distinguish objects
	CPtr<CDnnBlob> focalForce;

	// -1
	CPtr<CDnnBlob> minusOne;

	// The handle for acceptable minimum and maximum probability values (so that separation can be performed correctly)
	CPtr<CDnnBlob> minProbValue;
	CPtr<CDnnBlob> maxProbValue;

	// Calculates the function gradient
	void calculateGradient( CFloatHandle correctClassProbabilityPerBatchHandle, int batchSize, int labelSize,
		CFloatHandle remainderVectorHandle, CFloatHandle entropyPerBatchHandle, CFloatHandle tempMatrixHandle,
		CConstFloatHandle label, CFloatHandle lossGradient );
};

} // namespace NeoML
