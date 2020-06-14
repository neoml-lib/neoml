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
#include <NeoML/Dnn/Layers/QualityControlLayer.h>
#include <NeoML/TraditionalML/VariableMatrix.h>

namespace NeoML {

// These layers put the accuracy stats into output blobs
// To access the data you need to connect sink layers to these layers' outputs

// Calculates the number of objects classified correctly (accumulated over several iterations)
// The first input contains the classification result, the second - the correct class labels
class NEOML_API CAccuracyLayer : public CQualityControlLayer {
	NEOML_DNN_LAYER( CAccuracyLayer )
public:
	explicit CAccuracyLayer( IMathEngine& mathEngine );

	virtual void Serialize( CArchive& archive ) override;

protected:
	virtual void Reshape() override;
	virtual void OnReset() override;
	virtual void RunOnceAfterReset() override;

private:
	// the number of iterations for which the error is accumulated
	int iterationsCount;
	// the accumulated classification accuracy
	double collectedAccuracy;
};

//---------------------------------------------------------------------------------------------------------------------

// Collects the data for a confusion matrix over several iterations
// The first input contains the classification result, the second - the correct class labels
class NEOML_API CConfusionMatrixLayer : public CQualityControlLayer {
	NEOML_DNN_LAYER( CConfusionMatrixLayer )
public:
	explicit CConfusionMatrixLayer( IMathEngine& mathEngine );

	virtual void Serialize( CArchive& archive ) override;

	// Accessing the matrix
	const CVariableMatrix<float>& GetMatrix() const { return confusionMatrix; }
	// Resets all matrix elements to 0
	void ResetMatrix() { confusionMatrix.Set( 0 ); }

protected:
	virtual void Reshape() override;
	virtual void OnReset() override { confusionMatrix.Set( 0 ); }
	virtual void RunOnceAfterReset() override;

private:
	// Confusion matrix
	CVariableMatrix<float> confusionMatrix;
};

} // namespace NeoML
