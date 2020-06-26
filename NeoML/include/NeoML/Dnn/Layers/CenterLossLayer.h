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

// The layer calculates the center loss function that penalizes large variance inside a class
// It may be used as an additional loss function for training discriminative features
// See the paper: http://ydwen.github.io/papers/WenECCV16.pdf
// Accepts two inputs: #0 contains the features, #1 contains the class labels in sparse format (int)
class NEOML_API CCenterLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CCenterLossLayer )
public:
	explicit CCenterLossLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The number of classes
	// The class labels will have to be in the range 0..numberOfClasses - 1
	int GetNumberOfClasses() const { return numberOfClasses; }
	void SetNumberOfClasses( int _numberOfClasses ) { numberOfClasses = _numberOfClasses; }

	// The rate of centers convergence - the multiplier used for calculating 
	// the moving mean of the class centers for each subsequent batch
	// In the paper referred to the multiplier is called alpha
	float GetClassCentersConvergenceRate() const
		{ return classCentersConvergenceRate->GetData().GetValue(); }
	void SetClassCentersConvergenceRate( float _classCentersConvergenceRate )
		{ classCentersConvergenceRate->GetData().SetValue( _classCentersConvergenceRate ); }

	// Gets the blob containing the class centers. Available only after the network ran at least once.
	// The resulting blob has only two dimensions larger than 1: BatchWidth and ChannelsCount;
	// BatchWidth corresponds to classes, ChannelsCount to features
	// Therefore it contains numberOfClasses vectors, each of the length equal to the number of features used by the layer
	CPtr<const CDnnBlob> GetClassCenters() { return classCentersBlob.Ptr(); }
	// Checks if the class centers blob is already filled in
	bool HasClassCenters() { return classCentersBlob != 0; }

protected:
	void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize, CConstIntHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient ) override;

private:
	// The number of classes
	int numberOfClasses;
	// The centers convergence rate
	CPtr<CDnnBlob> classCentersConvergenceRate;
	// The unit multiplier
	CPtr<CDnnBlob> oneMult;
	// The internal blobs
	CPtr<CDnnBlob> classCentersBlob;

	void updateCenters(const CFloatHandle& tempDiffHandle);
};

} // namespace NeoML
