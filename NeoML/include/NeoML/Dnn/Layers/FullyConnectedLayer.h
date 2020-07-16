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
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// CFullyConnectedLayer implements a fully-connected layer
class NEOML_API CFullyConnectedLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CFullyConnectedLayer )
public:
	explicit CFullyConnectedLayer( IMathEngine& mathEngine, const char* name = nullptr );

	void Serialize( CArchive& archive ) override;

	// The number of elements ("neurons") of the fully-connected layer
	int GetNumberOfElements() const { return numberOfElements; }
	void SetNumberOfElements(int newNumberOfElements);

	// Retrieves or sets the weights data (the data blob is copied)
	// The dimensions of the blob are NumOfElements * InputHeight * InputWidth * InputChannelsCount
	// If the weights have not been initialized, an empty blob will be returned; pass an empty blob to reset the weights
	CPtr<CDnnBlob> GetWeightsData() const;
	void SetWeightsData(CDnnBlob* newWeights);

	// Retrieves or sets the free term (the data blob is copied)
	// The free term blob should be of NumOfElements size
	// If the free term has not been initialized, an empty blob will be returned; pass an empty blob to reset the free term
	CPtr<CDnnBlob> GetFreeTermData() const;
	void SetFreeTermData(CDnnBlob* newFreeTerms);

	// Applies the batch normalization parameters to the internal parameters of the layer
	// The layer will then return the same output 
	// that was previously returned by the combination of this layer with batch normalization
	void ApplyBatchNormalization(CBatchNormalizationLayer& batchNorm);

	// Indicates if the free term should be set to zero ("no bias")
	bool IsZeroFreeTerm() const { return isZeroFreeTerm; }
	void SetZeroFreeTerm(bool _isZeroFreeTerm);

protected:
	virtual ~CFullyConnectedLayer();

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;
	void FilterLayerParams( float threshold ) override;

	// The filter. The pointer is valid only if the desired parameters are known (either defined externally or obtained on reshape)
	CPtr<CDnnBlob>& Weights() { return paramBlobs[0]; }
	CPtr<CDnnBlob>& FreeTerms() { return paramBlobs[1]; }	// the free term matrix

	CPtr<CDnnBlob>& WeightsDiff() { return paramDiffBlobs[0]; }
	CPtr<CDnnBlob>& FreeTermsDiff() { return paramDiffBlobs[1]; }

	const CPtr<CDnnBlob>& Weights() const { return paramBlobs[0]; }
	const CPtr<CDnnBlob>& FreeTerms() const { return paramBlobs[1]; }	// the free term matrix

private:
	int numberOfElements; // the number of elements (neurons) of the fully-connected layer
	bool isZeroFreeTerm; // indicates if the free term should be set to zero
};

} // namespace NeoML
