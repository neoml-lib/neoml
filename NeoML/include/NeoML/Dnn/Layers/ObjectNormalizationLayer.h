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
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Normalizes data in each object (ObjectSize = BD_Height x BD_Width x BD_Depth x BD_Channels).
// After that multiplies each object by scale and adds bias to the result.
// Scale and bias are trainable vector of the ObjectSize length.

// The final formula is
//  f(x) = (x - mean(x)) / sqrt(var(x) + eps) * scale + bias

class NEOML_API CObjectNormalizationLayer : public CBaseInPlaceLayer {
	NEOML_DNN_LAYER( CObjectNormalizationLayer )
public:
	explicit CObjectNormalizationLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive );

	// Returns or sets epsilon, added to variance
	// By default, epsilon is equal to 1e-5
	// This value must be positive
	void SetEpsilon( float newEpsilon );
	float GetEpsilon() const;

	// Returns or sets the scale
	// The blob should be of ObjectSize size
	// May be null if the scale has not been initialized (or must be reset)
	CPtr<CDnnBlob> GetScale() const;
	void SetScale( const CPtr<CDnnBlob>& newScale );

	// Returns or sets the bias
	// The blob should be of ObjectSize size
	// May be null if the bias has not been initialized (or must be reset)
	CPtr<CDnnBlob> GetBias() const;
	void SetBias( const CPtr<CDnnBlob>& newBias );

protected:
	void OnReshaped() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	CPtr<CDnnBlob> epsilon;
	CPtr<CDnnBlob> invObjectSize;

	// The training parameters names
	enum TParamName {
		PN_Scale = 0, // scale
		PN_Bias, // bias

		PN_Count,
	};

	// Internal (untrainable) parameters
	enum TInternalParamName {
		IPN_NegAverage = 0, // the average across the objects multiplied by -1
		IPN_InvSqrtVariance, // 1 / sqrt(variance)

		IPN_Count,
	};

	CPtr<CDnnBlob> internalParams;
	CPtr<CDnnBlob> normalizedInput;
	CPtr<CDnnBlob> outputDiffBackup;

	void calcMean();
	void calcVar();
	void normalizeInput();
	void applyScaleAndBias();

	// The pointer is valid only when the desired parameters are known: either set externally or are filled in on reshape
	CPtr<CDnnBlob>& Scale() { return paramBlobs[PN_Scale]; }
	CPtr<CDnnBlob>& Bias() { return paramBlobs[PN_Bias]; }

	CPtr<CDnnBlob>& ScaleDiff() { return paramDiffBlobs[PN_Scale]; }
	CPtr<CDnnBlob>& BiasDiff() { return paramDiffBlobs[PN_Bias]; }

	const CPtr<CDnnBlob>& Scale() const { return paramBlobs[PN_Scale]; }
	const CPtr<CDnnBlob>& Bias() const { return paramBlobs[PN_Bias]; }
};

} // namespace NeoML
