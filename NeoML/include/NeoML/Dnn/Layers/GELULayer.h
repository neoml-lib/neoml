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

// Activation layer with formula x * Ф(x),
// where Ф(x) - cumulative distribution function of the standard normal distribution N(0, 1)
class NEOML_API CGELULayer : public CBaseLayer {
	NEOML_DNN_LAYER( CGELULayer )
public:
	explicit CGELULayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// CDF can be calculated using the error function (slow) or using an approximation. The precise method is used by default.
	enum TCalculationMode {
		// x * 0.5( 1 + erf( x / sqrt(2) ) )
		CM_Precise,
		// x * sigmoid(1.702x)
		CM_SigmoidApproximate
	};

	// Changes GELU calculation mode
	void SetCalculationMode( TCalculationMode );
	// Returns current calculation mode
	TCalculationMode GetCalculationMode() const { return mode; }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return TInputBlobs; }

private:
	TCalculationMode mode = CM_SigmoidApproximate;

	// 1
	CFloatHandleVar oneVar;
	// 0.5
	CFloatHandleVar halfVar;
	// 1/sqrt(2)
	CFloatHandleVar sqrt2InvVar;
	// 1/sqrt(2pi)
	CFloatHandleVar sqrt2PiInvVar;
	// 1.702f
	CFloatHandleVar approxScaleVar;

	CPtr<CDnnBlob> erfMemoization;

	void runPrecise();
	void runFastApproximate();
	void backwardPrecise();
	void backwardFastApproximate();
};

NEOML_API CLayerWrapper<CGELULayer> Gelu();

} // namespace NeoML
