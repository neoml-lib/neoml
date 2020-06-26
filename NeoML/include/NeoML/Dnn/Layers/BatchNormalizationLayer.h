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
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// Batch normalization layer (see the paper: http://arxiv.org/pdf/1502.03167.pdf)
class NEOML_API CBatchNormalizationLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CBatchNormalizationLayer )
public:
	explicit CBatchNormalizationLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Clears the statistics, keeping only the final parameters
	// May help to reduce memory consumption (both at runtime and needed for serialization)
	void ClearStatistics();

	// If true, "channel-based" statistics is gathered, that is, the data for each channel is averaged across the other dimensions
	// If false, the statistics is averaged across the batch (BatchLength * BatchWidth * ListSize dimensions)
	bool IsChannelBased() const { return isChannelBased; }
	void SetChannelBased(bool _isChannelBased);

	// Convergence rate for slow statistics (gathered across several batches)
	// This value may be from (0; 1] interval
	// The smaller this value, the more statistics takes previous data into account (~ 1 / rate)
	float GetSlowConvergenceRate() const { return slowConvergenceRate->GetData().GetValue(); }
	void SetSlowConvergenceRate(float rate);

	// The final normalization parameters
	CPtr<CDnnBlob> GetFinalParams() { updateFinalParams(); return finalParams == 0 ? 0 : finalParams->GetCopy(); }
	void SetFinalParams(const CPtr<CDnnBlob>& _params);

	// Indicates if the free term should be set to zero ("no bias")
	bool IsZeroFreeTerm() const { return isZeroFreeTerm; }
	void SetZeroFreeTerm(bool _isZeroFreeTerm) { isZeroFreeTerm  = _isZeroFreeTerm; }

	// Indicates if the final params weights should be used for initialization
	// After initialization the value will be reset to false automatically
	// IMPORTANT: never set this to true without first calling SetFinalsParams, otherwise the layer will be initialized with garbage data
	bool IsUsingFinalParamsForInitialization() const { return useFinalParamsForInitialization; }
	void UseFinalParamsForInitialization( bool use ) { useFinalParamsForInitialization = use; }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	bool isChannelBased;
	bool isZeroFreeTerm; // indicates if the free term is zero
	CPtr<CDnnBlob> slowConvergenceRate; // the convergence rate for slow statistics
	CPtr<CDnnBlob> finalParams; // the final linear operation parameters (gamma, beta)

	// The variables used to calculate statistics
	CPtr<CDnnBlob> varianceEpsilon;
	CPtr<CDnnBlob> fullBatchInv;
	CPtr<CDnnBlob> varianceNorm;
	CPtr<CDnnBlob> residual;

	CPtr<CDnnBlob> normalized;

	CPtr<CDnnBlob> varianceMult;

	// The training parameters names
	enum TParamName {
		PN_Gamma = 0,		// gamma
		PN_Beta,			// beta

		PN_Count,
	};

	// Internal (untrainable) parameters
	enum TInternalParamName {
		IPN_Average = 0,		// the average across the batch
		IPN_Variance,			// the variance across the batch
		IPN_InvSqrtVariance,	// 1 / sqrt(variance)
		IPN_SlowAverage,		// the average across several batches
		IPN_SlowVariance,		// the variance estimate across several batches

		IPN_Count,
	};
	CPtr<CDnnBlob> internalParams;

	bool useFinalParamsForInitialization; // indicates if final params should be used for initialization

	bool checkAndCreateParams();
	void getFullBatchAndObjectSize(int& fullBatchSize, int& objectSize);
	void runWhenLearning();
	void runWhenNoLearning();
	void processInput(const CPtr<CDnnBlob>& inputBlob, const CPtr<CDnnBlob>& paramBlob);
	void calculateAverage();
	void calculateVariance();
	void calculateNormalized();
	void updateSlowParams(bool isInit);
	void backwardWhenLearning();
	void backwardWhenNoLearning();

	bool isFinalParamDirty; // indicates if final params need updating
	void updateFinalParams();

	void initializeFromFinalParams();
};

} // namespace NeoML
