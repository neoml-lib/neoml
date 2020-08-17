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
#include <NeoML/Dnn/Layers/LossLayer.h>
#include <NeoML/Dnn/Layers/RecurrentLayer.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>

namespace NeoML {

class CCrfCalculationLayer;
class CCrfInternalLossLayer;
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
// Conditional random field (CRF) implementation
// The layer inputs:
// #0 is the sequence of vectors containing features that describe the objects
// #1 is the sequence of correct class labels (for training)
//
// The outputs:
// #0 contains the optimal class sequence found by Viterbi algorithm (not during training)
// #1 is the non-normalized logarithm of probability of optimal class sequence ending in this position
// #2 is the non-normalized logarithm of probability of the correct class in this position
//
// For training this layer, CCrfLossLayer should be used; its #0 input is connected to #1 output of this layer, 
// its #1 input - to the #2 output of this layer
class NEOML_API CCrfLayer : public CRecurrentLayer {
	NEOML_DNN_LAYER( CCrfLayer )
public:
	explicit CCrfLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The layer inputs and outputs
	enum TInput {
		I_Features = 0,
		I_Label = 1,
	};
	enum TOutput {
		O_BestPrevClass = 0,
		O_ClassSeqLogProb = 1,
		O_LabelLogProb = 2
	};
	// The number of classes (labels) in the CRF
	int GetNumberOfClasses() const;
	void SetNumberOfClasses(int numberOfClasses);
	// The "zero" class with which the ends of sequences are filled
	int GetPaddingClass() const;
	void SetPaddingClass(int paddingClass);

	// The dropout rate (using the variational tied weights dropout, see https://arxiv.org/abs/1512.05287)
	float GetDropoutRate() const { return dropOutLayer == 0 ? 0.f : dropOutLayer->GetDropoutRate(); }
	void SetDropoutRate(float newDropoutRate);

	// Enables calculation of the O_BestPrevClass output during training. Disabled by default.
	bool GetBestPrevClassEnabled() const;
	void SetBestPrevClassEnabled( bool enabled );

private:
	CPtr<CFullyConnectedLayer> hiddenLayer;
	CPtr<CDropoutLayer> dropOutLayer;
	CPtr<CCrfCalculationLayer> calculator;
	CPtr<CBackLinkLayer> backLink;

	void buildLayer(float dropOut);
};

// The layer that implements the basic CRF functionality - the Viterbi algorithm for optimal class sequence,
// calculating the logarithm of the optimal sequence probability and the logarithm of the weighted class sequence probability
// The layer inputs:
// #0 - the non-normalized logarithm of class probability in the given position
// #1 - the non-normalized logarithm of the probability of the optimal class sequence ending in the given position
// #2 - the correct class for the given position
//
// The layer outputs: 
// #0 - for each class contains the previous class in the optimal sequence according to Viterbi method (optional during training, see SetBestPrevClassEnabled)
// #1 - the non-normalized logarithm of the probability of the optimal class sequence ending in the given position
// #2 - the non-normalized logarithm of the probability of the correct class in the given position
class NEOML_API CCrfCalculationLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CCrfCalculationLayer )
public:
	explicit CCrfCalculationLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The layer inputs and outputs
	enum TInput {
		I_ClassLogProb = 0,
		I_ClassSeqLogProb = 1,
		I_Label = 2,
	};
	enum TOutput {
		O_BestPrevClass = 0,
		O_ClassSeqLogProb = 1,
		O_LabelLogProb = 2
	};

	// The "zero" class with which the ends of the sequences are filled
	int GetPaddingClass() const { return paddingClass; }
	void SetPaddingClass(int _paddingClass) { paddingClass = _paddingClass; }

	// Enables calculation of the O_BestPrevClass output during training
	bool GetBestPrevClassEnabled() const { return doCalculateBestPrevClass; }
	void SetBestPrevClassEnabled( bool enabled ) { doCalculateBestPrevClass = enabled; }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	// The "zero" class (no label); reserved, currently not in use
	// May be used in future for mask implementation
	int paddingClass;
	// The class transition estimates matrix
	// Each row corresponds to the i + 1 class, each column to the class with the same number as the column
	CPtr<CDnnBlob>& Transitions() { return paramBlobs[0]; }
	CPtr<CDnnBlob>& TransitionsDiff() { return paramDiffBlobs[0]; }
	const CPtr<CDnnBlob>& Transitions() const { return paramBlobs[0]; }
	mutable CPtr<CDnnBlob> prevLabels; // previous sequence labels
	CPtr<CDnnBlob> tempSumBlob; // temporary blob with the sum of logarithms of all possible sequences probabilities
	bool doCalculateBestPrevClass; // enables calculation of the O_BestPrevClass output during training
	CPtr<CDnnBlob> discardedBestPrevClassMax; // buffer for discarded by-product max values of the O_BestPrevClass calculation during training
	
	CPtr<CDnnBlob> getPrevLabels() const;
	void calcLabelProbability();
	// Indicates if the current step is the first (may also be the only one)
	bool isFirstStep() const;
};

///////////////////////////////////////////////////////////////////////////////////
// CCrfLossLayer is the loss layer that estimates the CRF error 
// as -log of the probability of the correct class sequence
// The layer inputs:
// #0 - the previous class in an optimal sequence (not used for calculations, only for constructing the network)
// #1 - the non-normalized logarithm of probability of optimal class sequence ending in this position
// #2 - the non-normalized logarithm of probability that the correct class is in this position
class NEOML_API CCrfLossLayer : public CCompositeLayer {
	NEOML_DNN_LAYER( CCrfLossLayer )
public:
	explicit CCrfLossLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The layer inputs
	enum TInput {
		I_BestPrevClass = 0, // a placeholder for constructing the network
		I_ClassSeqLogProb = 1,
		I_LabelLogProb = 2
	};
	// Total loss weight
	float GetLossWeight() const;
	void SetLossWeight(float _lossWeight);
	// Gets the last loss value
	float GetLastLoss() const;

	// The maximum loss gradient value
	// The system may not function as intended with very large loss gradient,
	// so we don't recommend changing this value
	float GetMaxGradientValue() const;
	void SetMaxGradientValue(float maxValue);

private:
	CPtr<CCrfInternalLossLayer> internalLossLayer;
	void buildLayer();
};

// The inputs:
// #0 (the data) - the non-normalized logarithm of the probability of optimal sequence including this position
// #1 (the labels) - the non-normalized logarithm of the probability of the correct class sequence
class NEOML_API CCrfInternalLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CCrfInternalLossLayer )
public:
	explicit CCrfInternalLossLayer( IMathEngine& mathEngine ) : CLossLayer( mathEngine, "FmlCnnCrfLossLayer", true ) {}

	void Serialize( CArchive& archive ) override;

protected:
	virtual void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient0, CFloatHandle lossGradient1) override;
};

///////////////////////////////////////////////////////////////////////////////////

// CBestSequenceLayer implements a layer that retrieves the optimal sequence 
// based on the CCrfLayer outputs
// The inputs: 
// #0 - for each class contains the previous class of an optimal sequence determined using Viterbi algorithm
// #1 - the non-normalized logarithm of the probability of optimal class sequence ending in this position
// The output #0 contains the optimal class sequence
class NEOML_API CBestSequenceLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CBestSequenceLayer )
public:
	explicit CBestSequenceLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "FmlCnnBestSequenceLayer", false ) {}

	void Serialize( CArchive& archive ) override;
	// The layer inputs
	enum TIntput {
		I_BestPrevClass = 0, // a placeholder for constructing the network
		I_ClassSeqLogProb = 1,
	};

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

//-----------------------------------------------------------------------------------------------

inline int CCrfLayer::GetNumberOfClasses() const 
{ 
	return hiddenLayer->GetNumberOfElements(); 
}

inline void CCrfLayer::SetNumberOfClasses(int numberOfClasses) 
{
	hiddenLayer->SetNumberOfElements(numberOfClasses);
	backLink->SetDimSize(BD_Channels, numberOfClasses);
}

inline int CCrfLayer::GetPaddingClass() const 
{ 
	return calculator->GetPaddingClass(); 
}

inline void CCrfLayer::SetPaddingClass(int paddingClass) 
{ 
	calculator->SetPaddingClass(paddingClass); 
}

inline bool CCrfLayer::GetBestPrevClassEnabled() const
{
	return calculator->GetBestPrevClassEnabled();
}

inline void CCrfLayer::SetBestPrevClassEnabled( bool enabled )
{
	calculator->SetBestPrevClassEnabled( enabled );
}

inline float CCrfLossLayer::GetLossWeight() const
{
	return internalLossLayer->GetLossWeight();
}

inline void CCrfLossLayer::SetLossWeight(float lossWeight)
{
	internalLossLayer->SetLossWeight(lossWeight);
}

inline float CCrfLossLayer::GetLastLoss() const
{
	return internalLossLayer->GetLastLoss();
}

inline float CCrfLossLayer::GetMaxGradientValue() const
{
	return internalLossLayer->GetMaxGradientValue();
}

inline void CCrfLossLayer::SetMaxGradientValue(float maxValue)
{
	return internalLossLayer->SetMaxGradientValue(maxValue);
}

} // namespace NeoML
