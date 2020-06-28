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
#include <NeoML/TraditionalML/LdGraph.h>
#include <NeoML/TraditionalML/VariableMatrix.h>

namespace NeoML {

// The layer that implements connectionist temporal classification (CTC) 
// for training recurrent networks to recognize sequences
class NEOML_API CCtcLossLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CCtcLossLayer )
public:
	explicit CCtcLossLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The layer may have 2 to 5 inputs
	enum TInput {
		// The activations of the previous layer, of the dimensions:
		//  (BatchLength=the counter) * (BatchWidth=batch size) * (the number of classes)
		I_Result = 0,

		// The class labels, a blob with integer data, of the dimensions:
		//  (BatchLength=the maximum label length) * (BatchWidth=batch size) * 1
		I_Labels = 1,

		// The class labels length (optional), a blob with integer data, of the dimensions:
		//  (BatchLength=1) * (BatchWidth=batch size) * 1
		I_LabelsLengths = 2,

		// The input sequence length (optional), a blob with integer data, of the dimensions:
		//  (BatchLength=1) * (BatchWidth=batch size) * 1
		I_InputLengths = 3,

		// Training sample weights (optional), of the dimensions:
		//  (BatchLength=1) * (BatchWidth=batch size)* 1
		I_LabelWeights = 4
	};

	// The blank label, to be used as a space between other labels:
	int GetBlankLabel() const { return blankLabel; }
	void SetBlankLabel(int _blankLabel) { blankLabel = _blankLabel; }

	// Total loss weight
	float GetLossWeight() const { return lossWeight->GetData().GetValue(); }
	void SetLossWeight(float _lossWeight) { lossWeight->GetData().SetValue(_lossWeight); }

	// Gets the last loss value
	float GetLastLoss() const { return loss->GetData().GetValue(); }

	// The maximum loss gradient value
	// The system may not function as intended with very large loss gradient,
	// so we don't recommend changing this value
	float GetMaxGradientValue() const { return maxGradient->GetData().GetValue(); }
	void SetMaxGradientValue(float maxValue);

	// Indicates if the blank labels may be skipped when aligning
	bool GetAllowBlankLabelSkips() { return allowBlankLabelSkip; }
	void SetAllowBlankLabelSkips( bool enabled ) { allowBlankLabelSkip = enabled; }

	const CPtr<CDnnBlob> GetLastGradient() const { return lossGradient; }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CPtr<CDnnBlob> lossWeight; // scale multiplier for the loss function
	CPtr<CDnnBlob> loss; // the loss value on the last tep
	CPtr<CDnnBlob> lossDivider; // the averaging factor for calculating the loss value
	CPtr<CDnnBlob> lossGradientDivider; // the averaging factor for calculating the loss gradient (taking lossWeight into account)
	CPtr<CDnnBlob> weights;	// the vector weights

	CPtr<CDnnBlob> minGradient;
	CPtr<CDnnBlob> maxGradient;

	int blankLabel; // the blank label
	CPtr<CDnnBlob> paddedLabels; // the sequence of labels separated by blanks { 2 * LabelLength + 1, BW, 1 } int
	CPtr<CDnnBlob> nonBlanksMask; // 0.0 for the blanks, 1.0 for the rest { 2 * LabelLength + 1 }
	CPtr<CDnnBlob> labelRows; // the matrix row numbers for filling in paddedLabels
	CPtr<CDnnBlob> logAlpha, logBeta; // the blobs with logarithms of prefix and suffix sequence probabilities { InputLength, 2 * LabelLength + 1, BW }
	CPtr<CDnnBlob> blankSkipMask; // the blobs with masks for elements with blanks removed { 2 * LabelLength + 1, BW }
	CPtr<CDnnBlob> logBetaPrev2; // the blob with logarithms of suffixes shifted 2 steps back in time { 2 * LabelLength + 1, BW }
	CPtr<CDnnBlob> resultProb; // softmax(I_Result)	{ InputLength, BW, Classes }
	CPtr<CDnnBlob> resultLogProb; // log(softmax(I_Result)) { InputLength, BW, Classes }
	CPtr<CDnnBlob> paddingResultValue; // contains the padding for the result { Classes }

	CPtr<CDnnBlob> resultProbWindow;
	CPtr<CDnnBlob> resultLogProbWindow;
	CPtr<CDnnBlob> logAlphaWindow;
	CPtr<CDnnBlob> logAlphaPrevWindow;
	CPtr<CDnnBlob> logBetaWindow;
	// total logarithm of the probability of sequences with a given alignment at a given moment
	CPtr<CDnnBlob> logAlphaBeta; // { 2 * LabelLength + 1, BW }
	
	CPtr<CDnnBlob> lossGradient; // loss function gradient { InputLength, BW, Classes }
	CPtr<CDnnBlob> lossGradientWindow;
	CPtr<CDnnBlob> probSum;     // a temporary blob for calculating gradient
	CPtr<CDnnBlob> rowIndices;  // the row indices blob

	// The arrays of indices for initializing beta on a backward pass
	// They point at where the labels end in the length and position dimensions in a batch
	CPtr<CDnnBlob> endOfLabelPosition;
	CPtr<CDnnBlob> endOfLabelSample;

	// The integer constant -1
	CPtr<CDnnBlob> minusOneInt;
	// The array of float zeros of BatchWidth size
	CPtr<CDnnBlob> batchOfZeros;

	bool allowBlankLabelSkip; // indicates if blanks between different labels may be skipped

	void calculateForwardVariables();
	void calculateBackwardVariables( CDnnBlob* labelsLength, CDnnBlob* inputsLengths );
	void calculateGradient(CFloatHandle totalLogProb);
	void calculateBlankSkipMasks();
	void applyInputLengthsPadding( CDnnBlob* inputLengths, CDnnBlob* paddingBlob,
		CDnnBlob* targetBlob, CDnnBlob* targetBlobWindow );
};

///////////////////////////////////////////////////////////////////////////////////
// The layer that builds a linear division graph (LDG) on the sequence recognition results

struct CCtcGLDArc {
public:
	const int Begin;
	const int End;
	int Label;
	float LogProb;

	CCtcGLDArc(int begin, int end) : Begin(begin), End(end) {}

	typedef float Quality;
	int InitialCoord() const { return Begin; }
	int FinalCoord() const { return End; }
	Quality ArcQuality() const { return LogProb; }
};

class NEOML_API CCtcDecodingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CCtcDecodingLayer )
public:
	explicit CCtcDecodingLayer( IMathEngine& mathEngine );

	// The layer inputs
	enum TInput {
		// The activations of the previous layer
		//  (BatchLength=the counter) * (BatchWidth=batch size) * (the number of classes).
		I_Result = 0,
		// Input sequences length (optional), integer data
		//  (BatchLength=1) * (BatchWidth=batch size) * 1
		I_InputLengths = 1,
	};
	// The blank label to be used as a space
	int GetBlankLabel() const { return blankLabel; }
	void SetBlankLabel(int _blankLabel) { blankLabel = _blankLabel; }
	// The probability threshold for blanks, when building an LDG
	float GetBlankProbabilityThreshold() const { return blankProbabilityThreshold; }
	void SetBlankProbabilityThreshold(float threshold) { blankProbabilityThreshold = threshold; }
	// The probability threshold for cutting off arcs when building an LDG
	float GetArcProbabilityThreshold() const { return arcProbabilityThreshold; }
	void SetArcProbabilityThreshold(float threshold) { arcProbabilityThreshold = threshold; }

	// Sequence length
	int GetSequenceLength() const { return lastResults[I_Result]->GetBatchLength(); }
	// The number of sequences in the batch
	int GetBatchWidth() const { return lastResults[I_Result]->GetBatchWidth(); }
	// The number of labels used
	int GetLabelsCount() const { return lastResults[I_Result]->GetChannelsCount(); }

	// Builds a linear division graph using the recognition results
	bool BuildGLD(int sequenceNumber, CLdGraph<CCtcGLDArc>& gld) const;

	void GetBestSequence(int sequenceNumber, CArray<int>& bestLabelSequence) const;

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	int blankLabel; // the blank label
	float blankProbabilityThreshold; // the blank probability threshold for LDG building
	float arcProbabilityThreshold; // the arc probability threshold for LDG building
	CPtr<CDnnBlob> transposedResult; // the transposed log(softmax(0))
	CPtr<CDnnBlob> resultLogProb; // the window blob for one sequence in the transposedResult
	CPtr<CDnnBlob> bestLabels; // the best labels along each dimension
	CObjectArray<CDnnBlob> lastResults; // the copies of input blobs from the last run
};

} // namespace NeoML
