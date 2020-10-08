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

// Based on Alex Graves, "Supervised Sequence Labelling with Recurrent Neural Networks", chapter 7.

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/CtcLayer.h>
#include <float.h>

namespace NeoML {

static const float MaxGradientValue = 1e+6;
static const float logZero = -FLT_MAX / 4;
// log(1-). Using 0.0 may lead to denormalized numbers
static const float logOneNeg = -FLT_MIN * 2;

/////////////////////////////////////////////////////////////////////////////////////////
// These two methods are only called from Reshape, so their efficiency is not critical
// May be rewritten as CMathEngine primitives if needed

// Fills a matrix with the two-dimensional arithmetic progression values:
//  result[i, j] = start + i * rowStep + j * colStep
static void matrixFillLinear( IMathEngine& mathEngine,
	const CIntHandle& resultHandle, int height, int width,
	int start, int rowStep, int colStep )
{
	CArray<int> result;
	result.SetSize( width * height );

	for( int i = 0; i < height; i++ ) {
		for( int j = 0; j < width; j++ ) {
			result[i * width + j] = start + i * rowStep + j * colStep;
		}
	}

	mathEngine.DataExchangeTyped( resultHandle, result.GetPtr(), result.Size() );
}

// Fills an array with the arithmetic progression values:
//  result[i] = start + i * step
static void vectorFillLinear( IMathEngine& mathEngine,
	const CIntHandle& resultHandle, int vectorSize, int start, int step )
{
	matrixFillLinear( mathEngine, resultHandle, 1, vectorSize, start, 0, step );
}

/////////////////////////////////////////////////////////////////////////////////////////
// CCtcLossLayer

CCtcLossLayer::CCtcLossLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnCtcLossLayer", false ),
	lossWeight( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	loss( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	lossDivider( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	lossGradientDivider( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	minGradient( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	maxGradient( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	blankLabel( 0 ),
	minusOneInt( CDnnBlob::CreateVector( mathEngine, CT_Int, 1 ) ),
	allowBlankLabelSkip( false )
{
	SetLossWeight(1.);
	loss->GetData().SetValue( 0 );
	minGradient->GetData().SetValue( -MaxGradientValue );
	maxGradient->GetData().SetValue( MaxGradientValue );
	minusOneInt->GetData<int>().SetValue( -1 );
}

void CCtcLossLayer::SetMaxGradientValue(float maxValue)
{
	NeoAssert(maxValue > 0);

	minGradient->GetData().SetValue(-maxValue);
	maxGradient->GetData().SetValue(maxValue);
}

void CCtcLossLayer::Reshape()
{
	CheckInputs();

	CheckArchitecture(!GetDnn()->IsRecurrentMode(),
		GetName(), "ctc loss layer inside the recurrent composite layer" );
	CheckArchitecture( GetInputCount() >= 2 && GetInputCount() <= 5,
		GetName(), "CCtcLossLayer must have two to five inputs" );

	const CBlobDesc& labels = inputDescs[I_Labels];
	const bool hasLabelsLengths = GetInputCount() > I_LabelsLengths;
	const bool hasInputLengths = GetInputCount() > I_InputLengths;

	const int batchWidth = labels.BatchWidth();
	const int labelsMaxLength = labels.BatchLength();

	CheckArchitecture( inputDescs[I_Result].BatchWidth() == inputDescs[I_Labels].BatchWidth(), 
		GetName(), "loss layer result batch size doesn't match labels batch size" );
	CheckArchitecture( inputDescs[I_Labels].BatchLength() >= 1 && inputDescs[I_Labels].ObjectSize() == 1, 
		GetName(), "incorrect label size" );
	CheckArchitecture( allowBlankLabelSkip || hasLabelsLengths || labelsMaxLength * 2 + 1 <= inputDescs[I_Result].BatchLength(),
		GetName(), "too small input length" );
	if( hasLabelsLengths ) {
		CheckArchitecture( inputDescs[I_LabelsLengths].BatchLength() == 1 &&
			inputDescs[I_LabelsLengths].BatchWidth() == batchWidth &&
			inputDescs[I_LabelsLengths].ObjectSize() == 1,
			GetName(), "CCtcLossLayer: incorrect labels lengths blob dimensions" );
	}
	if( hasInputLengths ) {
		CheckArchitecture( inputDescs[I_InputLengths].BatchLength() == 1 &&
			inputDescs[I_InputLengths].BatchWidth() == batchWidth &&
			inputDescs[I_InputLengths].ObjectSize() == 1,
			GetName(), "CCtcLossLayer: incorrect inputs lengths blob dimensions" );
	}
	if( GetInputCount() > I_LabelWeights ) {
		CheckArchitecture( inputDescs[I_Result].BatchWidth() == inputDescs[I_LabelWeights].BatchWidth(),
			GetName(), "weights batch size doesn't match result batch size" );
		CheckArchitecture( inputDescs[I_LabelWeights].BatchLength() == 1 && inputDescs[I_LabelWeights].ObjectSize() == 1,
			GetName(), "weight's batchLength and objectSize must have be equal to 1" );
	} else {
		weights = CDnnBlob::CreateVector( MathEngine(), CT_Float, inputDescs[I_Result].BatchWidth() );
		weights->Fill( 1.f );
	}

	// Create a blob with label sequences separated by spaces
	const int paddedLabelsMaxLength = labelsMaxLength * 2 + 1;
	paddedLabels = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, paddedLabelsMaxLength, batchWidth, 1 );

	// The table for recalculating label index when inserting spaces
	labelRows = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, labelsMaxLength, 1, 1 );
	vectorFillLinear( MathEngine(), labelRows->GetData<int>(), labelsMaxLength, 1, 2 );

	if( allowBlankLabelSkip ) {
		nonBlanksMask = CDnnBlob::CreateVector( MathEngine(), CT_Float, inputDescs[I_Labels].BatchLength() * 2 + 1 );
		nonBlanksMask->Fill( 0.0f );
		CFloatHandleStackVar fillValuesVariable(MathEngine(), 2);
		const float fillValues[] = { 0.0f, 1.0f };
		MathEngine().DataExchangeTyped( fillValuesVariable.GetHandle(), fillValues, 2 );
		MathEngine().AddVectorToMatrixRows(1, nonBlanksMask->GetData(), nonBlanksMask->GetData(),
			inputDescs[I_Labels].BatchLength(), 2, fillValuesVariable);

		blankSkipMask = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, paddedLabels->GetBatchLength(), 
			inputDescs[I_Result].BatchWidth() );
	}

	// A table for the indices in the input activation matrix
	rowIndices = paddedLabels->GetClone();
	matrixFillLinear( MathEngine(),
		rowIndices->GetData<int>(), paddedLabelsMaxLength, batchWidth, 0, 0, 1 );

	// Index arrays for initializing beta on the backward pass
	if( hasLabelsLengths ) {
		endOfLabelPosition = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, 1, batchWidth, 1 );
		endOfLabelSample = endOfLabelPosition->GetClone();
		vectorFillLinear( MathEngine(), endOfLabelSample->GetData<int>(), batchWidth, 0, 1 );
		batchOfZeros = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchWidth, 1 );
		batchOfZeros->Fill( 0.f );
	}

	// Create blobs for logarithms of probability of prefixes logAlpha and suffixes logBeta
	logAlpha = CDnnBlob::CreateDataBlob(MathEngine(), CT_Float, inputDescs[I_Result].BatchLength(), 
		paddedLabels->GetBatchLength(), inputDescs[I_Result].BatchWidth());
	logBeta = CDnnBlob::CreateDataBlob(MathEngine(), CT_Float, inputDescs[I_Result].BatchLength(), 
		paddedLabels->GetBatchLength(), inputDescs[I_Result].BatchWidth());

	// The padding blob
	const int classesCount = inputDescs[I_Result].ObjectSize();
	paddingResultValue = CDnnBlob::CreateVector( MathEngine(), CT_Float, classesCount );

	resultProb = CDnnBlob::CreateBlob( MathEngine(), inputDescs[I_Result] );
	resultLogProb = resultProb->GetClone();

	resultProbWindow = CDnnBlob::CreateWindowBlob(resultProb, 1);
	probSum = resultProbWindow->GetClone();
	resultLogProbWindow = CDnnBlob::CreateWindowBlob(resultLogProb, 1);
	logAlphaWindow = CDnnBlob::CreateWindowBlob(logAlpha, 1);
	logAlphaPrevWindow = CDnnBlob::CreateWindowBlob(logAlpha, 1);
	logBetaWindow = CDnnBlob::CreateWindowBlob(logBeta, 1);
	logAlphaBeta = logBetaWindow->GetClone();
	logBetaPrev2 = logBetaWindow->GetClone();
	lossGradient = 0;
	lossGradientWindow = 0;

	CFloatHandleStackVar tempLossDivider( MathEngine() );
	tempLossDivider.SetValue( 1.f / inputDescs[I_Result].BatchWidth() );
	MathEngine().VectorEltwiseMultiply( tempLossDivider.GetHandle(), lossWeight->GetData(),
		lossGradientDivider->GetData(), 1 );
	// Change the sign before the lossDivider:
	MathEngine().VectorNegSum( tempLossDivider.GetHandle(), 1, lossDivider->GetData() );
}

// Calculates the logarithms of prefix probability logAlpha on a forward pass
void CCtcLossLayer::calculateForwardVariables()
{
	NeoAssert(paddedLabels->GetBatchLength() == logAlpha->GetBatchWidth());

	const int T = logAlpha->GetBatchLength();
	const int U = logAlpha->GetBatchWidth();
	// Initialize
	resultLogProbWindow->SetParentPos(0);
	logAlphaWindow->SetParentPos(0);
	// The sequence may start with a space or with the first element
	MathEngine().VectorFill(logAlphaWindow->GetObjectData( 0 ), 0.f, 
		logAlphaWindow->GetObjectSize() * 2);
	MathEngine().VectorFill(
		logAlphaWindow->GetObjectData( 2 ), logZero,
		logAlphaWindow->GetObjectSize() * (U - 2) );
	// Add the logarithm of probability of label recognition
	MathEngine().AddMatrixElementsToVector(resultLogProbWindow->GetData(), 
		resultLogProbWindow->GetBatchWidth(), resultLogProbWindow->GetObjectSize(),
		rowIndices->GetData<int>(), paddedLabels->GetData<int>(),
		logAlphaWindow->GetData(), logAlphaWindow->GetDataSize());

	const CPtr<CDnnBlob>& logAlphaShift2Buffer = logAlphaBeta;

	// Align the result sequence T elements long with the labels and spaces sequence U elements long
	for(int t = 1; t < T; t++) {
		resultLogProbWindow->SetParentPos(t);
		logAlphaWindow->SetParentPos(t);
		logAlphaPrevWindow->SetParentPos(t - 1);
		// Add up the alternative pairings after the previous moment in time
		MathEngine().VectorCopy( logAlphaWindow->GetObjectData( 0 ),
			logAlphaPrevWindow->GetObjectData( 0 ), logAlphaWindow->GetObjectSize() );
		MathEngine().VectorEltwiseLogSumExp( logAlphaPrevWindow->GetObjectData( 0 ),
			logAlphaPrevWindow->GetObjectData( 1 ), logAlphaWindow->GetObjectData( 1 ),
			logAlphaWindow->GetObjectSize() * (U - 1) );
		if( allowBlankLabelSkip ) {
			// If label[i] != blank and label[i] != label[i-2], the labels may be put together with no space:
			// label[i-2];label[i] -> label[i-2];blank;label[i]
			MathEngine().VectorAdd( logAlphaPrevWindow->GetObjectData( 0 ),
				blankSkipMask->GetObjectData( 0 ), logAlphaShift2Buffer->GetData(),
				logAlphaWindow->GetObjectSize() * (U - 2) );
			MathEngine().VectorEltwiseLogSumExp( logAlphaWindow->GetObjectData( 2 ),
				logAlphaShift2Buffer->GetData(), logAlphaWindow->GetObjectData( 2 ),
				logAlphaWindow->GetObjectSize() * (U - 2) );
		}
		// Add the logarithm of probability of label recognition
		MathEngine().AddMatrixElementsToVector(resultLogProbWindow->GetData(), 
			resultLogProbWindow->GetBatchWidth(), resultLogProbWindow->GetObjectSize(),
			rowIndices->GetData<int>(), paddedLabels->GetData<int>(), 
			logAlphaWindow->GetData(), logAlphaWindow->GetDataSize());
	}
}

// Calculates the logarithms of suffixes probability logBeta on a backward pass
// The difference in calculating logAlpha[t] and logBeta[t] stems from the fact
// that logAlpha[t] + logBeta[t] must be equal to the logarithm of probability of recognizing the sequence
void CCtcLossLayer::calculateBackwardVariables( CDnnBlob* labelsLengths, CDnnBlob* inputsLengths )
{
	NeoAssert(paddedLabels->GetBatchLength() == logBeta->GetBatchWidth());

	CFloatHandleStackVar logZeroVal( MathEngine() );
	logZeroVal.SetValue( logZero );
	CFloatHandleStackVar logOneNegVal( MathEngine() );
	logOneNegVal.SetValue( logOneNeg );
	
	const int T = logBeta->GetBatchLength();
	const int U = logBeta->GetBatchWidth();
	const int batchWidth = logBeta->GetObjectSize();
	// Initialize
	resultLogProbWindow->SetParentPos(T - 1);
	logBetaWindow->SetParentPos(T - 1);
	// Use the logAlphaBeta and nonBlanksMask arrays as temporary buffers
	const CPtr<CDnnBlob>& logBetaWindowBuffer = logAlphaBeta;
	const CPtr<CDnnBlob>& logBetaShift2Buffer = logBetaPrev2;

	// The sequence may end either with a space or with the actual last element
	// Therefore logBetaWindow is filled with logZero everywhere except two positions per sample
	// which should be filled with zero
	MathEngine().VectorFill(
		logBetaWindow->GetData(), logZero, logBetaWindow->GetDataSize() );
	if( labelsLengths == 0 ) {
		// Fixed length. Fill the two last positions with zero
		MathEngine().VectorFill(
			logBetaWindow->GetObjectData( U - 2 ), 0.f, 2 * batchWidth );
	} else {
		// Variable length. Fill the (2 * labelsLengths[j]) and (2 * labelsLengths[j] - 1) positions with zero
		MathEngine().VectorAdd(
			labelsLengths->GetData<int>(), labelsLengths->GetData<int>(),
			endOfLabelPosition->GetData<int>(), batchWidth );
		MathEngine().SetVectorToMatrixElements(
			logBetaWindow->GetData(), U, batchWidth,
			endOfLabelPosition->GetData<int>(), endOfLabelSample->GetData<int>(),
			batchOfZeros->GetData(), batchWidth );
		MathEngine().VectorAddValue( endOfLabelPosition->GetData<int>(), endOfLabelPosition->GetData<int>(),
			batchWidth, minusOneInt->GetData<int>() );
		if( inputsLengths != 0 ) {
			CArray<int> buffer;
			buffer.SetSize( batchWidth );
			CPtr<CDnnBlob> addition = inputsLengths->GetCopy();
			inputsLengths->CopyTo( buffer.GetPtr() );
			for( int i = 0; i < buffer.Size(); ++i ) {
				buffer[i] = ( buffer[i] < T ) ? 1 : 0;
			}
			MathEngine().DataExchangeTyped( addition->GetData<int>(), buffer.GetPtr(), batchWidth );
			MathEngine().VectorAdd( endOfLabelPosition->GetData<int>(), addition->GetData<int>(), 
				endOfLabelPosition->GetData<int>(), batchWidth );
		}
		MathEngine().SetVectorToMatrixElements(
			logBetaWindow->GetData(), U, batchWidth,
			endOfLabelPosition->GetData<int>(), endOfLabelSample->GetData<int>(),
			batchOfZeros->GetData(), batchWidth );
	}

	for(int t = T - 2; t >= 0; t--) {
		resultLogProbWindow->SetParentPos(t + 1);
		logBetaWindow->SetParentPos(t + 1);
		logBetaWindowBuffer->CopyFrom(logBetaWindow);
		logBetaWindow->SetParentPos(t);
		// Add the logarithm of probability of label recognition
		MathEngine().AddMatrixElementsToVector(resultLogProbWindow->GetData(), 
			resultLogProbWindow->GetBatchWidth(), resultLogProbWindow->GetObjectSize(),
			rowIndices->GetData<int>(), paddedLabels->GetData<int>(), 
			logBetaWindowBuffer->GetData(), logBetaWindowBuffer->GetDataSize());
		// Add up the alternative pairings after the previous moment in time
		MathEngine().VectorEltwiseLogSumExp(logBetaWindowBuffer->GetObjectData( 0 ),
			logBetaWindowBuffer->GetObjectData( 1 ), logBetaWindow->GetObjectData( 0 ),
			logBetaWindow->GetObjectSize() * (U - 1));
		if( allowBlankLabelSkip ) {
			// If label[i] != blank and label[i] != label[i+2], the labels may be put together with no space:
			// label[i];label[i+2] -> label[i];blank;label[i+2]
			MathEngine().VectorAdd( logBetaWindowBuffer->GetObjectData( 2 ),
				blankSkipMask->GetObjectData( 0 ), logBetaShift2Buffer->GetData(),
				logBetaWindow->GetObjectSize() * (U - 2) );
			MathEngine().VectorEltwiseLogSumExp( logBetaWindow->GetObjectData( 0 ),
				logBetaShift2Buffer->GetData(), logBetaWindow->GetObjectData( 0 ),
				logBetaWindow->GetObjectSize() * (U - 2) );
		}
		MathEngine().VectorCopy( logBetaWindow->GetObjectData( U - 1 ),
			logBetaWindowBuffer->GetObjectData( U - 1 ), logBetaWindow->GetObjectSize() );
	}
}

// Calculates the loss function gradient
void CCtcLossLayer::calculateGradient(CFloatHandle totalLogProb)
{
	if(lossGradient == 0) {
		lossGradient = inputBlobs[I_Result]->GetClone();
		lossGradientWindow = CDnnBlob::CreateWindowBlob(lossGradient, 1);
	}
	const int T = logAlpha->GetBatchLength(); // the number of "time moments"

// See Alex Graves "Supervised Sequence Labelling with Recurrent Neural Networks", 
// chapter 7.4.1
	for(int t = 0; t < T; t++) {
		probSum->Fill(logZero);
		resultProbWindow->SetParentPos(t);
		logAlphaWindow->SetParentPos(t);
		logBetaWindow->SetParentPos(t);
		lossGradientWindow->SetParentPos(t);
		MathEngine().VectorAdd(logAlphaWindow->GetData(), logBetaWindow->GetData(),
			logAlphaBeta->GetData(), logAlphaBeta->GetDataSize());

		if( allowBlankLabelSkip ) {
			// For the sake of computational stability
			MathEngine().MatrixLogSumExpByColumns( logAlphaBeta->GetData(),
				logAlphaBeta->GetBatchWidth(), logAlphaBeta->GetObjectSize(),
				totalLogProb, inputBlobs[I_Result]->GetBatchWidth() );
		}

		MathEngine().EltwiseLogSumExpVectorToMatrixElements(probSum->GetData(), 
			probSum->GetBatchWidth(), probSum->GetObjectSize(),
			rowIndices->GetData<int>(), paddedLabels->GetData<int>(), 
			logAlphaBeta->GetData(), logAlphaBeta->GetDataSize());

		MathEngine().SubVectorFromMatrixColumns(probSum->GetData(), probSum->GetData(),
			probSum->GetBatchWidth(), probSum->GetObjectSize(), totalLogProb);
		MathEngine().VectorExp(probSum->GetData(), probSum->GetData(), 
			probSum->GetDataSize());

		MathEngine().VectorSub(resultProbWindow->GetData(), probSum->GetData(),
			lossGradientWindow->GetData(), lossGradientWindow->GetDataSize());
	}
	// For the elements after the input sequence end the gradient is 0
	if( inputBlobs.Size() > I_InputLengths ) {
		paddingResultValue->Fill( 0.0f );
		applyInputLengthsPadding( inputBlobs[I_InputLengths], paddingResultValue, lossGradient, lossGradientWindow );
	}
}

// Calculates the space-skipping mask. Formula: mask[i] = log(I{i < len(paddedLabel) - 2} * I{paddedLabel[i] != paddedLabel[i+2]})
void CCtcLossLayer::calculateBlankSkipMasks()
{
	CFloatHandleStackVar logZeroVal( MathEngine() );
	MathEngine().DataExchangeTyped<float>( logZeroVal, &logZero, 1 );
	// The two last elements are equal to logZero
	MathEngine().VectorFill( blankSkipMask->GetObjectData( paddedLabels->GetBatchLength() - 2 ), 1.0f, 
		blankSkipMask->GetObjectSize() * 2 );
	const int effectiveMaskSize = (paddedLabels->GetBatchLength() - 2) * blankSkipMask->GetObjectSize();
	MathEngine().VectorEqual( paddedLabels->GetData<int>(),
		paddedLabels->GetObjectData<int>( 2 * paddedLabels->GetBatchWidth() ), blankSkipMask->GetData(),
		effectiveMaskSize  );
	MathEngine().VectorMultiply( blankSkipMask->GetData(), blankSkipMask->GetData(),
		blankSkipMask->GetDataSize(), logZeroVal );
}

// Fills the sequence ends (the elements after inputLengths[i]) in targetBlob with the paddingBlob values
// [[t_11, t_12, t_13, t_14, t_15], [t_21, t_22, t_23, t_24, t_25]]; [i_1 = 3, i_2 = 4]; pad 
//		=> [[t_11, t_12, t_13, pad, pad], [t_21, t_22, t_23, t_24, pad]]
void CCtcLossLayer::applyInputLengthsPadding( CDnnBlob* inputLengths, CDnnBlob* paddingBlob, 
	CDnnBlob* targetBlob, CDnnBlob* targetBlobWindow )
{
	NeoAssert( paddingBlob->GetDataSize() == targetBlobWindow->GetObjectSize() );
	if( inputLengths == 0 ) {
		return;
	}
	const int maxInputLength = targetBlob->GetBatchLength();
	const int batchWidth = targetBlob->GetBatchWidth();
	const int classesCount = targetBlob->GetObjectSize();

	CArray<int> buffer;
	buffer.SetSize( batchWidth );
	inputLengths->CopyTo( buffer.GetPtr() );
	int minLength = maxInputLength;
	for( int i = 0; i < buffer.Size(); ++i ) {
		if( buffer[i] < minLength ) {
			minLength = buffer[i];
		}
	}
	for( int timePosition = maxInputLength - 1; timePosition >= minLength; --timePosition ) {
		targetBlobWindow->SetParentPos( timePosition );
		for( int batchIndex = 0; batchIndex < batchWidth; ++batchIndex ) {
			if( buffer[batchIndex] <= timePosition ) {
				MathEngine().VectorCopy( targetBlobWindow->GetObjectData( batchIndex ), paddingBlob->GetData(), classesCount );
			}
		}
	}
}

void CCtcLossLayer::RunOnce()
{
	CDnnBlob* labelsLengths =
		inputBlobs.Size() <= I_LabelsLengths ? 0 : inputBlobs[I_LabelsLengths];
	CDnnBlob* inputLengths =
		inputBlobs.Size() <= I_InputLengths ? 0 : inputBlobs[I_InputLengths];

	const int batchWidth = inputBlobs[I_Labels]->GetBatchWidth();

	if( inputBlobs.Size() > I_LabelWeights ) {
		weights = inputBlobs[I_LabelWeights];
	}
	// Insert spaces
	CIntHandleStackVar fillValue( MathEngine() );
	fillValue.SetValue(blankLabel);
	MathEngine().MatrixSpreadRows(inputBlobs[I_Labels]->GetData<int>(), 
		inputBlobs[I_Labels]->GetBatchLength(), inputBlobs[I_Labels]->GetBatchWidth(),
		paddedLabels->GetData<int>(), paddedLabels->GetBatchLength(), 
		labelRows->GetData<int>(), fillValue);

	if( allowBlankLabelSkip ) {
		calculateBlankSkipMasks();
	}

	// Calculate the log(softmax) for the input results
	MathEngine().MatrixSoftmaxByRows(inputBlobs[I_Result]->GetData(), 
		inputBlobs[I_Result]->GetObjectCount(), inputBlobs[I_Result]->GetObjectSize(), 
		resultProb->GetData());
	MathEngine().VectorLog(resultProb->GetData(), resultLogProb->GetData(), 
		resultLogProb->GetDataSize());

	// Fill the ends with blanks
	paddingResultValue->Fill( logZero );
	paddingResultValue->GetData().SetValueAt( blankLabel, logOneNeg );
	applyInputLengthsPadding( inputLengths, paddingResultValue, resultLogProb, resultLogProbWindow );

	// Calculate the logarithm of prefix probability with a forward pass
	calculateForwardVariables();
	// Calculate the logarithm of suffix probability with a backward pass
	calculateBackwardVariables( labelsLengths, inputLengths );
	
	CFloatHandleStackVar totalLogProb(MathEngine(), batchWidth);
	// Maximize the total logarithm of the probability of recognizing the correct sequence
	logAlphaWindow->SetParentPos(0);
	logBetaWindow->SetParentPos(0);
	MathEngine().VectorAdd(logAlphaWindow->GetData(), logBetaWindow->GetData(), 
		logAlphaBeta->GetData(), logAlphaBeta->GetDataSize());

	// Calculate the total logarithm of the probability of recognizing
	// the correct sequence by adding across all possible pairings
	NeoAssert(logAlphaBeta->GetObjectSize() == batchWidth);
	MathEngine().MatrixLogSumExpByColumns(logAlphaBeta->GetData(),
		logAlphaBeta->GetBatchWidth(), logAlphaBeta->GetObjectSize(),
		totalLogProb, batchWidth);

	// Take weights into account
	MathEngine().VectorDotProduct(weights->GetData(), totalLogProb, batchWidth,
		loss->GetData());
	// lossDivider is negative, so loss = -totalLogProb
	MathEngine().VectorMultiply(loss->GetData(), loss->GetData(), 1,
		lossDivider->GetData());

	// Calculate the loss function gradient
	if(IsBackwardPerformed()) {
		calculateGradient(totalLogProb);
	}
}

void CCtcLossLayer::BackwardOnce()
{
	// Take weights into account
	MathEngine().Multiply1DiagMatrixByMatrix( lossGradient->GetBatchLength(), weights->GetData(),
		lossGradient->GetBatchWidth(), lossGradient->GetData(), lossGradient->GetObjectSize(),
		inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize() );
	MathEngine().VectorMultiply(inputDiffBlobs[I_Result]->GetData(), 
		inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize(), 
		lossGradientDivider->GetData());
	// In case of "huge" gradients the system behavior may be incorrect,
	// so cut these values down
	MathEngine().VectorMinMax(inputDiffBlobs[I_Result]->GetData(), 
		inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize(), 
		minGradient->GetData(), maxGradient->GetData());
}

static const int CtcLossLayerVersion = 2000;

void CCtcLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CtcLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << GetLossWeight();
		archive << maxGradient->GetData().GetValue();
		archive << blankLabel;
		archive << allowBlankLabelSkip;
	} else if( archive.IsLoading() ) {
		float tmp;
		archive >> tmp;
		SetLossWeight(tmp);
		float maxGradientValue = MaxGradientValue;
		archive >> maxGradientValue;
		minGradient->GetData().SetValue( -maxGradientValue );
		maxGradient->GetData().SetValue( maxGradientValue );
		loss->GetData().SetValue( 0 );
		archive >> blankLabel;
		archive >> allowBlankLabelSkip;
		ForceReshape();
	} else {
		NeoAssert( false );
	}
}

CLayerWrapper<CCtcLossLayer> CtcLoss( int blankLabel, bool allowBlankLabelSkip,
	float lossWeight )
{
	return CLayerWrapper<CCtcLossLayer>( "CtcLoss", [=]( CCtcLossLayer* result ) {
		result->SetBlankLabel( blankLabel );
		result->SetAllowBlankLabelSkips( allowBlankLabelSkip );
		result->SetLossWeight( lossWeight );
	} );
}

///////////////////////////////////////////////////////////////////////////////////
// CCtcDecodingLayer

CCtcDecodingLayer::CCtcDecodingLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnCtcDecodingLayer", false ),
	blankLabel(0),
	blankProbabilityThreshold(0.01f),
	arcProbabilityThreshold(0.01f)
{
}

void CCtcDecodingLayer::Reshape()
{
	CheckInputs();
	CBlobDesc transposedDesc = inputDescs[I_Result];
	transposedDesc.SetDimSize(BD_BatchLength, inputDescs[I_Result].BatchWidth());
	transposedDesc.SetDimSize(BD_BatchWidth, inputDescs[I_Result].BatchLength());
	transposedResult = CDnnBlob::CreateBlob(MathEngine(), CT_Float, transposedDesc);
	resultLogProb = CDnnBlob::CreateWindowBlob(transposedResult, 1);
	bestLabels = CDnnBlob::CreateDataBlob(MathEngine(), CT_Int, transposedResult->GetBatchLength(), 
		transposedResult->GetBatchWidth(), 1);
	lastResults.DeleteAll();
}

void CCtcDecodingLayer::RunOnce()
{
	MathEngine().TransposeMatrix(1, inputBlobs[I_Result]->GetData(), 
		inputBlobs[I_Result]->GetBatchLength(), 1, inputBlobs[I_Result]->GetBatchWidth(),
		inputBlobs[I_Result]->GetObjectSize(), transposedResult->GetData(), 
		transposedResult->GetDataSize());
	// Calculate log(softmax) of the input results
	MathEngine().MatrixSoftmaxByRows(transposedResult->GetData(), 
		transposedResult->GetObjectCount(), transposedResult->GetObjectSize(), 
		transposedResult->GetData());
	MathEngine().VectorLog(transposedResult->GetData(), transposedResult->GetData(), 
		transposedResult->GetDataSize());

	// Find the most probable labels for each coordinate
	CFloatHandleStackVar buffer(MathEngine(), transposedResult->GetObjectCount());
	MathEngine().FindMaxValueInRows(transposedResult->GetData(), 
		transposedResult->GetObjectCount(), transposedResult->GetObjectSize(), buffer, 
		bestLabels->GetData<int>(), bestLabels->GetDataSize());

	inputBlobs.CopyTo(lastResults);
}

void CCtcDecodingLayer::BackwardOnce()
{
}

static void integrateResultMatrix(CVariableMatrix<float>& resultMatrix)
{
	for(int i = 1; i < resultMatrix.SizeX(); i++) {
		float* prevCol = resultMatrix.Column(i - 1);
		float* curCol = resultMatrix.Column(i);
		for(int j = 0; j < resultMatrix.SizeY(); j++) {
			curCol[j] += prevCol[j];
		}
	}
}

bool CCtcDecodingLayer::BuildGLD(int sequenceNumber, CLdGraph<CCtcGLDArc>& gld) const
{
	gld.DeleteAll();
	int sequenceLength = lastResults[I_Result]->GetBatchLength();
	int labelsCount = lastResults[I_Result]->GetChannelsCount();
	// Find the total logarithm of probability of each label 
	// from the start to the current index in the sequence
	resultLogProb->SetParentPos(sequenceNumber);
	CVariableMatrix<float> resultMatrix(sequenceLength, labelsCount);
	resultLogProb->CopyTo( resultMatrix.Column( 0 ) );
	integrateResultMatrix(resultMatrix);
	// Find the best labels for each index
	CArray<int> bestLabelsArray;
	bestLabelsArray.SetSize(sequenceLength);
	MathEngine().DataExchangeTyped( bestLabelsArray.GetPtr(), bestLabels->GetData<const int>( {sequenceNumber} ), sequenceLength );
	// Find the possible start and end of blank label ranges
	float labelLogProbThreshold = logf(arcProbabilityThreshold);
	float blankLogProbThreshold = logf(blankProbabilityThreshold);
	CDynamicBitSet<> blankStarts, blankEnds;
	blankEnds.Set(0);
	blankStarts.Set(sequenceLength);
	int certainBlankStart = NotFound;
	for(int x = 0; x < sequenceLength; x++) {
		if(bestLabelsArray[x] == blankLabel) {
			if(certainBlankStart == NotFound) {
				// The start of highly probable blank range
				blankStarts.Set(x);
				certainBlankStart = x;
			}
		} else {
			if(certainBlankStart != NotFound) {
				// The end of highly probable blank range
				blankEnds.Set(x);
				float blankLogProb = resultMatrix(x - 1, blankLabel); 
				if(certainBlankStart > 0) {
					blankLogProb -= resultMatrix(certainBlankStart - 1, blankLabel);
				}
				auto arc = FINE_DEBUG_NEW CCtcGLDArc(certainBlankStart, x);
				arc->Label = blankLabel;
				arc->LogProb = blankLogProb;
				gld.InsertArc(arc);
			} else {
				// Out of the probable blank range
				float blankLogProb = resultMatrix(x, blankLabel);
				if(x > 0) {
					blankLogProb -= resultMatrix(x - 1, blankLabel);
				}
				if(blankLogProb >= blankLogProbThreshold) {
					blankStarts.Set(x);
					blankEnds.Set(x + 1);
					auto arc = FINE_DEBUG_NEW CCtcGLDArc(x, x + 1);
					arc->Label = blankLabel;
					arc->LogProb = blankLogProb;
					gld.InsertArc(arc);
				}
			}
			certainBlankStart = NotFound;
		}
	}
	// Build an arc for the final highly probable blank range
	if(certainBlankStart != NotFound) {
		float blankLogProb = resultMatrix(sequenceLength - 1, blankLabel); 
		if(certainBlankStart > 0) {
			blankLogProb -= resultMatrix(certainBlankStart - 1, blankLabel);
		}
		auto arc = FINE_DEBUG_NEW CCtcGLDArc(certainBlankStart, sequenceLength);
		arc->Label = blankLabel;
		arc->LogProb = blankLogProb;
		gld.InsertArc(arc);
	}
	// Build a linear division graph
	for(int x = 0; x < sequenceLength; x++) {
		if(!blankEnds.Has(x)) {
			continue;
		}
		for(int y = x + 1; y <= sequenceLength; y++ ) {
			if(!blankStarts.Has(y)) {
				continue;
			}
			for(int l = 0; l < labelsCount; l++) {
				if(l == blankLabel) {
					continue;
				}
				float labelLogProb = resultMatrix(y - 1, l);
				if(x > 0) {
					labelLogProb -= resultMatrix(x - 1, l);
				}
				if(labelLogProb >= labelLogProbThreshold) {
					auto arc = FINE_DEBUG_NEW CCtcGLDArc(x, y);
					arc->Label = l;
					arc->LogProb = labelLogProb;
					gld.InsertArc(arc);
				}
			}
		}
	}
	gld.CalculateBestPathQuality(-FLT_MAX / 2);
	return gld.VerifyPath(0, sequenceLength);
}

void CCtcDecodingLayer::GetBestSequence(int sequenceNumber, CArray<int>& bestLabelSequence) const
{
	// Find the best label for each index in the sequence
	int sequenceLength = lastResults[I_Result]->GetBatchLength();
	if( lastResults.Size() > I_InputLengths ) {
		CArray<int> lengths;
		lengths.SetSize( lastResults[I_InputLengths]->GetDataSize() );
		lastResults[I_InputLengths]->CopyTo( lengths.GetPtr() );
		sequenceLength = min( lengths[sequenceNumber], sequenceLength );
	}
	CArray<int> bestLabelsArray;
	bestLabelsArray.SetSize(sequenceLength);
	MathEngine().DataExchangeTyped(bestLabelsArray.GetPtr(), bestLabels->GetData<const int>( {sequenceNumber} ), sequenceLength);
	
	bestLabelSequence.DeleteAll();
	for(int i = 0; i < bestLabelsArray.Size(); i++) {
		int l = bestLabelsArray[i];
		if(l != blankLabel && 
			(i == 0 || l != bestLabelsArray[i - 1]) ) 
		{
			bestLabelSequence.Add(l);
		}
	}
}

static const int CtcDecodingLayerVersion = 2000;

void CCtcDecodingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CtcDecodingLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << blankLabel;
		archive << blankProbabilityThreshold;
		archive << arcProbabilityThreshold;
	} else if( archive.IsLoading() ) {
		archive >> blankLabel;
		archive >> blankProbabilityThreshold;
		archive >> arcProbabilityThreshold;
		ForceReshape();
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
