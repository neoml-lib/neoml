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

#include <common.h>
#pragma hdrstop

#include <CpuMathEngine.h>
#include <CpuMathEnginePrivate.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

static const float logZero = -FLT_MAX / 4;

// Fills a matrix with the two-dimensional arithmetic progression values:
//  result[i, j] = start + i * rowStep + j * colStep
static void matrixFillLinear( const CIntHandle& resultHandle, int height, int width,
	int start, int rowStep, int colStep )
{
	std::vector<int> result;
	result.resize( width * height );
	for( int i = 0; i < height; i++ ) {
		for( int j = 0; j < width; j++ ) {
			result[i * width + j] = start + i * rowStep + j * colStep;
		}
	}
	resultHandle.GetMathEngine()->DataExchangeTyped( resultHandle, result.data(), result.size() );
}

// Fills an array with the arithmetic progression values:
//  result[i] = start + i * step
static void vectorFillLinear( const CIntHandle& resultHandle, int vectorSize, int start, int step )
{
	matrixFillLinear( resultHandle, 1, vectorSize, start, 0, step );
}

// Calculates the space-skipping mask. Formula: mask[i] = log(I{i < len(paddedLabel) - 2} * I{paddedLabel[i] != paddedLabel[i+2]})
static void calcBlankSkipMask( int padLabelLen, int batchSize, const CConstIntHandle& padLabels,
	const CFloatHandle& blankSkipMask )
{
	IMathEngine& mathEngine = *padLabels.GetMathEngine();
	CFloatHandleStackVar logZeroVar( mathEngine );
	logZeroVar.SetValue( logZero );
	mathEngine.VectorFill( blankSkipMask + batchSize * ( padLabelLen - 2 ), 1.f, batchSize * 2 );
	const int effectiveMaskSize = ( padLabelLen - 2 ) * batchSize;
	mathEngine.VectorEqual( padLabels, padLabels + 2 * batchSize, blankSkipMask, effectiveMaskSize );
	mathEngine.VectorMultiply( blankSkipMask, blankSkipMask, padLabelLen * batchSize, logZeroVar );
}

static void fillPadding( int maxSeqLen, int batchSize, int classCount, int blankLabel,
	const CFloatHandle& dataHandle, const CConstIntHandle& seqLensHandle )
{
	float* data = GetRaw( dataHandle );
	const int* seqLens = GetRaw( seqLensHandle );
	const float fillValue = blankLabel == -1 ? 0.f : -FLT_MAX / 4;

	for( int b = 0; b < batchSize; ++b ) {
		const int currLen = seqLens[b];
		float* currData = data + ( currLen * batchSize + b ) * classCount;
		for( int seq = currLen; seq < maxSeqLen; ++seq ) {
			vectorFill( currData, fillValue, classCount );
			if( blankLabel != -1 ) {
				currData[blankLabel] = -FLT_MIN * 2;
			}
			currData += batchSize * classCount;
		}
	}
}

static void subVectorFromMatrixRows(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	IMathEngine& mathEngine = *matrixHandle.GetMathEngine();
	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;

	for(int i = 0; i < matrixHeight; i++) {
		mathEngine.VectorSub( matrix, vectorHandle, result, matrixWidth );
		matrix += matrixWidth;
		result += matrixWidth;
	}
}

static void matrixLogSumExpByColumns( const CConstFloatHandle& matrixHandle, int height, int width,
	const CFloatHandle& resultHandle, int resultSize )
{
	IMathEngine& mathEngine = *matrixHandle.GetMathEngine();

	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;

	CFloatHandleStackVar temp( mathEngine, height * width );
	CFloatHandleStackVar tempVec( mathEngine, width );

	// Find maximum in each column
	{
		CIntHandleStackVar indices( mathEngine, width );
		mathEngine.FindMaxValueInColumns( 1, matrix, height, width, result, indices, width );
	}

	// Subtract the maximum and save the result to a temporary variable
	subVectorFromMatrixRows( matrix, temp, height, width, result );

	// exp
	mathEngine.VectorExp( temp, temp, height * width );

	// Add up the rows, putting the result into tempVec
	mathEngine.SumMatrixRows( 1, tempVec, temp, height, width );

	// log
	mathEngine.VectorLog( tempVec, tempVec, width );

	// Add the logarithm to the maximum
	mathEngine.VectorAdd( result, tempVec, result, width );
}

static void calcGradient( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstFloatHandle& resultProb, const CConstFloatHandle& logAlpha, const CConstFloatHandle& logBeta,
	const CConstIntHandle& rowIndices, const CConstIntHandle& padLabels, const CConstIntHandle& resultLens,
	const CFloatHandle& totalLogProb, const CFloatHandle& lossGradient )
{
	IMathEngine& mathEngine = *padLabels.GetMathEngine();
	const int T = resultLen; // the number of "time moments"
	const int U = padLabelLen;

	// See Alex Graves "Supervised Sequence Labelling with Recurrent Neural Networks",
	// chapter 7.4.1
	CFloatHandleStackVar probSum( mathEngine, batchSize * classCount );
	CFloatHandleStackVar logAlphaBeta( mathEngine, U * batchSize );
	for( int t = 0; t < T; ++t ) {
		mathEngine.VectorFill( probSum, logZero, batchSize * classCount );
		CConstFloatHandle resultProbWindow = resultProb + t * batchSize * classCount;
		CConstFloatHandle logAlphaWindow = logAlpha + t * U * batchSize;
		CConstFloatHandle logBetaWindow = logBeta + t * U * batchSize;
		CFloatHandle lossGradientWindow = lossGradient + t * batchSize * classCount;
		mathEngine.VectorAdd( logAlphaWindow, logBetaWindow, logAlphaBeta, U * batchSize );

		if( skipBlanks ) {
			// For the sake of computational stability
			matrixLogSumExpByColumns( logAlphaBeta, U, batchSize, totalLogProb, batchSize );
		}

		mathEngine.EltwiseLogSumExpVectorToMatrixElements( probSum, batchSize, classCount,
			rowIndices, padLabels, logAlphaBeta, U * batchSize );
		mathEngine.SubVectorFromMatrixColumns( probSum, probSum, batchSize, classCount, totalLogProb );
		mathEngine.VectorExp( probSum, probSum, batchSize * classCount );
		mathEngine.VectorSub( resultProbWindow, probSum, lossGradientWindow, batchSize * classCount );
	}

	if( !resultLens.IsNull() ) {
		fillPadding( resultLen, batchSize, classCount, -1, lossGradient, resultLens );
	}
}

// --------------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::CtcLossForward( int resultLen, int batchSize, int classCount, int labelLen, int blankLabel, bool skipBlanks,
	const CConstFloatHandle& result, const CConstIntHandle& labels,
	const CConstIntHandle& labelLens, const CConstIntHandle& resultLens, const CConstFloatHandle& labelWeights,
	const CFloatHandle& loss, const CFloatHandle& lossGradient )
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( labels.GetMathEngine() == this );
	ASSERT_EXPR( labelLens.IsNull() || labelLens.GetMathEngine() == this );
	ASSERT_EXPR( resultLens.IsNull() || resultLens.GetMathEngine() == this );
	ASSERT_EXPR( labelWeights.IsNull() || labelWeights.GetMathEngine() == this );
	ASSERT_EXPR( loss.GetMathEngine() == this );
	ASSERT_EXPR( lossGradient.IsNull() || lossGradient.GetMathEngine() == this );

	const int padLabelLen = labelLen * 2 + 1;

	CIntHandleStackVar padLabels( *this, padLabelLen * batchSize );
	{
		CIntHandleStackVar fillValue( *this );
		fillValue.SetValue( blankLabel );
		CIntHandleStackVar labelRows( *this, labelLen );
		vectorFillLinear( labelRows, labelLen, 1, 2 );
		MatrixSpreadRows( labels, labelLen, batchSize, padLabels, padLabelLen, labelRows, fillValue );
	}

	CFloatHandleStackVar blankSkipMask( *this, padLabelLen * batchSize );
	if( skipBlanks ) {
		calcBlankSkipMask( padLabelLen, batchSize, padLabels, blankSkipMask );
	}

	CFloatHandleStackVar resultProb( *this, resultLen * batchSize * classCount );
	MatrixSoftmaxByRows( result, resultLen * batchSize, classCount, resultProb );
	CFloatHandleStackVar resultLogProb( *this, resultLen * batchSize * classCount );
	VectorLog( resultProb, resultLogProb, resultLen * batchSize * classCount );

	if( !resultLens.IsNull() ) {
		fillPadding( resultLen, batchSize, classCount, blankLabel, resultLogProb, resultLens );
	}

	CIntHandleStackVar rowIndices( *this, padLabelLen * batchSize );
	matrixFillLinear( rowIndices, padLabelLen, batchSize, 0, 0, 1 );

	CFloatHandleStackVar logAlpha( *this, resultLen * padLabelLen * batchSize );
	ctcCalcForwardVariables( resultLen, batchSize, classCount, padLabelLen, skipBlanks, rowIndices,
		padLabels, blankSkipMask, resultLogProb, logAlpha );
	CFloatHandleStackVar logBeta( *this, resultLen * padLabelLen * batchSize );
	ctcCalcBackwardVariables( resultLen, batchSize, classCount, padLabelLen, skipBlanks, rowIndices,
		padLabels, blankSkipMask, resultLogProb, resultLens, labelLens, logBeta );

	CFloatHandleStackVar totalLogProb( *this, batchSize );
	CFloatHandleStackVar logAlphaBeta( *this, padLabelLen * batchSize );
	VectorAdd( logAlpha, logBeta, logAlphaBeta, padLabelLen * batchSize );
	matrixLogSumExpByColumns( logAlphaBeta, padLabelLen, batchSize, totalLogProb, batchSize );

	if( !labelWeights.IsNull() ) {
		VectorDotProduct( labelWeights, totalLogProb, batchSize, loss );
	} else {
		VectorSum( totalLogProb, batchSize, loss );
	}
	loss.SetValue( -loss.GetValue() / batchSize );

	if( !lossGradient.IsNull() ) {
		calcGradient( resultLen, batchSize, classCount, padLabelLen, skipBlanks, resultProb,
			logAlpha, logBeta, rowIndices, padLabels, resultLens, totalLogProb, lossGradient );
	}
}

// Calculates the logarithms of prefix probabilities logAlpha on a forward pass
void CCpuMathEngine::ctcCalcForwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstIntHandle& rowIndices, const CConstIntHandle& padLabels, const CConstFloatHandle& blankSkipMask,
	const CConstFloatHandle& resultLogProb, const CFloatHandle& logAlpha )
{
	IMathEngine& mathEngine = *padLabels.GetMathEngine();
	const int T = resultLen;
	const int U = padLabelLen;

	// The sequence may start with a space or with the first element
	mathEngine.VectorFill( logAlpha, 0.f, batchSize * 2 );
	mathEngine.VectorFill( logAlpha + batchSize * 2, logZero, batchSize * ( U - 2 ) );
	// Add the logarithm of probability of label recognition
	mathEngine.AddMatrixElementsToVector( resultLogProb, batchSize, classCount, rowIndices, padLabels,
		logAlpha, U * batchSize );

	// Align the result sequence T elements long with the labels and spaces sequence U elements long
	CFloatHandleStackVar logAlphaShift2Buffer( mathEngine, U * batchSize );
	for( int t = 1; t < T; ++t ) {
		CConstFloatHandle resultLogProbWindow = resultLogProb + t * batchSize * classCount;
		CFloatHandle logAlphaWindow = logAlpha + t * U * batchSize;
		CFloatHandle logAlphaPrevWindow = logAlpha + ( t - 1 ) * U * batchSize;
		// Add up the alternative pairings after the previous moment in time
		mathEngine.VectorCopy( logAlphaWindow, logAlphaPrevWindow, batchSize );
		vectorEltwiseLogSumExp( logAlphaPrevWindow, logAlphaPrevWindow + batchSize,
			logAlphaWindow + batchSize, batchSize * ( U - 1 ) );

		if( skipBlanks ) {
			// If label[i] != blank and label[i] != label[i-2], the labels may be put together with no space:
			// label[i-2];label[i] -> label[i-2];blank;label[i]
			mathEngine.VectorAdd( logAlphaPrevWindow, blankSkipMask, logAlphaShift2Buffer, batchSize * ( U - 2 ) );
			vectorEltwiseLogSumExp( logAlphaWindow + batchSize * 2, logAlphaShift2Buffer,
				logAlphaWindow + batchSize * 2, batchSize * ( U - 2 ) );
		}

		// Add the logarithm of probability of label recognition
		mathEngine.AddMatrixElementsToVector( resultLogProbWindow, batchSize, classCount,
			rowIndices, padLabels, logAlphaWindow, batchSize * U );
	}
}

// Calculates the logarithms of suffixes probability logBeta on a backward pass
// The difference in calculating logAlpha[t] and logBeta[t] stems from the fact
// that logAlpha[t] + logBeta[t] must be equal to the logarithm of probability of recognizing the sequence
void CCpuMathEngine::ctcCalcBackwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstIntHandle& rowIndices, const CConstIntHandle& padLabels, const CConstFloatHandle& blankSkipMask,
	const CConstFloatHandle& resultLogProb, const CConstIntHandle& resultLens, const CConstIntHandle& labelLens,
	const CFloatHandle& logBeta )
{
	const int T = resultLen;
	const int U = padLabelLen;

	// Initialize
	CConstFloatHandle resultLogProbWindow = resultLogProb + ( T - 1 ) * batchSize * classCount;
	CFloatHandle logBetaWindow = logBeta + ( T - 1 ) * U * batchSize;

	// The sequence may end either with a space or with the actual last element
	// Therefore logBetaWindow is filled with logZero everywhere except two positions per sample
	// which should be filled with zero
	VectorFill( logBetaWindow, logZero, U * batchSize );
	if( labelLens.IsNull() ) {
		// Fixed length. Fill the two last positions with zero
		VectorFill( logBetaWindow + ( U - 2 ) * batchSize, 0.f, batchSize * 2 );
	} else {
		// Varying length. Fill the (2 * labelsLengths[j]) and (2 * labelsLengths[j] - 1) positions with zero
		CIntHandleStackVar endOfLabelPos( *this, batchSize );
		VectorAdd( labelLens, labelLens, endOfLabelPos, batchSize );

		CIntHandleStackVar endOfLabelSample( *this, batchSize );
		vectorFillLinear( endOfLabelSample, batchSize, 0, 1 );

		CFloatHandleStackVar batchOfZeros( *this, batchSize );
		VectorFill( batchOfZeros, 0.f, batchSize );

		SetVectorToMatrixElements( logBetaWindow, U, batchSize, endOfLabelPos, endOfLabelSample,
			batchOfZeros, batchSize );
		CIntHandleStackVar minusOneInt( *this );
		minusOneInt.SetValue( -1 );
		VectorAddValue( endOfLabelPos, endOfLabelPos, batchSize, minusOneInt );

		if( !resultLens.IsNull() ) {
			std::vector<int> buffer;
			buffer.resize( batchSize );
			CIntHandleStackVar addition( *this, batchSize );
			VectorCopy( addition, resultLens, batchSize );
			DataExchangeTyped( buffer.data(), resultLens, batchSize );
			for( size_t i = 0; i < buffer.size(); ++i ) {
				buffer[i] = ( buffer[i] < T ) ? 1 : 0;
			}
			DataExchangeTyped( addition.GetHandle(), buffer.data(), batchSize );
			VectorAdd( endOfLabelPos, addition, endOfLabelPos, batchSize );
		}
		SetVectorToMatrixElements( logBetaWindow, U, batchSize, endOfLabelPos, endOfLabelSample,
			batchOfZeros, batchSize );
	}

	CFloatHandleStackVar logBetaWindowBuffer( *this, U * batchSize );
	for( int t = T - 2; t >= 0; --t ) {
		resultLogProbWindow = resultLogProb + ( t + 1 ) * batchSize * classCount;
		VectorCopy( logBetaWindowBuffer, logBeta + ( t + 1 ) * U * batchSize, U * batchSize );
		logBetaWindow = logBeta + t * U * batchSize;

		// Add the logarithm of probability of label recognition
		AddMatrixElementsToVector( resultLogProbWindow, batchSize, classCount, rowIndices, padLabels,
			logBetaWindowBuffer, U * batchSize );
		// Add up the alternative pairings after the previous moment in time
		vectorEltwiseLogSumExp( logBetaWindowBuffer, logBetaWindowBuffer + batchSize,
			logBetaWindow, batchSize * ( U - 1 ) );
		if( skipBlanks ) {
			// If label[i] != blank and label[i] != label[i+2], the labels may be put together with no space:
			// label[i];label[i+2] -> label[i];blank;label[i+2]
			CFloatHandleStackVar logBetaShift2Buffer( *this, ( U - 2 ) * batchSize );
			VectorAdd( logBetaWindowBuffer + 2 * batchSize, blankSkipMask, logBetaShift2Buffer, batchSize * ( U - 2 ) );
			vectorEltwiseLogSumExp( logBetaWindow, logBetaShift2Buffer, logBetaWindow, batchSize * ( U - 2) );
		}
		VectorCopy( logBetaWindow + batchSize * ( U - 1 ), logBetaWindowBuffer + batchSize * ( U - 1 ), batchSize );
	}
}

} // namespace NeoML
