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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <vector>

#include <CudaMathEngine.h>
#include <CudaCommon.h>
#include <CudaDevice.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>

#include <Kernels/CudaDnnCtcKernels.h>

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

// --------------------------------------------------------------------------------------------------------------------

void CCudaMathEngine::CtcLossForward( int resultLen, int batchSize, int classCount, int labelLen, int blankLabel, bool skipBlanks,
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
		ctcFillPadding( resultLen, batchSize, classCount, blankLabel, resultLogProb, resultLens );
	}

	CIntHandleStackVar rowIndices( *this, padLabelLen * batchSize );
	matrixFillLinear( rowIndices, padLabelLen, batchSize, 0, 0, 1 );

	CFloatHandleStackVar logAlpha( *this, resultLen * padLabelLen * batchSize );
	ctcCalcForwardVariables( resultLen, batchSize, classCount, padLabelLen, skipBlanks, rowIndices,
		padLabels, blankSkipMask, resultLogProb, logAlpha );
	CFloatHandleStackVar logBeta( *this, resultLen * padLabelLen * batchSize );
	ctcCalcBackwardVariables( resultLen, batchSize, classCount, padLabelLen, skipBlanks,
		padLabels, blankSkipMask, resultLogProb, resultLens, labelLens, logBeta );

	CFloatHandleStackVar totalLogProb( *this, batchSize );
	CFloatHandleStackVar logAlphaBeta( *this, padLabelLen * batchSize );
	VectorAdd( logAlpha, logBeta, logAlphaBeta, padLabelLen * batchSize );
	MatrixLogSumExpByColumns( logAlphaBeta, padLabelLen, batchSize, totalLogProb, batchSize );

	if( !labelWeights.IsNull() ) {
		VectorDotProduct( labelWeights, totalLogProb, batchSize, loss );
	} else {
		VectorSum( totalLogProb, batchSize, loss );
	}
	loss.SetValue( -loss.GetValue() / batchSize );

	if( !lossGradient.IsNull() ) {
		ctcCalcGradient( resultLen, batchSize, classCount, padLabelLen, skipBlanks, resultProb,
			logAlpha, logBeta, padLabels, resultLens, totalLogProb, lossGradient );
		if( !resultLens.IsNull() ) {
			ctcFillPadding( resultLen, batchSize, classCount, -1, lossGradient, resultLens );
		}
	}
}

// Calculates the logarithms of prefix probabilities logAlpha on a forward pass
void CCudaMathEngine::ctcCalcForwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstIntHandle& rowIndices, const CConstIntHandle& padLabels, const CConstFloatHandle& blankSkipMask,
	const CConstFloatHandle& resultLogProb, const CFloatHandle& logAlpha )
{
	const int U = padLabelLen;

	// The sequence may start with a space or with the first element
	VectorFill( logAlpha, 0.f, batchSize * 2 );
	VectorFill( logAlpha + batchSize * 2, logZero, batchSize * ( U - 2 ) );
	// Add the logarithm of probability of label recognition
	AddMatrixElementsToVector( resultLogProb, batchSize, classCount, rowIndices, padLabels,
		logAlpha, U * batchSize );

	// Align the result sequence T elements long with the labels and spaces sequence U elements long
	int blockCount, threadCount;
	getCudaTaskGrid( blockCount, threadCount, batchSize );
	CtcCalcForwardVariableKernel<<<blockCount, threadCount>>>( resultLen, batchSize, classCount, padLabelLen, skipBlanks,
		GetRaw( padLabels ), GetRaw( blankSkipMask ), GetRaw( resultLogProb ), GetRaw( logAlpha ) );
}

// Calculates the logarithms of suffixes probability logBeta on a backward pass
// The difference in calculating logAlpha[t] and logBeta[t] stems from the fact
// that logAlpha[t] + logBeta[t] must be equal to the logarithm of probability of recognizing the sequence
void CCudaMathEngine::ctcCalcBackwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstIntHandle& padLabels, const CConstFloatHandle& blankSkipMask,
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
	int blockCount, threadCount;
	getCudaTaskGrid( blockCount, threadCount, batchSize );
	CtcCaclBackwardVariableKernel<<<blockCount, threadCount>>>( resultLen, batchSize, classCount, padLabelLen, skipBlanks,
		GetRaw( padLabels ), GetRaw( blankSkipMask ), GetRaw( resultLogProb ),
		GetRaw( logBetaWindowBuffer.GetHandle() ), GetRaw( logBeta ) );
}

void CCudaMathEngine::ctcCalcGradient( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstFloatHandle& resultProb, const CConstFloatHandle& logAlpha, const CConstFloatHandle& logBeta,
	const CConstIntHandle& padLabels, const CConstIntHandle& resultLens,
	const CFloatHandle& totalLogProb, const CFloatHandle& lossGradient )
{
	// See Alex Graves "Supervised Sequence Labelling with Recurrent Neural Networks",
	// chapter 7.4.1
	int blockCount;
	int threadCount;
	CFloatHandleStackVar probSum( *this, batchSize * classCount );
	getCudaTaskGrid( blockCount, threadCount, batchSize );
	CtcCalcGradientKernel<<<blockCount, threadCount>>>( resultLen, batchSize, classCount, padLabelLen, skipBlanks, logZero,
		GetRaw( padLabels ), GetRaw( resultProb ), GetRaw( logAlpha ), GetRaw( logBeta ), GetRaw( totalLogProb ),
		GetRaw( probSum.GetHandle() ), GetRaw( lossGradient ) );
}

void CCudaMathEngine::ctcFillPadding( int maxSeqLen, int batchSize, int classCount, int blankLabel,
	const CFloatHandle& dataHandle, const CConstIntHandle& seqLensHandle )
{
	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3D( blockCount, threadCount, maxSeqLen, batchSize, classCount );
	CtcFillPaddingKernel<<<blockCount, threadCount>>>( maxSeqLen, batchSize, classCount, blankLabel,
		GetRaw( dataHandle ), GetRaw( seqLensHandle ) );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
