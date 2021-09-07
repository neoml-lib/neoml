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
// log(1-). Using 0.0 may lead to denormalized numbers
static const float logOneNeg = -FLT_MIN * 2;

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
	CFloatHandleStackVar resultLogProbMask( *this, resultLen * padLabelLen * batchSize );
	
	{
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid3D( blockCount, threadCount, resultLen, padLabelLen, batchSize );
		CtcCalcResultLogProbMaskKernel<<<blockCount, threadCount>>>( resultLen, batchSize, classCount, padLabelLen, blankLabel, logZero, logOneNeg,
			GetRaw( resultLens ), GetRaw( padLabels.GetHandle() ), GetRaw( resultProb.GetHandle() ), GetRaw( resultLogProbMask.GetHandle() ) );
	}

	CFloatHandleStackVar logAlphaBeta( *this, resultLen * padLabelLen * batchSize );
	{
		CFloatHandle logAlpha = logAlphaBeta.GetHandle();
		ctcCalcForwardVariables( resultLen, batchSize, classCount, padLabelLen, skipBlanks,
			blankSkipMask, resultLogProbMask, logAlpha );
		CFloatHandleStackVar logBeta( *this, resultLen * padLabelLen * batchSize );
		ctcCalcBackwardVariables( resultLen, batchSize, classCount, padLabelLen, skipBlanks,
			blankSkipMask, resultLogProbMask, resultLens, labelLens, logBeta );
		VectorAdd( logAlpha, logBeta, logAlphaBeta, resultLen * padLabelLen * batchSize );
	}

	const int totalLogProbBatch = ( skipBlanks && !lossGradient.IsNull() ) ? resultLen : 1;
	CFloatHandleStackVar totalLogProb( *this, totalLogProbBatch * batchSize );
	{
		int heightNorm = (padLabelLen + CtcMatrixLogSumExpByColumnsCombine - 1) / CtcMatrixLogSumExpByColumnsCombine;
		heightNorm = alignXSizeForWarp(heightNorm);
		dim3 blockCount;
		dim3 threadCount;
		// Rows over the X instead of Y axis, so we could reduce by X
		getCudaTaskGrid3DMinZYX(1, 1, 1024, blockCount, threadCount, totalLogProbBatch, batchSize, heightNorm);
		blockCount.x = 1;

		const int sharedSize = threadCount.x * threadCount.y * threadCount.z * sizeof(float);
		CtcMatrixLogSumExpByColumnsKernel<<<blockCount, threadCount, sharedSize>>>(
			totalLogProbBatch, GetRaw(logAlphaBeta.GetHandle()), padLabelLen, batchSize, GetRaw(totalLogProb.GetHandle()), heightNorm);
	}

	if( !labelWeights.IsNull() ) {
		VectorDotProduct( labelWeights, totalLogProb, batchSize, loss );
	} else {
		VectorSum( totalLogProb, batchSize, loss );
	}
	loss.SetValue( -loss.GetValue() / batchSize );

	if( !lossGradient.IsNull() ) {
		ctcCalcGradient( resultLen, batchSize, classCount, padLabelLen, skipBlanks, resultProb,
			logAlphaBeta, padLabels, resultLens, totalLogProb, lossGradient );
		if( !resultLens.IsNull() ) {
			ctcFillPadding( resultLen, batchSize, classCount, lossGradient, resultLens );
		}
	}
}

// Calculates the logarithms of prefix probabilities logAlpha on a forward pass
void CCudaMathEngine::ctcCalcForwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstFloatHandle& blankSkipMask, const CConstFloatHandle& resultLogProbMask, const CFloatHandle& logAlpha )
{
	const int U = padLabelLen;

	// The sequence may start with a space or with the first element
	VectorFill( logAlpha, 0.f, batchSize * 2 );
	VectorFill( logAlpha + batchSize * 2, logZero, batchSize * ( U - 2 ) );
	// Add the logarithm of probability of label recognition
	VectorAdd( logAlpha, resultLogProbMask, logAlpha, batchSize * U );

	// Align the result sequence T elements long with the labels and spaces sequence U elements long
	const int padLabelLenForWarp = alignXSizeForWarp( padLabelLen );
	dim3 blockCount, threadCount;
	getCudaTaskGrid2DMinYX( 1, 1024, blockCount, threadCount, batchSize, padLabelLenForWarp );
	blockCount.x = 1;
	CtcCalcForwardVariableKernel<<<blockCount, threadCount>>>( resultLen, batchSize, classCount, padLabelLen, skipBlanks,
		GetRaw( blankSkipMask ), GetRaw( resultLogProbMask ), GetRaw( logAlpha ) );
}

// Calculates the logarithms of suffixes probability logBeta on a backward pass
// The difference in calculating logAlpha[t] and logBeta[t] stems from the fact
// that logAlpha[t] + logBeta[t] must be equal to the logarithm of probability of recognizing the sequence
void CCudaMathEngine::ctcCalcBackwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstFloatHandle& blankSkipMask, const CConstFloatHandle& resultLogProbMask,
	const CConstIntHandle& resultLens, const CConstIntHandle& labelLens, const CFloatHandle& logBeta )
{
	const int T = resultLen;
	const int U = padLabelLen;

	// Initialize
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

	const int padLabelLenForWarp = alignXSizeForWarp( padLabelLen );
	dim3 blockCount, threadCount;
	getCudaTaskGrid2DMinYX( 1, 1024, blockCount, threadCount, batchSize, padLabelLenForWarp );
	blockCount.x = 1;
	CtcCalcBackwardVariableKernel<<<blockCount, threadCount>>>( resultLen, batchSize, classCount, padLabelLen, skipBlanks,
		GetRaw( blankSkipMask ), GetRaw( resultLogProbMask ), GetRaw( logBeta ) );
}

void CCudaMathEngine::ctcCalcGradient( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const CConstFloatHandle& resultProb, const CConstFloatHandle& logAlphaBeta,
	const CConstIntHandle& padLabels, const CConstIntHandle& resultLens,
	const CFloatHandle& totalLogProb, const CFloatHandle& lossGradient )
{
	// See Alex Graves "Supervised Sequence Labelling with Recurrent Neural Networks",
	// chapter 7.4.1
	CFloatHandleStackVar probSum( *this, resultLen * batchSize * classCount );
	{
		VectorFill( probSum, logZero, resultLen * batchSize * classCount );
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid2D( blockCount, threadCount, resultLen, batchSize );
		CtcCalcProbSumKernel<<<blockCount, threadCount>>>( resultLen, batchSize, classCount, padLabelLen,
			GetRaw( padLabels ), GetRaw( logAlphaBeta ), GetRaw( probSum.GetHandle() ) );
	}
	{
		dim3 blockCount;
		dim3 threadCount;
		getCudaTaskGrid3D( blockCount, threadCount, resultLen, batchSize, classCount );
		CtcCalcGradientKernel<<<blockCount, threadCount>>>( resultLen, batchSize, classCount, skipBlanks,
			GetRaw( resultProb ), GetRaw( totalLogProb ), GetRaw( probSum.GetHandle() ), GetRaw( lossGradient ) );
	}
}

void CCudaMathEngine::ctcFillPadding( int maxSeqLen, int batchSize, int classCount,
	const CFloatHandle& dataHandle, const CConstIntHandle& seqLensHandle )
{
	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid3D( blockCount, threadCount, maxSeqLen, batchSize, classCount );
	CtcFillPaddingKernel<<<blockCount, threadCount>>>( maxSeqLen, batchSize, classCount,
		GetRaw( dataHandle ), GetRaw( seqLensHandle ) );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
