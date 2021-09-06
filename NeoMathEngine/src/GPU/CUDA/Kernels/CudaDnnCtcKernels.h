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

#include <CudaMathEngineDnnConvs.h>
#include <Kernels/CudaGrid.h>
#include <Kernels/CudaReduce.h>

namespace NeoML {

#define CTC_LOG_ONE_NEG ( -FLT_MIN * 2 )
#define CTC_LOG_ZERO ( -FLT_MAX / 4 )

__global__ void CtcFillPaddingKernel( int maxSeqLen, int batchSize, int classCount, int blankLabel,
	float* data, const int* seqLens )
{
	int seq, b, classIndex;
	if( GetCudaTaskIndex3D( maxSeqLen, batchSize, classCount, seq, b, classIndex ) && seq >= seqLens[b] ) {
		const float fillValue = blankLabel == -1 ? 0.f
			: ( classIndex == blankLabel ? CTC_LOG_ONE_NEG : CTC_LOG_ZERO );
		data[( seq * batchSize + b ) * classCount + classIndex] = fillValue;
	}
}

__global__ void CtcCalcResultLogProbMaskKernel( int resultLen, int batchSize, int classCount, int padLabelLen, int blankLabel,
	const int* resultLens, const int* padLabels, const float* resultProb, float* resultLogProbMask )
{
	int t;
	int u;
	int b;
	if( GetCudaTaskIndex3D( resultLen, padLabelLen, batchSize, t, u, b ) ) {
		resultLogProbMask += ( t * padLabelLen + u ) * batchSize + b;
		resultProb += ( t * batchSize + b ) * classCount;
		if( resultLens != nullptr ) {
			resultLens += b;
		}
		padLabels += u * batchSize + b;
		const int col = *padLabels;
		if( resultLens == nullptr || t < *resultLens ) {
			if( col >= 0 && col < classCount ) {
				*resultLogProbMask = logf( resultProb[col] );
			} else {
				*resultLogProbMask = 0.f;
			}
		} else {
			*resultLogProbMask = ( col == blankLabel ) ? CTC_LOG_ONE_NEG : CTC_LOG_ZERO;
		}
	}
}

__global__ void CtcCalcForwardVariableKernel( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const float* blankSkipMask, const float* resultLogProbMask, float* logAlpha )
{
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	if( b < batchSize ) {
		const int T = resultLen;
		const int U = padLabelLen;
		for( int t = 1; t < T; ++t ) {
			const float* resultLogProbWindow = resultLogProbMask + t * padLabelLen * batchSize;
			float* logAlphaWindow = logAlpha + t * U * batchSize;
			float* logAlphaPrevWindow = logAlpha + ( t - 1 ) * U * batchSize;

			// Add up the alternative pairings after the previous moment in time
			for( int u = threadIdx.x; u < padLabelLen; u += blockDim.x ) {
				const int idx = u * batchSize + b;
				float value = logAlphaPrevWindow[idx];
				if( u != 0 ) {
					value = LogSumExpFunc( value, logAlphaPrevWindow[idx - batchSize] );
					if( skipBlanks && u > 1 ) {
						// If label[i] != blank and label[i] != label[i-2], the labels may be put together with no space:
						// label[i-2];label[i] -> label[i-2];blank;label[i]
						value = LogSumExpFunc( value, logAlphaPrevWindow[idx - batchSize * 2] + blankSkipMask[idx - batchSize * 2] );
					}
				}
				value += resultLogProbWindow[idx];
				logAlphaWindow[idx] = value;
			}

			// logAlpha[T - 1] must be calculated in order to calculate logAlpha[T]
			if( blockDim.x > warpSize && t < T - 1 ) {
				__syncthreads();
			}
		}
	}
}

__global__ void CtcCalcBackwardVariableKernel( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const float* blankSkipMask, const float* resultLogProbMask, float* logBetaWindowBuffer, float* logBeta )
{
	int b;
	if( GetCudaTaskIndex( batchSize, b ) ) {
		const int T = resultLen;
		const int U = padLabelLen;

		for( int t = T - 2; t >= 0; --t ) {
			const float* resultLogProbWindow = resultLogProbMask + ( t + 1 ) * padLabelLen * batchSize;
			for( int u = 0; u < U; ++u ) {
				const int idx = u * batchSize + b;
				logBetaWindowBuffer[idx] = logBeta[( t + 1 ) * U * batchSize + idx];
			}
			float* logBetaWindow = logBeta + t * U * batchSize;

			// Add the logarithm of probability of label recognition
			for( int u = 0; u < U; ++u ) {
				const int idx = u * batchSize + b;
				logBetaWindowBuffer[idx] += resultLogProbWindow[idx];
			}

			// Add up the alternative pairings after the previous moment in time
			for( int u = 0; u < U - 1; ++u ) {
				const int idx = u * batchSize + b;
				logBetaWindow[idx] = LogSumExpFunc( logBetaWindowBuffer[idx], logBetaWindowBuffer[idx + batchSize] );
			}

			if( skipBlanks ) {
				// If label[i] != blank and label[i] != label[i+2], the labels may be put together with no space:
				// label[i];label[i+2] -> label[i];blank;label[i+2]
				for( int u = 0; u < U - 2; ++u ) {
					const int idx = u * batchSize + b;
					logBetaWindow[idx] = LogSumExpFunc( logBetaWindow[idx], logBetaWindowBuffer[2 * batchSize + idx] + blankSkipMask[idx] );
				}
			}

			logBetaWindow[( U - 1 ) * batchSize + b] = logBetaWindowBuffer[( U - 1 ) * batchSize + b];
		}
	}
}

__global__ void CtcCalcProbSumKernel( int resultLen, int batchSize, int classCount, int padLabelLen,
	const int* padLabels, const float* logAlphaBeta, float* probSum )
{
	int t;
	int b;
	if( GetCudaTaskIndex2D( resultLen, batchSize, t, b ) ) {
		padLabels += b;
		logAlphaBeta += t * padLabelLen * batchSize + b;
		probSum += ( t * batchSize + b ) * classCount;
		for( int u = 0; u < padLabelLen; ++u ) {
			const int classIdx = *padLabels;
			probSum[classIdx] = LogSumExpFunc( probSum[classIdx], *logAlphaBeta );
			padLabels += batchSize;
			logAlphaBeta += batchSize;
		}
	}
}

__global__ void CtcCalcGradientKernel( int resultLen, int batchSize, int classCount, bool skipBlanks,
	const float* resultProb, const float* totalLogProb, const float* probSum, float* lossGradient )
{
	int t, b, c;
	if( GetCudaTaskIndex3D( resultLen, batchSize, classCount, t, b, c ) ) {
		const int offset = ( t * batchSize + b ) * classCount + c;
		resultProb += offset;
		totalLogProb += skipBlanks ? t * batchSize + b : b;
		probSum += offset;
		lossGradient += offset;

		*lossGradient = *resultProb - ExponentFunc( *probSum - *totalLogProb );
	}
}

} // namespace NeoML
