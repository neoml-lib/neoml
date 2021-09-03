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

__global__ void CtcFillPaddingKernel( int maxSeqLen, int batchSize, int classCount, int blankLabel,
	float* data, const int* seqLens )
{
	int seq, b, classIndex;
	if( GetCudaTaskIndex3D( maxSeqLen, batchSize, classCount, seq, b, classIndex ) && seq >= seqLens[b] ) {
		const float fillValue = blankLabel == -1 ? 0.f
			: ( classIndex == blankLabel ? -FLT_MIN * 2 : -FLT_MAX / 4 );
		data[( seq * batchSize + b ) * classCount + classIndex] = fillValue;
		if( ( seq * batchSize + b ) * classCount + classIndex == 0 ) {
			::printf( "WTF\n" );
		}
	}
}

__global__ void CtcCalcForwardVariableKernel( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const int* padLabels, const float* blankSkipMask, const float* resultLogProb, float* logAlpha )
{
	int b;
	if( GetCudaTaskIndex( batchSize, b ) ) {
		const int T = resultLen;
		const int U = padLabelLen;
		for( int t = 1; t < T; ++t ) {
			const float* resultLogProbWindow = resultLogProb + t * batchSize * classCount + b * classCount;
			float* logAlphaWindow = logAlpha + t * U * batchSize;
			float* logAlphaPrevWindow = logAlpha + ( t - 1 ) * U * batchSize;

			// Add up the alternative pairings after the previous moment in time
			logAlphaWindow[b] = logAlphaPrevWindow[b];
			for( int u = 0; u < U - 1; ++u ) {
				logAlphaWindow[batchSize * ( u + 1 ) + b] = LogSumExpFunc( logAlphaPrevWindow[batchSize * u + b],
					logAlphaPrevWindow[batchSize * ( u + 1 ) + b] );
			}

			if( skipBlanks ) {
				// If label[i] != blank and label[i] != label[i-2], the labels may be put together with no space:
				// label[i-2];label[i] -> label[i-2];blank;label[i]
				for( int u = 0; u < U - 2; ++u ) {
					logAlphaWindow[batchSize * ( u + 2 ) + b] = LogSumExpFunc(
						logAlphaWindow[batchSize * ( u + 2 ) + b],
						logAlphaPrevWindow[u * batchSize + b] + blankSkipMask[u * batchSize + b]
					);
				}
			}

			// Add the logarithm of probability of label recognition
			for( int u = 0; u < U; ++u ) {
				const int idx = u * batchSize + b;
				const int col = padLabels[idx];
				if( col >= 0 && col < classCount ) {
					logAlphaWindow[idx] += resultLogProbWindow[col];
				}
			}
		}
	}
}

__global__ void CtcCalcBackwardVariableKernel( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
	const int* padLabels, const float* blankSkipMask, const float* resultLogProb,
	float* logBetaWindowBuffer, float* logBeta )
{
	int b;
	if( GetCudaTaskIndex( batchSize, b ) ) {
		const int T = resultLen;
		const int U = padLabelLen;

		for( int t = T - 2; t >= 0; --t ) {
			const float* resultLogProbWindow = resultLogProb + ( t + 1 ) * batchSize * classCount + b * classCount;
			for( int u = 0; u < U; ++u ) {
				const int idx = u * batchSize + b;
				logBetaWindowBuffer[idx] = logBeta[( t + 1 ) * U * batchSize + idx];
			}
			float* logBetaWindow = logBeta + t * U * batchSize;

			// Add the logarithm of probability of label recognition
			for( int u = 0; u < U; ++u ) {
				const int idx = u * batchSize + b;
				const int col = padLabels[idx];
				if( col >= 0 && col < classCount ) {
					logBetaWindowBuffer[idx] += resultLogProbWindow[col];
				}
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

const int CtcCalcTotalLogProbCombine = 2;
__global__ void CtcCalcTotalLogProbKernel( int batchSize, const float* __restrict__ matrix, int height, int width, float* result, int heightNorm )
{
	extern __shared__  float buffer[];
	float& my = buffer[( threadIdx.z * blockDim.y + threadIdx.y ) * blockDim.x + threadIdx.x];

	my = -FLT_MAX;

	int combineCount = ( height + blockDim.x - 1 ) / blockDim.x;

	int index;
	int step;
	int count = GetCudaTaskCountAndIndexX( height, combineCount, index, step );
	index *= width;
	step *= width;

	int xPos;
	int yPos;
	int zPos;
	GetCudaTaskIndex3D( batchSize, width, heightNorm, zPos, xPos, yPos );
	if( xPos < width && zPos < batchSize && count > 0 ) {
		matrix += zPos * height * width;
		result += zPos * width;

		matrix += xPos; // get the correct column
						// find the maximum
		my = matrix[index];
		for( int j = 1; j < count; ++j ) {
			float val = matrix[index + j * step];
			if( val > my ) {
				my = val;
			}
		}
	}

	float maxVal = ReduceMaxXSharedBuffer( buffer );

	// Add up the needed part
	if( xPos < width && zPos < batchSize && count > 0 ) {
		my = expf( matrix[index] - maxVal );
		for( int j = 1; j < count; ++j ) {
			my += expf( matrix[index + j * step] - maxVal );
		}
	} else {
		my = 0.f;
	}

	float sumVal = ReduceSumXSharedBuffer( buffer );

	if( xPos < width && zPos < batchSize && threadIdx.x == 0 ) {
		result[xPos] = maxVal + log( sumVal );
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
