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

__global__ void CtcFillPaddingKernel( int maxSeqLen, int batchSize, int classCount, float* data, const int* seqLens )
{
	int seq, b, classIndex;
	if( GetCudaTaskIndex3D( maxSeqLen, batchSize, classCount, seq, b, classIndex ) && seq >= seqLens[b] ) {
		data[( seq * batchSize + b ) * classCount + classIndex] = 0.f;
	}
}


const int CtcMatrixLogSumExpByColumnsCombine = 2;
__global__ void CtcMatrixLogSumExpByColumnsKernel(int batchSize, const float* __restrict__ matrix, int height, int width,
	float* result, int heightNorm)
{
	extern __shared__  float buffer[];
	float& my = buffer[(threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];

	my = -FLT_MAX;

	int combineCount = (height + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	int count = GetCudaTaskCountAndIndexX(height, combineCount, index, step);
	index *= width;
	step *= width;

	int xPos;
	int yPos;
	int zPos;
	GetCudaTaskIndex3D(batchSize, width, heightNorm, zPos, xPos, yPos);
	if(xPos < width && zPos < batchSize && count > 0) {
		matrix += zPos * height * width;
		result += zPos * width;

		matrix += xPos; // get the correct column
						// find the maximum
		my = matrix[index];
		for(int j = 1; j < count; ++j) {
			float val = matrix[index + j * step];
			if(val > my) {
				my = val;
			}
		}
	}

	float maxVal = ReduceMaxXSharedBuffer(buffer);

	// Add up the needed part
	if(xPos < width && zPos < batchSize && count > 0) {
		my = expf(matrix[index] - maxVal);
		for(int j = 1; j < count; ++j) {
			my += expf(matrix[index + j * step] - maxVal);
		}
	} else {
		my = 0.f;
	}

	float sumVal = ReduceSumXSharedBuffer(buffer);

	if(xPos < width && zPos < batchSize && threadIdx.x == 0) {
		result[xPos] = maxVal + log(sumVal);
	}
}

__global__ void CtcCalcResultLogProbMaskKernel( int resultLen, int batchSize, int classCount, int padLabelLen, int blankLabel,
	float logZero, float logOneNeg, const int* resultLens, const int* padLabels, const float* resultProb, float* resultLogProbMask )
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
			*resultLogProbMask = ( col == blankLabel ) ? logOneNeg : logZero;
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
			for( int u = threadIdx.x; u < U; u += blockDim.x ) {
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
	const float* blankSkipMask, const float* resultLogProbMask, float* logBeta )
{
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	if( b < batchSize ) {
		const int T = resultLen;
		const int U = padLabelLen;

		for( int t = T - 2; t >= 0; --t ) {
			const float* resultLogProbWindow = resultLogProbMask + ( t + 1 ) * U * batchSize;
			float* logBetaPrevWindow = logBeta + ( t + 1 ) * U * batchSize;
			float* logBetaWindow = logBeta + t * U * batchSize;

			for( int u = threadIdx.x; u < U; u += blockDim.x ) {
				const int idx = u * batchSize + b;
				float value = logBetaPrevWindow[idx] + resultLogProbWindow[idx];
				if( u != U - 1 ) {
					const int uPrevIdx = idx + batchSize;
					value = LogSumExpFunc( value, logBetaPrevWindow[uPrevIdx] + resultLogProbWindow[uPrevIdx] );
					if( skipBlanks && u < U - 2 ) {
						const int uPrev2Idx = idx + 2 * batchSize;
						value = LogSumExpFunc( value, logBetaPrevWindow[uPrev2Idx] + resultLogProbWindow[uPrev2Idx] + blankSkipMask[idx] );
					}
				}
				logBetaWindow[idx] = value;
			}

			if( blockDim.x > warpSize && t > 0 ) {
				__syncthreads();
			}
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
