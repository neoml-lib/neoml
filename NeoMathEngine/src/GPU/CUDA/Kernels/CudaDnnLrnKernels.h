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

#include <CudaMathEngineDnnPoolings.h>
#include <Kernels/CudaGrid.h>
#include <Kernels/CudaReduce.h>

namespace NeoML {

__global__ void LrnKernel( const float* input, float* invSum, float* invSumBeta, float* output, int vectorCount, int vectorSize, int windowSize,
	float bias, float alpha, float beta )
{
	int vectorIndex, channelIndex;
	if( !GetCudaTaskIndex2D( vectorCount, vectorSize, vectorIndex, channelIndex ) ) {
		return;
	}

	const int firstC = max( 0, channelIndex - ( windowSize - 1 ) / 2 );
	const int lastC = min( vectorSize - 1, channelIndex + windowSize / 2 );

	input += vectorIndex * vectorSize;
	invSum += vectorIndex * vectorSize + channelIndex;
	invSumBeta += vectorIndex * vectorSize + channelIndex;
	output += vectorIndex * vectorSize + channelIndex;

	float res = 0;

	for( int i = firstC; i <= lastC; ++i ) {
		res += input[i] * input[i];
	}

	res = 1.f / ( bias + alpha * res / windowSize );
	*invSum = res;
	res = powf( res, beta );
	*invSumBeta = res;

	*output = res * input[channelIndex];
}

__global__ void LrnBackwardKernel( const float* input, const float* output, const float* outputDiff, const float* invSum,
	const float* invSumBeta, float* inputDiff, int vectorCount, int vectorSize, int windowSize, float alpha, float beta )
{
	int vectorIndex, channelIndex;
	if( !GetCudaTaskIndex2D( vectorCount, vectorSize, vectorIndex, channelIndex ) ) {
		return;
	}

	// (windowSize - 1) / 2 and windowSize / 2 are switched because it's backward
	const int firstC = max( 0, channelIndex - windowSize / 2 );
	const int lastC = min( vectorSize - 1, channelIndex + ( windowSize - 1 ) / 2 );

	input += vectorIndex * vectorSize + channelIndex;
	output += vectorIndex * vectorSize;
	outputDiff += vectorIndex * vectorSize;
	invSum += vectorIndex * vectorSize;
	invSumBeta += vectorIndex * vectorSize + channelIndex;
	inputDiff += vectorIndex * vectorSize + channelIndex;

	float res = 0;
	for( int i = firstC; i <= lastC; ++i ) {
		res += output[i] * outputDiff[i] * invSum[i];
	}

	res *= -2.f * alpha * beta * *input / windowSize;

	*inputDiff = *invSumBeta * outputDiff[channelIndex] + res;
}

} // namespace NeoML
