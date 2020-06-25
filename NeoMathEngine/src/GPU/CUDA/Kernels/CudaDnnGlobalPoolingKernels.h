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
#include <cfloat>

namespace NeoML {

__device__ inline void MergeBuffers(int maxCount, float* buffer, float* maxIndexBuffer,
	const float* buffer0, const float* maxIndexBuffer0, const float* buffer1, const float* maxIndexBuffer1)
{
	while(maxCount-- > 0) {
		if(*buffer0 > *buffer1) {
			*buffer++ = *buffer0++;
			*maxIndexBuffer++ = *maxIndexBuffer0++;
		} else {
			*buffer++ = *buffer1++;
			*maxIndexBuffer++ = *maxIndexBuffer1++;
		}
	}
}

const int BlobGlobalMaxPoolingCombine = 8;
__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* sourceData,
	int* maxIndicesData, float* resultData, int poolSize, int maxCount, int poolSizeNorm )
{
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& maxIndices = desc.MaxIndices;
	const CCudaBlobDesc& result = desc.Result;

	int bufferStep = 2 * maxCount;

	// Initialize the data
	extern __shared__ float sharedData[];

	float* buffer = sharedData + (threadIdx.y * blockDim.x * 2 + threadIdx.x) * bufferStep;
	float* maxIndexBuffer = buffer + maxCount;
	// The pointer to the end of the buffer (the rest will be used to merge)
	float* bufferEnd = sharedData + (threadIdx.y * blockDim.x * 2 + blockDim.x) * bufferStep;

	for(int i = 0; i < maxCount; ++i) {
		buffer[i] = -FLT_MAX;
		maxIndexBuffer[i] = -1.f;
	}

	int totalChannels = source.Channels();
	// Find the position and other indices
	int bc;
	int index;
	if(GetCudaTaskIndex2D(source.ObjectCount() * totalChannels, poolSizeNorm, bc, index)) {
		// Find the maximum of the 'combine' values and put them into the buffer
		int combine = (poolSize + blockDim.x - 1) / blockDim.x;
		int step;
		int count = GetCudaTaskCountAndIndex(poolSize, combine, index, step);

		int b = bc / totalChannels;
		int c = bc % totalChannels;

		const float* curSourceData = sourceData + ( b * poolSize + index ) * totalChannels + c;
		for(int i = 0; i < count; ++i) {
			float nextValue = __ldg(curSourceData);
			float nextIndex = (float)index;
			for(int j = 0; j < maxCount; ++j) {
				if(nextValue >= buffer[j]) {
					float preValue = buffer[j];
					float preIndex = maxIndexBuffer[j];
					buffer[j] = nextValue;
					maxIndexBuffer[j] = nextIndex;
					nextValue = preValue;
					nextIndex = preIndex;
				}
			}

			index += step;
			curSourceData += step * totalChannels;
		}
	}
	// Merge the buffers
	int toMergeCount = blockDim.x;
	int threadNum = threadIdx.x;
	while(toMergeCount > 1) {
		__syncthreads();

		bool isOdd = (toMergeCount % 2) != 0;
		int mergedBufferCount = toMergeCount / 2;

		if((threadNum % 2) == 0) {
			threadNum /= 2;
			if(threadNum == mergedBufferCount) {
				// The buffer number was odd; for this thread (last) no merge is required
				buffer += mergedBufferCount * bufferStep;
				maxIndexBuffer += mergedBufferCount * bufferStep;
			} else {
				float* resultBuffer = bufferEnd + bufferStep * threadNum;
				float* resultMaxIndexBuffer = resultBuffer + maxCount;
				MergeBuffers(maxCount, resultBuffer, resultMaxIndexBuffer,
					buffer, maxIndexBuffer, buffer + bufferStep, maxIndexBuffer + bufferStep);
				buffer = resultBuffer;
				maxIndexBuffer = resultMaxIndexBuffer;

				if(isOdd) {
					buffer -= bufferStep;
					maxIndexBuffer -= bufferStep;
				}
			}
			bufferEnd += bufferStep * mergedBufferCount;
		}

		toMergeCount = mergedBufferCount;
		if(isOdd) {
			++toMergeCount;
		}
	}

	if(threadIdx.x == 0) {
		// Copy the result into final blobs
		int channelNum = bc % totalChannels;
		int batchNum = bc / totalChannels;
		if(batchNum < result.ObjectCount() && channelNum < totalChannels) {
			float* curResultData = GetBlobPtr(result, resultData, batchNum, 0, 0, channelNum);
			int* maxIndexData = GetBlobPtr(maxIndices, maxIndicesData, batchNum, 0, 0, channelNum);
			for(int i = 0; i < maxCount; ++i) {
				*curResultData = *buffer++;
				*maxIndexData = *maxIndexBuffer++;

				curResultData += totalChannels;
				maxIndexData += totalChannels;
			}
		}
	}
}

const int BlobGlobalMaxPoolingBackwardCombine = 8;
__global__ void BlobGlobalMaxPoolingBackwardKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* __restrict__ outputDiffData,
	const int* maxIndicesData, float* inputDiffData, int poolSize, int maxCount, int fullSize )
{
	int index;
	int step;
	int count = GetCudaTaskCountAndIndex(fullSize, BlobGlobalMaxPoolingBackwardCombine, index, step);

	int totalChannels = desc.Result.Channels();

	for(int i = 0; i < count; ++i) {
		int channel = index % totalChannels;
		int batchNum = index / totalChannels / maxCount;

		int inputIndex = maxIndicesData[index];
		if( inputIndex >= 0 ) {
			inputDiffData[(batchNum * poolSize + inputIndex ) * totalChannels + channel] = __ldg(outputDiffData + index);
		}

		index += step;
	}
}

} // namespace NeoML
