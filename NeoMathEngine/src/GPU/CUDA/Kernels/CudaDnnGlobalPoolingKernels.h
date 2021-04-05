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
__global__ void BlobGlobalMaxPoolingHeapKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* sourceData,
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

__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingSortKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* sourceData,
	int* maxIndicesData, int* indicesSorted, float* resultData, int poolSize, int maxCount, int poolSizeNorm, int numBins )
{
	extern __shared__ float sharedData[];

	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& maxIndices = desc.MaxIndices;
	const CCudaBlobDesc& result = desc.Result;

	unsigned int histSize = 1 << numBins;
	int* sharedHistogram = (int*)( sharedData + ( threadIdx.x * ( blockDim.y + 1 ) + threadIdx.y ) * histSize );
	int* sumSharedHistogram = (int*)( sharedData + ( threadIdx.x * ( blockDim.y + 1 ) + blockDim.y ) * histSize );
	int inOffset = 0;
	int outOffset = source.BlobSize();

	int totalChannels = source.Channels();
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int b = x / totalChannels;
	int c = x % totalChannels;
	int combine = ( poolSize + blockDim.y - 1 ) / blockDim.y;
	int count = min( combine, poolSize - threadIdx.y * combine );

	for( int bin = 0; bin < 32 / numBins; bin += numBins ) {
		for( int i = 0; i < histSize; ++i ) {
			sharedHistogram[i] = 0;
		}

		// build histogram for each thread
		if( b < source.ObjectCount() && c < totalChannels ) {
			int curIndex = inOffset + ( b * poolSize + threadIdx.y * combine ) * totalChannels + c;

			for( int i = 0; i < count; ++i ) {
				int& sourceIndex = indicesSorted[curIndex];
				if( bin == 0 ) {
					sourceIndex = curIndex;
				}
				unsigned int value = ( unsigned int )__ldg( sourceData + sourceIndex );
				unsigned int histValue = ( value >> bin ) & ( histSize - 1 );
				sharedHistogram[histValue] += 1;
				curIndex += totalChannels;
			}
		}

		__syncthreads();

		// prefix sum for threads in two steps

		// up-sweep step
		for( int step = 2; step <= blockDim.y; step *= 2 ) {
			if( ( threadIdx.y + 1 ) % step == 0 ) {
				for( int j = 0; j < histSize; ++j ) {
					sharedHistogram[j] += ( sharedHistogram - step / 2 * histSize )[j];
				}
			}
			__syncthreads();
		}

		// down-sweep step
		if( threadIdx.y == blockDim.y - 1 ) {
			sumSharedHistogram[0] = 0;
			for( int j = 0; j < histSize - 1; ++j ) {
				sumSharedHistogram[j + 1] = sumSharedHistogram[j] + sharedHistogram[j];
				sharedHistogram[j] = 0;
			}
			sharedHistogram[histSize - 1] = 0;
		}
		__syncthreads();

		for( int step = blockDim.y; step > 1; step /= 2 ) {
			if( ( threadIdx.y + 1 ) % step == 0 ) {
				for( int j = 0; j < histSize; ++j ) {
					int t = ( sharedHistogram - step / 2 * histSize )[j];
					( sharedHistogram - step / 2 * histSize )[j] = sharedHistogram[j];
					sharedHistogram[j] += t;
				}
			}
			__syncthreads();
		}

		if( b < source.ObjectCount() && c < totalChannels ) {
			int curIndex = inOffset + ( b * poolSize + threadIdx.y * combine ) * totalChannels + c;
			int batchIndex = b * poolSize * totalChannels;

			for( int i = 0; i < count; ++i ) {
				int sourceIndex = indicesSorted[curIndex];
				unsigned int value = ( unsigned int )__ldg( sourceData + sourceIndex );
				unsigned int histValue = ( value >> bin ) & ( histSize - 1 );
				int newIndex = outOffset + batchIndex + ( sharedHistogram[histValue] + sumSharedHistogram[histValue] ) * totalChannels + c;
				indicesSorted[newIndex] = count; // sourceIndex;
				curIndex += totalChannels;
				sharedHistogram[histValue]++;
			}
		}
		__syncthreads();

		inOffset ^= source.BlobSize();
		outOffset ^= source.BlobSize();
	}


	if( b < source.ObjectCount() && c < totalChannels && threadIdx.y == 0 ) {
		int sortedIndex = outOffset + (( b + 1 ) * poolSize - 1 ) * totalChannels + c;
		int outIndex = b * maxCount * totalChannels + c;
		for( int i = 0; i < maxCount; ++i ) {
			int index = indicesSorted[sortedIndex];
			maxIndicesData[outIndex] = ( index - c ) / totalChannels - b * poolSize;
			resultData[outIndex] = sourceData[index];
			sortedIndex -= totalChannels;
			outIndex += totalChannels;
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
