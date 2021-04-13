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

struct IndexedHeap {
	float* data;
	float* indices;
	int size;
	__device__ IndexedHeap( float* d, float* ind, int maxCount ) : data( d ), indices( ind ), size( maxCount ) {}

	__device__ void swap( int a, int b ) {
		float temp = data[a];
		data[a] = data[b];
		data[b] = temp;
		temp = indices[a];
		indices[a] = indices[b];
		indices[b] = temp;
	}

	__device__ bool is_less( int left, int right ) {
		return ( data[left] == data[right] ) ? indices[left] > indices[right] : data[left] < data[right];
	}

	__device__ void heapify( int node, int size ) {
		while (true) {
			const int left = 2 * node + 1;
			const int right = left + 1;
			int smallest = node;
			if ( left < size && is_less( left, smallest ) ) {
				smallest = left;
			}
			if ( right < size && is_less( right, smallest ) ) {
				smallest = right;
			}
			if ( smallest == node ) {
				break;
			}
			swap( smallest, node );
			node = smallest;
		}
	}

	__device__ void build_heap() {
		for ( int node = ( size - 1 ) / 2; node >= 0; node-- ) {
			heapify( node, size );
		}
	}

	__device__ void insert( float val, int ind ) {
		if( ( data[0] == val ) ? indices[0] > ind : data[0] < val ) {
			data[0] = val;
			indices[0] = ind;
			heapify( 0, size );
		}
	}

	__device__ void sort() {
		for ( int cnt = size - 1; cnt > 0; cnt-- ) {
			swap( cnt, 0 );
			heapify( 0, cnt );
		}
	}
};

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
	IndexedHeap heap( buffer, maxIndexBuffer, maxCount );

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
			heap.insert( __ldg( curSourceData ), index );
			index += step;
			curSourceData += step * totalChannels;
		}
	}
	heap.sort();

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

__device__ inline unsigned FloatToUnsigned( const float* ptr ) {
	unsigned res = *( unsigned* )ptr;
	unsigned sign = 1 << ( sizeof( float ) * 8 - 1 );
	if( res & sign ) {
		res = ~res;
	} else {
		res ^= sign;
	}
	return res;
}

__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingLocalSortKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* sourceData,
	int* indicesSorted1, int* indicesSorted2, int poolSize, int bin, int histSize, int* local, int* global )
{
	extern __shared__ float sharedData[];

	const CCudaBlobDesc& source = desc.Source;

	int* sharedHistogram = (int*)( sharedData + ( threadIdx.x * blockDim.y + threadIdx.y ) * histSize );

	int totalChannels = source.Channels();
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int* globalSums = global + x * ( gridDim.y + 1 ) * histSize;
	int* localSums = local + x * gridDim.y * histSize;
	int b = x / totalChannels;
	int c = x % totalChannels;
	int countPerBlock = ( poolSize + gridDim.y - 1 ) / gridDim.y;
	int countPerThread = ( countPerBlock + blockDim.y - 1 ) / blockDim.y;
	int count = min( countPerThread, min( countPerBlock, poolSize - (int)blockIdx.y * countPerBlock ) - (int)threadIdx.y * countPerThread );

	for( int i = 0; i < histSize; ++i ) {
		sharedHistogram[i] = 0;
	}

	// build histogram for each thread
	if( b < source.ObjectCount() && c < totalChannels ) {
		int curIndex = ( b * poolSize + blockIdx.y * countPerBlock + threadIdx.y * countPerThread ) * totalChannels + c;

		for( int i = 0; i < count; ++i ) {
			int& sourceIndex = indicesSorted1[curIndex];
			if( bin == 0 ) {
				sourceIndex = curIndex;
			}
			unsigned int value = FloatToUnsigned( sourceData + sourceIndex );
			unsigned int histValue = ~( value >> bin ) & ( histSize - 1 );
			sharedHistogram[histValue] += 1;
			curIndex += totalChannels;
		}
	}

	__syncthreads();

	// merge histograms by prefix sum in two steps

	// up-sweep step
	int bitSum = ( threadIdx.y + 1 ) & 1;
	for( int step = 2; step <= blockDim.y; step *= 2 ) {
		if( bitSum == 0 ) {
			for( int j = 0; j < histSize; ++j ) {
				sharedHistogram[j] += ( sharedHistogram - step / 2 * histSize )[j];
			}
		}
		bitSum += ( threadIdx.y + 1 ) & step;
		__syncthreads();
	}

	// down-sweep step
	if( threadIdx.y == blockDim.y - 1 ) {
		localSums[blockIdx.y * histSize] = 0;
		for( int j = 0; j < histSize - 1; ++j ) {
			localSums[blockIdx.y * histSize + j + 1] = localSums[blockIdx.y * histSize + j] + sharedHistogram[j];
			globalSums[blockIdx.y * histSize + j] = sharedHistogram[j];
			sharedHistogram[j] = 0;
		}
		globalSums[blockIdx.y * histSize + histSize - 1] = sharedHistogram[histSize - 1];
		sharedHistogram[histSize - 1] = 0;
	}
	__syncthreads();

	for( int step = blockDim.y; step > 1; step /= 2 ) {
		bitSum -= ( threadIdx.y + 1 ) & step;
		if( bitSum == 0 ) {
			for( int j = 0; j < histSize; ++j ) {
				int t = ( sharedHistogram - step / 2 * histSize )[j];
				( sharedHistogram - step / 2 * histSize )[j] = sharedHistogram[j];
				sharedHistogram[j] += t;
			}
		}
		__syncthreads();
	}

	// local sort in block
	if( b < source.ObjectCount() && c < totalChannels ) {
		int globalBlockIndex = ( b * poolSize + blockIdx.y * countPerBlock ) * totalChannels + c;
		int curIndex = ( b * poolSize + blockIdx.y * countPerBlock + threadIdx.y * countPerThread ) * totalChannels + c;

		for( int i = 0; i < count; ++i ) {
			int sourceIndex = indicesSorted1[curIndex];
			unsigned int value = FloatToUnsigned( sourceData + sourceIndex );
			unsigned int histValue = ~( value >> bin ) & ( histSize - 1 );
			int newIndex = globalBlockIndex + ( sharedHistogram[histValue] + localSums[blockIdx.y * histSize + histValue] ) * totalChannels;
			indicesSorted2[newIndex] = sourceIndex;
			curIndex += totalChannels;
			sharedHistogram[histValue]++;
		}
	}
}

__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingGlobalScanKernel( const CCudaGlobalMaxPoolingDescInternal desc, int histSize, int* global, int blockCountY )
{
	extern __shared__ float sharedData[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int* globalSums = global + x * ( blockCountY + 1 ) * histSize;
	int* sharedHistogram = ( int* )( sharedData + ( threadIdx.x * blockDim.y + threadIdx.y ) * histSize );
	
	int countPerThread = ( blockCountY + blockDim.y - 1 ) / blockDim.y;
	int index = threadIdx.y * countPerThread;
	int count = min( countPerThread, blockCountY - ( int )index );

	for( int i = 0; i < histSize; ++i ) {
		int temp = 0;
		sharedHistogram[i] = 0;
		for( int j = 0; j < count; ++j ) {
			sharedHistogram[i] += globalSums[( index + j ) * histSize + i];
			globalSums[( index + j ) * histSize + i] = temp;
			temp = sharedHistogram[i];
		}
	}

	__syncthreads();

	// prefix sum for threads in block in two steps

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
		globalSums[blockCountY * histSize] = 0;
		for( int j = 0; j < histSize - 1; ++j ) {
			// overall sum for histogram value
			globalSums[blockCountY * histSize + j + 1] = globalSums[blockCountY * histSize + j] + sharedHistogram[j];
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

	// global positions of block histogram values
	for( int i = 0; i < histSize; ++i ) {
		for( int j = 0; j < count; ++j ) {
			globalSums[( index + j ) * histSize + i] += sharedHistogram[i];
		}
	}
}

__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingGlobalShuffleKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* sourceData,
	int* indicesSorted1, int* indicesSorted2, int bin, int histSize, int poolSize, int* local, int* global, float* resultData, int* resultIndices, int maxCount, bool isLast )
{
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& maxIndices = desc.MaxIndices;
	const CCudaBlobDesc& result = desc.Result;

	int totalChannels = source.Channels();
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int b = x / totalChannels;
	int c = x % totalChannels;
	int* globalSums = global + x * ( gridDim.y + 1 ) * histSize;
	int* localSums = local + x * gridDim.y * histSize;
	int countPerBlock = ( poolSize + gridDim.y - 1 ) / gridDim.y;
	int countPerThread = ( countPerBlock + blockDim.y - 1 ) / blockDim.y;
	int localPos = threadIdx.y * countPerThread;
	int count = min( countPerThread, min( countPerBlock, poolSize - ( int )blockIdx.y * countPerBlock ) - localPos );
	int index = ( b * poolSize + blockIdx.y * countPerBlock + localPos ) * totalChannels + c;

	// global sort using local and global positions of histogram values
	if( b < source.ObjectCount() && c < totalChannels ) {
		for( int i = 0; i < count; i++ ) {
			int sourceIndex = indicesSorted1[index];
			unsigned int value = FloatToUnsigned( sourceData + sourceIndex );
			unsigned int histValue = ~( value >> bin ) & ( histSize - 1 );
			int localIndex = localPos + i - localSums[blockIdx.y * histSize + histValue];
			int globalIndex = globalSums[blockIdx.y * histSize + histValue] + globalSums[gridDim.y * histSize + histValue];
			int pos = globalIndex + localIndex;
			indicesSorted2[( b * poolSize + pos ) * totalChannels + c] = sourceIndex;
			if( isLast && pos < maxCount ) {
				int resultIndex = ( b * maxCount + pos ) * totalChannels + c;
				resultIndices[resultIndex] = ( sourceIndex - c ) / totalChannels - b * poolSize;
				resultData[resultIndex] = sourceData[sourceIndex];
			}
			index += totalChannels;
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
