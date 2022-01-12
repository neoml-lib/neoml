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

enum HeapType { MinHeap, MaxHeap };

struct IndexedValue {
	float val;
	float ind;

	__device__ bool isLess( const IndexedValue& other ) {
		return ( val == other.val ) ? ind > other.ind : val < other.val;
	}

	__device__ void swap( IndexedValue& other ) {
		float tmp = val;
		val = other.val;
		other.val = tmp;
		tmp = ind;
		ind = other.ind;
		other.ind = tmp;
	}
};

template<HeapType T>
struct Heap {
	IndexedValue* data;
	int size;
	__device__ Heap( float* _data, int maxCount ) : data( reinterpret_cast<IndexedValue*>( _data ) ), size( maxCount ) {}

	__device__ bool isLess( int left, int right ) {
		if( T == HeapType::MinHeap ) {
			return data[left].isLess( data[right] );
		} else {
			return data[right].isLess( data[left] );
		}
	}

	__device__ void heapify( int node, int size ) {
		while (true) {
			const int left = 2 * node + 1;
			const int right = left + 1;
			int smallest = node;
			if ( left < size && isLess( left, smallest ) ) {
				smallest = left;
			}
			if ( right < size && isLess( right, smallest ) ) {
				smallest = right;
			}
			if ( smallest == node ) {
				break;
			}
			data[smallest].swap( data[node] );
			node = smallest;
		}
	}

	__device__ void buildHeap() {
		for ( int node = ( size - 1 ) / 2; node >= 0; node-- ) {
			heapify( node, size );
		}
	}

	__device__ void sort() {
		for ( int cnt = size - 1; cnt > 0; cnt-- ) {
			data[0].swap( data[cnt] );
			heapify( 0, cnt );
		}
	}

	__device__ IndexedValue root() {
		return data[0];
	}

	__device__ void replaceRoot( IndexedValue entry ) {
		data[0] = entry;
		heapify( 0, size );
	}

	__device__ void insert( IndexedValue entry ) {
		if( data[0].isLess( entry ) ) {
			replaceRoot( entry );
		}
	}
};

__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingHeapKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* sourceData,
	int* maxIndicesData, float* resultData, int poolSize, int maxCount )
{
	const CCudaBlobDesc& source = desc.Source;

	// Initialize the data
	extern __shared__ float sharedData[];

	int totalChannels = source.Channels();
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int b = y / totalChannels;
	int c = y % totalChannels;

	int threadCountX = blockDim.x;
	int bufferStep = 2 * maxCount;
	float* localHeap = sharedData + ( threadIdx.y * ( blockDim.x + 1 ) + threadIdx.x ) * bufferStep;
	float* globalHeap = sharedData + ( threadIdx.y * ( blockDim.x + 1 ) + blockDim.x ) * bufferStep;

	for( int i = 0; i < maxCount; ++i ) {
		localHeap[2 * i] = -FLT_MAX;
		localHeap[2 * i + 1] = -1;
	}

	Heap<HeapType::MinHeap> heap( localHeap, maxCount );

	if( b < source.ObjectCount() && c < totalChannels ) {
		const float* curSourceData = sourceData + ( b * poolSize + threadIdx.x ) * totalChannels + c;
		for( int ind = threadIdx.x; ind < poolSize; ind += threadCountX ) {
			heap.insert( { __ldg( curSourceData ), float( ind ) } );
			curSourceData += threadCountX * totalChannels;
		}
		heap.sort();
	}
	__syncthreads();

	if( threadIdx.x == 0 && b < source.ObjectCount() && c < totalChannels  ) {
		for(int i = 0; i < maxCount; ++i) {
			globalHeap[2 * i] = -FLT_MAX;
			globalHeap[2 * i + 1] = -1;
		}

		// add max from each thread to min heap
		Heap<HeapType::MinHeap> minHeap( globalHeap, maxCount );
		for( int i = 0; i < threadCountX; ++i ) {
			minHeap.insert( { localHeap[i * bufferStep], float( i * bufferStep ) } );
		}

		// build max heap and extract maximum maxCount times
		Heap<HeapType::MaxHeap> maxHeap( globalHeap, maxCount );
		maxHeap.buildHeap();
		for( int i = 0, resIndex = b * maxCount * totalChannels + c; i < maxCount; ++i, resIndex += totalChannels ) {
			const IndexedValue& root = maxHeap.root();
			resultData[resIndex] = root.val;
			int rootIndex = ( int )root.ind;
			if( rootIndex == -1 ) {
				maxIndicesData[resIndex] = -1;
			} else {
				IndexedValue* threadMax = reinterpret_cast< IndexedValue* >( localHeap + rootIndex );
				maxIndicesData[resIndex] = ( int )threadMax->ind;
				IndexedValue* threadNextMax = reinterpret_cast< IndexedValue* >( localHeap + rootIndex + 2 );
				maxHeap.replaceRoot( { threadNextMax->val, float( rootIndex + 2 ) } );
			}
		}
	}
}

__device__ inline unsigned FloatToUnsigned( const float* ptr )
{
	unsigned res = *( unsigned* )ptr;
	unsigned sign = 1U << ( sizeof( float ) * 8 - 1 );
	if( res & sign ) {
		res = ~res;
	} else {
		res ^= sign;
	}
	return res;
}

__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingLocalSortKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* sourceData,
	int* indicesSorted1, int* indicesSorted2, int poolSize, int bin, int histSize, int* local, int* global  )
{
	extern __shared__ float sharedData[];

	const CCudaBlobDesc& source = desc.Source;

	int* startHistogram = (int*)( sharedData + threadIdx.x * blockDim.y * histSize );
	int* threadHistogram = (int*)( sharedData + ( threadIdx.x * blockDim.y + threadIdx.y ) * histSize );

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
		threadHistogram[i] = 0;
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
			threadHistogram[histValue] += 1;
			curIndex += totalChannels;
		}
	}

	__syncthreads();

	// merge histograms by prefix sum in two steps

	// up-sweep step
	int offset = 1;
	for( int d = blockDim.y >> 1; d > 0; d >>= 1 ) {
		if( threadIdx.y < d ) {
			int* left = startHistogram + ( offset * ( 2 * threadIdx.y + 1 ) - 1 ) * histSize;
			int* right = startHistogram + ( offset * ( 2 * threadIdx.y + 2 ) - 1 ) * histSize;
			for( int j = 0; j < histSize; ++j, ++right, ++left ) {
				*right += *left;
			}
		}
		offset *= 2;
		__syncthreads();
	}

	// down-sweep step
	if( threadIdx.y == blockDim.y - 1 ) {
		localSums[blockIdx.y * histSize] = 0;
		for( int j = 0; j < histSize - 1; ++j ) {
			localSums[blockIdx.y * histSize + j + 1] = localSums[blockIdx.y * histSize + j] + threadHistogram[j];
			globalSums[blockIdx.y * histSize + j] = threadHistogram[j];
			threadHistogram[j] = 0;
		}
		globalSums[blockIdx.y * histSize + histSize - 1] = threadHistogram[histSize - 1];
		threadHistogram[histSize - 1] = 0;
	}
	__syncthreads();

	for( int d = 1; d < blockDim.y; d *= 2 ) {
		offset >>= 1;
		if( threadIdx.y < d ) {
			int* left = startHistogram + ( offset * ( 2 * threadIdx.y + 1 ) - 1 ) * histSize;
			int* right = startHistogram + ( offset * ( 2 * threadIdx.y + 2 ) - 1 ) * histSize;
			for( int j = 0; j < histSize; ++j, ++left, ++right ) {
				int t = *left;
				*left = *right;
				*right += t;
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
			int newIndex = globalBlockIndex + ( threadHistogram[histValue] + localSums[blockIdx.y * histSize + histValue] ) * totalChannels;
			indicesSorted2[newIndex] = sourceIndex;
			curIndex += totalChannels;
			threadHistogram[histValue]++;
		}
	}
}

__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingGlobalScanKernel( const CCudaGlobalMaxPoolingDescInternal desc, int histSize, int* global, int blockCountY )
{
	extern __shared__ float sharedData[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int* globalSums = global + x * ( blockCountY + 1 ) * histSize;
	int* startHistogram = ( int* )( sharedData + threadIdx.x * blockDim.y * histSize );
	int* threadHistogram = ( int* )( sharedData + ( threadIdx.x * blockDim.y + threadIdx.y ) * histSize );
	
	int countPerThread = ( blockCountY + blockDim.y - 1 ) / blockDim.y;
	int index = threadIdx.y * countPerThread;
	int count = min( countPerThread, blockCountY - ( int )index );

	for( int i = 0; i < histSize; ++i ) {
		int temp = 0;
		threadHistogram[i] = 0;
		for( int j = 0; j < count; ++j ) {
			threadHistogram[i] += globalSums[( index + j ) * histSize + i];
			globalSums[( index + j ) * histSize + i] = temp;
			temp = threadHistogram[i];
		}
	}

	__syncthreads();

	// prefix sum for threads in block in two steps

	// up-sweep step
	int offset = 1;
	for( int d = blockDim.y >> 1; d > 0; d >>= 1 ) {
		if( threadIdx.y < d ) {
			int* left = startHistogram + ( offset * ( 2 * threadIdx.y + 1 ) - 1 ) * histSize;
			int* right = startHistogram + ( offset * ( 2 * threadIdx.y + 2 ) - 1 ) * histSize;
			for( int j = 0; j < histSize; ++j, ++right, ++left ) {
				*right += *left;
			}
		}
		offset *= 2;
		__syncthreads();
	}

	// down-sweep step
	if( threadIdx.y == blockDim.y - 1 ) {
		globalSums[blockCountY * histSize] = 0;
		for( int j = 0; j < histSize - 1; ++j ) {
			// overall sum for histogram value
			globalSums[blockCountY * histSize + j + 1] = globalSums[blockCountY * histSize + j] + threadHistogram[j];
			threadHistogram[j] = 0;
		}
		threadHistogram[histSize - 1] = 0;
	}
	__syncthreads();

	for( int d = 1; d < blockDim.y; d *= 2 ) {
		offset >>= 1;
		if( threadIdx.y < d ) {
			int* left = startHistogram + ( offset * ( 2 * threadIdx.y + 1 ) - 1 ) * histSize;
			int* right = startHistogram + ( offset * ( 2 * threadIdx.y + 2 ) - 1 ) * histSize;
			for( int j = 0; j < histSize; ++j, ++left, ++right ) {
				int t = *left;
				*left = *right;
				*right += t;
			}
		}
		__syncthreads();
	}
	// global positions of each block histogram values
	for( int i = 0; i < histSize; ++i ) {
		for( int j = 0; j < count; ++j ) {
			globalSums[( index + j ) * histSize + i] += threadHistogram[i];
		}
	}
}

__launch_bounds__( 1024, 1 )
__global__ void BlobGlobalMaxPoolingGlobalShuffleKernel( const CCudaGlobalMaxPoolingDescInternal desc, const float* sourceData,
	int* indicesSorted1, int* indicesSorted2, int bin, int histSize, int poolSize, int* local, int* global, float* resultData, int* resultIndices, int maxCount, bool isFirst, bool isLast )
{
	const CCudaBlobDesc& source = desc.Source;

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
		// initialize result
		if( isFirst ) {
			int index = blockIdx.y * blockDim.y + threadIdx.y;
			int step = gridDim.y * blockDim.y;
			for( ; index < maxCount; index += step ) {
				int resultIndex = ( b * maxCount + index ) * totalChannels + c;
				resultIndices[resultIndex] = -1;
				resultData[resultIndex] = -FLT_MAX;
			}
		}

		for( int i = 0; i < count; i++ ) {
			int sourceIndex = indicesSorted1[index];
			unsigned int value = FloatToUnsigned( sourceData + sourceIndex );
			unsigned int histValue = ~( value >> bin ) & ( histSize - 1 );
			int localIndex = localPos + i - localSums[blockIdx.y * histSize + histValue];
			int globalIndex = globalSums[blockIdx.y * histSize + histValue] + globalSums[gridDim.y * histSize + histValue];
			int pos = globalIndex + localIndex;
			if( isLast && pos < maxCount ) {
				// fill result
				int resultIndex = ( b * maxCount + pos ) * totalChannels + c;
				resultIndices[resultIndex] = ( sourceIndex - c ) / totalChannels - b * poolSize;
				resultData[resultIndex] = sourceData[sourceIndex];
			} else {
				indicesSorted2[( b * poolSize + pos ) * totalChannels + c] = sourceIndex;
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
