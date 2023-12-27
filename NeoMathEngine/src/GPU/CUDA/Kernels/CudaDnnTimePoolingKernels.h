/* Copyright © 2017-2023 ABBYY

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

const int BlobMaxOverTimePoolingCombine = 8;
__global__ void BlobMaxOverTimePoolingKernel( const CCudaMaxOverTimePoolingDescInternal desc, const float* sourceData,
	int* maxIndicesData, float* resultData )
{
	extern __shared__ CValueWithIndex buf[];

	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	const int objectSize = source.ObjectSize();
	const int seqElemSize = source.BatchWidth() * objectSize;

	int x;
	int pos;
	if( !GetCudaTaskIndex2D( result.BlobSize(), desc.FilterLen, pos, x ) ) {
		return;
	}

	const int seqNum = pos / seqElemSize;
	const int srcPos = pos % seqElemSize;
	const int srcSeqNumEnd = seqNum * desc.StrideLen + desc.FilterLen;
	int srcSeqNum = seqNum * desc.StrideLen + x;

	CValueWithIndex& val = buf[( threadIdx.z * blockDim.y + threadIdx.y ) * blockDim.x + threadIdx.x];
	// NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum
	val.Index = srcSeqNum;
	val.Value = __ldg( sourceData + srcSeqNum * seqElemSize + srcPos );
	srcSeqNum += blockDim.x;

	while( srcSeqNum < srcSeqNumEnd ) {
		float candidate = __ldg( sourceData + srcSeqNum * seqElemSize + srcPos );
		if( candidate > val.Value ) {
			val.Value = candidate;
			val.Index = srcSeqNum;
		}
		srcSeqNum += blockDim.x;
	}

	CValueWithIndex res = ReduceMaxWithIndexXSharedBuffer( buf );

	resultData[pos] = res.Value;
	maxIndicesData[pos] = res.Index;
}

__global__ void BlobMaxOverTimePoolingKernel( const CCudaMaxOverTimePoolingDescInternal desc, const float* sourceData,
	float* resultData )
{
	extern __shared__ float buffer[];

	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	const int objectSize = source.ObjectSize();
	const int seqElemSize = source.BatchWidth() * objectSize;

	int x;
	int pos;
	if( !GetCudaTaskIndex2D( result.BlobSize(), desc.FilterLen, pos, x ) ) {
		return;
	}

	const int seqNum = pos / seqElemSize;
	const int srcPos = pos % seqElemSize;
	const int srcSeqNumEnd = seqNum * desc.StrideLen + desc.FilterLen;
	int srcSeqNum = seqNum * desc.StrideLen + x;

	float& val = buffer[( threadIdx.z * blockDim.y + threadIdx.y ) * blockDim.x + threadIdx.x];
	// NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum
	val = __ldg( sourceData + srcSeqNum * seqElemSize + srcPos );
	srcSeqNum += blockDim.x;

	while( srcSeqNum < srcSeqNumEnd ) {
		const float candidate = __ldg( sourceData + srcSeqNum * seqElemSize + srcPos );
		if( candidate > val ) {
			val = candidate;
		}
		srcSeqNum += blockDim.x;
	}
	resultData[pos] = ReduceMaxXSharedBuffer( buffer );
}

struct CStoreSet {
	__device__ void Execute( float& acc, const float& value )
	{
		acc = value;
	}
};

struct CStoreAtomicAdd {
	__device__ void Execute( float& acc, const float& value )
	{
		atomicAdd( &acc, value );
	}
};

static const int BlobMaxOverTimePoolingBackwardCombine = 8;
template<class Store>
__global__ void BlobMaxOverTimePoolingBackwardKernel( Store store, const CCudaMaxOverTimePoolingDescInternal desc, const float* resultDiff,
	const int* maxIndices, float* sourceDiff )
{
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	int index;
	int step;
	const int count = GetCudaTaskCountAndIndex( result.BlobSize(), BlobMaxOverTimePoolingBackwardCombine, index, step );

	const int objectSize = source.ObjectSize();
	const int seqElemSize = source.BatchWidth() * objectSize;

	for( int i = 0; i < count; ++i ) {
		int pos = index % seqElemSize;

		store.Execute( sourceDiff[maxIndices[index] * seqElemSize + pos], __ldg( resultDiff + index ) );

		index += step;
	}
}

} // namespace NeoML
