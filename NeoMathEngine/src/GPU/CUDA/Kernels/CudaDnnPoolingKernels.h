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
#include <cfloat>

namespace NeoML {

__global__ void BlobMaxPoolingKernel( const CCudaMaxPoolingDescInternal desc, const float* sourceData,
	int* maxIndices, float* resultData )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	const int totalChannels = result.Depth() * result.Channels();

	int num;
	int j;
	int channel;
	if( !GetCudaTaskIndex3D( result.ObjectCount(), result.Height() * result.Width(), totalChannels, num, j, channel ) ) {
		return;
	}

	const int i = j % result.Width();
	j /= result.Width();

	const int sourceRowSize = source.Width() * totalChannels;
	const int sourceItemSize = totalChannels;

	const int sourceJ = j * desc.StrideHeight;
	const int sourceI = i * desc.StrideWidth;

	const float* sourcePtr = GetBlobPtr( source, sourceData, num, sourceJ, sourceI, channel );
	const int resultPos = GetBlobPos( result, num, j, i, channel );

	int startIndexPos = GetBlobPos( source, 0, sourceJ, sourceI, channel );

	float resultValue = -FLT_MAX;
	int index = startIndexPos;

	for( int jStep = 0; jStep < desc.FilterHeight; ++jStep ) {
		const float* sourceItemPtr = sourcePtr;
		for( int iStep = 0; iStep < desc.FilterWidth; ++iStep ) {
			float value = __ldg( sourceItemPtr );
			if( resultValue < value ) {
				resultValue = value;
				index = startIndexPos + iStep * sourceItemSize;
			}
			sourceItemPtr += sourceItemSize;
		}
		sourcePtr += sourceRowSize;
		startIndexPos += sourceRowSize;
	}

	resultData[resultPos] = resultValue;
	if( maxIndices != 0 ) {
		maxIndices[resultPos] = index;
	}
}

const int BlobMaxPoolingBackwardCombine = 16;
__global__ void BlobMaxPoolingBackwardKernel( const CCudaMaxPoolingDescInternal desc, bool isAtomic, const float* resultDiff,
	const int* maxIndices, float* sourceDiff, int batchNorm )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	const int totalChannels = result.Depth() * result.Channels();

	int b;
	int hw;
	int channel;
	if( !GetCudaTaskIndex3D( batchNorm, result.Height() * result.Width(), totalChannels, b, hw, channel ) ) {
		return;
	}

	b *= BlobMaxPoolingBackwardCombine;
	const int bLast = min( b + BlobMaxPoolingBackwardCombine, result.ObjectCount() );
	const int count = bLast - b;

	const int batchStep = result.ObjectSize();
	const int index = hw * totalChannels + channel + b * batchStep;
	const float* resultPtr = resultDiff + index;
	const int* indicesPtr = maxIndices + index;

	const int sourceObjectSize = source.ObjectSize();
	sourceDiff += b * sourceObjectSize;

	for( int k = 0; k < count; ++k ) {
		const int i = *indicesPtr;
		const float value = __ldg( resultPtr );
		if( isAtomic ) {
			atomicAdd( sourceDiff + i, value );
		} else {
			sourceDiff[i] = value;
		}
		resultPtr += batchStep;
		indicesPtr += batchStep;
		sourceDiff += sourceObjectSize;
	}
}

//------------------------------------------------------------------------------------------------------------

__global__ void BlobMeanPoolingKernel( const CCudaMeanPoolingDescInternal desc, const float* sourceData, float* resultData )
{
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	const int totalChannels = result.Depth() * result.Channels();

	int b;
	int hw;
	int channel;
	if( !GetCudaTaskIndex3D( result.ObjectCount(), result.Height() * result.Width(), totalChannels, b, hw, channel ) ) {
		return;
	}

	const int i = hw % result.Width();
	const int j = hw / result.Width();

	const float* sourcePtr = GetBlobPtr( source, sourceData, b, j * desc.StrideHeight, i * desc.StrideWidth, channel );
	float* resultPtr = GetBlobPtr( result, resultData, b, 0, hw, channel );
	*resultPtr = 0;

	const int sourceRowSize = source.Width() * totalChannels;
	const int sourceItemSize = totalChannels;

	for( int jStep = 0; jStep < desc.FilterHeight; ++jStep ) {
		const float* sourceItemPtr = sourcePtr;
		for( int iStep = 0; iStep < desc.FilterWidth; ++iStep ) {
			*resultPtr += __ldg( sourceItemPtr );
			sourceItemPtr += sourceItemSize;
		}
		sourcePtr += sourceRowSize;
	}
	*resultPtr /= desc.FilterHeight * desc.FilterWidth;
}

__global__ void BlobMeanPoolingBackwardKernel( const CCudaMeanPoolingDescInternal desc, const float* resultDiff,
	float* sourceDiff, bool isAtomic )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	const int resultGeomSize = result.Height() * result.Width();
	const int totalChannels = result.Depth() * result.Channels();

	int b;
	int pos;
	int channel;
	if( !GetCudaTaskIndex3D( result.ObjectCount(), resultGeomSize, totalChannels, b, pos, channel ) ) {
		return;
	}

	const int resultShift = ( b * resultGeomSize + pos ) * totalChannels + channel;
	const float value = __ldg( resultDiff + resultShift ) / desc.FilterHeight / desc.FilterWidth;

	// result position
	const int iOut = pos % result.Width();
	const int jOut = pos / result.Width();

	// source position
	const int jStart = jOut * desc.StrideHeight;
	const int iStart = iOut * desc.StrideWidth;

	float* sourcePtr = sourceDiff + ( ( b * source.Height() + jStart ) * source.Width() + iStart ) * totalChannels + channel;

	const int sourceRowSize = source.Width() * totalChannels;

	if( isAtomic ) {
		for( int j = 0; j < desc.FilterHeight; ++j ) {
			float* sourceColumnData = sourcePtr;
			for( int i = 0; i < desc.FilterWidth; ++i ) {
				atomicAdd( sourceColumnData, value );
				sourceColumnData += totalChannels;
			}
			sourcePtr += sourceRowSize;
		}
	} else {
		for( int j = 0; j < desc.FilterHeight; ++j ) {
			float* sourceColumnData = sourcePtr;
			for( int i = 0; i < desc.FilterWidth; ++i ) {
				*sourceColumnData = value;
				sourceColumnData += totalChannels;
			}
			sourcePtr += sourceRowSize;
		}
	}
}

} // namespace NeoML
