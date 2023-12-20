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

__global__ void Blob3dMaxPoolingKernel( const CCuda3dMaxPoolingDescInternal desc, const float* __restrict__ sourceData,
	int* __restrict__ maxIndices, float* __restrict__ resultData )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	const int resultGeomSize = result.Depth() * result.Height() * result.Width();
	const int resultObjectSize = result.Channels() * resultGeomSize;
	const int totalChannels = source.Channels();

	int b;
	int pos;
	int channel;
	if( !GetCudaTaskIndex3D( result.ObjectCount(), resultGeomSize, totalChannels, b, pos, channel ) ) {
		return;
	}

	const int resultShift = b * resultObjectSize + pos * totalChannels + channel;

	const int sourceHW = source.Height() * source.Width();
	const int sourceGeomSize = source.Depth() * sourceHW;
	const float* sourcePtr = sourceData + b * totalChannels * sourceGeomSize;

	// Output position
	const int kOut = pos % result.Depth();
	pos /= result.Depth();
	const int iOut = pos % result.Width();
	const int jOut = pos / result.Width();

	// Input position
	const int jStart = jOut * desc.StrideHeight;
	const int iStart = iOut * desc.StrideWidth;
	const int kStart = kOut * desc.StrideDepth;

	float maxValue = -FLT_MAX;
	int maxIndex = 0;
	int jIndex = jStart * source.Width() * source.Depth() * totalChannels + channel;

	for( int j = 0; j < desc.FilterHeight; ++j ) {
		int iIndex = jIndex + iStart * source.Depth() * totalChannels;
		for( int i = 0; i < desc.FilterWidth; ++i ) {
			int index = iIndex + kStart * totalChannels;
			for( int k = 0; k < desc.FilterDepth; ++k ) {
				const float value = __ldg( sourcePtr + index );
				if( value >= maxValue ) {
					maxIndex = index;
					maxValue = value;
				}
				index += totalChannels;
			}
			iIndex += source.Depth() * totalChannels;
		}
		jIndex += source.Width() * source.Depth() * totalChannels;
	}

	resultData[resultShift] = maxValue;
	if( maxIndices != 0 ) {
		maxIndices[resultShift] = maxIndex - channel;
	}
}

__global__ void Blob3dMaxPoolingBackwardKernel( const CCuda3dMaxPoolingDescInternal desc, const float* __restrict__ resultDiff,
	const int* __restrict__ maxIndices, float* __restrict__ sourceDiff, bool isAtomic )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	const int resultGeomSize = result.Depth() * result.Height() * result.Width();

	int b;
	int pos;
	int channel;
	if( !GetCudaTaskIndex3D( result.ObjectCount(), resultGeomSize, result.Channels(), b, pos, channel ) ) {
		return;
	}

	float* sourcePtr = sourceDiff + b * source.ObjectSize();

	const int resultShift = ( b * resultGeomSize + pos ) * result.Channels() + channel;
	const int index = maxIndices[resultShift] + channel;
	const float value = __ldg( resultDiff + resultShift );

	if( isAtomic ) {
		atomicAdd( sourcePtr + index, value );
	} else {
		sourcePtr[index] = value;
	}
}

__global__ void Blob3dMeanPoolingKernel( const CCuda3dMeanPoolingDescInternal desc, const float* sourceData,
	float* resultData )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	const int resultGeomSize = result.Depth() * result.Height() * result.Width();
	const int totalChannels = result.Channels();
	const int resultObjectSize = totalChannels * resultGeomSize;

	int b;
	int pos;
	int channel;
	if( !GetCudaTaskIndex3D( result.ObjectCount(), resultGeomSize, result.Channels(), b, pos, channel ) ) {
		return;
	}

	const int resultShift = b * resultObjectSize + pos * totalChannels + channel;

	const int sourceGeomSize = source.Depth() * source.Height() * source.Width();
	const float* sourcePtr = sourceData + b * totalChannels * sourceGeomSize;

	// Output position
	const int kOut = pos % result.Depth();
	pos /= result.Depth();
	const int iOut = pos % result.Width();
	const int jOut = pos / result.Width();

	// Input position
	const int jStart = jOut * desc.StrideHeight;
	const int iStart = iOut * desc.StrideWidth;
	const int kStart = kOut * desc.StrideDepth;

	float sumValue = 0;
	int jIndex = jStart * source.Width() * source.Depth() * totalChannels + channel;

	for( int j = 0; j < desc.FilterHeight; ++j ) {
		int iIndex = jIndex + iStart * source.Depth() * totalChannels;
		for( int i = 0; i < desc.FilterWidth; ++i ) {
			int index = iIndex + kStart * totalChannels;
			for( int k = 0; k < desc.FilterDepth; ++k ) {
				sumValue += __ldg( sourcePtr + index );
				index += totalChannels;
			}
			iIndex += source.Depth() * totalChannels;
		}
		jIndex += source.Width() * source.Depth() * totalChannels;
	}

	resultData[resultShift] = sumValue / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth;
}

__global__ void Blob3dMeanPoolingBackwardKernel( const CCuda3dMeanPoolingDescInternal desc, const float* resultDiff,
	float* sourceDiff, bool isAtomic )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	const int resultGeomSize = result.Depth() * result.Height() * result.Width();
	const int totalChannels = result.Channels();

	int b;
	int pos;
	int channel;
	if( !GetCudaTaskIndex3D( result.ObjectCount(), resultGeomSize, totalChannels, b, pos, channel ) ) {
		return;
	}

	const int resultShift = ( b * resultGeomSize + pos ) * totalChannels + channel;
	const float value = __ldg( resultDiff + resultShift ) / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth;

	// Output position
	const int kOut = pos % result.Depth();
	pos /= result.Depth();
	const int iOut = pos % result.Width();
	const int jOut = pos / result.Width();

	// Input position
	const int jStart = jOut * desc.StrideHeight;
	const int iStart = iOut * desc.StrideWidth;
	const int kStart = kOut * desc.StrideDepth;

	float* sourcePtr = sourceDiff + ( ( ( b * source.Height() + jStart ) * source.Width() + iStart ) * source.Depth() + kStart ) * totalChannels + channel;
	const int sourceWDC = source.Width() * source.Depth() * source.Channels();

	if( isAtomic ) {
		for( int j = 0; j < desc.FilterHeight; ++j ) {
			float* sourceColumnData = sourcePtr;
			for( int i = 0; i < desc.FilterWidth; ++i ) {
				float* sourcePixelData = sourceColumnData;
				for( int k = 0; k < desc.FilterDepth; ++k ) {
					atomicAdd( sourcePixelData, value );
					sourcePixelData += totalChannels;
				}
				sourceColumnData += source.Depth() * totalChannels;
			}
			sourcePtr += sourceWDC;
		}
	} else {
		for( int j = 0; j < desc.FilterHeight; ++j ) {
			float* sourceColumnData = sourcePtr;
			for( int i = 0; i < desc.FilterWidth; ++i ) {
				float* sourcePixelData = sourceColumnData;
				for( int k = 0; k < desc.FilterDepth; ++k ) {
					*sourcePixelData = value;
					sourcePixelData += totalChannels;
				}
				sourceColumnData += source.Depth() * totalChannels;
			}
			sourcePtr += sourceWDC;
		}
	}
}

} // namespace NeoML
