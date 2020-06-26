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

__global__ void BlobMaxPoolingKernel( const CCudaMaxPoolingDescInternal desc, const float* sourceData,
	int* maxIndices, float* resultData )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	int totalChannels = result.Depth() * result.Channels();

	int num, j, i, channel;
	int sourceRowSize;
	int sourceItemSize;

	const float* sourcePtr;
	int resultPos;

	if(!GetCudaTaskIndex3D(result.ObjectCount(), result.Height() * result.Width(), totalChannels, num, j, channel)) {
		return;
	}

	i = j % result.Width();
	j /= result.Width();

	sourceRowSize = source.Width() * totalChannels;
	sourceItemSize = totalChannels;

	int sourceJ = j * desc.StrideHeight;
	int sourceI = i * desc.StrideWidth;

	sourcePtr = GetBlobPtr(source, sourceData, num, sourceJ, sourceI, channel);
	resultPos = GetBlobPos(result, num, j, i, channel);

	int startIndexPos = GetBlobPos(source, 0, sourceJ, sourceI, channel);

	float resultValue = -FLT_MAX;
	int index = startIndexPos;

	for(int jStep = 0; jStep < desc.FilterHeight; ++jStep) {
		const float* sourceItemPtr = sourcePtr;
		for(int iStep = 0; iStep < desc.FilterWidth; ++iStep) {
			float value = __ldg(sourceItemPtr);
			if(resultValue < value) {
				resultValue = value;
				index = startIndexPos + iStep * sourceItemSize;
			}
			sourceItemPtr += sourceItemSize;
		}
		sourcePtr += sourceRowSize;
		startIndexPos += sourceRowSize;
	}

	resultData[resultPos] = resultValue;
	if(maxIndices != 0) {
		maxIndices[resultPos] = index;
	}
}

const int BlobMaxPoolingBackwardCombine = 16;
__global__ void BlobMaxPoolingBackwardKernel( const CCudaMaxPoolingDescInternal desc, bool isAtomic, float* outputDiffData,
	int* maxIndicesData, float* inputDiffData, int batchNorm )
{
	const CCudaBlobDesc& outputDiff = desc.Result;

	int b;
	int index;

	int inputObjectSize = desc.Source.ObjectSize();
	int totalChannels = outputDiff.Depth() * outputDiff.Channels();
	int channel;
	int hw;
	if(!GetCudaTaskIndex3D(batchNorm, outputDiff.Height() * outputDiff.Width(), totalChannels, b, hw, channel)) {
		return;
	}

	index = hw * totalChannels + channel;

	int batchStep = outputDiff.ObjectSize();

	b *= BlobMaxPoolingBackwardCombine;
	int bLast = b + BlobMaxPoolingBackwardCombine;
	if(bLast > outputDiff.ObjectCount()) {
		bLast = outputDiff.ObjectCount();
	}
	int count = bLast - b;
	index += b * batchStep;

	const float* outputDiffPtr = outputDiffData + index;
	const int* indicesPtr = maxIndicesData + index;
	inputDiffData += b * inputObjectSize;

	for(int k = 0; k < count; ++k) {
		int inputIndex = *indicesPtr;
		float value = __ldg(outputDiffPtr);
		if(isAtomic) {
			atomicAdd(inputDiffData + inputIndex, value);
		} else {
			inputDiffData[inputIndex] = value;
		}
		outputDiffPtr += batchStep;
		indicesPtr += batchStep;
		inputDiffData += inputObjectSize;
	}
}

__global__ void BlobMeanPoolingKernel( const CCudaMeanPoolingDescInternal desc, const float* sourceData, float* resultData )
{
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& result = desc.Result;

	int totalChannels = result.Depth() * result.Channels();

	int sourceRowSize;
	int sourceItemSize;

	const float* sourcePtr;
	float* resultPtr;

	sourceRowSize = source.Width() * totalChannels;
	sourceItemSize = totalChannels;

	int b;
	int channel;
	int hw;
	if(!GetCudaTaskIndex3D(result.ObjectCount(), result.Height() * result.Width(), totalChannels, b, hw, channel)) {
		return;
	}

	int i = hw % result.Width();
	int j = hw / result.Width();

	sourcePtr = GetBlobPtr(source, sourceData, b, j * desc.StrideHeight, i * desc.StrideWidth, channel);
	resultPtr = GetBlobPtr(result, resultData, b, 0, hw, channel);
	*resultPtr = 0;

	for(int jStep = 0; jStep < desc.FilterHeight; ++jStep) {
		const float* sourceItemPtr = sourcePtr;
		for(int iStep = 0; iStep < desc.FilterWidth; ++iStep) {
			*resultPtr += __ldg(sourceItemPtr);
			sourceItemPtr += sourceItemSize;
		}
		sourcePtr += sourceRowSize;
	}

	*resultPtr /= desc.FilterHeight * desc.FilterWidth;
}

__global__ void BlobMeanPoolingBackwardKernel( const CCudaMeanPoolingDescInternal desc, const float* outputDiffData,
	float* inputDiffData, bool isAtomic )
{
	const CCudaBlobDesc& outputDiff = desc.Result;
	const CCudaBlobDesc& inputDiff = desc.Source;

	int b;
	int channel;
	int pos;

	int outputGeomSize = outputDiff.Height() * outputDiff.Width();
	int totalChannels = outputDiff.Depth() * outputDiff.Channels();

	if(!GetCudaTaskIndex3D(outputDiff.ObjectCount(), outputGeomSize, totalChannels, b, pos, channel)) {
		return;
	}

	int outputShift = (b * outputGeomSize + pos ) * totalChannels + channel;
	float value = __ldg(outputDiffData + outputShift) / desc.FilterHeight / desc.FilterWidth;

	// Output position
	int iOut = pos % outputDiff.Width();
	int jOut = pos / outputDiff.Width();

	// Input position
	int jStart = jOut * desc.StrideHeight;
	int iStart = iOut * desc.StrideWidth;

	float* curInputDiffData = inputDiffData + ( ( b * inputDiff.Height() + jStart ) * inputDiff.Width() + iStart ) * totalChannels + channel;

	int inputRowSize = inputDiff.Width() * totalChannels;

	if(isAtomic) {
		for(int j = 0; j < desc.FilterHeight; ++j) {
			float* inputColumnData = curInputDiffData;
			for(int i = 0; i < desc.FilterWidth; ++i) {
				atomicAdd(inputColumnData, value);
				inputColumnData += totalChannels;
			}
			curInputDiffData += inputRowSize;
		}
	} else {
		for( int j = 0; j < desc.FilterHeight; ++j ) {
			float* inputColumnData = curInputDiffData;
			for( int i = 0; i < desc.FilterWidth; ++i ) {
				*inputColumnData = value;
				inputColumnData += totalChannels;
			}
			curInputDiffData += inputRowSize;
		}
	}
}

} // namespace NeoML
