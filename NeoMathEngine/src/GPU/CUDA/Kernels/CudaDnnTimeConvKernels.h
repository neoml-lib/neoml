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

namespace NeoML {

const int BlobTimeConvolutionPrepareCombine = 16;
__global__ void BlobTimeConvolutionPrepareKernel( const CCudaTimeConvolutionDescInternal desc,
	const float* sourceData, int xSizeNorm, float* preparedData )
{
	const CCudaBlobDesc& filter = desc.Filter;
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	int h = blockIdx.z * blockDim.z + threadIdx.z;
	int seqNumber = blockIdx.y * blockDim.y + threadIdx.y;

	if( h >= filter.Height() || seqNumber >= result.BatchLength() ) {
		return;
	}

	int inputSeqNumber = seqNumber * desc.Stride + h * desc.Dilation - desc.Padding;

	int objectSize = source.ObjectSize();

	int sourceShift = inputSeqNumber * source.BatchWidth() * objectSize;

	int resultShift = objectSize * filter.Height() * result.BatchWidth() * seqNumber + objectSize * h;
	int resultStep = objectSize * filter.Height();

	const float* inputData = (0 <= inputSeqNumber && inputSeqNumber < source.BatchLength())
		? (sourceData + sourceShift) : 0;
	float* outputData = preparedData + resultShift;

	// Pass over x
	int index;
	int step;
	int count = GetCudaTaskCountAndIndex(result.BatchWidth() * objectSize,
		BlobTimeConvolutionPrepareCombine, index, step);

	for( int i = 0; i < count; ++i, index += step ) {
		int batch = index / objectSize;
		int pos = index % objectSize;
		outputData[batch * resultStep + pos] = (inputData == 0) ? 0 : __ldg(inputData + index);
	}
}

const int BlobTimeConvolutionBackwardUnpackCombine = 64;
__global__ void BlobTimeConvolutionBackwardUnpackKernel( const CCudaTimeConvolutionDescInternal desc, float* outputDiffData,
	const float* filterData, float* inputDiffData, int xSizeNorm, int combineCount, const float* data )
{
	const CCudaBlobDesc& inputDiff = desc.Source;
	const CCudaBlobDesc& filter = desc.Filter;
	const CCudaBlobDesc& outputDiff = desc.Result;

	int batch = blockIdx.y * blockDim.y + threadIdx.y;
	if( batch >= inputDiff.ObjectCount() ) {
		return;
	}

	int objectSize = inputDiff.ObjectSize();

	int seqNum = batch / inputDiff.BatchWidth();
	int batchNum = batch % inputDiff.BatchWidth();

	int index;
	int step;
	int count = GetCudaTaskCountAndIndex(objectSize, combineCount, index, step);

	// Initialize the sums
	float sums[BlobTimeConvolutionBackwardUnpackCombine];
	for( int i = 0; i < count; ++i ) {
		sums[i] = 0;
	}

	for( int filterY = 0; filterY < filter.Height(); filterY++ ) {
		int inSeqNumFirst = seqNum - filterY * desc.Dilation;
		if( inSeqNumFirst < -desc.Padding ) {
			break; // the next values can only be smaller
		}
		if( ( inSeqNumFirst + desc.Padding ) % desc.Stride != 0 ) {
			continue; // this row is not affected by the current filter row
		}
		int outSeqNum = ( inSeqNumFirst + desc.Padding ) / desc.Stride;
		if( outSeqNum >= outputDiff.BatchLength() ) {
			continue;
		}
		const float* from = data + ( (outSeqNum * inputDiff.BatchWidth() + batchNum) * filter.Height() + filterY) * objectSize;
		int curIndex = index;
		for(int i = 0; i < count; ++i, curIndex += step) {
			sums[i] += __ldg(from + curIndex);
		}
	}

	// Write the results
	float* curInputDiffData = inputDiffData + (seqNum * inputDiff.BatchWidth() + batchNum) * objectSize;
	for( int i = 0; i < count; ++i, index += step ) {
		curInputDiffData[index] = sums[i];
	}
}

__global__ void blobTimeConvolutionLearnFilterKernel( CCudaTimeConvolutionDescInternal desc,
	const float* __restrict__ input, const float* __restrict__ outputDiff, float* filterDiff )
{
	const int objectSize = desc.Filter.Channels();
	const int filterHeight = desc.Filter.Height();
	const int filterCount = desc.Filter.ObjectCount();

	const int inputLength = desc.Source.BatchLength();
	const int outputLength = desc.Result.BatchLength();

	const int batchWidth = desc.Source.BatchWidth();

	int index;
	if( GetCudaTaskIndex( desc.Filter.BlobSize(), index ) ) {
		filterDiff += index;
		float res = 0;

		const int filterChannel = index % objectSize;
		index /= objectSize;
		const int filterRow = index % filterHeight;
		const int filterNum = index / filterHeight;

		for( int outL = 0; outL < outputLength; ++outL ) {
			int inL = outL * desc.Stride - desc.Padding + filterRow * desc.Dilation;
			if( inL < 0 || inL >= inputLength ) {
				continue;
			}

			const float* currOutputDiff = outputDiff + outL * batchWidth * filterCount + filterNum;
			const float* currInput = input + inL * batchWidth * objectSize + filterChannel;
			for( int b = 0; b < batchWidth; ++b ) {
				res += *currOutputDiff * *currInput;
				currInput += objectSize;
				currOutputDiff += filterCount;
			}
		}

		*filterDiff += res;
	}
}

} // namespace NeoML
