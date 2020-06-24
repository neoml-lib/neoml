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

__global__ void Blob3dMaxPoolingKernel( const CCuda3dMaxPoolingDescInternal desc, const float* __restrict__ sourceData,
	int* maxIndicesData, float* resultData )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	int b;
	int channel;
	int pos;

	int resultGeomSize = result.Depth() * result.Height() * result.Width();
	int resultObjectSize = result.Channels() * resultGeomSize;
	int totalChannels = source.Channels();

	if(!GetCudaTaskIndex3D( result.ObjectCount(), resultGeomSize, totalChannels, b, pos, channel)) {
		return;
	}

	int resultShift = b * resultObjectSize + pos * totalChannels + channel;

	int inputHW = source.Height() * source.Width();
	int inputGeom = source.Depth() * inputHW;
	const float* inputData = sourceData + b * totalChannels * inputGeom;

	// Output position
	int kOut = pos % result.Depth();
	pos /= result.Depth();
	int iOut = pos % result.Width();
	int jOut = pos / result.Width();

	// Input position
	int jStart = jOut * desc.StrideHeight;
	int iStart = iOut * desc.StrideWidth;
	int kStart = kOut * desc.StrideDepth;

	float maxValue = -FLT_MAX;
	int maxIndex = 0;

	int jIndex = jStart * source.Width() * source.Depth() * totalChannels + channel;
	for(int j = 0; j < desc.FilterHeight; ++j) {
		int iIndex = jIndex + iStart * source.Depth() * totalChannels;
		for(int i = 0; i < desc.FilterWidth; ++i) {
			int index = iIndex + kStart * totalChannels;
			for( int k = 0; k < desc.FilterDepth; ++k ) {
				float value = __ldg(inputData + index);
				if(value >= maxValue) {
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
	if( maxIndicesData != 0 ) {
		maxIndicesData[resultShift] = maxIndex - channel;
	}
}

__global__ void Blob3dMaxPoolingBackwardKernel( const CCuda3dMaxPoolingDescInternal desc, const float* outputDiffData,
	const int* maxIndicesData, float* inputDiffData, bool isAtomic )
{
	const CCudaBlobDesc& outputDiff = desc.Result;
	const CCudaBlobDesc& inputDiff = desc.Source;

	int b;
	int pos;
	int channel;

	int outputGeomSize = outputDiff.Depth() * outputDiff.Height() * outputDiff.Width();

	if(!GetCudaTaskIndex3D(outputDiff.ObjectCount(), outputGeomSize, outputDiff.Channels(), b, pos, channel)) {
		return;
	}

	float* curInputDiffData = inputDiffData + b * inputDiff.ObjectSize();

	int outputShift = (b * outputGeomSize + pos) * outputDiff.Channels() + channel;
	int index = maxIndicesData[outputShift] + channel;
	float value = __ldg(outputDiffData + outputShift);

	if(isAtomic) {
		atomicAdd(curInputDiffData + index, value);
	} else {
		curInputDiffData[index] = value;
	}
}

__global__ void Blob3dMeanPoolingKernel( const CCuda3dMeanPoolingDescInternal desc, const float* sourceData,
	float* resultData )
{
	const CCudaBlobDesc& result = desc.Result;
	const CCudaBlobDesc& source = desc.Source;

	int b;
	int channel;
	int pos;

	int resultGeomSize = result.Depth() * result.Height() * result.Width();
	int totalChannels = result.Channels();
	int resultObjectSize = totalChannels * resultGeomSize;

	if(!GetCudaTaskIndex3D(result.ObjectCount(), resultGeomSize, result.Channels(), b, pos, channel)) {
		return;
	}

	int resultShift = b * resultObjectSize + pos * totalChannels + channel;

	int inputGeom = source.Depth() * source.Height() * source.Width();
	const float* inputData = sourceData + b * totalChannels * inputGeom;

	// Output position
	int kOut = pos % result.Depth();
	pos /= result.Depth();
	int iOut = pos % result.Width();
	int jOut = pos / result.Width();

	// Input position
	int jStart = jOut * desc.StrideHeight;
	int iStart = iOut * desc.StrideWidth;
	int kStart = kOut * desc.StrideDepth;

	float sumValue = 0;

	int jIndex = jStart * source.Width() * source.Depth() * totalChannels + channel;
	for(int j = 0; j < desc.FilterHeight; ++j) {
		int iIndex = jIndex + iStart * source.Depth() * totalChannels;
		for(int i = 0; i < desc.FilterWidth; ++i) {
			int index = iIndex + kStart * totalChannels;
			for( int k = 0; k < desc.FilterDepth; ++k ) {
				sumValue += __ldg(inputData + index);
				index += totalChannels;
			}
			iIndex += source.Depth() * totalChannels;
		}
		jIndex += source.Width() * source.Depth() * totalChannels;
	}

	resultData[resultShift] = sumValue / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth;
}

__global__ void Blob3dMeanPoolingBackwardKernel( const CCuda3dMeanPoolingDescInternal desc, const float* outputDiffData,
	float* inputDiffData, bool isAtomic )
{
	const CCudaBlobDesc& outputDiff = desc.Result;
	const CCudaBlobDesc& inputDiff = desc.Source;

	int b;
	int channel;
	int pos;

	int outputGeomSize = outputDiff.Depth() * outputDiff.Height() * outputDiff.Width();
	int totalChannels = outputDiff.Channels();

	if(!GetCudaTaskIndex3D(outputDiff.ObjectCount(), outputGeomSize, totalChannels, b, pos, channel)) {
		return;
	}

	int outputShift = (b * outputGeomSize + pos ) * totalChannels + channel;
	float value = __ldg(outputDiffData + outputShift) / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth;

	// Output position
	int kOut = pos % outputDiff.Depth();
	pos /= outputDiff.Depth();
	int iOut = pos % outputDiff.Width();
	int jOut = pos / outputDiff.Width();

	// Input position
	int jStart = jOut * desc.StrideHeight;
	int iStart = iOut * desc.StrideWidth;
	int kStart = kOut * desc.StrideDepth;

	float* curInputDiffData = inputDiffData + ( ( ( b * inputDiff.Height() + jStart ) * inputDiff.Width() + iStart ) * inputDiff.Depth() + kStart ) * totalChannels + channel;

	int inputWDC = inputDiff.Width() * inputDiff.Depth() * inputDiff.Channels();

	if(isAtomic) {
		for(int j = 0; j < desc.FilterHeight; ++j) {
			float* inputColumnData = curInputDiffData;
			for(int i = 0; i < desc.FilterWidth; ++i) {
				float* inputPixelData = inputColumnData;
				for( int k = 0; k < desc.FilterDepth; ++k ) {
					atomicAdd(inputPixelData, value);
					inputPixelData += totalChannels;
				}
				inputColumnData += inputDiff.Depth() * totalChannels;
			}
			curInputDiffData += inputWDC;
		}
	} else {
		for( int j = 0; j < desc.FilterHeight; ++j ) {
			float* inputColumnData = curInputDiffData;
			for( int i = 0; i < desc.FilterWidth; ++i ) {
				float* inputPixelData = inputColumnData;
				for( int k = 0; k < desc.FilterDepth; ++k ) {
					*inputPixelData = value;
					inputPixelData += totalChannels;
				}
				inputColumnData += inputDiff.Depth() * totalChannels;
			}
			curInputDiffData += inputWDC;
		}
	}
}

} // namespace NeoML
