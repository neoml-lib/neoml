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

// Channelwise convolution kernel. Calculates one result element per thread
__global__ void BlobChannelwiseConvolutionKernel( const CCudaChannelwiseConvolutionDescInternal desc,
	const float* __restrict__ sourceData, const float* __restrict__ filterData, const float* __restrict__ freeTerm,
	float* __restrict__ resultData )
{
	const CCudaBlobDesc& source = desc.Source;
	const CCudaBlobDesc& filter = desc.Filter;
	const CCudaBlobDesc& result = desc.Result;

	int taskY;
	int taskX;

	if( GetCudaTaskIndex2D( result.ObjectCount() * result.Height(), result.Width() * result.Channels(), taskY, taskX ) ) {
		float* resultPtr = resultData + taskY * result.Channels() * result.Width() + taskX;

		const int channel = taskX % result.Channels();
		const int outCol = taskX / result.Channels();
		const int outRow = taskY % result.Height();
		const int b = taskY / result.Height();

		const int rowStart = outRow * desc.StrideHeight - desc.PaddingHeight;
		const int colStart = outCol * desc.StrideWidth - desc.PaddingWidth;

		const int rowEnd = rowStart + filter.Height();
		const int colEnd = colStart + filter.Width();

		const float* filterPtr = filterData + channel;
		const float* sourcePtr = sourceData + b * source.ObjectSize() + channel;

		if( freeTerm != 0 ) {
			*resultPtr = freeTerm[channel];
		} else {
			*resultPtr = 0;
		}

		for( int filterRow = max( 0, -rowStart ); filterRow < filter.Height() - max( 0, rowEnd - source.Height() );
			++filterRow )
		{
			for( int filterCol = max( 0, -colStart ); filterCol < filter.Width() - max( 0, colEnd - source.Width() );
				++filterCol )
			{
				*resultPtr += filterPtr[( filterRow * filter.Width() + filterCol ) * filter.Channels()]
					* sourcePtr[( ( rowStart + filterRow ) * source.Width() + ( colStart + filterCol ) ) * source.Channels()];
			}
		}
	}
}

// The kernel for reverse channelwise convolution
// Each thread calculates one result element
__global__ void BlobChannelwiseConvolutionBackwardKernel( const CCudaChannelwiseConvolutionDescInternal desc,
	const float* __restrict__ sourceData, const float* __restrict__ filterData, float* __restrict__ resultData )
{
	const CCudaBlobDesc& outputDiff = desc.Result;
	const CCudaBlobDesc& filter = desc.Filter;
	const CCudaBlobDesc& inputDiff = desc.Source;

	int taskX;
	int taskY;

	if( GetCudaTaskIndex2D( inputDiff.ObjectCount() * inputDiff.Height(), inputDiff.Width() * inputDiff.Channels(), taskY, taskX ) ) {
		resultData += taskY * inputDiff.Width() * inputDiff.Channels() + taskX;
		const int channel = taskX % inputDiff.Channels();
		const int outCol = taskX / inputDiff.Channels();
		const int outRow = taskY % inputDiff.Height();
		const int b = taskY / inputDiff.Height();

		const int lastFilterX = min( ( outCol + desc.PaddingWidth ) / desc.StrideWidth + 1, outputDiff.Width() );
		const int lastFilterY = min( ( outRow + desc.PaddingHeight ) / desc.StrideHeight + 1, outputDiff.Height() );

		float res = 0;

		const float* filterPtr = filterData + channel;
		const float* sourcePtr = sourceData + b * outputDiff.ObjectSize() + channel;

		for( int sourceY = lastFilterY - 1;
			sourceY >= 0 && outRow < sourceY * desc.StrideHeight - desc.PaddingHeight + filter.Height();
			--sourceY )
		{
			for( int sourceX = lastFilterX - 1;
				sourceX >= 0 && outCol < sourceX * desc.StrideWidth - desc.PaddingWidth + filter.Width();
				sourceX-- )
			{
				const int inFilterX = outCol + desc.PaddingWidth - sourceX * desc.StrideWidth;
				const int inFilterY = outRow + desc.PaddingHeight - sourceY * desc.StrideHeight;

				res += filterPtr[( inFilterY * filter.Width() + inFilterX ) * filter.Channels()]
					* sourcePtr[( sourceY * outputDiff.Width() + sourceX ) * outputDiff.Channels()];
			}
		}

		*resultData = res;
	}
}

// The kernel for calculating channelwise convolution filter gradient
// Each thread calculates one result element
__global__ void BlobChannelwiseConvolutionLearnAddKernel( const CCudaChannelwiseConvolutionDescInternal desc,
	const float* __restrict__ inputData, const float* __restrict__ outputDiffData, float* __restrict__ filterDiffData )
{
	const CCudaBlobDesc& input = desc.Source;
	const CCudaBlobDesc& filterDiff = desc.Filter;
	const CCudaBlobDesc& outputDiff = desc.Result;

	int fb;
	int hw;
	if( GetCudaTaskIndex2D( filterDiff.Height() * filterDiff.Width(), filterDiff.Channels(), hw, fb ) ) {
		int fy = hw / filterDiff.Width();
		int fx = hw % filterDiff.Width();
		float* filtData = filterDiffData + hw * filterDiff.Channels() + fb;

		const float* curOutputDiffData = outputDiffData + fb;
		const float* curInputData = inputData + fb;
		for( int b = 0; b < outputDiff.ObjectCount(); b++ ) {
			const float* outputDiffDataPtr = curOutputDiffData;
			for( int y = 0; y < outputDiff.Height(); y++ ) {
				const int inputY = -desc.PaddingHeight + desc.StrideHeight * y + fy;
				if( 0 <= inputY && inputY < input.Height() ) {
					for( int x = 0; x < outputDiff.Width(); x++ ) {
						const int inputX = -desc.PaddingWidth + desc.StrideWidth * x + fx;
						if( 0 <= inputX && inputX < input.Width() ) {
							*filtData += curInputData[( inputY * input.Width() + inputX ) * input.Channels()] * *outputDiffDataPtr;
						}
						outputDiffDataPtr += outputDiff.Channels();
					}
				} else {
					outputDiffDataPtr += outputDiff.Width() * outputDiff.Channels();
				}
			}
			curOutputDiffData += outputDiff.ObjectSize();
			curInputData += input.ObjectSize();
		}
	}
}

} // namespace NeoML
