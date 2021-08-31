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

// ====================================================================================================================
// Convolution implemented with a temporary matrix

// The filter is represented by a matrix FilterCount x (FilterHeight * FilterWidth * FilterDepth * InputChannels)
// The temporary matrix is of (BatchSize * ResultHeight * ResultWidth * ResultDepth) height
// and (FilterHeight * FilterWidth * FilterDepth * InputChannels) width
// Each row of this matrix contains the values that will be affected by the filter on the step equal to the row index
// Then the convolution consists of multiplying the temporary matrix by the transposed filter

// One thread can write into a temporary matrix row not more than this number of elements
const int BuildTempMatrixCombine = 16;

__launch_bounds__(1024, 1)
__global__ void BuildTempMatrixKernel( const CCuda3dConvolutionDescInternal desc,
	const float* __restrict__ input, int matrixHeight, int matrixWidth, float* __restrict__ matrix,
	int matrixWidthNorm, int heightOffset )
{
	const int filterSize = desc.Filter.ObjectSize();
	const int inputChannels = desc.Source.Channels();
	const int filterHeight = desc.Filter.Height();
	const int filterWidth = desc.Filter.Width();
	const int filterDepth = desc.Filter.Depth();
	const int outputHeight = desc.Result.Height();
	const int outputWidth = desc.Result.Width();
	const int outputDepth = desc.Result.Depth();
	const int inputHeight = desc.Source.Height();
	const int inputWidth = desc.Source.Width();
	const int inputDepth = desc.Source.Depth();

	const int strideHeight = desc.StrideHeight;
	const int strideWidth = desc.StrideWidth;
	const int strideDepth = desc.StrideDepth;

	const int paddingHeight = desc.PaddingHeight;
	const int paddingWidth = desc.PaddingWidth;
	const int paddingDepth = desc.PaddingDepth;

	int matrixRow;
	int matrixCol;

	GetCudaTaskIndex2D( matrixHeight, matrixWidthNorm, matrixRow, matrixCol );
	if( matrixRow >= matrixHeight ) {
		return;
	}

	int step;
	int count = GetCudaTaskCountAndIndex( matrixWidth, BuildTempMatrixCombine, matrixCol, step );
	matrix += matrixRow * matrixWidth + matrixCol;
	matrixRow += heightOffset;

	const int outPixel = matrixRow % outputDepth;
	const int outCol = ( matrixRow / outputDepth ) % outputWidth;
	const int outRow = ( matrixRow / ( outputDepth * outputWidth ) ) % outputHeight;
	const int batch = matrixRow / ( outputDepth * outputWidth * outputHeight );
	const int inputRowStart = outRow * strideHeight - paddingHeight;
	const int inputColStart = outCol * strideWidth - paddingWidth;
	const int inputPixelStart = outPixel * strideDepth - paddingDepth;

	for( int i = 0; i < count; ++i ) {
		const int channel = matrixCol % inputChannels;
		int filterGeom = matrixCol / inputChannels;
		const int fitlerPixel = filterGeom % filterDepth;
		filterGeom /= filterDepth;
		const int filterCol = filterGeom % filterWidth;
		const int filterRow = filterGeom / filterWidth;

		const int inputRow = inputRowStart + filterRow;
		const int inputCol = inputColStart + filterCol;
		const int inputPixel = inputPixelStart + fitlerPixel;

		if( inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth && inputPixel >= 0 && inputPixel < inputDepth ) {
			*matrix = input[( ( ( batch * inputHeight + inputRow ) * inputWidth + inputCol ) * inputDepth + inputPixel ) * inputChannels + channel];
		} else {
			*matrix = 0.;
		}

		matrixCol += step;
		matrix += step;
	}
}

// ====================================================================================================================
// Reverse convolution implemented with a temporary matrix

// The temporary matrix is obtained by multiplying the gradient matrices of the output and the filter
// Then the input gradient is found by the operation inverse to creating a temporary matrix for convolution

// The types of operation used to build the input gradient
enum TBackwardOperationType {
	BOT_AtomicAdd, // atomic addition (default)
	BOT_Add, // addition (used when the convolution stride is bigger than the filter size)
	BOT_Set // assignment (used when the convolution stride is equal to the filter size and the whole input is covered)
};

// One thread processes no more than combine elements of one temporary matrix row
const int BuildInputFromTempMatrixCombine = 16;
__global__ void BuildInputFromTempMatrixKernel( const CCuda3dConvolutionDescInternal desc, const float* __restrict__ tempMatrix,
	int matrixHeight, int matrixWidth, float* result, TBackwardOperationType operation, int widthNorm, int heightOffset )
{
	int tempRow;
	int tempCol;
	GetCudaTaskIndex2D( matrixHeight, widthNorm, tempRow, tempCol );
	if( tempRow >= matrixHeight ) {
		return;
	}
	tempMatrix += tempRow * matrixWidth;
	tempRow += heightOffset;

	const int outPixel = tempRow % desc.Result.Depth();
	tempRow /= desc.Result.Depth();
	const int outCol = tempRow % desc.Result.Width();
	tempRow /= desc.Result.Width();
	const int outRow = tempRow % desc.Result.Height();
	const int b = tempRow / desc.Result.Height();

	int step;
	int count = GetCudaTaskCountAndIndex( matrixWidth, BuildInputFromTempMatrixCombine, tempCol, step );
	tempMatrix += tempCol;

	const int inputChannels = desc.Source.Channels();
	const int filterDepth = desc.Filter.Depth();
	const int filterWidth = desc.Filter.Width();
	const int inputDepth = desc.Source.Depth();
	const int inputWidth = desc.Source.Width();
	const int inputHeight = desc.Source.Height();

	result += b * desc.Source.ObjectSize();

	int filterChannel = tempCol % inputChannels;
	tempCol /= inputChannels;
	int filterPixel = tempCol % filterDepth;
	tempCol /= filterDepth;
	int filterCol = tempCol % filterWidth;
	int filterRow = tempCol / filterWidth;

	for( int i = 0; i < count; ++i ) {
		const int inPixel = outPixel * desc.StrideDepth + filterPixel - desc.PaddingDepth;
		const int inCol = outCol * desc.StrideWidth + filterCol - desc.PaddingWidth;
		const int inRow = outRow * desc.StrideHeight + filterRow - desc.PaddingHeight;
		const int inputIndex = ( ( inRow * inputWidth + inCol ) * inputDepth + inPixel ) * inputChannels + filterChannel;

		if( inPixel >= 0 && inPixel < inputDepth && inCol >= 0 && inCol < inputWidth && inRow >= 0 && inRow < inputHeight ) {
			switch( operation ) {
				case BOT_AtomicAdd:
					atomicAdd( result + inputIndex, *tempMatrix );
					break;
				case BOT_Add:
					result[inputIndex] += *tempMatrix;
					break;
				case BOT_Set:
					result[inputIndex] = *tempMatrix;
					break;
			}
		}

		filterChannel += step;
		filterPixel += filterChannel / inputChannels;
		filterChannel = filterChannel % inputChannels;
		filterCol += filterPixel / filterDepth;
		filterPixel = filterPixel % filterDepth;
		filterRow += filterCol / filterWidth;
		filterCol = filterCol % filterWidth;
		tempMatrix += step;
	}
}

} // namespace NeoML
