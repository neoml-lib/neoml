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

// The filter is represented by a matrix FilterCount x (FilterHeight * FilterWidth * InputChannels)
// The temporary matrix is of (BatchSize * ResultHeight * ResultWidth) height
// and (FilterHeight * FilterWidth * InputChannels) width
// Each row of this matrix contains the values that will be affected by the filter on the step equal to the row index
// Then the convolution consists of multiplying the temporary matrix by the transposed filter

// Build the matrix for the forward pass
__global__ void BuildTempMatrixKernel( const CCudaConvolutionDescInternal desc,
	const float* __restrict__ sourceData, int resultOffset, int resultSize, float* __restrict__ resultData )
{
	const int strideHeight = desc.StrideHeight;
	const int strideWidth = desc.StrideWidth;

	const int paddingHeight = desc.PaddingHeight;
	const int paddingWidth = desc.PaddingWidth;

	const int dilationHeight = desc.DilationHeight;
	const int dilationWidth = desc.DilationWidth;

	int b;
	int x;
	int y;
	int xy;
	int c;
	if( GetCudaTaskIndex2D( resultSize, desc.Source.Depth() * desc.Source.Channels(), xy, c ) ) {
		resultData += xy * desc.Filter.ObjectSize() + c;
		xy += resultOffset;
		x = xy % desc.Result.Width();
		y = xy / desc.Result.Width();
		b = y / desc.Result.Height();
		y = y % desc.Result.Height();
		sourceData += b * desc.Source.ObjectSize() + c;

		int startX = strideWidth * x + -paddingWidth;
		int startY = strideHeight * y + -paddingHeight;
		int inputY = startY;
		for( int fy = 0; fy < desc.Filter.Height(); fy++ ) {
			if( 0 <= inputY && inputY < desc.Source.Height() ) {
				int inputX = startX;
				const float* sourceDataPtr = sourceData + inputY * desc.Source.Width() * desc.Source.Channels() * desc.Source.Depth();
				for( int fx = 0; fx < desc.Filter.Width(); fx++ ) {
					if( 0 <= inputX && inputX < desc.Source.Width() ) {
						*resultData = sourceDataPtr[inputX * desc.Source.Channels() * desc.Source.Depth()];
					} else {
						*resultData = 0;
					}
					resultData += desc.Source.Channels() * desc.Source.Depth();
					inputX += dilationWidth;
				}
			} else {
				for( int fx = 0; fx < desc.Filter.Width(); fx++ ) {
					*resultData = 0;
					resultData += desc.Source.Channels() * desc.Source.Depth();
				}
			}
			inputY += dilationHeight;
		}
	}
}

// ====================================================================================================================
// Convolution implementation with the filter of 3x3 size, stride 1 and dilation 1 for a small number of channels

// The whole output is divided into blocks of 1 x 8 x 1 size (H x W x CH)
// One result contains one such block

// Calculate the convolution with the 1 x 3 filter over the block
// The input block of 1 x 10 size is passed via src*
// The result is added to res*
#define CONV1x3_STRIDE1_8ELEMS( res0, res1, src0, src1, src2, flt ) \
	res0.x += flt.x * src0.x + flt.y * src0.y + flt.z * src0.z; \
	res0.y += flt.x * src0.y + flt.y * src0.z + flt.z * src0.w; \
	res0.z += flt.x * src0.z + flt.y * src0.w + flt.z * src1.x; \
	res0.w += flt.x * src0.w + flt.y * src1.x + flt.z * src1.y; \
	res1.x += flt.x * src1.x + flt.y * src1.y + flt.z * src1.z; \
	res1.y += flt.x * src1.y + flt.y * src1.z + flt.z * src2.x; \
	res1.z += flt.x * src1.z + flt.y * src2.x + flt.z * src2.y; \
	res1.w += flt.x * src2.x + flt.y * src2.y + flt.z * src2.z;

// Load 4 float values located consecutively by width dimension
inline __device__ void load4Floats( float4& res, const float* from, int coord, const int channels, const int maxCoord )
{
	res.x = ( coord < 0 || coord >= maxCoord ) ? 0.f : *from;
	from += channels;
	coord++;
	res.y = ( coord < 0 || coord >= maxCoord ) ? 0.f : *from;
	from += channels;
	coord++;
	res.z = ( coord < 0 || coord >= maxCoord ) ? 0.f : *from;
	from += channels;
	coord++;
	res.w = ( coord < 0 || coord >= maxCoord ) ? 0.f : *from;
	from += channels;
	coord++;
}

// Load 3 float values located consecutively by width dimension
inline __device__ void load3Floats( float3& res, const float* from, int coord, const int channels, const int maxCoord )
{
	res.x = ( coord < 0 || coord >= maxCoord ) ? 0.f : *from;
	from += channels;
	coord++;
	res.y = ( coord < 0 || coord >= maxCoord ) ? 0.f : *from;
	from += channels;
	coord++;
	res.z = ( coord < 0 || coord >= maxCoord ) ? 0.f : *from;
	from += channels;
	coord++;
}

__launch_bounds__(512, 2)
__global__ void Conv3x3s1d1Kernel1x8( const CCudaConvolutionDescInternal desc,
	const float* __restrict__ input, const float* __restrict__ filter, const float* __restrict__ freeTerm,
	float* __restrict__ result, int widthNorm )
{
	const int filterCount = desc.Result.Channels();
	const int objectCount = desc.Result.ObjectCount();
	const int outputHeight = desc.Result.Height();
	const int outputWidth = desc.Result.Width();
	const int inputChannels = desc.Source.Depth() * desc.Source.Channels();

	int b;
	int outCol;
	int filterIndex;
	if( GetCudaTaskIndex3D( objectCount * outputHeight, widthNorm, filterCount, b, outCol, filterIndex ) ) {
		const int inputWidth = desc.Source.Width();
		const int inputHeight = desc.Source.Height();
		const int inputRowSize = inputWidth * inputChannels;
		outCol *= 8;

		result += ( b * outputWidth + outCol ) * filterCount + filterIndex;

		const int outRow = b % outputHeight;
		b /= outputHeight;

		const int inCol = outCol - desc.PaddingWidth;
		const int inRow = outRow - desc.PaddingHeight;

		input += b * inputHeight * inputRowSize + inRow * inputRowSize + inCol * inputChannels;

		const float initValue = freeTerm == 0 ? 0.f : freeTerm[filterIndex];
		filter += filterIndex * 9 * inputChannels;

		float4 res0{ initValue, initValue, initValue, initValue };
		float4 res1{ initValue, initValue, initValue, initValue };

		float4 src0;
		float3 src1;
		float3 src2;

		for( int c = 0; c < inputChannels; ++c ) {
			if( inRow >= 0 && inRow < inputHeight ) {
				float3 flt{ *filter, *( filter + inputChannels ), *( filter + 2 * inputChannels ) };
				load4Floats( src0, input, inCol, inputChannels, inputWidth );
				load3Floats( src1, input + 4 * inputChannels, inCol + 4, inputChannels, inputWidth );
				load3Floats( src2, input + 7 * inputChannels, inCol + 7, inputChannels, inputWidth );
				CONV1x3_STRIDE1_8ELEMS( res0, res1, src0, src1, src2, flt );
			}
			if( inRow + 1 >= 0 && inRow + 1 < inputHeight ) {
				float3 flt{ *(filter + 3 * inputChannels), *( filter + 4 * inputChannels ), *( filter + 5 * inputChannels ) };
				load4Floats( src0, input + inputRowSize, inCol, inputChannels, inputWidth );
				load3Floats( src1, input + inputRowSize + 4 * inputChannels, inCol + 4, inputChannels, inputWidth );
				load3Floats( src2, input + inputRowSize + 7 * inputChannels, inCol + 7, inputChannels, inputWidth );
				CONV1x3_STRIDE1_8ELEMS( res0, res1, src0, src1, src2, flt );

			}
			if( inRow + 2 >= 0 && inRow + 2 < inputHeight ) {
				float3 flt{ *(filter + 6 * inputChannels), *( filter + 7 * inputChannels ), *( filter + 8 * inputChannels ) };
				load4Floats( src0, input + 2 * inputRowSize, inCol, inputChannels, inputWidth );
				load3Floats( src1, input + 2 * inputRowSize + 4 * inputChannels, inCol + 4, inputChannels, inputWidth );
				load3Floats( src2, input + 2 * inputRowSize + 7 * inputChannels, inCol + 7, inputChannels, inputWidth );
				CONV1x3_STRIDE1_8ELEMS( res0, res1, src0, src1, src2, flt );
			}
			++input;
			++filter;
		}

		const int rem = min( 8, outputWidth - outCol );
		*result = res0.x;
		if( rem > 1 ) {
			result[filterCount] = res0.y;
			if( rem > 2 ) {
				result[2 * filterCount] = res0.z;
				if( rem > 3 ) {
					result[3 * filterCount] = res0.w;
					if( rem > 4 ) {
						result[4 * filterCount] = res1.x;
						if( rem > 5 ) {
							result[5 * filterCount] = res1.y;
							if( rem > 6 ) {
								result[6 * filterCount] = res1.z;
								if( rem == 8 ) {
									result[7 * filterCount] = res1.w;
								}
							}
						}
					}
				}
			}
		}
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
__global__ void BuildInputFromTempMatrixKernel( const CCudaConvolutionDescInternal desc,
	const float* __restrict__ tempMatrix, int matrixHeight, int matrixWidth, float* result,
	TBackwardOperationType operation, int widthNorm )
{
	int matrixRow;
	int matrixCol;
	GetCudaTaskIndex2D( matrixHeight, widthNorm, matrixRow, matrixCol );
	if( matrixRow >= matrixHeight ) {
		return;
	}
	tempMatrix += matrixRow * matrixWidth;

	const int outCol = matrixRow % desc.Result.Width();
	matrixRow /= desc.Result.Width();
	const int outRow = matrixRow % desc.Result.Height();
	const int b = matrixRow / desc.Result.Height();

	int step;
	int count = GetCudaTaskCountAndIndex( matrixWidth, BuildInputFromTempMatrixCombine, matrixCol, step );
	tempMatrix += matrixCol;

	const int inputChannels = desc.Source.Channels() * desc.Source.Depth();
	const int filterWidth = desc.Filter.Width();
	const int inputWidth = desc.Source.Width();
	const int inputHeight = desc.Source.Height();

	result += b * desc.Source.ObjectSize();

	int filterChannel = matrixCol % inputChannels;
	matrixCol /= inputChannels;
	int filterCol = matrixCol % filterWidth;
	int filterRow = matrixCol / filterWidth;

	for( int i = 0; i < count; ++i ) {
		const int inCol = outCol * desc.StrideWidth + filterCol * desc.DilationWidth - desc.PaddingWidth;
		const int inRow = outRow * desc.StrideHeight + filterRow * desc.DilationHeight - desc.PaddingHeight;
		const int inputIndex = ( inRow * inputWidth + inCol ) * inputChannels + filterChannel;

		if( inCol >= 0 && inCol < inputWidth && inRow >= 0 && inRow < inputHeight ) {
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
		filterCol += filterChannel / inputChannels;
		filterChannel = filterChannel % inputChannels;
		filterRow += filterCol / filterWidth;
		filterCol = filterCol % filterWidth;
		tempMatrix += step;
	}
}

} // namespace NeoML
