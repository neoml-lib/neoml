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

#include <Kernels/CudaGrid.h>
#include <CudaBlobDesc.h>
#include <Kernels/CudaReduce.h>

namespace NeoML {

template<class T>
struct CCudaBlobDescArray {
	int Count;
	CCudaBlobDesc Descs[MaxBlobDescs];
	T* Data[MaxBlobDescs];
	int Widths[MaxBlobDescs];
};

const int BlobMergeByDimCombine = 16;

template<class T>
__global__ void BlobMergeByDimKernel(int height, int width, CCudaBlobDescArray<T> from, CCudaBlobDesc to, T* toData, int heightNorm)
{
	int j;
	int i;
	if(!GetCudaTaskIndex2D(heightNorm, width, j, i)) {
		return;
	}

	j *= BlobMergeByDimCombine;
	int jLast = j + BlobMergeByDimCombine;
	if(jLast > height) {
		jLast = height;
	}

	int count = jLast - j;

	T* curToData = toData + j * width + i;

	int fromIndex = 0;
	int fromI = i;
	while(fromI >= from.Widths[fromIndex]) {
		fromI -= from.Widths[fromIndex];
		++fromIndex;
	}

	int fromWidth = from.Widths[fromIndex];
	const T* fromData = from.Data[fromIndex] + j * fromWidth + fromI;

	for(int k = 0; k < count; ++k) {
		*curToData = __ldg(fromData);
		curToData += width;
		fromData += fromWidth;
	}
}

const int BlobSplitByDimCombine = 16;

template<class T>
__global__ void BlobSplitByDimKernel(int height, int width, CCudaBlobDesc from, const T* fromData, CCudaBlobDescArray<T> to, int heightNorm)
{
	int j;
	int i;
	if(!GetCudaTaskIndex2D(heightNorm, width, j, i)) {
		return;
	}


	j *= BlobSplitByDimCombine;
	int jLast = j + BlobSplitByDimCombine;
	if(jLast > height) {
		jLast = height;
	}

	int count = jLast - j;

	const T* curFromData = fromData + j * width + i;

	int toIndex = 0;
	int toI = i;
	while(toI >= to.Widths[toIndex]) {
		toI -= to.Widths[toIndex];
		++toIndex;
	}

	int toWidth = to.Widths[toIndex];
	T* toData = to.Data[toIndex] + j * toWidth + toI;
	for(int k = 0; k < count; ++k) {
		*toData = __ldg(curFromData);
		curFromData += width;
		toData += toWidth;
	}
}

__global__ void BlobResizeImageKernel( const CCudaBlobDesc from, const float* __restrict__ fromData, int deltaLeft,
	int deltaTop, float defaultValue, const CCudaBlobDesc to, float* toData )
{
	const int geom = to.Height() * to.Width();
	const int totalChannels = to.Channels() * to.Depth();

	int num;
	int currGeom;
	int ch;
	if( GetCudaTaskIndex3D( to.ObjectCount(), geom, totalChannels, num, currGeom, ch ) ) {
		toData += num * totalChannels * geom + totalChannels * currGeom + ch;

		const int xFrom = currGeom % to.Width() - deltaLeft;
		const int yFrom = currGeom / to.Width() - deltaTop;
		if( xFrom >= 0 && yFrom >= 0 && xFrom < from.Width() && yFrom < from.Height() ) {
			fromData += num * totalChannels * from.Height() * from.Width() + totalChannels * ( xFrom + yFrom * from.Width() ) + ch;
			*toData = *fromData;
		} else {
			*toData = defaultValue;
		}
	}
}

const int BlobGetSubSequenceCombine = 16;
__global__ void BlobGetSubSequenceKernel( CCudaBlobDesc from, const float* fromData, int* index, CCudaBlobDesc to,
	float* toData, int startPos, bool isRev, int objectSizeNorm )
{
	int seqPos;
	int seqNum;
	int i;

	GetCudaTaskIndex3D( to.BatchLength(), to.BatchWidth(), objectSizeNorm, seqPos, seqNum, i );

	if( seqPos >= to.BatchLength() || seqNum >= to.BatchWidth() ) {
		return;
	}

	int objectSize = from.ObjectSize() * from.ListSize();

	int fromSeqPos = isRev ? startPos - seqPos : startPos + seqPos;
	int fromPos = fromSeqPos * from.BatchWidth() + seqNum;
	const float* curFromData = fromData + fromPos * objectSize;
	int toPos = seqPos * to.BatchWidth() + seqNum;
	float* curToData = toData + toPos * objectSize;

	int step;
	int count = GetCudaTaskCountAndIndex(objectSize, BlobGetSubSequenceCombine, i, step);

	if(i == 0 && count > 0 && index != 0) {
		index[toPos] = fromPos;
	}

	for(int k = 0; k < count; ++k) {
		curToData[i] = __ldg(curFromData + i);
		i += step;
	}
}

__global__ void Upsampling2DForwardKernel(
	int heightCopyCount, int widthCopyCount, int pixelSize,
	int batchSize, int inputHeight, int inputRowSize, const float* input,
	int resultHeight, int resultRowSize, float* result )
{
	int resultI;
	int resultJ;
	if( !GetCudaTaskIndex2D( resultHeight, resultRowSize, resultI, resultJ ) ) {
		return;
	}
	const int inputI = resultI / heightCopyCount;
	const int inputJ = ( resultJ / pixelSize / widthCopyCount ) * pixelSize + resultJ % pixelSize;

	for( int batchIndex = 0; batchIndex < batchSize; ++batchIndex ) {
		*( result + resultI * resultRowSize + resultJ ) = *( input + inputI * inputRowSize + inputJ );
		input += inputHeight * inputRowSize;
		result += resultHeight * resultRowSize;
	}
}

__global__ void Upsampling2DBackwardKernel( 
	int heightCopyCount, int widthCopyCount, int pixelSize,
	int batchSize, int inputHeight, int inputRowSize,
	const float* input, int resultHeight, int resultRowSize, float* result )
{
	int inputI;
	int inputJ;
	if( !GetCudaTaskIndex2D( inputHeight, inputRowSize, inputI, inputJ ) ) {
		return;
	}
	const int resultI = inputI / heightCopyCount;
	const int resultJ = ( inputJ / pixelSize / widthCopyCount ) * pixelSize + inputJ % pixelSize;
	for( int batchIndex = 0; batchIndex < batchSize; ++batchIndex ) {
		// Atomic operation because several threads are writing into the same variables
		atomicAdd( result + resultI * resultRowSize + resultJ,
			*( input + inputI * inputRowSize + inputJ ) );
		input += inputHeight * inputRowSize;
		result += resultHeight * resultRowSize;
	}
}

static __global__ void BuildIntegerHistKernel( const int* numbers, int numbersCount, int* result )
{
	int index;
	if( GetCudaTaskIndex( numbersCount, index ) ) {
		const int currNumber = numbers[index];
		if( currNumber >= 0 ) {
			atomicAdd( result + currNumber, 1 );
		}
	}
}

const int MatrixRowsToVectorSquaredL2DistanceCombineCount = 16;
__global__ void MatrixRowsToVectorSquaredL2DistanceKernel( const float* matrix, int matrixHeight,
	int matrixWidth, const float* vector, float* result, int normalizedWidth )
{
	int rowIndex;
	int colIndex;
	if( !GetCudaTaskIndex2D( matrixHeight, normalizedWidth, rowIndex, colIndex ) ) {
		return;
	}
	int step;
	int count = GetCudaTaskCountAndIndex( matrixWidth, MatrixRowsToVectorSquaredL2DistanceCombineCount,
		colIndex, step );

	if( count == 0 ) {
		return;
	}

	float squareSum = 0.f;
	matrix += rowIndex * matrixWidth + colIndex;
	vector += colIndex;

	if( count > 0 ) {
		for( int i = 0; i < count; ++i ) {
			squareSum += ( *matrix - *vector ) * ( *matrix - *vector );
			matrix += step;
			vector += step;
		}

		atomicAdd( result + rowIndex, squareSum );
	}
}

// BP_CDHW -> BP_HWDC
inline __device__ int LegacyRepackIndex( int fromIndex, int channels, int height, int width )
{
	int x = fromIndex % width;
	fromIndex /= width;
	int y = fromIndex % height;
	fromIndex /= height;
	int c = fromIndex % channels;
	int b = fromIndex / channels;
	return c + channels * ( x + width * ( y + height * b ) );
}

template<class T>
__global__ void ReorgKernel( const T *input, int width, int height, int nFilters,
	int batchSize, int stride, bool isForward, T *output )
{
	// The index of the element being processed
	int index = 0;
	if( !GetCudaTaskIndex( width * height * nFilters * batchSize, index ) ) {
		return;
	}
	const int inputIndex = LegacyRepackIndex( index, nFilters * stride * stride, height / stride, width / stride );

	const int inputColumn = index % width;
	index = index / width;
	const int inputRow = index % height;
	index = index / height;
	const int inputFilter = index % nFilters;
	index = index / nFilters;
	const int inputImage = index % batchSize;

	const int outputNFilters = nFilters / ( stride * stride );

	const int outputFilter = inputFilter % outputNFilters;
	const int offset = inputFilter / outputNFilters;
	const int outputColumn = inputColumn * stride + offset % stride;
	const int outputRow = inputRow * stride + offset / stride;

	int outIndex = outputColumn + width * stride * ( outputRow + height * stride *
		( outputFilter + outputNFilters * inputImage ) );

	outIndex = LegacyRepackIndex( outIndex, nFilters, height, width );

	if( isForward ) {
		output[inputIndex] = input[outIndex];
	} else {
		output[outIndex] = input[inputIndex];
	}
}

__global__ void AddWidthIndexKernel( const float *input, int width, int height, int nFilters,
	int batchSize, bool isForward, float *output )
{
	// The index of the element being processed
	int index = 0;
	if( !GetCudaTaskIndex( width * height * nFilters * batchSize, index ) ) {
		return;
	}
	const int inputColumn = ( index / nFilters ) % width;
	output[index] = input[index] + ( isForward ? inputColumn : -inputColumn ); 
}

__global__ void AddWidthIndexKernel( const int *input, int width, int height, int nFilters,
	int batchSize, bool isForward, int *output )
{
	// The index of the element being processed
	int index = 0;
	if( !GetCudaTaskIndex( width * height * nFilters * batchSize, index ) ) {
		return;
	}
	const int inputColumn = ( index / nFilters ) % width;
	output[index] = input[index] + ( isForward ? inputColumn : -inputColumn );
}

__global__ void AddHeightIndexKernel( const float *input, int width, int height, int nFilters,
	int batchSize, bool isForward, float *output )
{
	// The index of the element being processed
	int index = 0;
	if( !GetCudaTaskIndex( width * height * nFilters * batchSize, index ) ) {
		return;
	}
	const int inputRow = ( index / ( width * nFilters ) ) % height;
	output[index] = input[index] + ( isForward ? inputRow : -inputRow );
}

__global__ void AddHeightIndexKernel( const int *input, int width, int height, int nFilters,
	int batchSize, bool isForward, int *output )
{
	// The index of the element being processed
	int index = 0;
	if( !GetCudaTaskIndex( width * height * nFilters * batchSize, index ) ) {
		return;
	}
	const int inputRow = ( index / ( width * nFilters ) ) % height;
	output[index] = input[index] + ( isForward ? inputRow : -inputRow );
}

} // namespace NeoML
