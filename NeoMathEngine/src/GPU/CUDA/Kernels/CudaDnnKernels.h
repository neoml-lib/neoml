/* Copyright © 2017-2024 ABBYY

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
struct CCudaBlobDescArray final {
	int Count = 0;
	CCudaBlobDesc Descs[MaxBlobDescs]{};
	T* Data[MaxBlobDescs]{};
	int Widths[MaxBlobDescs]{};
};

const int BlobMergeByDimCombine = 16;

template<class T>
__global__ void BlobMergeByDimKernel(int height, int width, CCudaBlobDescArray<T> from, CCudaBlobDesc to, T* toData, int heightNorm)
{
	int j = 0;
	int i = 0;
	if(!GetCudaTaskIndex2D(heightNorm, width, j, i)) {
		return;
	}

	j *= BlobMergeByDimCombine;
	int jLast = j + BlobMergeByDimCombine;
	if(jLast > height) {
		jLast = height;
	}

	const int count = jLast - j;
	T* curToData = toData + j * width + i;

	int fromIndex = 0;
	int fromI = i;
	while(fromI >= from.Widths[fromIndex]) {
		fromI -= from.Widths[fromIndex];
		++fromIndex;
	}

	const int fromWidth = from.Widths[fromIndex];
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
	int j = 0;
	int i = 0;
	if(!GetCudaTaskIndex2D(heightNorm, width, j, i)) {
		return;
	}

	j *= BlobSplitByDimCombine;
	int jLast = j + BlobSplitByDimCombine;
	if(jLast > height) {
		jLast = height;
	}

	const int count = jLast - j;
	const T* curFromData = fromData + j * width + i;

	int toIndex = 0;
	int toI = i;
	while(toI >= to.Widths[toIndex]) {
		toI -= to.Widths[toIndex];
		++toIndex;
	}

	const int toWidth = to.Widths[toIndex];
	T* toData = to.Data[toIndex] + j * toWidth + toI;
	for(int k = 0; k < count; ++k) {
		*toData = __ldg(curFromData);
		curFromData += width;
		toData += toWidth;
	}
}

__global__ void BlobResizeImageKernel( const CCudaBlobDesc from, const float* __restrict__ fromData, int deltaLeft,
	int deltaTop, int padding, float defaultValue, const CCudaBlobDesc to, float* toData )
{
	const int geom = to.Height() * to.Width();
	const int totalChannels = to.Channels() * to.Depth();

	int num = 0;
	int currGeom = 0;
	int ch = 0;
	if( GetCudaTaskIndex3D( to.ObjectCount(), geom, totalChannels, num, currGeom, ch ) ) {
		toData += num * totalChannels * geom + totalChannels * currGeom + ch;

		int xFrom = currGeom % to.Width() - deltaLeft;
		int yFrom = currGeom / to.Width() - deltaTop;
		if( padding == 1 ) { // Edge
			xFrom = xFrom < 0 ? 0 : ( xFrom >= from.Width() ? from.Width() - 1 : xFrom );
			yFrom = yFrom < 0 ? 0 : ( yFrom >= from.Height() ? from.Height() - 1 : yFrom );
		} else if( padding == 2 ) { // Reflect
			xFrom = xFrom < 0 ? -( xFrom % from.Width() )
				: ( xFrom >= from.Width() ? ( 2 * from.Width() - 2 - ( xFrom % from.Width() ) ) % from.Width() : xFrom );
			yFrom = yFrom < 0 ? -( yFrom % from.Height() )
				: ( yFrom >= from.Height() ? ( 2 * from.Height() - 2 - ( yFrom % from.Height() ) ) % from.Height() : yFrom );
		}

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
	int seqPos = 0;
	int seqNum = 0;
	int i = 0;
	GetCudaTaskIndex3D( to.BatchLength(), to.BatchWidth(), objectSizeNorm, seqPos, seqNum, i );

	if( seqPos >= to.BatchLength() || seqNum >= to.BatchWidth() ) {
		return;
	}

	const int objectSize = from.ObjectSize() * from.ListSize();
	const int fromSeqPos = isRev ? startPos - seqPos : startPos + seqPos;
	const int fromPos = fromSeqPos * from.BatchWidth() + seqNum;
	const float* const curFromData = fromData + fromPos * objectSize;
	const int toPos = seqPos * to.BatchWidth() + seqNum;
	float* const curToData = toData + toPos * objectSize;

	int step = 0;
	const int count = GetCudaTaskCountAndIndex(objectSize, BlobGetSubSequenceCombine, i, step);

	if(i == 0 && count > 0 && index != 0) {
		index[toPos] = fromPos;
	}

	for(int k = 0; k < count; ++k) {
		curToData[i] = __ldg(curFromData + i);
		i += step;
	}
}

template<class T>
__global__ void Upsampling2DForwardKernel(
	int heightCopyCount, int widthCopyCount, int pixelSize,
	int batchSize, int inputHeight, int inputRowSize, const T* input,
	int resultHeight, int resultRowSize, T* result )
{
	int resultI = 0;
	int resultJ = 0;
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
	int inputI = 0;
	int inputJ = 0;
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
	int index = 0;
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
	int rowIndex = 0;
	int colIndex = 0;
	if( !GetCudaTaskIndex2D( matrixHeight, normalizedWidth, rowIndex, colIndex ) ) {
		return;
	}

	int step = 0;
	const int count = GetCudaTaskCountAndIndex( matrixWidth, MatrixRowsToVectorSquaredL2DistanceCombineCount,
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
	const int x = fromIndex % width;
	fromIndex /= width;
	const int y = fromIndex % height;
	fromIndex /= height;
	const int c = fromIndex % channels;
	const int b = fromIndex / channels;
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

template<class T>
__global__ void SpaceToDepthKernel( const T* source, int dataRowCount, int dataRowWidth,
	int blockChannels, int blockSize, bool isForward, T* result )
{
	// number of elements in the single data row
	const int dataRowSize = blockSize * ( dataRowWidth * blockSize ) * blockChannels;

	int dataRowIndex = 0;
	int elementIndex = 0;
	if( !GetCudaTaskIndex2D( dataRowCount, dataRowSize, dataRowIndex, elementIndex ) ) {
		return;
	}

	// number of elements in a single row inside 3d-block
	const int blockRowSize = blockChannels * blockSize;

	// offset for switching to the next block inside data row
	const int sourceBlockOffset = isForward ? blockRowSize : blockSize * blockRowSize;
	const int resultBlockOffset = isForward ? blockSize * blockRowSize : blockRowSize;
	// offset for switching to the next row inside the 3d-block
	const int sourceBlockRowOffset = isForward ? dataRowWidth * blockRowSize : blockRowSize;
	const int resultBlockRowOffset = isForward ? blockRowSize : dataRowWidth * blockRowSize;

	const int pixelIndex = elementIndex / blockChannels;
	elementIndex %= blockChannels;
	const int inBlockX = pixelIndex % blockSize;
	const int inBlockY = ( pixelIndex / blockSize ) % blockSize;
	const int blockX = ( pixelIndex / blockSize / blockSize );

	source += dataRowIndex * dataRowSize + blockX * sourceBlockOffset + inBlockY * sourceBlockRowOffset
		+ inBlockX * blockChannels + elementIndex;
	result += dataRowIndex * dataRowSize + blockX * resultBlockOffset + inBlockY * resultBlockRowOffset
		+ inBlockX * blockChannels + elementIndex;
	*result = *source;
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

__global__ void QrnnFPoolingKernel( bool reverse, int sequenceLength, int objectSize,
	const float* z, const float* f, const float* h0, float* res )
{
	int index = 0;
	if( !GetCudaTaskIndex( objectSize, index ) ) {
		return;
	}

	const int nextObjectOffset = reverse ? -objectSize : objectSize;
	const int firstElemOffset = reverse ? ( objectSize * ( sequenceLength - 1 ) + index ) : index;
	z += firstElemOffset;
	f += firstElemOffset;
	res += firstElemOffset;

	if( h0 == nullptr ) {
		*res = *z * ( 1. - *f );
	} else {
		h0 += index;
		*res = *f * *h0 + ( 1. - *f ) * *z;
	}

	const float* hPrev = res;
	for( int step = 0; step < sequenceLength - 1; ++step ) {
		z += nextObjectOffset;
		f += nextObjectOffset;
		res += nextObjectOffset;
		*res = *f * *hPrev + ( 1. - *f ) * *z;
		hPrev = res;
	}
}

__global__ void QrnnFPoolingBackwardKernel( bool reverse, int sequenceLength, int objectSize,
	const float* z, const float* f, const float* h0, const float* out, float* outDiff,
	float* zDiff, float* fDiff )
{
	int index = 0;
	if( !GetCudaTaskIndex( objectSize, index ) ) {
		return;
	}

	const int nextObjectOffset = reverse ? -objectSize : objectSize;
	const int firstElemOffset = reverse ? ( objectSize * ( sequenceLength - 1 ) + index ) : index;
	z += firstElemOffset;
	f += firstElemOffset;
	out += firstElemOffset;
	outDiff += firstElemOffset;
	zDiff += firstElemOffset;
	fDiff += firstElemOffset;

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		*zDiff = *outDiff * ( 1.f - *f );
		*fDiff = *outDiff * ( *( out + nextObjectOffset ) - *z );
		*( outDiff + nextObjectOffset ) += *outDiff * *f;
		z += nextObjectOffset;
		f += nextObjectOffset;
		out += nextObjectOffset;
		outDiff += nextObjectOffset;
		zDiff += nextObjectOffset;
		fDiff += nextObjectOffset;
	}

	*zDiff = *outDiff * ( 1.f - *f );
	if( h0 == nullptr ) {
		*fDiff = - ( *z * *outDiff );
	} else {
		h0 += index;
		*fDiff = *outDiff * ( *h0 - *z );
	}
}

__global__ void QrnnIfPoolingKernel( bool reverse, int sequenceLength, int objectSize,
	const float* z, const float* f, const float* i, const float* h0, float* res )
{
	int index = 0;
	if( !GetCudaTaskIndex( objectSize, index ) ) {
		return;
	}

	const int nextObjectOffset = reverse ? -objectSize : objectSize;
	const int firstElemOffset = reverse ? ( objectSize * ( sequenceLength - 1 ) + index ) : index;
	z += firstElemOffset;
	f += firstElemOffset;
	i += firstElemOffset;
	res += firstElemOffset;

	if( h0 == nullptr ) {
		*res = *i * *z;
	} else {
		h0 += index;
		*res = *f * *h0 + *i * *z;
	}

	const float* hPrev = res;
	for( int step = 0; step < sequenceLength - 1; ++step ) {
		z += nextObjectOffset;
		f += nextObjectOffset;
		i += nextObjectOffset;
		res += nextObjectOffset;
		*res = *f * *hPrev + *i * *z;
		hPrev = res;
	}
}

__global__ void QrnnIfPoolingBackwardKernel( bool reverse, int sequenceLength, int objectSize,
	const float* z, const float* f, const float* i, const float* h0, const float* out, float* outDiff,
	float* zDiff, float* fDiff, float* iDiff )
{
	int index = 0;
	if( !GetCudaTaskIndex( objectSize, index ) ) {
		return;
	}

	const int nextObjectOffset = reverse ? -objectSize : objectSize;
	const int firstElemOffset = reverse ? ( objectSize * ( sequenceLength - 1 ) + index ) : index;
	z += firstElemOffset;
	f += firstElemOffset;
	i += firstElemOffset;
	out += firstElemOffset;
	outDiff += firstElemOffset;
	zDiff += firstElemOffset;
	fDiff += firstElemOffset;
	iDiff += firstElemOffset;

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		*zDiff = *outDiff * *i;
		*fDiff = *outDiff * *( out + nextObjectOffset );
		*iDiff = *outDiff * *z;
		*( outDiff + nextObjectOffset ) += *outDiff * *f;
		z += nextObjectOffset;
		f += nextObjectOffset;
		i += nextObjectOffset;
		out += nextObjectOffset;
		outDiff += nextObjectOffset;
		zDiff += nextObjectOffset;
		fDiff += nextObjectOffset;
		iDiff += nextObjectOffset;
	}

	*zDiff = *outDiff * *i;
	if( h0 == nullptr ) {
		*fDiff = 0.f;
	} else {
		h0 += index;
		*fDiff = *outDiff * *h0;
	}
	*iDiff = *outDiff * *z;
}

inline __device__ float cudaSigmoid( float x ) { return 1.f / ( 1.f + ExponentFunc( -x ) ); }
inline __device__ float cudaReLU( float x ) { return max( 0.f, x ); }

__global__ void IndRnnRecurrentKernel( bool reverse, int sequenceLength, int batchSize, int objectSize, int activation,
	const float* wx, const float* mask, const float* u, float* h )
{
	int batch = 0;
	int elem = 0;
	if( !GetCudaTaskIndex2D( batchSize, objectSize, batch, elem ) ) {
		return;
	}

	// AF_Sigmoid == 5
	// AF_ReLU == 2
	float ( *applyActivation )( float ) = ( activation == 5 ) ? cudaSigmoid : cudaReLU;

	const int inBatchOffset = batch * objectSize + elem;
	const float dropout = mask == nullptr ? 1.f : mask[inBatchOffset];
	const float weight = u[elem];
	const int totalSize = batchSize * objectSize;
	const int stepOffset = reverse ? -totalSize : totalSize;

	if( reverse ) {
		wx += ( sequenceLength - 1 ) * totalSize;
		h += ( sequenceLength - 1 ) * totalSize;
	}

	wx += inBatchOffset;
	h += inBatchOffset;

	float currRes = applyActivation( *wx );
	*h = currRes;

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		wx += stepOffset;
		h += stepOffset;
		currRes = *wx + weight * dropout * currRes;
		currRes = applyActivation( currRes );
		*h = currRes;
	}
}

inline __device__ float cudaSigmoidDiffOp( float out, float outDiff ) { return out * ( 1.f - out ) * outDiff; }
inline __device__ float cudaReLUDiffOp( float out, float outDiff ) { return out > 0.f ? outDiff : 0.f; }

__global__ void IndRnnRecurrentBackwardKernel( bool reverse, int sequenceLength, int batchSize, int objectSize, int activation,
	const float* mask, const float* u, const float* out, const float* outDiff, float* wxDiff )
{
	int batch = 0;
	int elem = 0;
	if( !GetCudaTaskIndex2D( batchSize, objectSize, batch, elem ) ) {
		return;
	}

	// AF_Sigmoid == 5
	// AF_ReLU == 2
	float ( *activationDiffOp )( float, float ) = ( activation == 5 ) ? cudaSigmoidDiffOp : cudaReLUDiffOp;

	const int inBatchOffset = batch * objectSize + elem;
	const float dropout = mask == nullptr ? 1.f : mask[inBatchOffset];
	const float weight = u[elem];
	const int totalSize = batchSize * objectSize;
	const int stepOffset = reverse ? -totalSize : totalSize;

	if( reverse ) {
		out += ( sequenceLength - 1 ) * totalSize;
		wxDiff += ( sequenceLength - 1 ) * totalSize;
		outDiff += ( sequenceLength - 1 ) * totalSize;
	}

	out += inBatchOffset;
	outDiff += inBatchOffset;
	wxDiff += inBatchOffset;

	float totalOutDiff = *outDiff;

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		float currOut = *out;
		float currWxDiff = activationDiffOp( currOut, totalOutDiff );
		*wxDiff = currWxDiff;

		outDiff += stepOffset;
		totalOutDiff = *outDiff + currWxDiff * weight * dropout;

		out += stepOffset;
		wxDiff += stepOffset;
	}

	float currOut = *out;
	*wxDiff = activationDiffOp( currOut, totalOutDiff );
}

__global__ void IndRnnRecurrentLearnKernel( bool reverse, int sequenceLength, int batchSize, int objectSize, int activation,
	const float* mask, const float* u, const float* out, const float* outDiff, float* uDiff )
{
	int batch = 0;
	int elem = 0;
	if( !GetCudaTaskIndex2D( batchSize, objectSize, batch, elem ) ) {
		return;
	}

	// AF_Sigmoid == 5
	// AF_ReLU == 2
	float ( *activationDiffOp )( float, float ) = ( activation == 5 ) ? cudaSigmoidDiffOp : cudaReLUDiffOp;

	const int inBatchOffset = batch * objectSize + elem;
	const float dropout = mask == nullptr ? 1.f : mask[inBatchOffset];
	const float weight = u[elem];
	const int totalSize = batchSize * objectSize;
	const int stepOffset = reverse ? -totalSize : totalSize;

	if( reverse ) {
		out += ( sequenceLength - 1 ) * totalSize;
		outDiff += ( sequenceLength - 1 ) * totalSize;
	}

	out += inBatchOffset;
	outDiff += inBatchOffset;

	float totalUDiff = 0;
	float totalOutDiff = *outDiff;
	float currOut = *out;

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		const float temp = activationDiffOp( currOut, totalOutDiff ) * dropout;
		outDiff += stepOffset;
		out += stepOffset;
		currOut = *out;
		totalUDiff += temp * currOut;
		totalOutDiff = *outDiff + temp * weight;
	}

	if( batchSize > 1 ) {
		atomicAdd( uDiff + elem, totalUDiff );
	} else {
		uDiff[elem] += totalUDiff;
	}
}

__global__ void BertConvKernel( const float* data, const float* kernel, int seqLen, int batchSize, int numHeads,
	int headSize, int kernelSize, float* output )
{
	const int batchNumHeads = batchSize * numHeads;
	const int taskCount = seqLen * batchNumHeads * headSize;

	int index = 0;
	if( !GetCudaTaskIndex( taskCount, index ) ) {
		return;
	}

	const int pad = ( kernelSize - 1 ) / 2;
	const int dataSeqStep = batchNumHeads * headSize;

	const int outputOffset = index;
	const int h = index % headSize;
	index /= headSize;

	const int b = index % batchNumHeads;
	const int seq = index / batchNumHeads;
	const int kernelOffset = index * kernelSize;

	const int kernelStart = max( 0, pad - seq );
	const int kernelEnd = min( kernelSize, seqLen + pad - seq );
	int dataOffset = h + b * headSize + ( seq - pad + kernelStart ) * dataSeqStep;
	float res = 0.f;

	for( int k = kernelStart; k < kernelEnd; ++k ) {
		res += data[dataOffset] * kernel[kernelOffset + k];
		dataOffset += dataSeqStep;
	}

	output[outputOffset] = res;
}

__global__ void BertConvBackwardDataKernel( const float* kernel, const float* outputDiff, int seqLen,
	int batchSize, int numHeads, int headSize, int kernelSize, float* dataDiff )
{
	const int batchNumHeads = batchSize * numHeads;
	const int taskCount = seqLen * batchNumHeads * headSize;

	int index = 0;
	if( !GetCudaTaskIndex( taskCount, index ) ) {
		return;
	}

	const int pad = ( kernelSize - 1 ) / 2;
	const int outputSeqStep = batchNumHeads * headSize;
	const int kernelSeqStep = batchNumHeads * kernelSize;

	const int dataOffset = index;
	const int h = index % headSize;
	index /= headSize;
	const int b = index % batchNumHeads;
	const int dataSeq = index / batchNumHeads;

	const int outputSeqStart = max( 0, dataSeq + pad - kernelSize + 1 );
	const int outputSeqEnd = min( seqLen, dataSeq + pad + 1 );
	int outputOffset = b * headSize + outputSeqStart * outputSeqStep + h;
	int kernelOffset = b * kernelSize + outputSeqStart * kernelSeqStep;
	float res = 0.f;

	for( int outputSeq = outputSeqStart; outputSeq < outputSeqEnd; ++outputSeq ) {
		const int posInKernel = dataSeq - ( outputSeq - pad );
		res += kernel[kernelOffset + posInKernel] * outputDiff[outputOffset];
		kernelOffset += kernelSeqStep;
		outputOffset += outputSeqStep;
	}

	dataDiff[dataOffset] = res;
}

__global__ void BertConvBackwardKernelKernel( const float* data, const float* outputDiff, int seqLen,
	int batchSize, int numHeads, int headSize, int kernelSize, float* kernelDiff )
{
	const int batchNumHeads = batchSize * numHeads;
	const int taskCount = seqLen * batchNumHeads * kernelSize;

	int index = 0;
	if( !GetCudaTaskIndex( taskCount, index ) ) {
		return;
	}

	const int kernelOffset = index;
	const int posInKernel = index % kernelSize;
	index /= kernelSize;

	const int outputSeq = index / batchNumHeads;
	const int pad = ( kernelSize - 1 ) / 2;
	const int dataSeq = ( outputSeq - pad ) + posInKernel;

	if( dataSeq < 0 || dataSeq >= seqLen ) {
		kernelDiff[kernelOffset] = 0.f;
		return;
	}

	const int b = index % batchNumHeads;
	const int dataSeqStep = batchNumHeads * headSize;
	int dataOffset = dataSeq * dataSeqStep + b * headSize;
	int outputOffset = outputSeq * dataSeqStep + b * headSize;
	float res = 0.f;

	for( int h = 0; h < headSize; ++h ) {
		res += data[dataOffset++] * outputDiff[outputOffset++];
	}

	kernelDiff[kernelOffset] = res;
}

__global__ void LinearInterpolationKernel( const float* data, float* result, int coords, int round,
	int objectCount, int scaledAxis, int objectSize, float scale )
{
	const int newSize = static_cast<int>( scaledAxis * scale );
	const int taskCount = objectCount * newSize * objectSize;

	int taskIndex = 0;
	if( !GetCudaTaskIndex( taskCount, taskIndex ) ) {
		return;
	}

	result += taskIndex;
	const int elem = taskIndex % objectSize;
	taskIndex /= objectSize;
	const int xNew = taskIndex % newSize;
	const int b = taskIndex / newSize;

	float xOld = 0;
	switch( coords ) {
		case 0: // HalfPixel
			xOld = ( xNew + 0.5f ) / scale - 0.5f;
			break;
		case 1: // PytorchHalfPixel
			xOld = ( newSize > 1 ) ? ( ( xNew + 0.5f ) / scale - 0.5f ) : 0.f;
			break;
		case 2: // AlignCorners
			xOld = static_cast<float>( xNew * ( scaledAxis - 1 ) ) / ( newSize - 1 );
			break;
		case 3:
			xOld = xNew / scale;
			break;
	}

	switch( round ) {
		case 0: // None
			break;
		case 1: // RoundPreferFloor
			if( static_cast<int>( xOld ) + 0.5f == xOld ) {
				xOld = ::floorf( xOld );
			} else {
				xOld = ::roundf( xOld );
			}
			break;
		case 2: // RoundPreferCeil
			xOld = ::roundf( xOld );
			break;
		case 3: // Floor
			xOld = ::floorf( xOld );
			break;
		case 4: // Ceil
			xOld = ::ceilf( xOld );
			break;
	}

	if( xOld <= 0 ) {
		*result = data[b * scaledAxis * objectSize + elem];
	} else if( xOld >= static_cast<float>( scaledAxis - 1 ) ) {
		*result = data[( b * scaledAxis + scaledAxis - 1 ) * objectSize + elem];
	} else {
		const int leftCoord = static_cast<int>( xOld );
		const float rightMul = xOld - ::floorf( xOld );
		const float leftMul = 1.f - rightMul;
		*result = leftMul * data[( b * scaledAxis + leftCoord ) * objectSize + elem]
			+ rightMul * data[( b * scaledAxis + ( leftCoord + 1 ) ) * objectSize + elem];
	}
}

template<class T>
__global__ void scatterNDKernel( const T* updates, const int* indices, T* data, const CCudaBlobDesc dataDesc,
	int updateCount, int indexDims, int objectSize )
{
	const int taskCount = updateCount * objectSize;

	int index = 0;
	if( !GetCudaTaskIndex( taskCount, index ) ) {
		return;
	}

	const int updateIndex = index / objectSize;
	const int elem = index % objectSize;

	indices += updateIndex * indexDims;
	updates += updateIndex * objectSize;

	int dataOffset = 0;
	int dimOffset = objectSize;
	for( int i = indexDims - 1; i >= 0; --i ) {
		dataOffset += indices[i] * dimOffset;
		dimOffset *= dataDesc.DimSize( i );
	}
	data[dataOffset + elem] = updates[elem];
}

} // namespace NeoML
