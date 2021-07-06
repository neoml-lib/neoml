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

#include <common.h>
#pragma hdrstop

#include <CpuMathEngine.h>
#include <CpuMathEnginePrivate.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Operations with blobs
template<class T>
void CCpuMathEngine::blobMergeByDimCommon( int dimNum, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount,
	const CBlobDesc& to, const CTypedMemoryHandle<T>& toData )
{
	int s[CBlobDesc::MaxDimensions];
	to.GetDimSizes( s );
	int objectCount = 1;
	for(int z  = 0; z < dimNum; z++) {
		objectCount *= s[z];
	}
	int objectSize = to.BlobSize() / objectCount;
	for(int x = 0; x < objectCount; x++) {
		CTypedMemoryHandle<T> output = toData + x * objectSize;
		for( int i = 0; i < fromCount; ++i ) {
			int fromLimits[CBlobDesc::MaxDimensions];
			from[i].GetDimSizes( fromLimits );
			int fromObjectSize = 1;
			for(int z = dimNum; z < CBlobDesc::MaxDimensions; z++) {
				fromObjectSize *= fromLimits[z];
			}
			VectorCopy( output, fromData[i] + x * fromObjectSize, fromObjectSize );
			output += fromObjectSize;
		}
	}
}

template<class T>
void CCpuMathEngine::blobMergeByDim0( const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CTypedMemoryHandle<T>& toData )
{
	CTypedMemoryHandle<T> output = toData;
	for( int i = 0; i < fromCount; ++i ) {
		int blobSize = from[i].BlobSize();
		VectorCopy( output, fromData[i], blobSize );
		output += blobSize;
	}
}

template<class T>
void CCpuMathEngine::blobMergeByDim( int dim, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CBlobDesc& to, const CTypedMemoryHandle<T>& toData )
{
	if(dim == 0) {
		return blobMergeByDim0(from, fromData, fromCount, toData);
	}
	return blobMergeByDimCommon(dim, from, fromData, fromCount, to, toData);
}

template<class T>
void CCpuMathEngine::blobSplitByDimCommon( int dimNum, const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount )
{
	int s[CBlobDesc::MaxDimensions];
	from.GetDimSizes( s );
	int objectCount = 1;
	for(int z  = 0; z < dimNum; z++) {
		objectCount *= s[z];
	}
	int objectSize = from.BlobSize() / objectCount;
	for(int x = 0; x < objectCount; x++) {
		CTypedMemoryHandle<T> input = fromData + x * objectSize;
		for( int i = 0; i < toCount; ++i ) {
			int toLimits[CBlobDesc::MaxDimensions];
			to[i].GetDimSizes( toLimits );
			int toObjectSize = 1;
			for(int z = dimNum; z < CBlobDesc::MaxDimensions; z++) {
				toObjectSize *= toLimits[z];
			}
			VectorCopy( toData[i] + x * toObjectSize, input, toObjectSize );
			input += toObjectSize;
		}
	}
}

template<class T>
void CCpuMathEngine::blobSplitByDim0( const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount )
{
	CTypedMemoryHandle<const T> input = fromData;
	for( int i = 0; i < toCount; ++i ) {
		int blobSize = to[i].BlobSize();
		VectorCopy( toData[i], input, blobSize );
		input += blobSize;
	}
}

template<class T>
void CCpuMathEngine::blobSplitByDim( int dim, const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount )
{
	if(dim == 0) {
		return blobSplitByDim0(fromData, to, toData, toCount);
	}
	return blobSplitByDimCommon(dim, from, fromData, to, toData, toCount);
}

void CCpuMathEngine::BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount, const CBlobDesc& to, const CFloatHandle& toData)
{
	ASSERT_EXPR(dim < BD_Count && fromCount <= MaxBlobDescs);
	blobMergeByDim(dim, from, fromData, fromCount, to, toData);
}

void CCpuMathEngine::BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CIntHandle* fromData, int fromCount, const CBlobDesc& to, const CIntHandle& toData)
{
	ASSERT_EXPR(dim < BD_Count && fromCount <= MaxBlobDescs);
	blobMergeByDim(dim, from, fromData, fromCount, to, toData);
}

void CCpuMathEngine::BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CFloatHandle& fromData, const CBlobDesc* to, const CFloatHandle* toData, int toCount)
{
	ASSERT_EXPR(dim < BD_Count && toCount <= MaxBlobDescs);
	blobSplitByDim(dim, from, fromData, to, toData, toCount);
}

void CCpuMathEngine::BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CIntHandle& fromData, const CBlobDesc* to, const CIntHandle* toData, int toCount)
{
	ASSERT_EXPR(dim < BD_Count && toCount <= MaxBlobDescs);
	blobSplitByDim(dim, from, fromData, to, toData, toCount);
}

void CCpuMathEngine::BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
	int deltaTop, int deltaBottom, float defaultValue, const CBlobDesc& to, const CFloatHandle& toData )
{
	int totalChannels = from.Depth() * from.Channels();

	const int outputDataSize = from.ObjectCount() * totalChannels * ( from.Width() + deltaLeft + deltaRight )
		* ( from.Height() + deltaTop + deltaBottom );
	ASSERT_EXPR( to.BlobSize() == outputDataSize );

	// If the size hasn't changed, copy the image
	if( deltaLeft == 0 && deltaRight == 0 && deltaTop == 0 && deltaBottom == 0 ) {
		VectorCopy( toData, fromData, outputDataSize );
		return;
	}

	// If the image size has increased, fill the toData with defaultValue
	if( deltaLeft > 0 || deltaRight > 0 || deltaTop > 0 || deltaBottom > 0 ) {
		VectorFill( toData, defaultValue, outputDataSize );
	}

	// The pointer to the current image (the element of a separate batch)
	CFloatHandle inputImage = fromData;
	CFloatHandle outputImage = toData;
	// The image size (used to offset pointers)
	const int inputImageSize = from.ObjectSize();
	const int outputImageSize = to.ObjectSize();
	// The image rows length
	const int inputRowSize = from.Width() * from.Depth() * from.Channels();
	const int outputRowSize = to.Width() * to.Depth() * to.Channels();
	for( int batch = 0; batch < from.ObjectCount(); ++batch ) {
		if( deltaLeft == 0 && deltaRight == 0 ) {
			ASSERT_EXPR( inputRowSize == outputRowSize );
			// If the image width isn't changed, copy together
			VectorCopy( CFloatHandle( outputImage + outputRowSize * max( 0, deltaTop ) ),
				CFloatHandle( inputImage + inputRowSize * max( 0, -deltaTop ) ),
				inputRowSize * ( from.Height() + min( 0, deltaTop ) + min( 0, deltaBottom ) ) );
		} else {
			// Otherwise copy the necessary parts row by row
			const int horizontalInputOffset = max( 0, -deltaLeft ) * totalChannels;
			const int horizontalOutputOffset = max( 0, deltaLeft ) * totalChannels;
			const int rowSizeToCopy = ( from.Width() + min( 0, deltaLeft ) + min( 0, deltaRight ) ) * totalChannels;
			for( int rowIndex = max( 0, -deltaTop ); rowIndex < from.Height() + min( 0, deltaBottom );
				++rowIndex ) {
				VectorCopy( CFloatHandle( outputImage + outputRowSize * ( rowIndex + deltaTop ) + horizontalOutputOffset ),
					CFloatHandle( inputImage + inputRowSize * rowIndex + horizontalInputOffset ),
					rowSizeToCopy );
			}
		}
		inputImage += inputImageSize;
		outputImage += outputImageSize;
	}
}

void CCpuMathEngine::BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData,
	const CIntHandle& indexHandle, const CBlobDesc& to, const CFloatHandle& toData,
	int startPos, bool isRev )
{
	ASSERT_EXPR( from.BatchWidth() == to.BatchWidth() && from.ObjectSize() == to.ObjectSize()
		&& from.ListSize() == to.ListSize() );

	int* indices = GetRaw( indexHandle );
	int batchWidth = from.BatchWidth();
	int objectSize = from.ObjectSize() * from.ListSize();
	int subSequenceLen = to.BatchLength();

	CFloatHandle curToData = toData;

	// Calculate the subsequence using sequenceLen
	for( int pos = 0; pos < subSequenceLen; ++pos ) {
		for( int seq = 0; seq < batchWidth; ++seq ) {
			int index = startPos;
			if( isRev ) {
				index -= pos;
			} else {
				index += pos;
			}
			index = index * batchWidth + seq;
			VectorCopy( curToData, CConstFloatHandle( fromData + index * objectSize ), objectSize );
			if( indices != 0 ) {
				*indices++ = index;
			}
			curToData += objectSize;
		}
	}
}

//------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::Upsampling2DForward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount, int widthCopyCount,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( heightCopyCount > 0 );
	ASSERT_EXPR( widthCopyCount > 0 );
	ASSERT_EXPR( input.BatchLength() == result.BatchLength() );
	ASSERT_EXPR( input.BatchWidth() == result.BatchWidth() );
	ASSERT_EXPR( input.Channels() == result.Channels() );
	ASSERT_EXPR( input.Depth() == result.Depth() );
	ASSERT_EXPR( input.Height() * heightCopyCount == result.Height() );
	ASSERT_EXPR( input.Width() * widthCopyCount == result.Width() );

	const int inputHeight = input.Height();
	const int inputWidth = input.Width();
	const int pixelSize = input.Depth() * input.Channels();

	const int resultRowSize = result.Width() * result.Depth() * result.Channels();
	const int objectCount = input.ObjectCount();

	CConstFloatHandle inputPtr = inputData;
	CFloatHandle outputPtr = resultData;

	for(int b = 0; b < objectCount; ++b) {
		for(int srcRowIndex = 0; srcRowIndex < inputHeight; ++srcRowIndex) {
			// Note the start of the output row with the index srcRowIndex * heightCopyCount
			CFloatHandle resultRowStart = outputPtr;

			// Fill the output row with the index srcRowIndex * heightCopyCount
			for(int srcColIndex = 0; srcColIndex < inputWidth; ++srcColIndex) {
				for(int w = 0; w < widthCopyCount; ++w) {
					VectorCopy(outputPtr, inputPtr, pixelSize);
					outputPtr += pixelSize;
				}
				inputPtr += pixelSize;
			}

			// Fill the rest heightCopyCount - 1 output rows with the indices
			// srcRowIndex * heightCopyCount + 1, ..., srcRowIndex * heightCopyCount + heightCopyCount - 1.
			for(int h = 0; h < heightCopyCount - 1; ++h) {
				VectorCopy(outputPtr, resultRowStart, resultRowSize);
				outputPtr += resultRowSize;
			}
		}
	}
}

void CCpuMathEngine::Upsampling2DBackward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount, int widthCopyCount,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( heightCopyCount > 0 );
	ASSERT_EXPR( widthCopyCount > 0 );
	ASSERT_EXPR( input.BatchLength() == result.BatchLength() );
	ASSERT_EXPR( input.BatchWidth() == result.BatchWidth() );
	ASSERT_EXPR( input.Channels() == result.Channels() );
	ASSERT_EXPR( input.Depth() == result.Depth() );
	ASSERT_EXPR( result.Height() * heightCopyCount == input.Height() );
	ASSERT_EXPR( result.Width() * widthCopyCount == input.Width() );

	const int objectCount = input.ObjectCount();
	const int pixelSize = input.Depth() * input.Channels();

	CFloatHandleStackVar temp( mathEngine(),
		objectCount * result.Height() * heightCopyCount * result.Width() * widthCopyCount * pixelSize);

	SumMatrixRows(objectCount * result.Height(), temp.GetHandle(), inputData,
		heightCopyCount, result.Width() * widthCopyCount * pixelSize);

	SumMatrixRows(objectCount * result.Height() * result.Width(), resultData, temp.GetHandle(),
		widthCopyCount, pixelSize);
}

void CCpuMathEngine::BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
	const CIntHandle& resultHandle, int maxNumber )
{
	VectorFill( resultHandle, 0, maxNumber );
	const int* numbers = GetRaw( numbersHandle );
	int* result = GetRaw( resultHandle );

	for( int i = 0; i < numbersCount; ++i ) {
		int currNumber = *numbers++;
		if( currNumber >= 0 ) {
			result[currNumber]++;
		}
	}
}

static inline int LegacyRepackIndex( int fromIndex, int channels, int height, int width )
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
static inline void ReorgFunc( const T* source, int stride, bool isForward, int batchSize, int channelsCount,
	int height, int width, T* result )
{
	const int outputChannels = channelsCount / ( stride * stride );
	for( int batch = 0; batch < batchSize; ++batch ) {
		for( int channel = 0; channel < channelsCount; ++channel ) {
			for( int h = 0; h < height; ++h ) {
				for( int w = 0; w < width; ++w ) {
					int inputIndex = w + width * ( h + height * ( channel + channelsCount * batch ) );
					inputIndex = LegacyRepackIndex( inputIndex, channelsCount * stride * stride, height / stride, width / stride );
					const int outputChannelId = channel % outputChannels;
					const int offset = channel / outputChannels;
					const int outputW = w*stride + offset % stride;
					const int outputH = h*stride + offset / stride;
					int outputIndex = outputW + width * stride * ( outputH + height * stride *
						( outputChannelId + outputChannels * batch ) );
					outputIndex = LegacyRepackIndex( outputIndex, channelsCount, height, width );
					if( isForward ) {
						result[inputIndex] = source[outputIndex];
					} else {
						result[outputIndex] = source[inputIndex];
					}
				}
			}
		}
	}
}

void CCpuMathEngine::Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	if( isForward ) {
		ReorgFunc( GetRaw( sourceData ), stride, isForward, source.ObjectCount(),
			source.Channels(), source.Height(), source.Width(), GetRaw( resultData ) );
	} else {
		ReorgFunc( GetRaw( sourceData ), stride, isForward, source.ObjectCount(),
			result.Channels(), result.Height(), result.Width(), GetRaw( resultData ) );
	}
}

void CCpuMathEngine::Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward, 
	const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	if( isForward ) {
		ReorgFunc( GetRaw( sourceData ), stride, isForward, source.ObjectCount(),
			source.Channels(), source.Height(), source.Width(), GetRaw( resultData ) );
	} else {
		ReorgFunc( GetRaw( sourceData ), stride, isForward, source.ObjectCount(),
			result.Channels(), result.Height(), result.Width(), GetRaw( resultData ) );
	}
}

void CCpuMathEngine::QrnnFPooling( bool reverse, int sequenceLength, int objectSize,
	const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& initialState,
	const CFloatHandle& result )
{
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( update.GetMathEngine() == this );
	ASSERT_EXPR( forget.GetMathEngine() == this );
	ASSERT_EXPR( initialState.IsNull() || initialState.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

	// Global means outside of OMP
	const float* globalZ = GetRaw( update );
	const float* globalF = GetRaw( forget );
	const float* globalH0 = initialState.IsNull() ? nullptr : GetRaw( initialState );
	float* globalRes = GetRaw( result );

	const int nextObjectOffset = reverse ? -objectSize : objectSize;

	if( reverse ) {
		const int firstElemOffset = ( sequenceLength - 1 ) * objectSize;
		globalZ += firstElemOffset;
		globalF += firstElemOffset;
		globalRes += firstElemOffset;
	}

	const int currThreadCount = IsOmpRelevant( objectSize, sequenceLength * objectSize ) ? threadCount : 1;
	NEOML_OMP_NUM_THREADS( currThreadCount )
	{
		int start;
		int count;
		if( OmpGetTaskIndexAndCount( objectSize, start, count ) ) {
			const float* z = globalZ + start;
			const float* f = globalF + start;
			const float* h0 = globalH0 == nullptr ? nullptr : globalH0 + start;
			float* res = globalRes + start;

			const int sseSize = count / 4;
			const int nonSseSize = count % 4;
			if( h0 == nullptr ) {
				NeoML::qrnnFPoolingFirstStep( z, f, res, sseSize, nonSseSize );
			} else {
				NeoML::qrnnFPoolingStep( z, f, h0, res, sseSize, nonSseSize );
			}

			const float* hPrev = res;
			for( int step = 0; step < sequenceLength - 1; ++step ) {
				z += nextObjectOffset;
				f += nextObjectOffset;
				res += nextObjectOffset;
				NeoML::qrnnFPoolingStep( z, f, hPrev, res, sseSize, nonSseSize );
				hPrev = res;
			}
		}
	}
}

void CCpuMathEngine::QrnnFPoolingBackward( bool reverse, int sequenceLength, int objectSize,
	const CConstFloatHandle& update, const CConstFloatHandle& forget,
	const CConstFloatHandle& initialState, const CConstFloatHandle& result, const CFloatHandle& resultDiff,
	const CFloatHandle& updateDiff, const CFloatHandle& forgetDiff )
{
	// This implementation isn't heavily optimized because it's backward for CPU
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( update.GetMathEngine() == this );
	ASSERT_EXPR( forget.GetMathEngine() == this );
	ASSERT_EXPR( initialState.IsNull() || initialState.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( updateDiff.GetMathEngine() == this );
	ASSERT_EXPR( forgetDiff.GetMathEngine() == this );

	CConstFloatHandle z = update;
	CConstFloatHandle f = forget;
	CConstFloatHandle h0 = initialState;
	CConstFloatHandle out = result;
	CFloatHandle outDiff = resultDiff;
	CFloatHandle zDiff = updateDiff;
	CFloatHandle fDiff = forgetDiff;

	const int nextObjectOffset = reverse ? -objectSize : objectSize;

	if( reverse ) {
		const int firstElemOffset = ( sequenceLength - 1 ) * objectSize;
		z += firstElemOffset;
		f += firstElemOffset;
		out += firstElemOffset;
		outDiff += firstElemOffset;
		zDiff += firstElemOffset;
		fDiff += firstElemOffset;
	}

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		// zDiff = outDiff * (1 - f) = outDiff - f * outDiff
		VectorEltwiseNegMultiply( outDiff, f, zDiff, objectSize );
		VectorAdd( zDiff, outDiff, zDiff, objectSize );
		// fDiff = outDifF * (prevOut - z) = outDiff * prevOut - outDiff * z
		VectorEltwiseNegMultiply( outDiff, z, fDiff, objectSize );
		VectorEltwiseMultiplyAdd( outDiff, out + nextObjectOffset, fDiff, objectSize );
		// Adding diff of recurrent part
		// prevOutDiff += outDiff * f
		VectorEltwiseMultiplyAdd( outDiff, f, outDiff + nextObjectOffset, objectSize );

		z += nextObjectOffset;
		f += nextObjectOffset;
		out += nextObjectOffset;
		outDiff += nextObjectOffset;
		zDiff += nextObjectOffset;
		fDiff += nextObjectOffset;
	}

	// Last step
	// zDiff = outDiff * (1 - f) = outDiff - f * outDiff
	VectorEltwiseNegMultiply( outDiff, f, zDiff, objectSize );
	VectorAdd( zDiff, outDiff, zDiff, objectSize );
	VectorEltwiseNegMultiply( outDiff, z, fDiff, objectSize );
	if( !h0.IsNull() ) {
		VectorEltwiseMultiplyAdd( outDiff, h0, fDiff, objectSize );
	}
}

void CCpuMathEngine::QrnnIfPooling( bool reverse, int sequenceLength, int objectSize,
	const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& input,
	const CConstFloatHandle& initialState, const CFloatHandle& result )
{
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( update.GetMathEngine() == this );
	ASSERT_EXPR( forget.GetMathEngine() == this );
	ASSERT_EXPR( input.GetMathEngine() == this );
	ASSERT_EXPR( initialState.IsNull() || initialState.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );

	// Global means outside of OMP
	const float* globalZ = GetRaw( update );
	const float* globalF = GetRaw( forget );
	const float* globalI = GetRaw( input );
	const float* globalH0 = initialState.IsNull() ? nullptr : GetRaw( initialState );
	float* globalRes = GetRaw( result );

	const int nextObjectOffset = reverse ? -objectSize : objectSize;

	if( reverse ) {
		const int firstElemOffset = ( sequenceLength - 1 ) * objectSize;
		globalZ += firstElemOffset;
		globalF += firstElemOffset;
		globalI += firstElemOffset;
		globalRes += firstElemOffset;
	}

	const int currThreadCount = IsOmpRelevant( objectSize, sequenceLength * objectSize ) ? threadCount : 1;
	NEOML_OMP_NUM_THREADS( currThreadCount )
	{
		int start;
		int count;
		if( OmpGetTaskIndexAndCount( objectSize, start, count ) ) {
			const float* z = globalZ + start;
			const float* f = globalF + start;
			const float* i = globalI + start;
			const float* h0 = globalH0 == nullptr ? nullptr : globalH0 + start;
			float* res = globalRes + start;

			const int sseSize = count / 4;
			const int nonSseSize = count % 4;
			if( h0 == nullptr ) {
				NeoML::vectorEltwiseMultiply( i, z, res, sseSize, nonSseSize );
			} else {
				NeoML::qrnnIfPoolingStep( z, f, i, h0, res, sseSize, nonSseSize );
			}

			const float* hPrev = res;
			for( int step = 0; step < sequenceLength - 1; ++step ) {
				z += nextObjectOffset;
				f += nextObjectOffset;
				i += nextObjectOffset;
				res += nextObjectOffset;
				NeoML::qrnnIfPoolingStep( z, f, i, hPrev, res, sseSize, nonSseSize );
				hPrev = res;
			}
		}
	}
}

void CCpuMathEngine::QrnnIfPoolingBackward( bool reverse, int sequenceLength, int objectSize,
	const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& input,
	const CConstFloatHandle& initialState, const CConstFloatHandle& result, const CFloatHandle& resultDiff,
	const CFloatHandle& updateDiff, const CFloatHandle& forgetDiff, const CFloatHandle& inputDiff )
{
	// This implementation isn't heavily optimized because it's backward for CPU
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( update.GetMathEngine() == this );
	ASSERT_EXPR( forget.GetMathEngine() == this );
	ASSERT_EXPR( input.GetMathEngine() == this );
	ASSERT_EXPR( initialState.IsNull() || initialState.GetMathEngine() == this );
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( updateDiff.GetMathEngine() == this );
	ASSERT_EXPR( forgetDiff.GetMathEngine() == this );
	ASSERT_EXPR( inputDiff.GetMathEngine() == this );

	CConstFloatHandle z = update;
	CConstFloatHandle f = forget;
	CConstFloatHandle i = input;
	CConstFloatHandle h0 = initialState;
	CConstFloatHandle out = result;
	CFloatHandle outDiff = resultDiff;
	CFloatHandle zDiff = updateDiff;
	CFloatHandle fDiff = forgetDiff;
	CFloatHandle iDiff = inputDiff;

	const int nextObjectOffset = reverse ? -objectSize : objectSize;

	if( reverse ) {
		const int firstElemOffset = ( sequenceLength - 1 ) * objectSize;
		z += firstElemOffset;
		f += firstElemOffset;
		i += firstElemOffset;
		out += firstElemOffset;
		outDiff += firstElemOffset;
		zDiff += firstElemOffset;
		fDiff += firstElemOffset;
		iDiff += firstElemOffset;
	}

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		// zDiff = outDiff * i
		VectorEltwiseMultiply( outDiff, i, zDiff, objectSize );
		// fDiff = outDiff * prevOut
		VectorEltwiseMultiply( outDiff, out + nextObjectOffset, fDiff, objectSize );
		// iDiff = outDiff * z
		VectorEltwiseMultiply( outDiff, z, iDiff, objectSize );
		// Adding diff of recurrent part
		// prevOutDiff += outDiff * f
		VectorEltwiseMultiplyAdd( outDiff, f, outDiff + nextObjectOffset, objectSize );

		z += nextObjectOffset;
		f += nextObjectOffset;
		i += nextObjectOffset;
		out += nextObjectOffset;
		outDiff += nextObjectOffset;
		zDiff += nextObjectOffset;
		fDiff += nextObjectOffset;
		iDiff += nextObjectOffset;
	}

	// Last step
	// zDiff = outDiff * i
	VectorEltwiseMultiply( outDiff, i, zDiff, objectSize );
	// iDiff = outDiff * z
	VectorEltwiseMultiply( outDiff, z, iDiff, objectSize );
	if( h0.IsNull() ) {
		// prevOut == 0
		// fDiff = outDiff * prevOut = 0
		VectorFill( fDiff, 0.f, objectSize );
	} else {
		// fDiff = outDiff * prevOut
		VectorEltwiseMultiply( outDiff, h0, fDiff, objectSize );
	}
}

void CCpuMathEngine::IndRnnRecurrent( bool reverse, int sequenceLength, int batchSize, int objectSize,
	const CConstFloatHandle& wx, const CConstFloatHandle& mask, const CConstFloatHandle& u,
	const CFloatHandle& h)
{
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( batchSize >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( wx.GetMathEngine() == this );
	ASSERT_EXPR( mask.IsNull() || mask.GetMathEngine() == this );
	ASSERT_EXPR( u.GetMathEngine() == this );
	ASSERT_EXPR( h.GetMathEngine() == this );

	const int stepOffset = reverse ? -batchSize * objectSize : batchSize * objectSize;
	const int firstStepOffset = reverse ? ( sequenceLength - 1 ) * batchSize * objectSize : 0;

	VectorSigmoid( wx + firstStepOffset, h + firstStepOffset, batchSize * objectSize );

	CConstFloatHandle hPrev = h + firstStepOffset;

	for( int step = 1; step < sequenceLength; ++step ) {
		CConstFloatHandle currWx = wx + firstStepOffset + step * stepOffset;
		CConstFloatHandle currMask = mask;
		CFloatHandle currH = h + firstStepOffset + step * stepOffset;
		CConstFloatHandle currHPrev = currH - stepOffset;
		for( int batch = 0; batch < batchSize; ++batch ) {
			if( mask.IsNull() ) {
				VectorEltwiseMultiply( currHPrev, u, currH, objectSize );
				VectorAdd( currH, currWx, currH, objectSize );
				VectorSigmoid( currH, currH, objectSize );
			} else {
				VectorEltwiseMultiply( currHPrev, currMask, currH, objectSize );
				VectorEltwiseMultiply( currH, u, currH, objectSize );
				VectorAdd( currH, currWx, currH, objectSize );
				VectorSigmoid( currH, currH, objectSize );
			}
			currWx += objectSize;
			currMask += objectSize;
			currH += objectSize;
			currHPrev += objectSize;
		}

		hPrev = h;
	}
}

void CCpuMathEngine::IndRnnRecurrentBackward( bool reverse, int sequenceLength, int batchSize, int objectSize,
	const CConstFloatHandle& mask, const CConstFloatHandle& u, const CConstFloatHandle& h, const CConstFloatHandle& hDiff,
	const CFloatHandle& wxDiff )
{
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( batchSize >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( mask.IsNull() || mask.GetMathEngine() == this );
	ASSERT_EXPR( u.GetMathEngine() == this );
	ASSERT_EXPR( h.GetMathEngine() == this );
	ASSERT_EXPR( hDiff.GetMathEngine() == this );
	ASSERT_EXPR( wxDiff.GetMathEngine() == this );

	const int stepOffset = reverse ? -batchSize * objectSize : batchSize * objectSize;
	const int firstStepOffset = reverse ? ( sequenceLength - 1 ) * batchSize * objectSize : 0;

	CFloatHandleStackVar totalHDiff( *this, batchSize * objectSize );
	VectorCopy( totalHDiff.GetHandle(), hDiff + firstStepOffset, batchSize * objectSize );

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		CConstFloatHandle currMask = mask;
		CConstFloatHandle currH = h + firstStepOffset + step * stepOffset;
		CConstFloatHandle currHDiff = hDiff + firstStepOffset + step * stepOffset;
		CFloatHandle currTotalHDiff = totalHDiff.GetHandle();
		CFloatHandle currWxDiff = wxDiff + firstStepOffset + step * stepOffset;

		for( int batch = 0; batch < batchSize; ++batch ) {
			VectorSigmoidDiffOp( currH, currTotalHDiff, currWxDiff, objectSize );
			VectorEltwiseMultiply( currWxDiff, u, currTotalHDiff, objectSize );
			if( !currMask.IsNull() ) {
				VectorEltwiseMultiply( currMask, currTotalHDiff, currTotalHDiff, objectSize );
			}
			VectorAdd( currHDiff + stepOffset, currTotalHDiff, currTotalHDiff, objectSize );

			if( !currMask.IsNull() ) {
				currMask += objectSize;
			}
			currH += objectSize;
			currHDiff += objectSize;
			currTotalHDiff += objectSize;
			currWxDiff += objectSize;
		}
	}

	const int lastStepOffset = reverse ? 0 : ( sequenceLength - 1 ) * stepOffset;
	VectorSigmoidDiffOp( h + lastStepOffset, totalHDiff.GetHandle(), wxDiff + lastStepOffset, batchSize * objectSize );
}

void CCpuMathEngine::IndRnnRecurrentLearn( bool reverse, int sequenceLength, int batchSize, int objectSize,
	const CConstFloatHandle& mask, const CConstFloatHandle& u, const CConstFloatHandle& h, const CConstFloatHandle& hDiff,
	const CFloatHandle& uDiff )
{
	ASSERT_EXPR( sequenceLength >= 1 );
	ASSERT_EXPR( batchSize >= 1 );
	ASSERT_EXPR( objectSize >= 1 );
	ASSERT_EXPR( mask.IsNull() || mask.GetMathEngine() == this );
	ASSERT_EXPR( u.GetMathEngine() == this );
	ASSERT_EXPR( h.GetMathEngine() == this );
	ASSERT_EXPR( hDiff.GetMathEngine() == this );
	ASSERT_EXPR( uDiff.GetMathEngine() == this );

	const int stepOffset = reverse ? -batchSize * objectSize : batchSize * objectSize;
	const int firstStepOffset = reverse ? ( sequenceLength - 1 ) * batchSize * objectSize : 0;

	CFloatHandleStackVar totalHDiff( *this, batchSize * objectSize );
	VectorCopy( totalHDiff.GetHandle(), hDiff + firstStepOffset, batchSize * objectSize );

	CFloatHandleStackVar buff( *this, objectSize );

	for( int step = 0; step < sequenceLength - 1; ++step ) {
		CConstFloatHandle currMask = mask;
		CConstFloatHandle currH = h + firstStepOffset + step * stepOffset;
		CConstFloatHandle currHDiff = hDiff + firstStepOffset + step * stepOffset;
		CFloatHandle currTotalHDiff = totalHDiff.GetHandle();

		for( int batch = 0; batch < batchSize; ++batch ) {
			VectorSigmoidDiffOp( currH, currTotalHDiff, buff.GetHandle(), objectSize );
			if( !currMask.IsNull() ) {
				VectorEltwiseMultiply( buff.GetHandle(), currMask, buff.GetHandle(), objectSize );
			}
			VectorEltwiseMultiplyAdd( buff.GetHandle(), currH + stepOffset, uDiff, objectSize );
			VectorEltwiseMultiply( buff.GetHandle(), u, buff.GetHandle(), objectSize );
			VectorAdd( currHDiff + stepOffset, buff.GetHandle(), currTotalHDiff, objectSize );

			if( !currMask.IsNull() ) {
				currMask += objectSize;
			}
			currH += objectSize;
			currHDiff += objectSize;
			currTotalHDiff += objectSize;
		}
	}
}

template<class T>
static inline void SpaceToDepthFunc( const T* source, int dataRowCount, int dataRowWidth,
	int blockChannels, int blockSize, bool isForward, T* result, int threadCount )
{
	// flattens 3d-block of size (blockSize x blockSize x channels)

	// number of elements in a single row inside 3d-block
	const int blockRowSize = blockChannels * blockSize;

	// offset for switching to the next data row
	const int dataRowSize = blockSize * ( dataRowWidth * blockSize ) * blockChannels;
	// offset for switching to the next block inside data row
	const int sourceBlockOffset = isForward ? blockRowSize : blockSize * blockRowSize;
	const int resultBlockOffset = isForward ? blockSize * blockRowSize : blockRowSize;
	// offset for switching to the next row inside the 3d-block
	const int sourceBlockRowOffset = isForward ? dataRowWidth * blockRowSize : blockRowSize;
	const int resultBlockRowOffset = isForward ? blockRowSize : dataRowWidth * blockRowSize;

	// iterate over data rows
	const int blobSize = dataRowCount * dataRowWidth * blockSize * blockRowSize;
	const int curThreadCount = IsOmpRelevant( dataRowCount, blobSize ) ? threadCount : 1;
	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int threadRowStart;
		int threadRowCount;
		if( OmpGetTaskIndexAndCount( dataRowCount, threadRowStart, threadRowCount ) ) {
			const T* sourcePtr = source + threadRowStart * dataRowSize;
			T* resultPtr = result + threadRowStart * dataRowSize;
			for( int dataRowIndex = 0; dataRowIndex < threadRowCount; ++dataRowIndex ) {
				const T* sourceRow = sourcePtr;
				T* resultRow = resultPtr;
				// iterate over blocks in data row
				for( int blockIndex = 0; blockIndex < dataRowWidth; ++blockIndex ) {
					const T* sourceBlock = sourceRow;
					T* resultBlock = resultRow;
					// iterate over rows of 3-dimensional (blockSize x blockSize x channels) block
					for( int blockRowIndex = 0; blockRowIndex < blockSize; ++blockRowIndex ) {
						// copy current row of 3d-block
						dataCopy( resultBlock, sourceBlock, blockRowSize );
						sourceBlock += sourceBlockRowOffset;
						resultBlock += resultBlockRowOffset;
					}
					// switching to the next block
					sourceRow += sourceBlockOffset;
					resultRow += resultBlockOffset;
				}
				sourcePtr += dataRowSize;
				resultPtr += dataRowSize;
			}
		}
	}
}

void CCpuMathEngine::SpaceToDepth( const CBlobDesc& source, const CConstFloatHandle& sourceData, int blockSize,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( source.Height() == result.Height() * blockSize );
	ASSERT_EXPR( source.Width() == result.Width() * blockSize );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( source.Channels() * blockSize * blockSize == result.Channels() );

	SpaceToDepthFunc( GetRaw( sourceData ), source.ObjectCount() * result.Height(), result.Width(), source.Channels(),
		blockSize, true, GetRaw( resultData ), threadCount );
}

void CCpuMathEngine::SpaceToDepth( const CBlobDesc& source, const CConstIntHandle& sourceData, int blockSize,
	const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( source.Height() == result.Height() * blockSize );
	ASSERT_EXPR( source.Width() == result.Width() * blockSize );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( source.Channels() * blockSize * blockSize == result.Channels() );

	SpaceToDepthFunc( GetRaw( sourceData ), source.ObjectCount() * result.Height(), result.Width(), source.Channels(),
		blockSize, true, GetRaw( resultData ), threadCount );
}

void CCpuMathEngine::DepthToSpace( const CBlobDesc& source, const CConstFloatHandle& sourceData, int blockSize,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( source.Height() * blockSize == result.Height() );
	ASSERT_EXPR( source.Width() * blockSize == result.Width() );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( source.Channels() == result.Channels() * blockSize * blockSize );

	SpaceToDepthFunc( GetRaw( sourceData ), source.ObjectCount() * source.Height(), source.Width(), result.Channels(),
		blockSize, false, GetRaw( resultData ), threadCount );
}

void CCpuMathEngine::DepthToSpace( const CBlobDesc& source, const CConstIntHandle& sourceData, int blockSize,
	const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( source.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( source.Height() * blockSize == result.Height() );
	ASSERT_EXPR( source.Width() * blockSize == result.Width() );
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( source.Channels() == result.Channels() * blockSize * blockSize );

	SpaceToDepthFunc( GetRaw( sourceData ), source.ObjectCount() * source.Height(), source.Width(), result.Channels(),
		blockSize, false, GetRaw( resultData ), threadCount );
}

} // namespace NeoML
