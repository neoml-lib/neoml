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

#include <common.h>
#pragma hdrstop

#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnPoolings.h>
#include <CpuMathEnginePrivate.h>
#include "CpuMathEngineDnnPooling.h"

namespace NeoML {

//------------------------------------------------------------------------------------------------------------
// Max pooling

CMaxPoolingDesc* CCpuMathEngine::InitMaxPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	const CBlobDesc& result )
{
	CCommonMaxPoolingDesc* desc = new CCommonMaxPoolingDesc( source, result, filterHeight, filterWidth, strideHeight, strideWidth );
	return desc;
}

void CCpuMathEngine::blobMaxPoolingWithIndices( const CCommonMaxPoolingDesc& desc, const float* sourceData,
	int* maxIndicesData, float* resultData )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int channels = result.Depth() * result.Channels();
	const int sourceRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;

	CFloatHandleStackVar buffer( *this, sourceRowSize );
	float* bufferRaw = GetRaw( buffer.GetHandle() );
	CIntHandleStackVar rowIndexBlob( *this, sourceRowSize );
	int* rowIndexBuffer = GetRaw( rowIndexBlob.GetHandle() );
	CIntHandleStackVar columnIndexBlob( *this, channels );
	int* columnIndexBuffer = GetRaw( columnIndexBlob.GetHandle() );

	int* maxIndicesPtr = maxIndicesData;
	const float* sourcePtr = sourceData;
	float* resultPtr = resultData;

	for( int i = 0; i < source.ObjectCount(); ++i ) {
		for( int j = 0; j < result.Height(); ++j ) {
			// Calculate maximums in columns over a strip of the window height
			const int currentStripRow = desc.StrideHeight * j;
			const float* currentStripStart = sourcePtr + currentStripRow * sourceRowSize;
			findMaxValueInColumns( bufferRaw, rowIndexBuffer, currentStripStart, desc.FilterHeight, sourceRowSize );
			// Calculate maximum over each window
			const float* currentbufferStart = bufferRaw;
			int currentWindowColumn = 0;
			for( int k = 0; k < result.Width(); ++k ) {
				findMaxValueInColumns( resultPtr, columnIndexBuffer, currentbufferStart, desc.FilterWidth, channels );
				for( int l = 0; l < channels; ++l ) {
					const int windowIndex = columnIndexBuffer[l] * channels + l;
					// Calculate the maximum element's index. It is the sum of the current strip offset, 
					// the number of the row in the strip, the window offset and the number of the column in the window
					*maxIndicesPtr = ( currentStripRow + rowIndexBuffer[windowIndex] ) * sourceRowSize + currentWindowColumn + windowIndex;
					++maxIndicesPtr;
				}
				currentbufferStart += windowStep;
				currentWindowColumn += windowStep;
				resultPtr += channels;
			}
		}
		sourcePtr += source.ObjectSize();
	}
}

void CCpuMathEngine::blobMaxPoolingWithoutIndices( const CCommon2DPoolingDesc& desc, int resultRowsToProcess,
	const float* sourceData, int sourceRowIndex, float* resultData, int resultRowIndex, float* bufferPtr )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int channels = result.Depth() * result.Channels();
	const int sourceRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;

	const int resultRowsAfterThisCall = resultRowIndex + resultRowsToProcess;
	const int firstImageIndex = resultRowIndex / result.Height();
	const int lastImageIndex = ( resultRowsAfterThisCall - 1 ) / result.Height();

	const float* sourcePtr = sourceData + ( firstImageIndex * source.Height() - sourceRowIndex ) * sourceRowSize;

	for( int i = firstImageIndex; i <= lastImageIndex; ++i ) {
		const int firstRowInImage = ( i == firstImageIndex ? resultRowIndex % result.Height() : 0 );
		const int lastRowInIamge = ( i == lastImageIndex ? ( resultRowsAfterThisCall - 1 ) % result.Height()
			: result.Height() - 1 );
		for( int j = firstRowInImage; j <= lastRowInIamge; ++j ) {
			// Calculate maximums in columns over a strip of the window height
			const float* currentStripStart = sourcePtr + sourceRowSize * desc.StrideHeight * j;
			findMaxValueInColumns( bufferPtr, currentStripStart, desc.FilterHeight, sourceRowSize );
			// Calculate maximum over the window
			const float* currentbufferStart = bufferPtr;
			for( int k = 0; k < result.Width(); ++k ) {
				findMaxValueInColumns( resultData, currentbufferStart, desc.FilterWidth, channels );
				currentbufferStart += windowStep;
				resultData += channels;
			}
		}
		sourcePtr += source.ObjectSize();
	}
}

void CCpuMathEngine::BlobMaxPooling( const CMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle* maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices == 0 || maxIndices->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* sourceDataRaw = GetRaw( sourceData );
	float* resultDataRaw = GetRaw( resultData );

	const CCommonMaxPoolingDesc& desc = static_cast<const CCommonMaxPoolingDesc&>( poolingDesc );

	if( maxIndices != nullptr ) {
		blobMaxPoolingWithIndices( desc, sourceDataRaw, GetRaw( *maxIndices ), resultDataRaw );
	} else {
		CFloatHandleStackVar buffer( *this, desc.Source.Width() * desc.Source.Depth() * desc.Source.Channels() );
		blobMaxPoolingWithoutIndices( desc, desc.Result.Height() * desc.Result.ObjectCount(), sourceDataRaw, 0,
			resultDataRaw, 0, GetRaw( buffer.GetHandle() ) );
	}
}

void CCpuMathEngine::BlobMaxPoolingBackward( const CMaxPoolingDesc& poolingDesc, const CConstFloatHandle& resultDiff,
	const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMaxPoolingDesc& desc = static_cast<const CCommonMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int* maxIndicesPtr = GetRaw( maxIndices );
	const float* resultPtr = GetRaw( resultDiff );
	float* sourcePtr = GetRaw( sourceDiff );

	vectorFill0( sourcePtr, source.BlobSize() );

	const int objectSize = source.ObjectSize();

	for( int i = 0; i < source.ObjectCount(); ++i ) {
		for( int j = 0; j < result.ObjectSize(); ++j ) {
			const int index = *maxIndicesPtr;
			sourcePtr[index] += *resultPtr;
			++maxIndicesPtr;
			++resultPtr;
		}
		sourcePtr += objectSize;
	}
}

//------------------------------------------------------------------------------------------------------------
// Mean pooling

CMeanPoolingDesc* CCpuMathEngine::InitMeanPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	const CBlobDesc& result )
{
	CCommonMeanPoolingDesc* desc = new CCommonMeanPoolingDesc( source, result, filterHeight, filterWidth, strideHeight, strideWidth );
	return desc;
}

void CCpuMathEngine::BlobMeanPooling( const CMeanPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMeanPoolingDesc& desc = static_cast<const CCommonMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;
	CFloatHandleStackVar buffer( mathEngine(), source.Width() * source.Depth() * source.Channels() );
	blobMeanPooling( desc, result.ObjectCount() * result.Height(), GetRaw( sourceData ), 0,
		GetRaw( resultData ), 0, GetRaw( buffer.GetHandle() ) );
}

void CCpuMathEngine::BlobMeanPoolingBackward( const CMeanPoolingDesc& poolingDesc,
	const CConstFloatHandle& resultDiff, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMeanPoolingDesc& desc = static_cast<const CCommonMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	vectorFill0( GetRaw( sourceDiff ), source.BlobSize() );

	const int channels = result.Depth() * result.Channels();
	const int sourceRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;
	CFloatHandleStackVar buffer( mathEngine(), sourceRowSize );

	CFloatHandle sourcePtr = sourceDiff;
	CConstFloatHandle resultPtr = resultDiff;

	for( int i = 0; i < result.ObjectCount(); ++i ) {
		for( int j = 0; j < result.Height(); ++j ) {
			CFloatHandle currentStripStart = sourcePtr + sourceRowSize * desc.StrideHeight * j;
			CFloatHandle bufferPtr = buffer.GetHandle();
			// Generate a row to be added to the source
			vectorFill0( GetRaw( bufferPtr ), sourceRowSize );
			for( int k = 0; k < result.Width(); ++k ) {
				AddVectorToMatrixRows( 1, bufferPtr, bufferPtr, desc.FilterWidth, channels, resultPtr );
				bufferPtr += windowStep;
				resultPtr += channels;
			}
			// Add the row to the source
			AddVectorToMatrixRows( 1, currentStripStart, currentStripStart, desc.FilterHeight, sourceRowSize, buffer.GetHandle() );
		}
		sourcePtr += source.ObjectSize();
	}
	// Multiply the diff by the inverse of the window size
	vectorMultiply( GetRaw( sourceDiff ), GetRaw( sourceDiff ), ( 1.f / desc.FilterHeight / desc.FilterWidth ), source.BlobSize() );
}

//------------------------------------------------------------------------------------------------------------
// GlobalMaxOverTime pooling

CGlobalMaxOverTimePoolingDesc* CCpuMathEngine::InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result )
{
	CCommonGlobalMaxOverTimePoolingDesc* desc = new CCommonGlobalMaxOverTimePoolingDesc( source, result );
	return desc;
}

void CCpuMathEngine::BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CIntHandle* maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices == 0 || maxIndices->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* sourceDataRaw = GetRaw( sourceData );
	float* resultDataRaw = GetRaw( resultData );

	const CCommonGlobalMaxOverTimePoolingDesc& desc = static_cast<const CCommonGlobalMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;

	if( maxIndices != 0 ) {
		findMaxValueInColumns( resultDataRaw, GetRaw( *maxIndices ), sourceDataRaw, source.BatchLength(), source.BatchWidth() * source.ObjectSize() );
	} else {
		findMaxValueInColumns( resultDataRaw, sourceDataRaw, source.BatchLength(), source.BatchWidth() * source.ObjectSize() );
	}
}

void CCpuMathEngine::BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CConstFloatHandle& resultDiff, const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonGlobalMaxOverTimePoolingDesc& desc = static_cast<const CCommonGlobalMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;

	float* sourcePtr = GetRaw( sourceDiff );
	const int* maxIndicesPtr = GetRaw( maxIndices );
	const float* resultPtr = GetRaw( resultDiff );

	vectorFill0( sourcePtr, source.BlobSize() );

	const int sourceObjectSize = source.BatchWidth() * source.ObjectSize();
	for( int i = 0; i < sourceObjectSize; ++i ) {
		sourcePtr[i + sourceObjectSize * *maxIndicesPtr++] = *resultPtr++;
	}
}

//------------------------------------------------------------------------------------------------------------
// BlobGlobalMax pooling

CGlobalMaxPoolingDesc* CCpuMathEngine::InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result )
{
	ASSERT_EXPR( result.ObjectCount() == source.ObjectCount() && maxIndices.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( maxIndices.ObjectSize() == result.ObjectSize() );

	CCommonGlobalMaxPoolingDesc* desc = new CCommonGlobalMaxPoolingDesc( source, result, maxIndices );
	return desc;
}

void CCpuMathEngine::BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc& poolingDesc,
	const CConstFloatHandle& resultDiff, const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonGlobalMaxPoolingDesc& desc = static_cast<const CCommonGlobalMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int* maxIndexPtr = GetRaw( maxIndices );
	const float* resultPtr = GetRaw( resultDiff );
	float* sourcePtr = GetRaw( sourceDiff );

	vectorFill0( sourcePtr, source.BlobSize() );

	const int poolSize = source.Height() * source.Width() * source.Depth();
	const int maxCount = result.Height() * result.Width() * result.Depth();
	const int objectSize = poolSize * source.Channels();

	for( int b = 0; b < source.ObjectCount(); ++b ) {
		for( int i = 0; i < maxCount; ++i ) {
			float* sourceChannelData = sourcePtr;
			for( int c = 0; c < result.Channels(); ++c ) {
				const int index = *maxIndexPtr++;
				if( index >= 0 ) {
					PRESUME_EXPR( index < poolSize );
					sourceChannelData[index * source.Channels()] = *resultPtr;
				}
				++resultPtr;
				++sourceChannelData;
			}
		}
		sourcePtr += objectSize;
	}
}

//------------------------------------------------------------------------------------------------------------
// Blob3dMax pooling

C3dMaxPoolingDesc* CCpuMathEngine::Init3dMaxPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& result )
{
	CCommon3dMaxPoolingDesc* desc = new CCommon3dMaxPoolingDesc( source, result, filterHeight, filterWidth, filterDepth,
		strideHeight, strideWidth, strideDepth );
	return desc;
}

void CCpuMathEngine::Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& poolingDesc, const CConstFloatHandle& resultDiff,
	const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dMaxPoolingDesc& desc = static_cast<const CCommon3dMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int* indexPtr = GetRaw( maxIndices );
	const float* resultDiffRaw = GetRaw( resultDiff );
	float* sourceDiffRaw = GetRaw( sourceDiff );

	vectorFill0( sourceDiffRaw, source.BlobSize() );

	for( int b = 0; b < source.ObjectCount(); ++b ) {
		for( int i = 0; i < result.GeometricalSize(); ++i ) {
			for( int channel = 0; channel < result.Channels(); ++channel ) {
				sourceDiffRaw[*indexPtr++ + channel] += *resultDiffRaw++;
			}
		}
		sourceDiffRaw += source.ObjectSize();
	}
}

//------------------------------------------------------------------------------------------------------------
// 3dMean pooling

C3dMeanPoolingDesc* CCpuMathEngine::Init3dMeanPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& result )
{
	CCommon3dMeanPoolingDesc* desc = new CCommon3dMeanPoolingDesc( source, result, filterHeight, filterWidth, filterDepth,
		strideHeight, strideWidth, strideDepth );
	return desc;
}

void CCpuMathEngine::Blob3dMeanPooling( const C3dMeanPoolingDesc& convDesc, const CConstFloatHandle& sourceData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dMeanPoolingDesc& desc = static_cast<const CCommon3dMeanPoolingDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const float* sourceObject = GetRaw( sourceData );
	float* resultJStart = GetRaw( resultData );

	const int sourceDepthSize = source.Depth() * source.Channels();
	const int sourceRowSize = source.Width() * sourceDepthSize;
	const int sourceObjectSize = source.Height() * sourceRowSize;

	const int resultDepthSize = result.Depth() * result.Channels();
	const int resultRowSize = result.Width() * resultDepthSize;

	for( int b = 0; b < result.ObjectCount(); ++b ) {
		// Go through all "cube blocks" and then go over values in each block
		// So we get a forward pass through the source and filterHeight * filterWidth passes through the result
		for( int j = 0; j < result.Height(); ++j ) {
			int sourceJIndex = j * desc.StrideHeight * sourceRowSize;
			for( int filterJ = 0; filterJ < desc.FilterHeight; ++filterJ ) {
				float* resultIStart = resultJStart;

				for( int i = 0; i < result.Width(); ++i ) {
					int sourceIIndex = sourceJIndex + i * desc.StrideWidth * sourceDepthSize;
					for( int filterI = 0; filterI < desc.FilterWidth; ++filterI ) {
						float* resultDataPtr = resultIStart;

						for( int k = 0; k < result.Depth(); ++k ) {
							int sourceIndex = sourceIIndex + k * desc.StrideDepth * source.Channels();
							for( int filterK = 0; filterK < desc.FilterDepth; ++filterK ) {
								if( ( filterJ == 0 ) && ( filterI == 0 ) && ( filterK == 0 ) ) {
									dataCopy( resultDataPtr, sourceObject + sourceIndex, result.Channels() );
								} else {
									vectorAdd( sourceObject + sourceIndex, resultDataPtr, resultDataPtr, result.Channels() );
								}
								sourceIndex += source.Channels();
							}
							resultDataPtr += result.Channels();
						}
						sourceIIndex += sourceDepthSize;
					}
					resultIStart += resultDepthSize;
				}
				sourceJIndex += sourceRowSize;
			}
			resultJStart += resultRowSize;
		}
		sourceObject += sourceObjectSize;
	}
	// Divide the result by filter volume
	vectorMultiply( GetRaw( resultData ), GetRaw( resultData ), ( 1.f / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth ), result.BlobSize() );
}

//------------------------------------------------------------------------------------------------------------
// MaxOverTime pooling

CMaxOverTimePoolingDesc* CCpuMathEngine::InitMaxOverTimePooling( const CBlobDesc& source,
	int filterLen, int strideLen, const CBlobDesc& result )
{
	const int outLen = ( source.BatchLength() - filterLen ) / strideLen + 1;
	ASSERT_EXPR( result.BatchLength() == outLen );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ObjectSize() == source.ObjectSize() );

	CCommonMaxOverTimePoolingDesc* desc = new CCommonMaxOverTimePoolingDesc( source, result, filterLen, strideLen );
	return desc;
}

void CCpuMathEngine::BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& poolingDesc,
	const CConstFloatHandle& resultDiff, const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMaxOverTimePoolingDesc& desc = static_cast<const CCommonMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int* indexPtr = GetRaw( maxIndices );
	const float* resultDiffRaw = GetRaw( resultDiff );
	float* sourceDiffRaw = GetRaw( sourceDiff );

	vectorFill0( sourceDiffRaw, source.BlobSize() );

	const int seqElemSize = source.ObjectSize() * source.BatchWidth();
	for( int l = 0; l < result.BatchLength(); ++l ) {
		for( int i = 0; i < seqElemSize; ++i ) {
			sourceDiffRaw[*indexPtr++ * seqElemSize + i] += *resultDiffRaw++;
		}
	}
}

//---------------------------------------------------------------------------------------------------

static void applyMeanPoolingBackwardSet( const float* resultDiff, int channels,
	int filterHeight, int filterWidth, int filterDepth, float* sourceDiff,
	int sourceRowSize, int sourceDepthSize )
{
	for( int j = 0; j < filterHeight; ++j ) {
		float* sourceDiffDepth = sourceDiff;
		for( int i = 0; i < filterWidth; ++i ) {
			float* sourceDiffPixel = sourceDiffDepth;
			for( int k = 0; k < filterDepth; ++k ) {
				dataCopy( sourceDiffPixel, resultDiff/*Pixel*/, channels );
				sourceDiffPixel += channels;
			}
			sourceDiffDepth += sourceDepthSize;
		}
		sourceDiff += sourceRowSize;
	}
}

static void applyMeanPoolingBackwardAdd( const float* resultDiff, int channels,
	int filterHeight, int filterWidth, int filterDepth, float* sourceDiff,
	int sourceRowSize, int sourceDepthSize )
{
	for( int j = 0; j < filterHeight; ++j ) {
		float* sourceDiffDepth = sourceDiff;
		for( int i = 0; i < filterWidth; ++i ) {
			float* sourceDiffPixel = sourceDiffDepth;
			for( int k = 0; k < filterDepth; ++k ) {
				vectorAdd( resultDiff/*Pixel*/, sourceDiffPixel, sourceDiffPixel, channels );
				sourceDiffPixel += channels;
			}
			sourceDiffDepth += sourceDepthSize;
		}
		sourceDiff += sourceRowSize;
	}
}

void CCpuMathEngine::Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& poolingDesc,
	const CConstFloatHandle& resultDiff, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dMeanPoolingDesc& desc = static_cast<const CCommon3dMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	float* sourceDiffRaw = GetRaw( sourceDiff );
	const float* resultDiffRaw = GetRaw( resultDiff );

	if( desc.FilterHeight != desc.StrideHeight || desc.FilterWidth != desc.StrideWidth || desc.FilterDepth != desc.StrideDepth ) {
		// Either the cube blocks for pooling have nonzero intersections and several diffs should be added up
		// or some of the data is skipped when pooling, and diff should be set to 0 for them
		vectorFill0( sourceDiffRaw, source.BlobSize() );
	}

	// The flag that indicates that the cube blocks for pooling have nonzero intersections
	bool isIntersect = desc.FilterHeight > desc.StrideHeight || desc.FilterWidth > desc.StrideWidth || desc.FilterDepth > desc.StrideDepth;

	const int sourceDepthSize = source.Depth() * source.Channels();
	const int sourceRowSize = sourceDepthSize * source.Width();
	const int sourceObjectSize = sourceRowSize * source.Height();

	float* sourceDiffPtr = sourceDiffRaw;
	const float* resultDiffPtr = resultDiffRaw;
	for( int b = 0; b < result.ObjectCount(); ++b ) {
		int jStart = 0;
		for( int j = 0; j < result.Height(); ++j ) {
			int iStart = jStart;
			for( int i = 0; i < result.Width(); ++i ) {
				int kStart = iStart;
				for( int k = 0; k < result.Depth(); ++k ) {
					if( isIntersect ) {
						applyMeanPoolingBackwardAdd( resultDiffPtr, result.Channels(),
							desc.FilterHeight, desc.FilterWidth, desc.FilterDepth, sourceDiffPtr + kStart,
							sourceRowSize, sourceDepthSize );
					} else {
						applyMeanPoolingBackwardSet( resultDiffPtr, result.Channels(),
							desc.FilterHeight, desc.FilterWidth, desc.FilterDepth, sourceDiffPtr + kStart,
							sourceRowSize, sourceDepthSize );
					}
					resultDiffPtr += result.Channels();
					kStart += source.Channels() * desc.StrideDepth;
				}
				iStart += sourceDepthSize * desc.StrideWidth;
			}
			jStart += sourceRowSize * desc.StrideHeight;
		}
		sourceDiffPtr += sourceObjectSize;
	}
	// Divide the sourceDiff by the filter volume
	vectorMultiply( sourceDiffRaw, sourceDiffRaw, ( 1.f / desc.FilterHeight / desc.FilterWidth / desc.FilterDepth ), source.BlobSize() );
}

//---------------------------------------------------------------------------------------------------

template<typename T>
static void addDimIndex( const CCpuMathEngine* const mathEngine, bool isForward,
	int objectCount, int indexedDimSize, int operateSize,
	const CTypedMemoryHandle<const T>& sourceData, const CTypedMemoryHandle<T>& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == mathEngine );
	ASSERT_EXPR( resultData.GetMathEngine() == mathEngine );
	CCpuExecutionScope scope;

	const T* sourceRaw = GetRaw( sourceData );
	T* resultRaw = GetRaw( resultData );

	for( int batch = 0; batch < objectCount; ++batch ) {
		for( int iDim = 0; iDim < indexedDimSize; ++iDim ) {
			const T value = static_cast<T>( isForward ? iDim : ( -iDim ) );
			vectorAddValue( sourceRaw, resultRaw, operateSize, value );
			sourceRaw += operateSize;
			resultRaw += operateSize;
		}
	}
}

//---------------------------------------------------------------------------------------------------

void CCpuMathEngine::AddWidthIndex( const CBlobDesc& source, const CConstFloatHandle& sourceData, bool isForward, const CFloatHandle& resultData )
{
	addDimIndex( this, isForward, source.ObjectCount() * source.Height(), source.Width(), source.Depth() * source.Channels(), sourceData, resultData);
}

void CCpuMathEngine::AddWidthIndex( const CBlobDesc& source, const CConstIntHandle& sourceData, bool isForward, const CIntHandle& resultData )
{
	addDimIndex( this, isForward, source.ObjectCount() * source.Height(), source.Width(), source.Depth() * source.Channels(), sourceData, resultData );
}

void CCpuMathEngine::AddHeightIndex( const CBlobDesc& source, const CConstFloatHandle& sourceData, bool isForward, const CFloatHandle& resultData )
{
	addDimIndex( this, isForward, source.ObjectCount(), source.Height(), source.Width() * source.Depth() * source.Channels(), sourceData, resultData );
}

void CCpuMathEngine::AddHeightIndex( const CBlobDesc& source, const CConstIntHandle& sourceData, bool isForward, const CIntHandle& resultData )
{
	addDimIndex( this, isForward, source.ObjectCount(), source.Height(), source.Width() * source.Depth() * source.Channels(), sourceData, resultData );
}

} // namespace NeoML
