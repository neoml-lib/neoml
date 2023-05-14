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

	for( int i = 0; i < source.ObjectCount(); ++i ) {
		const float* sourcePtr = sourceData + i * source.ObjectSize();
		float* resultPtr = resultData + i * result.ObjectSize();
		int* maxIndicesPtr = maxIndicesData + i * result.ObjectSize();
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
	}
}

void CCpuMathEngine::blobMaxPoolingWithoutIndices( const CCommonMaxPoolingDesc& desc,
	const float* sourceData, float* resultData )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int channels = result.Depth() * result.Channels();
	const int sourceRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;

	CFloatHandleStackVar buffer( *this, sourceRowSize );
	float* bufferPtr = GetRaw( buffer.GetHandle() );

	for( int i = 0; i < source.ObjectCount(); ++i ) {
		const float* sourcePtr = sourceData + i * source.ObjectSize();
		float* resultPtr = resultData + i * result.ObjectSize();
		for( int j = 0; j < result.Height(); ++j ) {
			// Calculate maximums in columns over a strip of the window height
			const float* currentStripStart = sourcePtr + sourceRowSize * desc.StrideHeight * j;
			findMaxValueInColumns( bufferPtr, currentStripStart, desc.FilterHeight, sourceRowSize );
			// Calculate maximum over the window
			const float* currentbufferStart = bufferPtr;
			for( int k = 0; k < result.Width(); ++k ) {
				findMaxValueInColumns( resultPtr, currentbufferStart, desc.FilterWidth, channels );
				currentbufferStart += windowStep;
				resultPtr += channels;
			}
		}
	}
}

void CCpuMathEngine::BlobMaxPooling( const CMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
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
		blobMaxPoolingWithoutIndices( desc, sourceDataRaw, resultDataRaw );
	}
}

void CCpuMathEngine::BlobMaxPoolingBackward( const CMaxPoolingDesc& poolingDesc, const CFloatHandle& resultDiff,
	const CIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMaxPoolingDesc& desc = static_cast<const CCommonMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	VectorFill( sourceDiff, 0, source.BlobSize() );

	for( int i = 0; i < result.ObjectCount(); ++i ) {
		CFloatHandle sourcePtr = sourceDiff + i * source.ObjectSize();
		CConstFloatHandle resultPtr = resultDiff + i * result.ObjectSize();
		CConstIntHandle maxIndicesPtr = maxIndices + i * result.ObjectSize();
		for( int j = 0; j < result.ObjectSize(); ++j ) {
			const int index = maxIndicesPtr.GetValue();
			sourcePtr.SetValueAt( index, sourcePtr.GetValueAt( index ) + resultPtr.GetValue() );
			++maxIndicesPtr;
			++resultPtr;
		}
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

void CCpuMathEngine::BlobMeanPooling( const CMeanPoolingDesc& poolingDesc, const CFloatHandle& sourceData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMeanPoolingDesc& desc = static_cast<const CCommonMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int channels = result.Depth() * result.Channels();
	const int sourceRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;
	CFloatHandleStackVar buffer( mathEngine(), sourceRowSize );
	for( int i = 0; i < source.ObjectCount(); ++i ) {
		CConstFloatHandle sourcePtr = sourceData + i * source.ObjectSize();
		CFloatHandle resultPtr = resultData + i * result.ObjectSize();
		for( int j = 0; j < result.Height(); ++j ) {
			// Calculate the sum of all rows in a strip of the window height
			CConstFloatHandle currentStripStart = sourcePtr + sourceRowSize * desc.StrideHeight * j;
			SumMatrixRows( 1, buffer.GetHandle(), currentStripStart, desc.FilterHeight, sourceRowSize );
			// Calculate the sum in each window
			CConstFloatHandle currentbufferStart = buffer.GetHandle();
			for( int k = 0; k < result.Width(); ++k ) {
				SumMatrixRows( 1, resultPtr, currentbufferStart, desc.FilterWidth, channels );
				currentbufferStart += windowStep;
				resultPtr += channels;
			}
		}
	}

	// Multiply the result by the inverse of the window size
	CFloatHandleStackVar filterSize( mathEngine(), 1 );
	filterSize.SetValue( 1.f / desc.FilterHeight / desc.FilterWidth );
	VectorMultiply( resultData, resultData, result.BlobSize(), filterSize );
}

void CCpuMathEngine::BlobMeanPoolingBackward( const CMeanPoolingDesc& poolingDesc, const CFloatHandle& resultDiff, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMeanPoolingDesc& desc = static_cast<const CCommonMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	VectorFill( sourceDiff, 0, source.BlobSize() );

	const int channels = result.Depth() * result.Channels();
	const int sourceRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;
	CFloatHandleStackVar buffer( mathEngine(), sourceRowSize );
	for( int i = 0; i < result.ObjectCount(); ++i ) {
		CFloatHandle sourcePtr = sourceDiff + i * source.ObjectSize();
		CConstFloatHandle resultPtr = resultDiff + i * result.ObjectSize();
		for( int j = 0; j < result.Height(); ++j ) {
			CFloatHandle currentStripStart = sourcePtr + sourceRowSize * desc.StrideHeight * j;
			CFloatHandle bufferPtr = buffer.GetHandle();
			// Generate a row to be added to the source
			VectorFill( bufferPtr, 0, sourceRowSize );
			for( int k = 0; k < result.Width(); ++k ) {
				AddVectorToMatrixRows( 1, bufferPtr, bufferPtr, desc.FilterWidth, channels, resultPtr );
				bufferPtr += windowStep;
				resultPtr += channels;
			}
			// Add the row to the source
			AddVectorToMatrixRows( 1, currentStripStart, currentStripStart, desc.FilterHeight, sourceRowSize, buffer.GetHandle() );
		}
	}

	// Multiply the diff by the inverse of the window size
	CFloatHandleStackVar filterSizeInv( mathEngine(), 1 );
	filterSizeInv.SetValue( 1.f / desc.FilterHeight / desc.FilterWidth );
	VectorMultiply( sourceDiff, sourceDiff, source.BlobSize(), filterSizeInv );
}

//------------------------------------------------------------------------------------------------------------
// GlobalMaxOverTime pooling

CGlobalMaxOverTimePoolingDesc* CCpuMathEngine::InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result )
{
	CCommonGlobalMaxOverTimePoolingDesc* desc = new CCommonGlobalMaxOverTimePoolingDesc( source, result );
	return desc;
}

void CCpuMathEngine::BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CFloatHandle& sourceData, const CIntHandle* maxIndices, const CFloatHandle& resultData )
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
	const CFloatHandle& sourceDiff, const CIntHandle& maxIndices, const CFloatHandle& resultDiff )
{
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonGlobalMaxOverTimePoolingDesc& desc = static_cast<const CCommonGlobalMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& result = desc.Source;

	const int* maxIndicesPtr = GetRaw( maxIndices );
	const float* sourcePtr = GetRaw( sourceDiff );
	float* resultPtr = GetRaw( resultDiff );

	vectorFill0( resultPtr, result.BlobSize() );

	const int objectSize = result.BatchWidth() * result.ObjectSize();
	for( int i = 0; i < objectSize; ++i ) {
		resultPtr[i + objectSize * *maxIndicesPtr++] = *sourcePtr++;
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
	const CFloatHandle& resultDiff, const CIntHandle& maxIndices, const CFloatHandle& sourceDiff )
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

void CCpuMathEngine::Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& poolingDesc, const CFloatHandle& resultDiff,
	const CIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dMaxPoolingDesc& desc = static_cast<const CCommon3dMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int* indexPtr = GetRaw( maxIndices );
	const float* resultPtr = GetRaw( resultDiff );
	float* sourcePtr = GetRaw( sourceDiff );

	vectorFill0( sourcePtr, source.BlobSize() );

	const int sourceObjectSize = source.ObjectSize();
	const int resultGeomSize = result.GeometricalSize();

	for( int b = 0; b < source.ObjectCount(); ++b ) {
		for( int i = 0; i < resultGeomSize; ++i ) {
			for( int channel = 0; channel < result.Channels(); ++channel ) {
				sourcePtr[*indexPtr++ + channel] += *resultPtr++;
			}
		}
		sourcePtr += sourceObjectSize;
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

//------------------------------------------------------------------------------------------------------------
// MaxOverTime pooling

CMaxOverTimePoolingDesc* CCpuMathEngine::InitMaxOverTimePooling( const CBlobDesc& source,
	int filterLen, int strideLen, const CBlobDesc& result )
{
	int outLen = ( source.BatchLength() - filterLen ) / strideLen + 1;
	ASSERT_EXPR( result.BatchLength() == outLen );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ObjectSize() == source.ObjectSize() );

	CCommonMaxOverTimePoolingDesc* desc = new CCommonMaxOverTimePoolingDesc( source, result, filterLen, strideLen );
	return desc;
}

void CCpuMathEngine::BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& poolingDesc,
	const CFloatHandle& resultDiff, const CIntHandle& maxIndices, const CFloatHandle& sourceDiff )
{
	ASSERT_EXPR( resultDiff.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( sourceDiff.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMaxOverTimePoolingDesc& desc = static_cast<const CCommonMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const float* resultPtr = GetRaw( resultDiff );
	const int* indexPtr = GetRaw( maxIndices );
	float* sourcePtr = GetRaw( sourceDiff );

	vectorFill0( sourcePtr, source.BlobSize() );

	const int seqElemSize = source.ObjectSize() * source.BatchWidth();
	for( int l = 0; l < result.BatchLength(); ++l ) {
		for( int i = 0; i < seqElemSize; ++i ) {
			sourcePtr[*indexPtr++ * seqElemSize + i] += *resultPtr++;
		}
	}
}

} // namespace NeoML
