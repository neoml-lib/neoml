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
	const int inputRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;

	CFloatHandleStackVar buffer( *this, inputRowSize );
	CIntHandleStackVar rowIndexBlob( *this, inputRowSize );
	int* rowIndexBuffer = GetRaw( rowIndexBlob.GetHandle() );
	CIntHandleStackVar columnIndexBlob( *this, channels );
	int* columnIndexBuffer = GetRaw( columnIndexBlob.GetHandle() );

	for( int i = 0; i < source.ObjectCount(); i++ ) {
		const float* inputPtr = sourceData + i * source.ObjectSize();
		float* outputPtr = resultData + i * result.ObjectSize();
		int* maxIndicesPtr = maxIndicesData + i * result.ObjectSize();
		for( int j = 0; j < result.Height(); j++ ) {
			// Calculate maximums in columns over a strip of the window height
			int currentStripRow = desc.StrideHeight * j;
			const float* currentStripStart = inputPtr + currentStripRow * inputRowSize;
			findMaxValueInColumns( GetRaw( buffer.GetHandle() ), rowIndexBuffer, currentStripStart,
				desc.FilterHeight, inputRowSize );
			// Calculate maximum over each window
			const float* currentbufferStart = GetRaw( buffer.GetHandle() );
			int currentWindowColumn = 0;
			for( int k = 0; k < result.Width(); k++ ) {
				findMaxValueInColumns( outputPtr, columnIndexBuffer, currentbufferStart,
					desc.FilterWidth, channels );
				for( int l = 0; l < channels; l++ ) {
					int windowIndex = columnIndexBuffer[l] * channels + l;
					// Calculate the maximum element's index. It is the sum of the current strip offset, 
					// the number of the row in the strip, the window offset and the number of the column in the window
					*maxIndicesPtr = ( currentStripRow + rowIndexBuffer[windowIndex] ) * inputRowSize + currentWindowColumn + windowIndex;
				}
				currentbufferStart += windowStep;
				currentWindowColumn += windowStep;
				outputPtr += channels;
			}
		}
	}
}

static inline void findMaxValueInColumns( float* resultHandle, const float* matrixHandle, int matrixHeight, int matrixWidth)
{
	if( matrixHeight == 1 ) {
		dataCopy( resultHandle, matrixHandle, matrixWidth );
		return;
	}

	const float* nextRow = matrixHandle + matrixWidth;
	vectorEltwiseMax(matrixHandle, nextRow, resultHandle, matrixWidth);

	for( int i = 2; i < matrixHeight; ++i ) {
		nextRow += matrixWidth;
		vectorEltwiseMax(resultHandle, nextRow, resultHandle, matrixWidth);
	}
}

void CCpuMathEngine::blobMaxPoolingWithoutIndices( const CCommonMaxPoolingDesc& desc,
	const float* sourceData, float* resultData )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int channels = result.Depth() * result.Channels();
	const int inputRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;
	CFloatHandleStackVar buffer( *this, inputRowSize );

	float* bufferPtr = GetRaw( buffer.GetHandle() );

	for( int i = 0; i < source.ObjectCount(); i++ ) {
		const float* inputPtr = sourceData + i * source.ObjectSize();
		float* outputPtr = resultData + i * result.ObjectSize();
		for( int j = 0; j < result.Height(); j++ ) {
			// Calculate maximums in columns over a strip of the window height
			const float* currentStripStart = inputPtr + inputRowSize * desc.StrideHeight * j;
			NeoML::findMaxValueInColumns( bufferPtr, currentStripStart,
				desc.FilterHeight, inputRowSize );
			// Calculate maximum over the window
			const float* currentbufferStart = bufferPtr;
			for( int k = 0; k < result.Width(); k++ ) {
				NeoML::findMaxValueInColumns( outputPtr, currentbufferStart, desc.FilterWidth, channels );
				currentbufferStart += windowStep;
				outputPtr += channels;
			}
		}
	}
}

void CCpuMathEngine::BlobMaxPooling( const CMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonMaxPoolingDesc& desc = static_cast<const CCommonMaxPoolingDesc&>( poolingDesc );

	if( maxIndicesData != 0 ) {
		blobMaxPoolingWithIndices( desc, GetRaw( sourceData ), GetRaw( *maxIndicesData ), GetRaw( resultData ) );
	} else {
		blobMaxPoolingWithoutIndices( desc, GetRaw( sourceData ), GetRaw( resultData ) );
	}
}

void CCpuMathEngine::BlobMaxPoolingBackward( const CMaxPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );

	const CCommonMaxPoolingDesc& desc = static_cast<const CCommonMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	VectorFill( inputDiffData, 0, inputDiff.BlobSize() );

	for( int i = 0; i < outputDiff.ObjectCount(); i++ ) {
		CFloatHandle inputPtr = inputDiffData + i * inputDiff.ObjectSize();
		CConstFloatHandle outputPtr = outputDiffData + i * outputDiff.ObjectSize();
		CConstIntHandle maxIndicesPtr = maxIndicesData + i * outputDiff.ObjectSize();
		for( int j = 0; j < outputDiff.ObjectSize(); j++ ) {
			int index = maxIndicesPtr.GetValue();
			maxIndicesPtr++;
			inputPtr.SetValueAt( index, inputPtr.GetValueAt( index ) + outputPtr.GetValue() );
			outputPtr++;
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

	const CCommonMeanPoolingDesc& desc = static_cast<const CCommonMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int channels = result.Depth() * result.Channels();
	const int inputRowSize = source.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;
	CFloatHandleStackVar buffer( mathEngine(), inputRowSize );
	for( int i = 0; i < source.ObjectCount(); i++ ) {
		CConstFloatHandle inputPtr = sourceData + i * source.ObjectSize();
		CFloatHandle outputPtr = resultData + i * result.ObjectSize();
		for( int j = 0; j < result.Height(); j++ ) {
			// Calculate the sum of all rows in a strip of the window height
			CConstFloatHandle currentStripStart = inputPtr + inputRowSize * desc.StrideHeight * j;
			SumMatrixRows( 1, buffer.GetHandle(), currentStripStart, desc.FilterHeight, inputRowSize );
			// Calculate the sum in each window
			CConstFloatHandle currentbufferStart = buffer.GetHandle();
			for( int k = 0; k < result.Width(); k++ ) {
				SumMatrixRows( 1, outputPtr, currentbufferStart, desc.FilterWidth, channels );
				currentbufferStart += windowStep;
				outputPtr += channels;
			}
		}
	}

	// Multiply the output by the inverse of the window size
	CFloatHandleStackVar filterSize( mathEngine(), 1 );
	filterSize.SetValue( 1.f / desc.FilterHeight / desc.FilterWidth );
	VectorMultiply( resultData, resultData, result.BlobSize(), filterSize );
}

void CCpuMathEngine::BlobMeanPoolingBackward( const CMeanPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCommonMeanPoolingDesc& desc = static_cast<const CCommonMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	VectorFill( inputDiffData, 0, inputDiff.BlobSize() );

	const int channels = outputDiff.Depth() * outputDiff.Channels();
	const int inputRowSize = inputDiff.Width() * channels;
	const int windowStep = desc.StrideWidth * channels;
	CFloatHandleStackVar inputBuffer( mathEngine(), inputRowSize );
	for( int i = 0; i < outputDiff.ObjectCount(); i++ ) {
		CFloatHandle inputPtr = inputDiffData + i * inputDiff.ObjectSize();
		CConstFloatHandle outputPtr = outputDiffData + i * outputDiff.ObjectSize();
		for( int j = 0; j < outputDiff.Height(); j++ ) {
			CFloatHandle currentStripStart = inputPtr + inputRowSize * desc.StrideHeight * j;
			CFloatHandle inputBufferPtr = inputBuffer.GetHandle();
			// Generate a row to be added to the input
			VectorFill( inputBufferPtr, 0, inputRowSize );
			for( int k = 0; k < outputDiff.Width(); k++ ) {
				AddVectorToMatrixRows( 1, inputBufferPtr, inputBufferPtr, desc.FilterWidth, channels, outputPtr );
				inputBufferPtr += windowStep;
				outputPtr += channels;
			}
			// Add the row to the input
			AddVectorToMatrixRows( 1, currentStripStart, currentStripStart, desc.FilterHeight, inputRowSize, inputBuffer.GetHandle() );
		}
	}

	// Multiply the diff by the inverse of the window size
	CFloatHandleStackVar filterSizeInv( mathEngine(), 1 );
	filterSizeInv.SetValue( 1.f / desc.FilterHeight / desc.FilterWidth );
	VectorMultiply( inputDiffData, inputDiffData, inputDiff.BlobSize(), filterSizeInv );
}

//------------------------------------------------------------------------------------------------------------
// GlobalMaxOverTime pooling

CGlobalMaxOverTimePoolingDesc* CCpuMathEngine::InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result )
{
	CCommonGlobalMaxOverTimePoolingDesc* desc = new CCommonGlobalMaxOverTimePoolingDesc( source, result );
	return desc;
}

void CCpuMathEngine::BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CFloatHandle& sourceData, const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonGlobalMaxOverTimePoolingDesc& desc = static_cast<const CCommonGlobalMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;

	if( maxIndicesData != 0 ) {
		findMaxValueInColumns( GetRaw( resultData ), GetRaw( *maxIndicesData ), GetRaw( sourceData ), source.BatchLength(), source.BatchWidth() * source.ObjectSize() );
	} else {
		findMaxValueInColumns( GetRaw( resultData ), GetRaw( sourceData ), source.BatchLength(), source.BatchWidth() * source.ObjectSize() );
	}
}

void CCpuMathEngine::BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CFloatHandle& sourceData, const CIntHandle& maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonGlobalMaxOverTimePoolingDesc& desc = static_cast<const CCommonGlobalMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& result = desc.Source;

	int objectSize = result.BatchWidth() * result.ObjectSize();

	const int* maxIndicesPtr = GetRaw( maxIndicesData );
	const float* outputPtr = GetRaw( sourceData );
	float* inputPtr = GetRaw( resultData );

	VectorFill( resultData, 0, result.BlobSize() );

	for( int i = 0; i < objectSize; ++i ) {
		inputPtr[i + objectSize * *maxIndicesPtr++] = *outputPtr++;
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
	const CFloatHandle& outputDiffData, const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCommonGlobalMaxPoolingDesc& desc = static_cast<const CCommonGlobalMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	VectorFill( inputDiffData, 0, inputDiff.BlobSize() );

	int poolSize = inputDiff.Height() * inputDiff.Width() * inputDiff.Depth();
	int maxCount = outputDiff.Height() * outputDiff.Width() * outputDiff.Depth();

	const float* outputDiffPtr = GetRaw( outputDiffData );
	const int* maxIndexPtr = GetRaw( maxIndicesData );
	float* inputDiffPtr = GetRaw( inputDiffData );
	int objectSize = poolSize * inputDiff.Channels();

	for( int b = 0; b < inputDiff.ObjectCount(); ++b ) {
		for( int i = 0; i < maxCount; ++i ) {
			float* inputDiffChannelData = inputDiffPtr;
			for( int c = 0; c < outputDiff.Channels(); ++c ) {
				int index = *maxIndexPtr++;
				if( index >= 0 ) {
					assert( index < poolSize );
					inputDiffChannelData[index * inputDiff.Channels()] = *outputDiffPtr;
				}
				++outputDiffPtr;
				++inputDiffChannelData;
			}
		}
		inputDiffPtr += objectSize;
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

void CCpuMathEngine::Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& poolingDesc, const CFloatHandle& outputDiffData,
	const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCommon3dMaxPoolingDesc& desc = static_cast<const CCommon3dMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	VectorFill( inputDiffData, 0, inputDiff.BlobSize() );
	const float* outputDiffPtr = GetRaw( outputDiffData );
	float* inputDiffPtr = GetRaw( inputDiffData );
	const int* indexPtr = GetRaw( maxIndicesData );

	int inputObjectSize = inputDiff.ObjectSize();
	int outputGeomSize = outputDiff.GeometricalSize();

	for( int b = 0; b < inputDiff.ObjectCount(); ++b ) {
		for( int i = 0; i < outputGeomSize; ++i ) {
			for( int channel = 0; channel < outputDiff.Channels(); ++channel ) {
				inputDiffPtr[*indexPtr++ + channel] += *outputDiffPtr++;
			}
		}
		inputDiffPtr += inputObjectSize;
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
	const CFloatHandle& outputDiffData, const CIntHandle& maxIndicesData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( inputDiffData.GetMathEngine() == this );

	const CCommonMaxOverTimePoolingDesc& desc = static_cast<const CCommonMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	int seqElemSize = inputDiff.ObjectSize() * inputDiff.BatchWidth();

	VectorFill( inputDiffData, 0, inputDiff.BlobSize() );

	const float* outputDiffDataPtr = GetRaw( outputDiffData );
	const int* indexDataPtr = GetRaw( maxIndicesData );
	float* inputDiffPtr = GetRaw( inputDiffData );

	for( int l = 0; l < outputDiff.BatchLength(); ++l ) {
		for( int i = 0; i < seqElemSize; ++i ) {
			inputDiffPtr[*indexDataPtr++ * seqElemSize + i] += *outputDiffDataPtr++;
		}
	}
}

} // namespace NeoML
