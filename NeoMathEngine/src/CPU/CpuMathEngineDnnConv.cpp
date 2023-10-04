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
#include <float.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEnginePrivate.h>
#include <CpuMathEngineDnnConv.h>
#include <NeoMathEngine/SimdMathEngine.h>
#include <CpuMathEngineDnnChannelwiseConv.h>

namespace NeoML {

// Returns the descriptor of the "flattened" blob with depth == 1 and channels = desc.depth * desc.channels
static inline CBlobDesc flatten( const CBlobDesc& desc )
{
	CBlobDesc res = desc;
	res.SetDimSize( BD_Depth, 1 );
	res.SetDimSize( BD_Channels, desc.Channels() * desc.Depth() );
	return res;
}

//------------------------------------------------------------------------------------------------------------

CConvolutionDesc* CCpuMathEngine::InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter, const CBlobDesc& result )
{
	ASSERT_EXPR( strideHeight > 0 );
	ASSERT_EXPR( strideWidth > 0 );
	ASSERT_EXPR( paddingHeight >= 0 );
	ASSERT_EXPR( paddingWidth >= 0 );
	ASSERT_EXPR( dilationHeight > 0 );
	ASSERT_EXPR( dilationWidth > 0 );
	ASSERT_EXPR( source.Channels() == filter.Channels() );
	ASSERT_EXPR( source.Depth() == filter.Depth() );
	ASSERT_EXPR( filter.Height() <= source.Height() + 2 * paddingHeight );
	ASSERT_EXPR( filter.Width() <= source.Width() + 2 * paddingWidth );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( result.BatchLength() == source.BatchLength() );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.Height() == 1 + ( source.Height() -
		( filter.Height() - 1 ) * dilationHeight + 2 * paddingHeight - 1 ) / strideHeight );
	ASSERT_EXPR( result.Width() == 1 + ( source.Width() -
		( filter.Width() - 1 ) * dilationWidth + 2 * paddingWidth - 1 ) / strideWidth );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( result.Depth() == 1 );

	CCpuConvolutionDesc* desc = new CCpuConvolutionDesc( *this, source, result, filter,
		paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth );
	if( simdMathEngine != nullptr ) {
		desc->SimdConvolutionDesc.reset( simdMathEngine->InitBlobConvolution( source, paddingHeight, paddingWidth,
			strideHeight, strideWidth, dilationHeight, dilationWidth, filter, result ) );
	}
	return desc;
}

// Creates a temporary blob with reordered input data that will be used to calculate convolution
// This method allows for nonzero dilation
void CCpuMathEngine::createDilationTemporaryBlob( const CCpuConvolutionDesc& desc, const float* inputData, int inputBatch,
	int outputColumnStart, int outputColumnCount, float* temporaryBlob )
{
	const CBlobDesc& inputBlob = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& output = desc.Result;

	const float* inputBlobPtr = inputData + inputBlob.ObjectSize() * inputBatch;
	float* tempBlobPtr = temporaryBlob;

	const int vectorSize = inputBlob.Depth() * inputBlob.Channels();
	if( desc.PaddingHeight > 0 || desc.PaddingWidth > 0 ) {
		// Padding is emulated by first filling the tempBlob by the padding value
		// and then writing over the required positions with the input data
		vectorFill0( tempBlobPtr, filter.Height() * filter.Width() * vectorSize * output.Height() * outputColumnCount );
	}

	for( int outputColumn = outputColumnStart; outputColumn < outputColumnStart + outputColumnCount; ++outputColumn ) {
		const int leftPos = -desc.PaddingWidth + outputColumn * desc.StrideWidth;
		if( leftPos + ( filter.Width() - 1 ) * desc.DilationWidth < 0 || leftPos >= inputBlob.Width() ) {
			// The current column is all padding
			continue;
		}

		for( int outputRow = 0; outputRow < output.Height(); ++outputRow ) {
			const int topPos = -desc.PaddingHeight + outputRow * desc.StrideHeight;
			if( topPos + ( filter.Height() - 1 ) * desc.DilationHeight < 0 || topPos >= inputBlob.Height() ) {
				// The current row is all padding
				continue;
			}

			float* tempRow = tempBlobPtr + ( ( outputColumn - outputColumnStart ) * output.Height() + outputRow )
				* filter.Height() * filter.Width() * vectorSize;

			for( int filterRow = 0; filterRow < filter.Height(); ++filterRow ) {
				const int verticalPos = topPos + desc.DilationHeight * filterRow;
				if( verticalPos < 0 || verticalPos >= inputBlob.Height() ) {
					// The current filter row only intersects with padding
					continue;
				}

				for( int filterColumn = 0; filterColumn < filter.Width(); ++filterColumn ) {
					const int horizontalPos = leftPos + desc.DilationWidth * filterColumn;
					if( horizontalPos < 0 || horizontalPos >= inputBlob.Width() ) {
						// The current element is padding
						continue;
					}

					const float* inputVector = inputBlobPtr + ( inputBlob.Width() * verticalPos + horizontalPos ) * vectorSize;
					float* tempVector = tempRow + ( filterRow * filter.Width() + filterColumn ) * vectorSize;
					dataCopy( tempVector, inputVector, vectorSize );
				}
			}
		}
	}
}

template<class TConvolutionDesc>
void CCpuMathEngine::createTemporaryBlob( const TConvolutionDesc& desc, const float* sourceData,
	int sourceBatch, int resultRowStart, int resultRowCount, float* tempBlob )
{
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	const int sourceChannelsCount = source.Depth() * source.Channels();
	const int windowRowSize = filter.Width() * sourceChannelsCount;
	const int sourceRowSize = source.Width() * source.Depth() * source.Channels();
	const int resultRowEnd = resultRowStart + resultRowCount;

	const float* sourcePtr = sourceData + source.ObjectSize() * sourceBatch;
	float* tempBlobPtr = tempBlob;

	if( desc.PaddingHeight > 0 || desc.PaddingWidth > 0 ) {
		// Padding is emulated by first filling the tempBlob by the padding value
		// and then writing over the required positions with the source data
		NeoML::vectorFill0( tempBlobPtr, filter.Height() * filter.Width() * source.Depth() * source.Channels() *
			result.Height() * resultRowCount );
	}

	// The source blob height - the number of filter windows that fit horizontally
	for( int j = resultRowStart; j < resultRowEnd; ++j ) {
		// Skip the top of the first window (padding)
		tempBlobPtr += desc.PaddingHeight * windowRowSize;
		// Calculate padding on the left and right
		const int paddingLeft = std::max( 0, ( desc.PaddingWidth - j * desc.StrideWidth ) * sourceChannelsCount );
		const int paddingRight = std::max( 0, ( filter.Width() - desc.PaddingWidth + j * desc.StrideWidth - source.Width() ) * sourceChannelsCount );
		// Copy the vertical strip of windows in a cycle. The start of the top window of the stripe.
		const float* currentWindowStart = sourcePtr + ( j * desc.StrideWidth - desc.PaddingWidth ) * sourceChannelsCount;
		// Copy the first window
		for( int k = 0; k < filter.Height() - desc.PaddingHeight; ++k ) {
			if( k < source.Height() ) {
				dataCopy( tempBlobPtr + paddingLeft, currentWindowStart + paddingLeft,
					( windowRowSize - paddingLeft - paddingRight ) );
				currentWindowStart += sourceRowSize;
			}
			tempBlobPtr += windowRowSize;
		}
		// The source blob width - the number of filter windows that fit vertically
		for( int k = 1; k < result.Height(); ++k ) {
			if( filter.Height() >= desc.StrideHeight ) {
				// The size of intersection for two vertically adjacent windows
				const int windowsIntersection = windowRowSize * ( filter.Height() - desc.StrideHeight );

				// If stride is smaller than the filter size, copy the adjacent filters intersection
				dataCopy( tempBlobPtr, tempBlobPtr - windowsIntersection, windowsIntersection );
				tempBlobPtr += windowsIntersection;
			} else {
				// If stride is larger than filter size, skip the rows that will not be in the next filter
				currentWindowStart += ( desc.StrideHeight - filter.Height() ) * sourceRowSize;
			}

			// The lower filter boundary - ( top padding + the source image height )
			// If this number is greater than 0, the filter intersects with the bottom padding
			int paddingBottom = std::max( 0, filter.Height() + k * desc.StrideHeight - desc.PaddingHeight - source.Height() );
			if( paddingBottom > std::min( desc.StrideHeight, filter.Height() ) ) {
				// The whole area to be copied next belongs to the bottom padding
				paddingBottom = std::min( desc.StrideHeight, filter.Height() );
			}
			// The paddingBottom now has only the bottom padding rows that are in the filter area

			// Copy the rows that are in filter area 
			// If strideHeight <= filterHeight the intersection has already been copied above
			// and we only need to copy additional strideHeight lower rows
			// If strideHeight > filterHeight the rows to be ignored have been skipped already 
			// and we need to copy filterHeight lower rows
			// The intersection with the bottom padding does not need copying 
			// because we've already filled temporaryBlob with the padding value
			for( int l = 0; l < std::min( desc.StrideHeight, filter.Height() ) - paddingBottom; ++l ) {
				dataCopy( tempBlobPtr + paddingLeft, currentWindowStart + paddingLeft,
					( windowRowSize - paddingLeft - paddingRight ) );
				currentWindowStart += sourceRowSize;
				tempBlobPtr += windowRowSize;
			}

			// temporaryBlob already filled with the padding value, so we only need to 
			// offset the pointer by the number of bottom padding elements that fit into the filter area
			tempBlobPtr += paddingBottom * windowRowSize;
		}
	}
}

void CCpuMathEngine::transposeResult( const CCpuConvolutionDesc& desc, const float* outputTransposedData,
	int batch, int resultStart, int resultCount, float* resultData )
{
	const CBlobDesc& result = desc.Result;

	const int resultPixelSize = result.Depth() * result.Channels();
	const int resultRowSize = resultPixelSize * result.Width();

	const float* inPtr = outputTransposedData;
	float* resultPtr = resultData + batch * result.ObjectSize() + resultStart * resultPixelSize;
	for( int i = 0; i < resultCount; ++i ) {
		float* outRowPtr = resultPtr;
		for( int j = 0; j < result.Height(); ++j ) {
			dataCopy( outRowPtr, inPtr, resultPixelSize );
			outRowPtr += resultRowSize;
			inPtr += resultPixelSize;
		}
		resultPtr += resultPixelSize;
	}
}

static inline void calcPaddings( const CCpuConvolutionDesc& desc, int width, int& startPaddingSize, int& endPaddingSize )
{
	const int startPos = -desc.PaddingWidth + width * desc.StrideWidth;
	startPaddingSize = std::min( desc.Filter.Width(), ( startPos < 0 ) ? 1 + ( -startPos - 1 ) / desc.DilationWidth : 0 );

	const int endPos = -desc.PaddingWidth + width * desc.StrideWidth + desc.DilationWidth * ( desc.Filter.Width() - 1 );
	endPaddingSize = ( desc.Source.Width() > endPos ) ? 0 :
		std::min( ( endPos - desc.Source.Width() ) / desc.DilationWidth + 1, desc.Filter.Width() );
}

void CCpuMathEngine::fillTempData( const float* sourceData, float* tempData, const CCpuConvolutionDesc& desc, int start, int count )
{
	const int channelsCount = desc.Filter.Depth() * desc.Filter.Channels();
	const int filterLineSize = desc.Filter.Width() * channelsCount;
	const int resultG = desc.Result.Width() * desc.Result.Height();

	for( int index = start; index < count + start; ++index ) {
		const int batch = index / resultG;
		const int height = ( index - batch * resultG ) / desc.Result.Width();
		const int width = ( index - batch * resultG ) % desc.Result.Width();

		int startPaddingSize = 0;
		int endPaddingSize = 0;
		calcPaddings( desc, width, startPaddingSize, endPaddingSize );
		const int dataSize = desc.Filter.Width() - startPaddingSize - endPaddingSize;

		const int sourceHeight = -desc.PaddingHeight + height * desc.StrideHeight;
		const int sourceWidth = -desc.PaddingWidth + width * desc.StrideWidth + startPaddingSize * desc.DilationWidth;

		const float* sourceDataPtr = sourceData + batch * desc.Source.ObjectSize() + ( sourceHeight * desc.Source.Width() + sourceWidth ) * channelsCount;
		float* tempStartPaddingPtr = tempData + ( index - start ) * desc.Filter.ObjectSize();
		float* tempDataPtr = tempStartPaddingPtr + startPaddingSize * channelsCount;
		float* tempEndPaddingPtr = tempDataPtr + dataSize * channelsCount;

		for( int h = 0; h < desc.Filter.Height(); ++h ) {
			if( 0 <= sourceHeight + h * desc.DilationHeight && sourceHeight + h * desc.DilationHeight < desc.Source.Height() ) {
				if( startPaddingSize > 0 ) {
					NeoML::vectorFill0( tempStartPaddingPtr, startPaddingSize * channelsCount );
				}

				if( desc.DilationWidth == 1 ) {
					if( dataSize > 0 ) {
						dataCopy( tempDataPtr, sourceDataPtr, dataSize * channelsCount );
					}
				} else {
					for( int i = 0; i < dataSize; ++i ) {
						dataCopy( tempDataPtr + i * channelsCount, sourceDataPtr + i * desc.DilationWidth * channelsCount, channelsCount );
					}
				}

				if( endPaddingSize > 0 ) {
					NeoML::vectorFill0( tempEndPaddingPtr, endPaddingSize * channelsCount );
				}
			} else {
				NeoML::vectorFill0( tempStartPaddingPtr, filterLineSize );
			}

			tempStartPaddingPtr += filterLineSize;
			tempDataPtr += filterLineSize;
			tempEndPaddingPtr += filterLineSize;
			sourceDataPtr += desc.DilationHeight * desc.Source.Width() * channelsCount;
		}
	}
}

void CCpuMathEngine::blobConvolutionForwardAlgo0( const CCpuConvolutionDesc& desc, const float* sourceData,
	const float* filterData, const CConstFloatHandle* freeTermData, float* resultData )
{
	const int filterObjectSize = desc.Filter.ObjectSize();
	const int filterObjectCount = desc.Filter.ObjectCount();

	const int resultItemCount = desc.Result.ObjectCount() * desc.Result.Width() * desc.Result.Height();
	const int cacheItemCount = std::max( 1, std::min( ceilTo( BlobConvolutionCacheSize / filterObjectSize, 16 ), resultItemCount ) );
	const int tempDataSize = cacheItemCount * filterObjectSize;

	CFloatHandleStackVar tempData( mathEngine(), tempDataSize );
	float* const tempDataPtr = GetRaw( tempData.GetHandle() );

	const int firstWidth = filterObjectSize;
	const int secondHeight = filterObjectCount;
	const int secondWidth = firstWidth;
	const int resultWidth = secondHeight;

	for( int index = 0; index < resultItemCount; ) {
		const int size = std::min( resultItemCount - index, cacheItemCount );

		const int firstHeight = size;
		auto mulDesc = desc.SmallMatricesMulDescsHeightArrays[CCpuConvolutionDesc::TSMMDA_Forward].Get( firstHeight,
			firstHeight, firstWidth, secondWidth, resultWidth );

		fillTempData( sourceData, tempDataPtr, desc, index, size );

		float* const resultDataPtr = resultData + index * resultWidth;

		multiplyMatrixByTransposedMatrix( /*first*/tempDataPtr, size, firstWidth, firstWidth,
			/*second*/filterData, secondHeight, secondWidth, /*result*/resultDataPtr, resultWidth, mulDesc );

		if( freeTermData != nullptr ) {
			addVectorToMatrixRows( resultDataPtr, resultDataPtr, size,
				resultWidth, resultWidth, resultWidth, GetRaw( *freeTermData ) );
		}
		index += size;
	}
}

void CCpuMathEngine::blobConvolutionForwardAlgo1( const CCpuConvolutionDesc& desc, const float* sourceData,
	const float* filterData, const CConstFloatHandle* freeTermData, float* resultData )
{
	const float* freeTermDataRaw = ( freeTermData == nullptr ) ? nullptr : GetRaw( *freeTermData );

	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	const int outputChannels = result.Depth() * result.Channels();
	const int outputTransposedDataRowSize = result.Height() * outputChannels;
	const int outputTransposedDataObjectSize = result.Width() * outputTransposedDataRowSize;
	const int tempBlobDataRowSize = result.Height() * filter.Height() * filter.Width() * source.Depth() * source.Channels();
	const int tempBlobDataObjectSize = result.Width() * tempBlobDataRowSize;

	CFloatHandleStackVar buffer( mathEngine(), outputTransposedDataObjectSize + tempBlobDataObjectSize );
	float* const outputTransposedPtr = GetRaw( buffer.GetHandle() );
	float* const tempBlobPtr = outputTransposedPtr + outputTransposedDataObjectSize;

	const int resultCount = result.Width();
	const int batchCount = source.ObjectCount();

	const int firstHeight = result.Height() * resultCount;
	const int firstWidth = filter.ObjectSize();
	const int secondHeight = filter.BatchWidth();
	const int secondWidth = filter.ObjectSize();
	const int resultWidth = secondHeight;
	auto mulDesc = desc.SmallMatricesMulDescsHeightArrays[CCpuConvolutionDesc::TSMMDA_Forward].Get( firstHeight,
		firstHeight, firstWidth, secondWidth, resultWidth, /*resultAdd*/( freeTermData != nullptr ) );

	for( int batch = 0; batch < batchCount; ++batch ) {
		// Fill the temporary matrix
		if( desc.DilationHeight > 1 || desc.DilationWidth > 1 ) {
			createDilationTemporaryBlob( desc, sourceData, batch, /*resultStart*/0, resultCount, tempBlobPtr );
		} else {
			createTemporaryBlob( desc, sourceData, batch, /*resultStart*/0, resultCount, tempBlobPtr );
		}
		// Apply the filter to the temporary matrix
		if( freeTermData != nullptr ) {
			setVectorToMatrixRows( outputTransposedPtr, firstHeight, outputChannels, freeTermDataRaw );

			multiplyMatrixByTransposedMatrixAndAdd( /*first*/tempBlobPtr, firstHeight, firstWidth, firstWidth,
				/*second*/filterData, secondHeight, secondWidth, /*result*/outputTransposedPtr, resultWidth, mulDesc );
		} else {
			multiplyMatrixByTransposedMatrix( /*first*/tempBlobPtr, firstHeight, firstWidth, firstWidth,
				/*second*/filterData, secondHeight, secondWidth, /*result*/outputTransposedPtr, resultWidth, mulDesc );
		}
		// Transpose the result
		transposeResult( desc, outputTransposedPtr, batch, /*resultStart*/0, resultCount, resultData );
	}
}

void CCpuMathEngine::BlobConvolution( const CConvolutionDesc& convDesc, const CConstFloatHandle& source,
	const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result )
{
	const CCpuConvolutionDesc& desc = static_cast<const CCpuConvolutionDesc&>( convDesc );
	CCpuExecutionScope scope;

	const float* sourceRaw = GetRaw( source );
	const float* filterRaw = GetRaw( filter );
	const float* freeTermRaw = ( freeTerm != nullptr ) ? GetRaw( *freeTerm ) : nullptr;
	float* resultRaw = GetRaw( result );

	if( desc.SimdConvolutionDesc != nullptr ) {
		simdMathEngine->BlobConvolution( *desc.SimdConvolutionDesc, sourceRaw, filterRaw, freeTermRaw, resultRaw );
		return;
	}

	switch( desc.ForwardAlgo ) {
		case CA_1:
		case CA_2:
		{
			const int64_t algo1DataSize = static_cast<int64_t>( desc.Result.Width() )
				* desc.Result.Height() * desc.Filter.ObjectSize() + desc.Result.ObjectSize();

			if( algo1DataSize <= BlobConvolutionCacheSize ) {
				blobConvolutionForwardAlgo1( desc, sourceRaw, filterRaw, freeTerm, resultRaw );
			} else {
				blobConvolutionForwardAlgo0( desc, sourceRaw, filterRaw, freeTerm, resultRaw );
			}
			break;
		}
		case CA_1x1:
		{
			const bool needsFlatten = ( desc.Source.Depth() != 1 );
			blob3dConvolution1x1x1( needsFlatten ? flatten( desc.Source ) : desc.Source, desc.Result,
				desc.StrideHeight, desc.StrideWidth, /*StrideDepth*/1, sourceRaw, filterRaw, freeTermRaw, resultRaw, nullptr );
			break;
		}
		default:
			ASSERT_EXPR( false );
	}
}

//------------------------------------------------------------------------------------------------------------

// This ring-buffer wraps the temporal memory.
// It stores the Source by Filter matrix-multiplication rows.
// It reduces temporal memory to size enough for al least exact 1 Result's matrix row calculation.
// Filter.Height and StrideHeight defines the number of items in the ring-buffer.
//
// NOTE: For neighbor Result's row may need the same buffer's items, except some first (no more need), 
//       some next items should be prepared freshly. This is reason why the ring-buffer is used:
//       It avoids to move underlying data (the begining pointer moves only).
// NOTE: For dilation != 1 it needs more temporal memory.
//       In some cases it needs the entire Source by Filter matrix-multiplication.
class CRingBuffer final {
public:
	explicit CRingBuffer( float *raw, int numItems, int itemSize ) :
		raw( raw ),
		itemSize( itemSize ),
		capacity( numItems )
	{}

	// Move the begining pointer of the ring-buffer in number of no-need rows.
	void PopFront( int numItemsToPop );
	// Get number of existing valid rows in the ring-buffer
	int GetNumExistItems() const { return size; }
	// Get real-optimal number of rows is needed to add to the ring-buffer for current calculations.
	int GetNumItemsToAdd( int maxNeedItems ) const;
	// Get pointer to positinon in the ring-buffer at adding items by external function (matrix-multiplication).
	// Account the number of new added items in the ring-buffer.
	float* GetBufPtrForAddingItems( int numItemsToAdd );
	// Get the offset of data in the ring-buffer corresponding to Source matrix row
	int GetSourceRowOffset( int sourceRow ) const { return ( sourceRow + startPos ) % capacity; }
	// Get read-only data of the ring-buffer corresponding to offset
	const float* GetRaw( int offset ) const { return raw + offset; }

private:
	void popFront( int numItems ) { startPos = ( size = std::max( 0, size - numItems ) ) ? ( ( startPos + numItems ) % capacity ) : 0; }

	float* const raw;          // Pointer to temporal memory beginning
	const int itemSize;        // Size of one matrix row in bytes
	const int capacity;        // Number of matrix rows in the ring-buffer
	int size = 0;              // Number of non-empty rows in the ring-buffer
	int startPos = 0;          // Offset in ring-buffer of data beginning
	int endPos = 0;            // Offset in ring-buffer of data ending, position to append new items
};

void CRingBuffer::PopFront( int numItemsToPop )
{
	ASSERT_EXPR( numItemsToPop > 0 );
	popFront( numItemsToPop );
}

int CRingBuffer::GetNumItemsToAdd( int maxNeedItems ) const
{
	const auto numItemsToAdd = ( size == 0 ) ? maxNeedItems : ( ( endPos >= startPos ) ? ( capacity - endPos ) : ( startPos - endPos ) );
	ASSERT_EXPR( numItemsToAdd > 0 && numItemsToAdd <= ( capacity - size ) );
	return std::min( maxNeedItems, numItemsToAdd );
}

float* CRingBuffer::GetBufPtrForAddingItems( int numItemsToAdd )
{
	float* const ptr = raw + ( size ? endPos : 0 ) * itemSize; // If there are no items, the buffer starts from scratch
	endPos = size ? ( ( endPos + numItemsToAdd ) % capacity ) : 0;
	size += numItemsToAdd;
	ASSERT_EXPR( size > 0 && size <= capacity );
	return ptr;
}

//------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::blobConvolutionBackwardAlgo1( const CCpuConvolutionDesc& desc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTerm, const CFloatHandle& resultData )
{
	const CBlobDesc& source = desc.Result;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Source;

	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( filter.BatchLength() == 1 );

	const int filterObjectSize = filter.ObjectSize();
	const int sourceChannelsCount = filter.BatchWidth();
	const int sourceRowSize = source.Width() * sourceChannelsCount;
	const int resultItemSize = result.Depth() * result.Channels();
	const int resultRowSize = result.Width() * resultItemSize;
	const int resultRowsCount = result.ObjectCount() * result.Height();

	// dilation == 1
	const int filterChannels = filter.Depth() * filter.Channels();
	const int filterRowSize = filter.Width() * filterChannels;
	const int filterColMin = -desc.PaddingWidth;
	const int filterColMax = result.Width() + desc.PaddingWidth - filter.Width();
	const int sourceRowStartShift = desc.PaddingHeight - filter.Height() + desc.StrideHeight;
	const int filterRowStartShift = filter.Height() - result.Height() - desc.PaddingHeight;

	// dilation > 1
	const int totalFilterHeight = ( filter.Height() - 1 ) * desc.DilationHeight + 1;
	const int totalFilterWidth = ( filter.Width() - 1 ) * desc.DilationWidth + 1;
	const int leftPosMin = -desc.PaddingWidth;
	const int leftPosMax = result.Width() + desc.PaddingWidth - totalFilterWidth;
	const int topPosMaxValShift = result.Height() + desc.PaddingHeight - totalFilterHeight;
	const int filterRowShift = filter.Width() * resultItemSize;

	// Raw data
	const float* const filterRaw = GetRaw( filterData );
	const float* const sourceRaw = GetRaw( sourceData );
	float* const resultRaw = GetRaw( resultData );

	const int ringBufRowSize = source.Width() * filterObjectSize;
	// Minimum possible size of the ring-buffer
	const int ringBufNeedNumRows = std::max( 1, ( filter.Height() * desc.DilationHeight + desc.StrideHeight ) / desc.StrideHeight );

	constexpr int maxCacheFitSize = 16 * 1024; // Empirical evaluated constant
	// Recommended size of the ring-buffer to cache-fit
	const int ringBufRealNumRows = std::max( ringBufNeedNumRows, std::min( source.Height(), maxCacheFitSize / ringBufRowSize ) );

	const int firstWidth = sourceChannelsCount;
	const int secondWidth = filterObjectSize;
	const int resultWidth = secondWidth;
	const auto& mulDescs = desc.SmallMatricesMulDescsHeightArrays[CCpuConvolutionDesc::TSMMDA_Backward];

	auto updateRingBuf = [&]( CRingBuffer& ringBuf, int batch, int sourceRowStart, int& prevBatchedSourceRowStart )
	{
		const int batchedSourceRowStart = batch * source.Height() + sourceRowStart;
		if( prevBatchedSourceRowStart < batchedSourceRowStart ) {
			const int sourceNeedHeight = std::min( source.Height() - sourceRowStart, ringBufNeedNumRows ); //min number of need rows
			// Move the begining pointer of the ring-buffer next to number of no-need rows
			ringBuf.PopFront( /*numItemsToPop*/ batchedSourceRowStart - prevBatchedSourceRowStart );
			const int sourceRowExistShift = ringBuf.GetNumExistItems();
			prevBatchedSourceRowStart = sourceNeedHeight ? batchedSourceRowStart : prevBatchedSourceRowStart;
			if( sourceRowExistShift < sourceNeedHeight ) { // Prepare only new last rows
				const int sourceRealHeight = ringBuf.GetNumItemsToAdd( std::min( source.Height() - sourceRowStart - sourceRowExistShift, ringBufRealNumRows ) );
				const int firstHeight = sourceRealHeight * source.Width();
				multiplyMatrixByMatrix(
					/*first*/( sourceRaw + ( batchedSourceRowStart + sourceRowExistShift ) * sourceRowSize ), firstHeight, firstWidth, firstWidth,
					/*second*/filterRaw, secondWidth, secondWidth, /*result*/ringBuf.GetBufPtrForAddingItems( sourceRealHeight ), resultWidth,
					mulDescs.Get( firstHeight, firstHeight, firstWidth, secondWidth, resultWidth, /*resultAdd*/false, /*trans1*/false, /*trans2*/false ) );

				// Make twice because of Ring-buffer, from end to start
				if( sourceNeedHeight > ( sourceRowExistShift + sourceRealHeight ) ) {
					const int sourceRowExistShiftRest = ringBuf.GetNumExistItems();
					const int sourceRealHeightRest = ringBuf.GetNumItemsToAdd( std::min( source.Height() - sourceRowStart - sourceRowExistShiftRest, ringBufRealNumRows ) );
					const int firstHeightRest = sourceRealHeight * source.Width();
					multiplyMatrixByMatrix(
						/*first*/( sourceRaw + ( batchedSourceRowStart + sourceRowExistShiftRest ) * sourceRowSize ), firstHeightRest, firstWidth, firstWidth,
						/*second*/filterRaw, secondWidth, secondWidth, /*result*/ringBuf.GetBufPtrForAddingItems( sourceRealHeightRest ), resultWidth,
						mulDescs.Get( firstHeightRest, firstHeightRest, firstWidth, secondWidth, resultWidth, /*resultAdd*/false, /*trans1*/false, /*trans2*/false ) );
					ASSERT_EXPR( sourceNeedHeight <= ( sourceRowExistShiftRest + sourceRealHeightRest ) );
				}
			}
		}
	}; //updateRingBuf

	// Container of temporal memory in stack
	const int ringBufBytesSize = ringBufRealNumRows * ringBufRowSize;
	CFloatHandleStackVar temp( mathEngine(), ringBufBytesSize );
	float* const tempRaw = NeoML::GetRaw( temp.GetHandle() );

	// Apply each own CRingBuffer for each Thread
	CRingBuffer ringBuf( tempRaw, ringBufRealNumRows, ringBufRowSize );
	int prevBatchedSourceRowStart = -1; // Previous Source matrix row offset for each Thread

	// Separate calculations for each row
	for( int resultRow = 0; resultRow < resultRowsCount; ++resultRow ) {
		float* const resultDataPtr = resultRaw + resultRow * resultRowSize;
		if( freeTerm != nullptr ) { // Set the free term
			setVectorToMatrixRows( resultDataPtr, result.Width(), resultItemSize, GetRaw( *freeTerm ) );
		} else {
			vectorFill0( resultDataPtr, resultRowSize );
		}
		const int batch = resultRow / result.Height();
		const int row = resultRow % result.Height();

		if( desc.DilationHeight == 1 && desc.DilationWidth == 1 ) {
			// Find all filters that affect the row
			const int sourceRowStart = std::max( 0, ( row + sourceRowStartShift ) / desc.StrideHeight );
			const int filterRowEnd = row - sourceRowStart * desc.StrideHeight + desc.PaddingHeight;
			if( 0 > filterRowEnd || filterRowEnd >= filter.Height() ) {
				continue;
			}

			updateRingBuf( ringBuf, batch, sourceRowStart, prevBatchedSourceRowStart );

			const int filterRowStart = std::max( 0, row + filterRowStartShift );
			int sourceRow = 0;
			for( int filterRow = filterRowEnd; filterRow >= filterRowStart; filterRow -= desc.StrideHeight, ++sourceRow ) {
				const int tempOffset = ringBuf.GetSourceRowOffset( sourceRow ) * ringBufRowSize + filterRow * filterRowSize;
				// The temp blob stores the filter rows multiplied by source; add them to the result rows in correct positions
				const float* tempRowData = ringBuf.GetRaw( tempOffset );
				for( int filterColumn = filterColMin; filterColumn <= filterColMax; filterColumn += desc.StrideWidth, tempRowData += filterObjectSize ) {
					const int minFilterColumn0 = std::min( 0, filterColumn );
					const int maxFilterColumn0 = std::max( 0, filterColumn );
					const int toCopy = std::min( filter.Width() + minFilterColumn0, result.Width() - maxFilterColumn0 );
					if( toCopy > 0 ) {
						float* resultPtr = resultDataPtr + maxFilterColumn0 * filterChannels;
						vectorAdd( resultPtr, tempRowData - minFilterColumn0 * filterChannels, resultPtr, toCopy * filterChannels );
					}
				} //filterColumn
			} //filterRow
		} else { //dilation
			const int topPosMin = std::max( row - totalFilterHeight + 1, -desc.PaddingHeight );
			const int topPosMax = std::min( row, topPosMaxValShift );
			const int sourceRowStart = std::max( 0, ( topPosMin + desc.PaddingHeight ) / desc.StrideHeight );

			updateRingBuf( ringBuf, batch, sourceRowStart, prevBatchedSourceRowStart );

			// Iterate through the filter top positions, starting to apply the filter once we intersect with the current row
			for( int topPos = topPosMin; topPos <= topPosMax; ++topPos ) {
				const int topPosPlusPaddingH = topPos + desc.PaddingHeight;
				if( topPosPlusPaddingH % desc.StrideHeight != 0 ) {
					// This position couldn't have been the filter top row
					continue;
				}
				const int rowMinusTopPos = row - topPos;
				if( rowMinusTopPos % desc.DilationHeight != 0 ) {
					// The filter that starts here doesn't intersect with the current row
					continue;
				}
				// Find all filters that affect the row
				const int filterRow = rowMinusTopPos / desc.DilationHeight; // The current filter row
				const int sourceRow = std::max( 0, topPosPlusPaddingH / desc.StrideHeight ) - sourceRowStart;
				// The pointer to the filter data at (topPos, leftPos) position
				const int tempOffset = ringBuf.GetSourceRowOffset( sourceRow ) * ringBufRowSize + filterRow * filterRowShift;
				const float* tempRowData = ringBuf.GetRaw( tempOffset );
				// Iterate through the filter left positions
				for( int leftPos = leftPosMin; leftPos <= leftPosMax; leftPos += desc.StrideWidth, tempRowData += filterObjectSize ) {
					// Apply the filter row starting at (topPos, leftPos) to the current row
					for( int filterColumn = 0; filterColumn < filter.Width(); ++filterColumn ) {
						const int resultColumn = leftPos + filterColumn * desc.DilationWidth;
						if( 0 <= resultColumn && resultColumn < result.Width() ) {
							float* resultPtr = resultDataPtr + resultColumn * resultItemSize;
							vectorAdd( resultPtr, tempRowData + filterColumn * resultItemSize, resultPtr, resultItemSize );
						}
					} //filterColumn
				} //leftPos
			} //topPos
		} //else dilation
	} //resultRow
}

// Creates a temporary outputDiff blob using the #2 algorithm
static void createTempBlobsLearnAlgo2( const CBlobDesc& source, const CBlobDesc& filter, CBlobDesc& tempDesc )
{
	tempDesc = source;
	tempDesc.SetDimSize( BD_BatchLength, 1 );
	tempDesc.SetDimSize( BD_BatchWidth, source.ObjectCount() + 1 );
	tempDesc.SetDimSize( BD_ListSize, 1 );
	tempDesc.SetDimSize( BD_Width, source.Width() + filter.Width() - 1 );
}

// Fills the temporary outputDiff blob using the #2 algorithm
void CCpuMathEngine::fillTempBlobsForLearnAlgo2( const CCpuConvolutionDesc& desc, const CConstFloatHandle& sourceData,
	const CBlobDesc& tempDesc, const CFloatHandle& tempHandle )
{
	const CBlobDesc& source = desc.Result;

	float* tempRaw = GetRaw( tempHandle );
	vectorFill0( tempRaw, tempDesc.ObjectSize() );

	const int sourceRowSize = source.Width() * source.Depth() * source.Channels();
	const int tempRowSize = tempDesc.ObjectSize() / tempDesc.Height();
	const int batchSize = source.ObjectCount();

	const float* sourceDataPtr = GetRaw( sourceData );
	for( int j = 0; j < batchSize; ++j ) {
		float* tempRawPtr = tempRaw + ( j + 1 ) * tempDesc.ObjectSize();
		for( int h = 0; h < source.Height(); ++h ) {
			dataCopy( tempRawPtr, sourceDataPtr, sourceRowSize );
			vectorFill0( tempRawPtr + sourceRowSize, tempRowSize - sourceRowSize );
			sourceDataPtr += sourceRowSize;
			tempRawPtr += tempRowSize;
		}
	}
}

void CCpuMathEngine::blobConvolutionBackwardAlgo2( const CCpuConvolutionDesc& desc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTerm, const CFloatHandle& resultData )
{
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Source;
	const CBlobDesc& source = desc.Result;

	ASSERT_EXPR( desc.StrideHeight == 1 );
	ASSERT_EXPR( desc.StrideWidth == 1 );
	ASSERT_EXPR( desc.PaddingHeight == 0 );
	ASSERT_EXPR( desc.PaddingWidth == 0 );
	ASSERT_EXPR( desc.DilationHeight == 1 );
	ASSERT_EXPR( desc.DilationWidth == 1 );

	CBlobDesc tempBlobDesc( CT_Float );
	createTempBlobsLearnAlgo2( source, filter, tempBlobDesc );
	CFloatHandleStackVar tempBlobForLearn( mathEngine(), tempBlobDesc.BlobSize() );
	fillTempBlobsForLearnAlgo2( desc, sourceData, tempBlobDesc, tempBlobForLearn.GetHandle() );

	// Repack the filter: switch batch & height, and reorder rows backward: end->start
	const int tempFilterObjectSize = filter.Width() * filter.BatchWidth() * filter.Depth() * filter.Channels();
	const int tempFilterDataSize = filter.Height() * tempFilterObjectSize;
	CFloatHandleStackVar tempFilter( mathEngine(), tempFilterDataSize );
	float* tempFilterRaw = GetRaw( tempFilter.GetHandle() );

	const int batchSize = source.ObjectCount();
	const int tempHeightWidth = tempBlobDesc.Height() * tempBlobDesc.Width();
	const int tempDepthChannels = tempBlobDesc.Depth() * tempBlobDesc.Channels();
	const int resultDepthChannels = result.Depth() * result.Channels();
	const int resultObjectSize = result.ObjectSize();

	const int firstWidth = filter.Width() * tempDepthChannels;
	const int secondWidth = filter.Depth() * filter.Channels();
	const int resultWidth = filter.Width() * resultDepthChannels;

	const float* filterRawPtr = GetRaw( filterData );
	for( int b = 0; b < filter.BatchWidth(); ++b ) {
		for( int h = 0; h < filter.Height(); ++h ) {
			for( int w = 0; w < filter.Width(); ++w ) {
				const int tempFilterPos = h * tempFilterObjectSize + ( ( filter.Width() - 1 - w ) * filter.BatchWidth() + b ) * secondWidth;
				dataCopy( tempFilterRaw + tempFilterPos, filterRawPtr, secondWidth );
				filterRawPtr += secondWidth;
			}
		}
	}
	float* const resultRaw = GetRaw( resultData );
	const float* const filterRaw = tempFilterRaw;
	const float* const tempRaw = GetRaw( tempBlobForLearn.GetHandle() );

	const auto& mulDescs = desc.SmallMatricesMulDescsHeightArrays[CCpuConvolutionDesc::TSMMDA_Backward];

	for( int j = 0; j < batchSize; ++j ) {
		float* resultRawPtr = resultRaw + j * resultObjectSize;
		if( freeTerm != nullptr ) {
			setVectorToMatrixRows( resultRawPtr, result.Height() * result.Width(), resultDepthChannels, GetRaw( *freeTerm ) );
		} else {
			vectorFill0( resultRawPtr, resultObjectSize );
		}
		const float* filterRawPtr = filterRaw;
		const float* const tempSourceStartPtr = tempRaw + ( j + 1 ) * tempBlobDesc.ObjectSize();

		for( int h = 0; h < filter.Height(); ++h ) {
			for( int w = 0; w < filter.Width(); ++w ) {
				float* const resultMatrix = resultRawPtr + w * resultDepthChannels;
				const float* const tempSourcePtr = tempSourceStartPtr + ( w - filter.Width() + 1 ) * tempDepthChannels;
				const int firstHeight = ( tempHeightWidth + filter.Width() - w - 1 ) / filter.Width();
				multiplyMatrixByMatrixAndAdd( /*first*/tempSourcePtr, firstHeight, firstWidth, firstWidth,
					/*second*/filterRawPtr, secondWidth, secondWidth, /*result*/resultMatrix, resultWidth,
					mulDescs.Get( firstHeight, firstHeight, firstWidth, secondWidth, resultWidth,
						/*resultAdd*/true, /*trans1*/false, /*trans2*/false ) );
			}
			resultRawPtr += result.Width() * resultDepthChannels;
			filterRawPtr += tempFilterObjectSize;
		}
	}
}

void CCpuMathEngine::BlobConvolutionBackward( const CConvolutionDesc& convDesc, const CConstFloatHandle& outputDiffData,
	const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& inputDiffData )
{
	CCpuExecutionScope scope;
	const CCpuConvolutionDesc& desc = static_cast<const CCpuConvolutionDesc&>( convDesc );

	switch( desc.BackwardAlgo ) {
		case CA_1:
			blobConvolutionBackwardAlgo1( desc, outputDiffData, filter, freeTerm, inputDiffData );
			break;
		case CA_2:
			blobConvolutionBackwardAlgo2( desc, outputDiffData, filter, freeTerm, inputDiffData );
			break;
		case CA_1x1:
		{
			bool needsFlatten = desc.Filter.Depth() != 1;
			C3dConvolutionDesc* blob3dConvDesc = InitBlob3dConvolution( needsFlatten ? flatten( desc.Source ) : desc.Source,
				/*paddingHeight*/0, /*paddingWidth*/0, /*paddingDepth*/0, desc.StrideHeight, desc.StrideWidth, /*strideDepth*/1,
				needsFlatten ? flatten( desc.Filter ) : desc.Filter, desc.Result );
			Blob3dConvolutionBackward( *blob3dConvDesc, outputDiffData, filter, freeTerm, inputDiffData );
			delete blob3dConvDesc;
			break;
		}
		default:
			ASSERT_EXPR( false );
	}
}

//------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::blobConvolutionLearnAlgo1( const CCpuConvolutionDesc& desc,
	const CConstFloatHandle& inputData, const CConstFloatHandle& outputDiffData, const CFloatHandle& filterDiffData,
	const CFloatHandle* freeTermDiffData, bool isFreeTermDiffFromInput )
{
	const float* inputDataRaw = GetRaw( inputData );
	const float* outputDiffDataRaw = GetRaw( outputDiffData );

	const CBlobDesc& input = desc.Source;
	const CBlobDesc& filterDiff = desc.Filter;
	const CBlobDesc& outputDiff = desc.Result;

	ASSERT_EXPR( filterDiff.Depth() == input.Depth() );
	ASSERT_EXPR( filterDiff.Channels() == input.Channels() );

	const int outputDiffTransHeight = outputDiff.Width() * outputDiff.Height();
	const int outputDiffTransWidth = outputDiff.Depth() * outputDiff.Channels();
	CFloatHandleStackVar outputDiffTrans( mathEngine(), outputDiffTransHeight * outputDiffTransWidth );

	const int tempBlobHolderHeight = outputDiffTransHeight;
	const int tempBlobHolderWidth = filterDiff.Height() * filterDiff.Width() * input.Depth() * input.Channels();
	CFloatHandleStackVar tempBlobHolder( mathEngine(), tempBlobHolderHeight * tempBlobHolderWidth );

	const int freeTermDiffSize = isFreeTermDiffFromInput ? filterDiff.Channels() : filterDiff.ObjectCount();
	const int filterDiffSize = filterDiff.BlobSize();
	CFloatHandleStackVar outputTemp( mathEngine(), filterDiffSize );

	float* const tempBlobHolderDataRaw = GetRaw( tempBlobHolder.GetHandle() );
	float* const outputDiffTransDataRaw = GetRaw( outputDiffTrans.GetHandle() );
	float* const outputTempDataRaw = GetRaw( outputTemp.GetHandle() );
	float* const filterDiffDataRaw = GetRaw( filterDiffData );
	float* const freeTermDiffDataRaw = ( freeTermDiffData != nullptr ) ? GetRaw( *freeTermDiffData ) : nullptr;

	const int firstHeight = outputDiffTransHeight;
	const int firstWidth = outputDiffTransWidth;
	const int secondWidth = tempBlobHolderWidth;
	const int resultWidth = secondWidth;
	auto mulDesc = desc.SmallMatricesMulDescsHeightArrays[CCpuConvolutionDesc::TSMMDA_Learn].Get( firstHeight,
		firstHeight, firstWidth, secondWidth, resultWidth, /*resultAdd*/false, /*trans1*/true, /*trans2*/false );

	for( int b = 0; b < outputDiff.ObjectCount(); ++b ) {
		if( desc.DilationHeight > 1 || desc.DilationWidth > 1 ) {
			createDilationTemporaryBlob( desc, inputDataRaw, b, 0, outputDiff.Width(), tempBlobHolderDataRaw );
		} else {
			createTemporaryBlob( desc, inputDataRaw, b, 0, outputDiff.Width(), tempBlobHolderDataRaw );
		}

		transposeMatrix( 1, outputDiffDataRaw + b * outputDiff.ObjectSize(),
			outputDiff.Height(), 1, outputDiff.Width(), outputDiff.Depth() * outputDiff.Channels(),
			outputDiffTransDataRaw );

		// Calculate diffs
		multiplyTransposedMatrixByMatrix( /*first*/outputDiffTransDataRaw, firstHeight, firstWidth,
			/*second*/tempBlobHolderDataRaw, secondWidth, /*result*/outputTempDataRaw, mulDesc );

		vectorAdd( filterDiffDataRaw, outputTempDataRaw, filterDiffDataRaw, filterDiffSize );

		if( freeTermDiffData != nullptr ) {
			// Train the free term (add diff to the accumulating data)
			const float* diffData;
			int diffDataHeight;
			int diffDataWidth;

			if( isFreeTermDiffFromInput ) {
				diffData = inputDataRaw + b * input.ObjectSize();
				diffDataHeight = input.Height();
				diffDataWidth = input.Width();
			} else {
				diffData = outputDiffTransDataRaw;
				diffDataHeight = outputDiff.Width();
				diffDataWidth = outputDiff.Height();
			}
			for( int j = 0; j < diffDataHeight; ++j ) {
				for( int k = 0; k < diffDataWidth; ++k ) {
					vectorAdd( freeTermDiffDataRaw, diffData, freeTermDiffDataRaw, freeTermDiffSize );
					diffData += freeTermDiffSize;
				}
			}
		}
	}
}

void CCpuMathEngine::blobConvolutionLearnAlgo2( const CCpuConvolutionDesc& desc, const CConstFloatHandle& inputData,
	const CConstFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData,
	bool isFreeTermDiffFromInput )
{
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filterDiff = desc.Filter;

	ASSERT_EXPR( desc.StrideHeight == 1 );
	ASSERT_EXPR( desc.StrideWidth == 1 );
	ASSERT_EXPR( desc.PaddingHeight == 0 );
	ASSERT_EXPR( desc.PaddingWidth == 0 );
	ASSERT_EXPR( desc.DilationHeight == 1 );
	ASSERT_EXPR( desc.DilationWidth == 1 );

	CBlobDesc tempBlobDesc( CT_Float );
	createTempBlobsLearnAlgo2( outputDiff, filterDiff, tempBlobDesc );
	CFloatHandleStackVar tempBlobForLearn( mathEngine(), tempBlobDesc.BlobSize() );
	fillTempBlobsForLearnAlgo2( desc, outputDiffData, tempBlobDesc, tempBlobForLearn.GetHandle() );

	const int tempBlobObjectSize = tempBlobDesc.ObjectSize();
	const int tempBlobDepthChannels = tempBlobDesc.Depth() * tempBlobDesc.Channels();
	const int filterDepthChannels = filterDiff.Depth()* filterDiff.Channels();
	const int inputDepthChannels = input.Depth() * input.Channels();
	const int freeTermDiffSize = isFreeTermDiffFromInput ? filterDiff.Channels() : filterDiff.ObjectCount();

	const float* const inputDataRaw = GetRaw( inputData );
	float* const filterDiffDataRaw = GetRaw( filterDiffData );
	float* const freeTermDiffDataRaw = ( freeTermDiffData != nullptr ) ? GetRaw( *freeTermDiffData ) : nullptr;
	float* const tempBlobForLearnRaw = GetRaw( tempBlobForLearn.GetHandle() );

	const int firstWidth = tempBlobDepthChannels;
	const int secondWidth = inputDepthChannels;
	const int resultWidth = filterDiff.ObjectSize();
	const auto& mulDescs = desc.SmallMatricesMulDescsHeightArrays[CCpuConvolutionDesc::TSMMDA_Learn];

	for( int j = 0; j < outputDiff.ObjectCount(); ++j ) {
		// filter diff
		float* filterMatrix = filterDiffDataRaw;
		for( int h = 0; h < filterDiff.Height(); ++h ) {
			for( int w = 0; w < filterDiff.Width(); ++w, filterMatrix += filterDepthChannels ) {
				const int firstHeight = ( input.Height() - filterDiff.Height() + 1 ) * input.Width() - w;
				const float* const inputMatrix = inputDataRaw + ( ( j * input.Height() + h ) * input.Width() + w ) * inputDepthChannels;
				multiplyTransposedMatrixByMatrixAndAdd( /*first*/tempBlobForLearnRaw + ( j + 1 ) * tempBlobObjectSize,
					firstHeight, firstWidth, firstWidth,
					/*second*/inputMatrix, secondWidth, secondWidth, /*result*/filterMatrix, resultWidth,
					mulDescs.Get( firstHeight, firstHeight, firstWidth, secondWidth, resultWidth,
						/*resultAdd*/true, /*trans1*/true, /*trans2*/false ) );
			}
		}

		if( freeTermDiffData != nullptr ) {
			// freeTerm diff
			// Train free term (add diff to the accumulating data)
			const float* diffData;
			int diffDataHeight;
			int diffDataWidth;

			if( isFreeTermDiffFromInput ) {
				diffData = inputDataRaw + j * input.ObjectSize();
				diffDataHeight = input.Height();
				diffDataWidth = input.Width();
			} else {
				diffData = GetRaw( outputDiffData ) + j * outputDiff.ObjectSize();
				diffDataHeight = outputDiff.Height();
				diffDataWidth = outputDiff.Width();
			}
			for( int m = 0; m < diffDataHeight; ++m ) {
				for( int k = 0; k < diffDataWidth; ++k ) {
					vectorAdd( freeTermDiffDataRaw, diffData, freeTermDiffDataRaw, freeTermDiffSize );
					diffData += freeTermDiffSize;
				}
			}
		}
	}
}

void CCpuMathEngine::BlobConvolutionLearnAdd( const CConvolutionDesc& convDesc, const CConstFloatHandle& input,
	const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle* freeTermDiff,
	bool isFreeTermDiffFromInput )
{
	CCpuExecutionScope scope;
	const CCpuConvolutionDesc& desc = static_cast<const CCpuConvolutionDesc&>( convDesc );

	switch( desc.BackwardAlgo ) {
		case CA_1:
			blobConvolutionLearnAlgo1( desc, input, outputDiff, filterDiff, freeTermDiff, isFreeTermDiffFromInput );
			break;
		case CA_2:
			blobConvolutionLearnAlgo2( desc, input, outputDiff, filterDiff, freeTermDiff, isFreeTermDiffFromInput );
			break;
		case CA_1x1:
		{
			bool needsFlatten = desc.Filter.Depth() != 1;
			C3dConvolutionDesc* blob3dConvDesc = InitBlob3dConvolution( needsFlatten ? flatten( desc.Source ) : desc.Source,
				/*paddingHeight*/0, /*paddingWidth*/0, /*paddingDepth*/0, desc.StrideHeight, desc.StrideWidth, /*strideDepth*/1,
				needsFlatten ? flatten( desc.Filter ) : desc.Filter, desc.Result );
			Blob3dConvolutionLearnAdd( *blob3dConvDesc, input, outputDiff, filterDiff, freeTermDiff, true );
			delete blob3dConvDesc;
			break;
		}
		default:
			ASSERT_EXPR( false );
	}
}

//------------------------------------------------------------------------------------------------------------

CChannelwiseConvolutionDesc* CCpuMathEngine::InitBlobChannelwiseConvolution( const CBlobDesc& source,
	int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
	const CBlobDesc& filter, const CBlobDesc* freeTerm, const CBlobDesc& result )
{
	ASSERT_EXPR( source.Depth() == 1 );
	ASSERT_EXPR( filter.Height() > paddingHeight );
	ASSERT_EXPR( filter.Height() <= source.Height() + 2 * paddingHeight );
	ASSERT_EXPR( filter.Width() > paddingWidth );
	ASSERT_EXPR( filter.Width() <= source.Width() + 2 * paddingWidth );
	ASSERT_EXPR( filter.ObjectCount() == 1 );
	ASSERT_EXPR( filter.Channels() == source.Channels() );
	ASSERT_EXPR( freeTerm == nullptr || freeTerm->BlobSize() == filter.Channels() );
	ASSERT_EXPR( result.BatchLength() == source.BatchLength() );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( result.Channels() == source.Channels() );
	const int expectedOutputHeight = ( source.Height() - filter.Height() + 2 * paddingHeight ) / strideHeight + 1;
	const int expectedOutputWidth = ( source.Width() - filter.Width() + 2 * paddingWidth ) / strideWidth + 1;
	ASSERT_EXPR( result.Height() == expectedOutputHeight );
	ASSERT_EXPR( result.Width() == expectedOutputWidth );

	CCommonChannelwiseConvolutionDesc* desc = new CCommonChannelwiseConvolutionDesc( paddingHeight, paddingWidth,
		strideHeight, strideWidth, source, filter, result );
	return desc;
}

void CCpuMathEngine::BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc& convDesc,
	const CConstFloatHandle& inputDiffData, const CConstFloatHandle& filterData, const CFloatHandle& outputDiffData )
{
	CCpuExecutionScope scope;

	const float* const inputDiffDataRaw = GetRaw( inputDiffData );
	const float* const filterDataRaw = GetRaw( filterData );
	float* const outputDiffDataRaw = GetRaw( outputDiffData );

	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );
	const CBlobDesc& input = desc.Result;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& output = desc.Source;

	const int inputGeo = input.Height() * input.Width();
	const int filterGeo = filter.Height() * filter.Width();
	const int inputBatch = input.Channels() * inputGeo;
	const int outputBatch = output.Channels() * output.Height() * output.Width();

	// Transpose the: HWC -> CHW
	CFloatHandleStackVar filterTransposed( mathEngine(), filter.BlobSize() );
	transposeMatrix( /*batchSize*/1, filterDataRaw, filterGeo,
		/*medium*/1, filter.Channels(), /*channels*/1, GetRaw( filterTransposed.GetHandle() ) );

	const int inputRepackedHeight = input.Height() * input.Width();
	const int inputRepackedWidth = input.Channels();
	CFloatHandleStackVar inputRepacked( mathEngine(), inputRepackedHeight * inputRepackedWidth );

	const int tempSize = inputGeo * filterGeo * input.Channels();
	CFloatHandleStackVar temp( mathEngine(), tempSize );

	const int outputRepackedHeight = output.Height() * output.Width();
	const int outputRepackedWidth = output.Channels();
	CFloatHandleStackVar outputRepacked( mathEngine(), outputRepackedHeight * outputRepackedWidth );

	float* const inputRepackedDataRaw = GetRaw( inputRepacked.GetHandle() );
	float* const outputRepackedDataRaw = GetRaw( outputRepacked.GetHandle() );
	float* const tempDataRaw = GetRaw( temp.GetHandle() );

	for( int batchIndex = 0; batchIndex < input.BatchWidth(); ++batchIndex ) {
		// Repack HWC -> CHW
		transposeMatrix( /*batchSize*/1, inputDiffDataRaw + batchIndex * inputBatch,
			inputGeo, /*medium*/1, input.Channels(), /*channels*/1, inputRepackedDataRaw );

		// Multiply the inputRepacked and filter matrices
		PRESUME_EXPR( temp.Size() >= inputRepackedWidth * inputGeo );
		batchMultiplyMatrixByTransposedMatrix( inputRepackedWidth, inputRepacked.GetHandle(), inputGeo, /*firstWidth*/1,
			filterTransposed.GetHandle(), filterGeo, temp.GetHandle(), nullptr );

		// Add the subvectors from the resulting matrix to the required positions in outputRepacked
		for( int step = 0; step < output.Height() * output.Channels(); ++step ) {
			float* outputDataPtr = outputRepackedDataRaw + step * output.Width();
			vectorFill0( outputDataPtr, output.Width() );

			const int channel = step / output.Height();
			const int row = step % output.Height();
			const int inputRowStart = std::max( 0, ( row + desc.PaddingHeight - filter.Height() + desc.StrideHeight ) / desc.StrideHeight );
			const int filterRowBackStart = row - inputRowStart * desc.StrideHeight + desc.PaddingHeight;
			if( 0 > filterRowBackStart || filterRowBackStart >= filter.Height() ) {
				continue;
			}
			const int filterRowBackEnd = std::max( 0, filter.Height() + row - output.Height() - desc.PaddingHeight );
			int inputRow = inputRowStart;
			for( int filterRow = filterRowBackStart;
				filterRow >= filterRowBackEnd;
				filterRow -= desc.StrideHeight, ++inputRow )
			{
				// The temp blob stores the rows of the filter multiplied by input; add them to the output rows in required positions
				const float* tempRowData = tempDataRaw + ( ( channel * input.Height() + inputRow )
					* input.Width() * filter.Height() + filterRow ) * filter.Width();

				for( int col = -desc.PaddingWidth;
					col <= output.Width() + desc.PaddingWidth - filter.Width();
					col += desc.StrideWidth )
				{
					int tempRowDataShift = 0;
					int toCopy = filter.Width();
					int pos = col;
					if( pos < 0 ) {
						tempRowDataShift = -pos;
						toCopy += pos;
						pos = 0;
					}
					if( pos + toCopy > output.Width() ) {
						toCopy = output.Width() - pos;
					}
					vectorAdd( outputDataPtr + pos, tempRowData + tempRowDataShift, outputDataPtr + pos, toCopy );
					tempRowData += filter.Height() * filter.Width();
				}
			}
		}

		// Repack CHW -> HWC
		transposeMatrix( /*batchSize*/1, outputRepackedDataRaw,
			outputRepackedWidth, /*medium*/1, outputRepackedHeight, /*channels*/1,
			outputDiffDataRaw + batchIndex * outputBatch );
	}
}

void CCpuMathEngine::BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputData,
	const CConstFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData )
{
	CCpuExecutionScope scope;

	const float* const inputDataRaw = GetRaw( inputData );
	const float* const filterDiffDataRaw = GetRaw( filterDiffData );
	const float* const outputDiffDataRaw = GetRaw( outputDiffData );

	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& filterDiff = desc.Filter;

	const int outputDiffTransHeight = outputDiff.Width() * outputDiff.Height();
	const int outputDiffTransWidth = input.Channels();
	CFloatHandleStackVar outputDiffTrans( mathEngine(), outputDiffTransHeight * outputDiffTransWidth );

	const int outputDiffTransRepackedHeight = outputDiffTransHeight;
	const int outputDiffTransRepackedWidth = outputDiffTransWidth;
	CFloatHandleStackVar outputDiffTransRepacked( mathEngine(), outputDiffTransRepackedHeight * outputDiffTransRepackedWidth );

	const int tempBlobHeight = outputDiffTransHeight * filterDiff.Height() * filterDiff.Width();
	const int tempBlobWidth = outputDiffTransWidth;
	CFloatHandleStackVar tempBlob( mathEngine(), tempBlobHeight * tempBlobWidth );

	const int tempBlobRepackedSize = tempBlobHeight * tempBlobWidth;
	CFloatHandleStackVar tempBlobRepacked( mathEngine(), tempBlobRepackedSize );

	const int filterTempSize = filterDiff.Channels() * filterDiff.Height() * filterDiff.Width();
	CFloatHandleStackVar filterTemp( mathEngine(), filterTempSize ); // a blob with one object diffs

	// The transposed filter diff
	CFloatHandleStackVar filterDiffTransposedHolder( mathEngine(), filterDiff.BlobSize() );
	float* const filterDiffTransposedRaw = GetRaw( filterDiffTransposedHolder.GetHandle() );
	CBlobDesc filterDiffTransposed( CT_Float );
	filterDiffTransposed.SetDimSize( BD_BatchWidth, filterDiff.Channels() );
	filterDiffTransposed.SetDimSize( BD_Height, filterDiff.Height() );
	filterDiffTransposed.SetDimSize( BD_Width, filterDiff.Width() );
	transposeMatrix( /*batchSize*/1, filterDiffDataRaw, filterDiff.Height() * filterDiff.Width(),
		/*medium*/1, filterDiff.Channels(), /*channels*/1, filterDiffTransposedRaw );

	float* const tempBlobDataRaw = GetRaw( tempBlob.GetHandle() );
	float* const tempBlobRepackedDataRaw = GetRaw( tempBlobRepacked.GetHandle() );
	float* const outputDiffTransDataRaw = GetRaw( outputDiffTrans.GetHandle() );
	float* const outputDiffTransRepackedDataRaw = GetRaw( outputDiffTransRepacked.GetHandle() );
	float* const filterTempDataRaw = GetRaw( filterTemp.GetHandle() );
	float* const filterDiffReductionDataRaw = GetRaw( filterDiffTransposedHolder.GetHandle() );

	const int batchCount = outputDiff.BatchWidth();
	for( int b = 0; b < batchCount; ++b ) {
		// Filling the matrix from the windows
		createTemporaryBlob( desc, inputDataRaw, b, /*resultRowStart*/0, outputDiff.Width(), tempBlobDataRaw );
		// Repack HWC -> CHW
		transposeMatrix( /*batchSize*/1, tempBlobDataRaw, tempBlobHeight,
			/*medium*/1, tempBlobWidth, /*channels*/1, tempBlobRepackedDataRaw );

		// Transpose the output blob HWC -> WHC
		transposeMatrix( /*batchSize*/1, outputDiffDataRaw + b * outputDiff.ObjectSize(),
			outputDiff.Height(), /*medium*/1, outputDiff.Width(), outputDiff.Channels(),
			outputDiffTransDataRaw );
		// Repack HWC -> CHW
		transposeMatrix( /*batchSize*/1, outputDiffTransDataRaw, outputDiffTransHeight,
			/*medium*/1, outputDiffTransWidth, /*channels*/1, outputDiffTransRepackedDataRaw );

		// Multiply matrices
		batchMultiplyTransposedMatrixByMatrix( outputDiffTransRepackedWidth,
			outputDiffTransRepackedDataRaw, outputDiffTransRepackedHeight, /*firstWidth*/1,
			tempBlobRepackedDataRaw, filterDiff.Height() * filterDiff.Width(),
			filterTempDataRaw, nullptr );

		// Update the accumulator
		vectorAdd( filterDiffReductionDataRaw, filterTempDataRaw, filterDiffReductionDataRaw, filterTempSize );

		if( freeTermDiffData != nullptr ) {
			// Train the free term (add diffs to the accumulated data)
			sumMatrixColumnsAdd( *freeTermDiffData, outputDiffTransRepacked.GetHandle(),
				outputDiffTransRepackedWidth, outputDiffTransRepackedHeight );
		}
	}

	TransposeMatrix( /*batchSize*/1, filterDiffTransposedHolder.GetHandle(), filterDiff.Channels(),
		/*medium*/1, filterDiff.Height() * filterDiff.Width(), /*channels*/1, filterDiffData, filterDiff.BlobSize() );
}

} // namespace NeoML

