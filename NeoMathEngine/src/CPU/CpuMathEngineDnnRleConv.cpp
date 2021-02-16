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

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/CrtAllocatedObject.h>

#include <CpuMathEngine.h>
#include <CpuMathEnginePrivate.h>
#include <CpuMathEngineOmp.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>

namespace NeoML {

// Finds the index of lowest nonzero bit
static inline unsigned long leastSignificantBitIndex( int x )
{
#if FINE_PLATFORM( FINE_WINDOWS )
	unsigned long res;
	_BitScanForward( &res, x );
	return res;
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_ANDROID ) 
	return __builtin_ctzl( x );
#else
	#error "Platform isn't supported!"
#endif
}

// Sets the lowest nonzero bit to zero
static inline unsigned long nullifyLeastSignificantBit( int x )
{
	return x & ~( 1 << leastSignificantBitIndex( x ) );
}

// (2^x)-1.
static inline unsigned long long pow2MinusOne( int x )
{
	return x == 64 ? 0xFFFFFFFFFFFFFFFF : ( ( 1ULL << x ) - 1 );
}

// ====================================================================================================================

// RLE convolution descriptor
struct CCpuRleConvolutionDesc : public CRleConvolutionDesc {
	float StrokeValue;
	float NonStrokeValue;

	int StrideHeight;
	int StrideWidth;

	// The filter and the diff converted to the optimal format (FilterHeight x FilterWidth x FilterCount).
	CFloatHandleVar ConvertedFilter;
	CFloatHandleVar ConvertedFilterDiff;

	CFloatHandleVar FilterConv; // filter convolutions for all different raster rows
	mutable bool UpdateNeeded; // indicates if FilterConv should be update (if the filter values changed)

	CBlobDesc Source;
	CBlobDesc Filter;
	CBlobDesc Result;

	CCpuRleConvolutionDesc( IMathEngine& mathEngine, float strokeValue, float nonStrokeValue, int strideHeight, int strideWidth,
		const CBlobDesc& source, const CBlobDesc& filter, const CBlobDesc& result );
};

CCpuRleConvolutionDesc::CCpuRleConvolutionDesc( IMathEngine& mathEngine, float strokeValue, float nonStrokeValue, int strideHeight,
		int strideWidth, const CBlobDesc& source, const CBlobDesc& filter, const CBlobDesc& result ) :
	StrokeValue( strokeValue ),
	NonStrokeValue( nonStrokeValue ),
	StrideHeight( strideHeight ),
	StrideWidth( strideWidth ),
	ConvertedFilter( mathEngine, filter.Height() * filter.Width() * filter.ObjectCount() ),
	ConvertedFilterDiff( mathEngine, filter.Height() * filter.Width() * filter.ObjectCount() ),
	FilterConv( mathEngine, ( 1 << filter.Width() ) * filter.Height() * filter.ObjectCount() ),
	UpdateNeeded( true ),
	Source( source ),
	Filter( filter ),
	Result( result )
{
}

static inline void updateFilterConv( IMathEngine& mathEngine, CCpuRleConvolutionDesc& desc, const CFloatHandle& filterData,
	const CFloatHandle* freeTermData )
{
	desc.UpdateNeeded = false;

	const int filterHeight = desc.Filter.Height();
	const int filterWidth = desc.Filter.Width();
	const int filterCount = desc.Filter.ObjectCount();

	mathEngine.TransposeMatrix( 1, filterData, filterCount, 1, filterHeight * filterWidth, 1,
		desc.ConvertedFilter.GetHandle(), desc.ConvertedFilter.Size() );

	int filterConvRowSize = filterHeight * filterCount;

	// Fill the zero filter with the minimum convolution value (all non-stroke positions)
	float* zeroFilterConvPtr = GetRaw( desc.FilterConv.GetHandle() );
	vectorFill0( zeroFilterConvPtr, filterConvRowSize );
	const float* convertFilterDataPtr = GetRaw( desc.ConvertedFilter.GetHandle() );
	for( int j = 0; j < filterHeight; ++j ) {
		for( int i = 0; i < filterWidth; ++i ) {
			alignedVectorAdd( zeroFilterConvPtr, convertFilterDataPtr, filterCount );
			convertFilterDataPtr += filterCount;
		}
		zeroFilterConvPtr += filterCount;
	}

	CFloatHandleStackVar nonStroke( mathEngine );
	nonStroke.SetValue( desc.NonStrokeValue );
	mathEngine.VectorMultiply( desc.FilterConv.GetHandle(), desc.FilterConv.GetHandle(), filterConvRowSize,
		nonStroke );

	// Fill in the rest of the convolutions as diff with the previous ones (start with zero that has been filled in already)
	float mult = desc.StrokeValue - desc.NonStrokeValue;
	float* filterConvBase = GetRaw( desc.FilterConv.GetHandle() );
	zeroFilterConvPtr = filterConvBase;
	convertFilterDataPtr = GetRaw( desc.ConvertedFilter.GetHandle() );
	for( int i = 1; i < ( 1 << filterWidth ); ++i ) {
		filterConvBase += filterConvRowSize;
		float* filterConvDst = filterConvBase;
		const float* filterConvSrc = zeroFilterConvPtr + filterConvRowSize * nullifyLeastSignificantBit( i );
		const float* filterSrc = convertFilterDataPtr
			+ filterCount * filterWidth * ( filterHeight - 1 ) + leastSignificantBitIndex( i ) * filterCount;
		for( int j = 0; j < filterHeight; ++j ) {
			alignedVectorMultiplyAndAdd( filterConvSrc, filterSrc, filterConvDst, filterCount, &mult );
			filterConvDst += filterCount;
			filterConvSrc += filterCount;
			filterSrc -= filterCount * filterWidth;
		}
	}

	if( freeTermData != 0 ) {
		// Add the free term to each convolution (the corresponding part)
		CFloatHandleStackVar freeTermConv( mathEngine, filterCount );
		CFloatHandle freeTermConvData = freeTermConv.GetHandle();

		CFloatHandleStackVar filterHeightInv( mathEngine );
		filterHeightInv.SetValue( 1.f / filterHeight );
		mathEngine.VectorMultiply( *freeTermData, freeTermConvData, freeTermConv.Size(), filterHeightInv );
		mathEngine.AddVectorToMatrixRows( 1, desc.FilterConv.GetHandle(), desc.FilterConv.GetHandle(),
			filterHeight * ( 1 << filterWidth ), filterCount, freeTermConvData );
	}
}

// ====================================================================================================================
CRleConvolutionDesc* CCpuMathEngine::InitBlobRleConvolution( const CBlobDesc& source, float strokeValue,
	float nonStrokeValue, int strideHeight, int strideWidth, const CBlobDesc& filter,
	const CBlobDesc& result )
{
	ASSERT_EXPR( strideHeight > 0 );
	ASSERT_EXPR( strideWidth > 0 );
	ASSERT_EXPR( source.Channels() == filter.Channels() );
	ASSERT_EXPR( source.Depth() == filter.Depth() );
	ASSERT_EXPR( filter.Height() <= source.Height() );
	ASSERT_EXPR( filter.Width() <= source.Width() );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( result.BatchLength() == source.BatchLength() );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.Height() == 1 + ( source.Height() - filter.Height() ) / strideHeight );
	ASSERT_EXPR( result.Width() == 1 + ( source.Width() - filter.Width() ) / strideWidth );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( result.Depth() == 1 );
	ASSERT_EXPR( filter.Width() <= MaxRleConvFilterWidth );
	ASSERT_EXPR( source.Width() <= MaxRleConvImageWidth );
	ASSERT_EXPR( source.Channels() == 1 );
	ASSERT_EXPR( ( filter.ObjectCount() % 4 ) == 0 );

	CCpuRleConvolutionDesc* desc = new CCpuRleConvolutionDesc( mathEngine(), strokeValue, nonStrokeValue, strideHeight, strideWidth,
		source, filter, result );
	return desc;
}

void CCpuMathEngine::BlobRleConvolution( const CRleConvolutionDesc& convDesc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTerm, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( freeTerm == 0 || freeTerm->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCpuRleConvolutionDesc& desc = static_cast<const CCpuRleConvolutionDesc&>( convDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Result;

	if( desc.UpdateNeeded ) {
		updateFilterConv( mathEngine(), const_cast<CCpuRleConvolutionDesc&>(desc), filterData, freeTerm );
	}

	const int filterHeight = filter.Height();
	const int filterWidth = filter.Width();
	const int filterCount = filter.ObjectCount();
	const int strideHeight = desc.StrideHeight;
	const int strideWidth = desc.StrideWidth;
	const int outputWidth = result.Width();
	const int outputRowSize = result.ObjectSize() / result.Height();
	const int heightCoveredByFilters = ( result.Height() - 1 ) * desc.StrideHeight + filterHeight;
	const int filterJCount = result.Height();
	const int filterConvStep = filterHeight * filterCount;
	const int filterLineMask = ( 1 << filterWidth ) - 1;
	const int objectCount = source.ObjectCount();
	const float* filterConvPtr = GetRaw( desc.FilterConv.GetHandle() );
	float* resultDataPtr = GetRaw( resultData );

	const int curThreadCount = IsOmpRelevant( objectCount ) ? threadCount : 1;

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int b = 0; b < objectCount; ++b ) {
		const CRleImage* inputImage = reinterpret_cast<CRleImage*>( GetRaw( sourceData + source.ObjectSize() * b ) );
		int imageStartPos = ( source.Width() - inputImage->Width ) / 2;
		int imageStartLine = ( source.Height() - inputImage->Height ) / 2;
		int imageStopLine = imageStartLine + inputImage->Height;

		const CRleStroke* inputData = inputImage->Lines;
		int lastJInit = 0;
		for( int lineNumber = 0; lineNumber < heightCoveredByFilters; ++lineNumber ) {
			unsigned long long line = 0; // bit representation of an image row

			if( imageStartLine <= lineNumber && lineNumber < imageStopLine ) {
				for( ; inputData->Start < MaxRleConvImageWidth; ++inputData ) {
					line |= pow2MinusOne( inputData->End - inputData->Start ) << ( inputData->Start + imageStartPos );
				}
				++inputData;
			}

			int firstJFilter = ( lineNumber - filterHeight + strideHeight ) / strideHeight;
			if( firstJFilter < 0 ) {
				firstJFilter = 0;
			}
			int lastJFilter = lineNumber / strideHeight + 1;
			if( lastJFilter > filterJCount ) {
				lastJFilter = filterJCount;
			}
			int jCount = lastJFilter - firstJFilter;

			float* outputObj = resultDataPtr + b * result.ObjectSize();

			if( lastJFilter > lastJInit ) {
				// Initialize the output little by little so that the cache does not overflow
				vectorFill0( outputObj + lastJInit * outputRowSize, outputRowSize * ( lastJFilter - lastJInit ) );
				lastJInit = lastJFilter;
			}

			// Traverse all filters that have this row
			float* output = outputObj + firstJFilter * outputRowSize;
			int filterLineNumber = max( 0, lineNumber - firstJFilter * strideHeight );
			const float* filterConvData = filterConvPtr + ( filterHeight - filterLineNumber - 1 ) * filterCount;

			for( int i = 0; i < outputWidth; ++i ) {
				int index = ( ( int ) ( line >> ( i * strideWidth ) ) & filterLineMask );
				const float* curFilterConvData = filterConvData + index * filterConvStep;
				float* curOutput = output;
				for( int j = 0; j < jCount; ++j ) {
					alignedVectorAdd( curOutput, curFilterConvData, filterCount );
					curFilterConvData += strideHeight * filterCount;
					curOutput += outputRowSize;
				}
				output += filterCount;
			}
		}
	}
}

void CCpuMathEngine::BlobRleConvolutionLearnAdd( const CRleConvolutionDesc& convDesc, const CFloatHandle& inputData,
	const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterDiffData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermDiffData == 0 || freeTermDiffData->GetMathEngine() == this );

	const CCpuRleConvolutionDesc& desc = static_cast<const CCpuRleConvolutionDesc&>( convDesc );
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& filterDiff = desc.Filter;
	const CBlobDesc& outputDiff = desc.Result;

	const int objectCount = outputDiff.ObjectCount();
	const int filterHeight = filterDiff.Height();
	const int filterWidth = filterDiff.Width();
	const int filterCount = filterDiff.ObjectCount();

	const float strokeValue = desc.StrokeValue;
	const float nonStrokeValue = desc.NonStrokeValue;
	const int heightCoveredByFilters = ( outputDiff.Height() - 1 ) * desc.StrideHeight + filterHeight;

	const int strideHeight = desc.StrideHeight;
	const int strideWidth = desc.StrideWidth;

	const int filterJCount = outputDiff.Height();

	TransposeMatrix( 1, filterDiffData, filterCount, 1, filterHeight * filterWidth, 1,
		desc.ConvertedFilterDiff.GetHandle(), desc.ConvertedFilterDiff.Size() );

	const int curThreadCount = IsOmpRelevant( objectCount ) ? threadCount : 1;

	unique_ptr<COmpReduction1DData> freeTermDiffItem( nullptr );
	unique_ptr<COmpReduction<COmpReduction1DData>> freeTermDiffReduction( nullptr );

	if( freeTermDiffData != nullptr ) {
		freeTermDiffItem.reset( new COmpReduction1DData( mathEngine(), *freeTermDiffData, desc.Filter.ObjectCount() ) );
		freeTermDiffReduction.reset( new COmpReduction<COmpReduction1DData>( curThreadCount, *freeTermDiffItem ) );
	}

	COmpReduction1DData filterDiffItem( mathEngine(), desc.ConvertedFilterDiff, filterHeight * filterWidth * filterCount );
	COmpReduction<COmpReduction1DData> filterDiffReduction( curThreadCount, filterDiffItem );

	CFloatHandleStackVar mults( mathEngine(), curThreadCount );
	float* multsPtr = GetRaw( mults.GetHandle() );
	const float* outputDiffDataRaw = GetRaw( outputDiffData );

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int b = 0; b < objectCount; ++b ) {
		const CRleImage* inputImage = reinterpret_cast<CRleImage*>( GetRaw( inputData + input.ObjectSize() * b ) );
		int imageStartPos = ( input.Width() - inputImage->Width ) / 2;
		int imageStartLine = ( input.Height() - inputImage->Height ) / 2;
		int imageStopLine = imageStartLine + inputImage->Height;

		float* filterDiffReductionPrivatePtr = GetRaw( filterDiffReduction.GetPrivate().Data );
		float* freeTermDiffReductionPrivatePtr = freeTermDiffData == nullptr ? nullptr :
			GetRaw( freeTermDiffReduction->GetPrivate().Data );

		const CRleStroke* inputDataPtr = inputImage->Lines;
		const float* outputDiffDataPtr = outputDiffDataRaw + outputDiff.ObjectSize() * b;
		// Iterate through the input rows
		for( int lineNumber = 0; lineNumber < heightCoveredByFilters; ++lineNumber ) {
			// Build the bit representation of an image row
			unsigned long long line = 0;
			if( imageStartLine <= lineNumber && lineNumber < imageStopLine ) {
				for( ; inputDataPtr->Start < MaxRleConvImageWidth; ++inputDataPtr ) {
					line |= pow2MinusOne( inputDataPtr->End - inputDataPtr->Start ) << ( inputDataPtr->Start + imageStartPos );
				}
				++inputDataPtr;
			}
			// Find the filter steps for this row
			int firstJFilter = ( lineNumber - filterHeight + strideHeight ) / strideHeight;
			if( firstJFilter < 0 ) {
				firstJFilter = 0;
			}
			int lastJFilter = lineNumber / strideHeight + 1;
			if( lastJFilter > filterJCount ) {
				lastJFilter = filterJCount;
			}
			// Iterate through all filter positions horizontally
			for( int outCol = 0; outCol < outputDiff.Width(); ++outCol ) {
				const float* currOutputDiff = outputDiffDataPtr + ( firstJFilter * outputDiff.Width() + outCol ) * outputDiff.Channels();
				// Iterate through the vertical filter positions that crossed the current input row
				for( int outRow = firstJFilter; outRow < lastJFilter; ++outRow ) {
					// The index of the filter row that goes over the current input row
					const int filterRow = lineNumber - strideHeight * outRow;
					float* currFilterDiff = filterDiffReductionPrivatePtr + filterRow * filterWidth * filterCount;
					for( int filterCol = 0; filterCol < filterWidth; ++filterCol ) {
						float* mult = multsPtr + OmpGetThreadNum();
						*mult = ( ( ( 1ULL << filterCol ) & line ) != 0 ) ? strokeValue : nonStrokeValue;
						alignedVectorMultiplyAndAdd( currFilterDiff, currOutputDiff, currFilterDiff, filterCount, mult );
						currFilterDiff += filterCount;
					}
					currOutputDiff += filterCount * outputDiff.Width();
				}
				line >>= strideWidth;
			}
		}

		if( freeTermDiffData != nullptr ) {
			// Calculate diff separately for the free terms
			for( int j = 0; j < outputDiff.Height(); ++j ) {
				for( int k = 0; k < outputDiff.Width(); ++k ) {
					alignedVectorAdd( freeTermDiffReductionPrivatePtr, outputDiffDataPtr, filterCount );
					outputDiffDataPtr += filterCount;
				}
			}
		}
	}

	if( freeTermDiffData != 0 ) {
		freeTermDiffReduction->Reduce();
	}
	filterDiffReduction.Reduce();

	TransposeMatrix( 1, desc.ConvertedFilterDiff.GetHandle(), filterHeight * filterWidth,
		1, filterCount, 1, filterDiffData, filterDiff.BlobSize() );

	desc.UpdateNeeded = true; // after learning, on the next forward pass FilterConv should be updated
}

} // namespace NeoML
