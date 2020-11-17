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

#include <array>
#include <vector>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

class CBlobConvolutionBase : public CCrtAllocatedObject {
public:
	virtual ~CBlobConvolutionBase() = default;
	virtual void ProcessConvolution( int threadCount, const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) = 0;
};

template<int FltCnt>
class CBlobConvolution : public CBlobConvolutionBase {
public:
	CBlobConvolution( IMathEngine* mathEngine,
		int channelCount, int filterHeight, int filterWidth, int sourceHeight, int sourceWidth, 
		int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth );
	~CBlobConvolution() override = default;

	void ProcessConvolution( int threadCount,
		const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) override;

private:
	struct CSize {
		int Height;
		int Width;
	};

	IMathEngine* mathEngine;

	const int ChCnt;
	const int FltH;
	const int FltW;
	const int SrcH;
	const int SrcW;
	const int PaddingH;
	const int PaddingW;
	const int StrideH;
	const int StrideW;
	const int DilationH;
	const int DilationW;
	const int ResH;
	const int ResW;

	// For some cases we will use FltCnt, rounded up to nearest integer multiple of 8
	static constexpr int FltCntM8 = ( FltCnt + 8 - 1 ) / 8 * 8;
	static constexpr size_t AvxAlignment = 32;

	const float* src;
	const float* flt;
	const float* freeTerm;
	float* res;

	// Length of one source line.
	const int SrcLineStride;
	// Distance in floats between two neighbor pixels in source.
	const int SrcXStep;
	const int SrcYStep;
	// Distance in floats between two neighbor pixels in window by horizontal.
	const int SrcXDilation;
	// Distance in floats between two neighbor pixels in window by horizontal.
	const int SrcYDilation;
	// Width of source window in floats
	const int SrcXWindowSize;
	const int ResLineStride;

	// Choose proper pixels in source and filter:
	// 0  1  2
	// 3  4  5
	// 6  7  8
	// Offset is relative to central pixel of source window
	const std::array<std::vector<int>, 16> SrcPixelsOffset;
	// Offset is relative to top left pixel of filter window
	const std::array<std::vector<int>, 16>  FltPixelsOffset;
	// In some cases when the width of the image is nearly equals to the width of optimized batch processing window,
	// we may faced to situation ( when dilation is higth ) when no one optimized batch ptocessing can be
	// applied. For such cases we will use optimized batch processing with narrower window but height greater then one.
	const CSize NarrowBatchProcessSize;
	const CSize WideBatchProcessSize;

	// Initialize NarrowBatchProcessSize and WideBatchProcessSize
	CSize getNarrowBatchProcessSize();
	CSize getWideBatchProcessSize();

	// Process one line of image. In case of narrow processing we will step through several lines.
	void processConvolutionLoop( int rxSize, bool useNarrowProcessing, const float*& srcPtr, float*& resPtr, int windowIndex );

	void batchProcessChannels( const float* srcPtr, const float* fltPtr,
		__m256& r00, __m256& r01, __m256& r02,
		__m256& r10, __m256& r11, __m256& r12,
		__m256& r20, __m256& r21, __m256& r22 );
	void batchProcessChannels( const float* srcPtr, const float* fltPtr, int srcNarrowStep,
		__m256& r00, __m256& r01, __m256& r02,
		__m256& r10, __m256& r11, __m256& r12,
		__m256& r20, __m256& r21, __m256& r22 );
	void singleProcessChannels( const float* srcPtr, const float* fltPtr, __m256& r0, __m256& r1, __m256& r2 );
	void singleProcessChannels( const float* srcPtr, const float* fltPtr, __m256& r0 );
	void singleProcessChannelsNarrow( const float* srcPtr, const float* fltPtr, __m256& r0, __m256& r1, __m256& r2 );


	// Process convolution for multiple result pixels ( number of pixels is defined by 'FastBatchProcessSize' member ).
	void batchProcess( const float* srcPtr, float* resPtr, int windowIndex, bool useNarrowProcessing );
	// Process convolution for single result pixel.
	void singleProcess( const float* srcPtr, float* resPtr, int windowIndex );
	void singleProcessNarrow( const float* srcPtr, float* resPtr, int windowIndex );

	// Rearrange filter and fill 'Filter' and 'FreeTerm' members.
	const float* rearrangeFilter( const float* filterData, CFloatHandleStackVar& Filter );
	const float* rearrangeFreeTerm( const float* freeTermData, CFloatHandleStackVar& FreeTerm );
	const std::array<std::vector<int>, 16> fillSrcPixelOffset();
	const std::array<std::vector<int>, 16> fillFltPixelOffset();

	// Circular rotation of three ymm registers to the left, step equals to six floats.
	static void rotateLeft6( __m256& y0, __m256& y1, __m256& y2 );
	// Circular rotation of three ymm registers to the left, step equals to two floats.
	static void rotateLeft2( __m256& y );

};

class CBlobConvolutionFabric : public CCrtAllocatedObject {
public:
	static bool IsBlobConvolutionAvailable( int FltCnt, int FltH, int FltW );
	static std::unique_ptr<CBlobConvolutionBase> GetProperInstance( IMathEngine* mathEngine, int FltCnt,
		int channelCount, int filterHeight, int filterWidth, int sourceHeight, int sourceWidth,
		int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CBlobConvolutionFabric::IsBlobConvolutionAvailable( int FltCnt, int FltH, int FltW )
{
	if( FltH != 3 || FltW != 3 ) {
		return false;
	}
	if( FltCnt == 24 ||
		FltCnt == 18 ||
		FltCnt == 6 ) {
		return true;
	}
	return false;
}

std::unique_ptr<CBlobConvolutionBase> CBlobConvolutionFabric::GetProperInstance( IMathEngine* mathEngine, int filterCount,
	int channelCount, int filterHeight, int filterWidth, int sourceHeight, int sourceWidth,
	int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth )
{
	switch( filterCount ) {
		case 24:
			return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<24>( mathEngine,
				channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
				paddingHeight, paddingWidth, strideHeight, strideWidth,
				dilationHeight, dilationWidth, resultHeight, resultWidth ) );
		case 18:
			return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<18>( mathEngine,
				channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
				paddingHeight, paddingWidth, strideHeight, strideWidth,
				dilationHeight, dilationWidth, resultHeight, resultWidth ) );
		case 6:
			return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<6>( mathEngine,
				channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
				paddingHeight, paddingWidth, strideHeight, strideWidth,
				dilationHeight, dilationWidth, resultHeight, resultWidth ) );
		default:
			return nullptr;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int FltCnt>
CBlobConvolution<FltCnt>::CBlobConvolution( IMathEngine* _mathEngine, int channelCount, int filterHeight, int filterWidth,
		int sourceHeight, int sourceWidth, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth ) :
	mathEngine( _mathEngine ),
	ChCnt( channelCount ),
	FltH( filterHeight ),
	FltW( filterWidth ),
	SrcH( sourceHeight ),
	SrcW( sourceWidth ),
	PaddingH( paddingHeight ),
	PaddingW( paddingWidth ),
	StrideH( strideHeight ),
	StrideW( strideWidth ),
	DilationH( dilationHeight ),
	DilationW( dilationWidth ),
	ResH( resultHeight ),
	ResW( resultWidth ),
	src( nullptr ),
	flt( nullptr ),
	freeTerm( nullptr ),
	res( nullptr ),
	SrcLineStride( SrcW * ChCnt ),
	SrcXStep( StrideW * ChCnt ),
	SrcYStep( StrideH * SrcLineStride ),
	SrcXDilation( DilationW * ChCnt ),
	SrcYDilation( DilationH * SrcLineStride ),
	SrcXWindowSize( FltW * SrcXDilation ),
	ResLineStride( ResW * FltCnt ),
	SrcPixelsOffset( fillSrcPixelOffset() ),
	FltPixelsOffset( fillFltPixelOffset() ),
	NarrowBatchProcessSize( getNarrowBatchProcessSize() ),
	WideBatchProcessSize( getWideBatchProcessSize() )
{
}

template<int FltCnt>
void CBlobConvolution<FltCnt>::ProcessConvolution( int threadCount,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
{
	CFloatHandleStackVar filterTempBuffer( *mathEngine, FltW * FltH * FltCntM8 * ChCnt );
	CFloatHandleStackVar freeTermTempBuffer( *mathEngine, FltCntM8 );

	src = sourceData;
	flt = rearrangeFilter( filterData, filterTempBuffer );
	freeTerm = rearrangeFreeTerm( freeTermData, freeTermTempBuffer );
	res = resultData;

	const int curThreadCount = IsOmpRelevant( ResH, ResH * ResW * FltCnt * FltW * FltH * ChCnt ) ? threadCount : 1;

	// Number of steps for each side of image, where filter is applied partially
	int PartialStepCountBeforeX = static_cast<const int>( std::ceil( static_cast<float>( PaddingW ) / StrideW ) );
	int PartialStepCountAfterX = static_cast<const int>( std::ceil( ( StrideW * ( std::ceil( static_cast<float>( SrcW ) / StrideW ) - 1 ) - SrcW + PaddingW + 1 ) / StrideW ) );
	int PartialStepCountBeforeY = static_cast<const int>( std::ceil( static_cast<float>( PaddingH ) / StrideH ) );
	int PartialStepCountAfterY = static_cast<const int>( std::ceil( ( StrideH * ( std::ceil( static_cast<float>( SrcH ) / StrideH ) - 1 ) - SrcH + PaddingH + 1 ) / StrideH ) );
	// For cases when filter window smaller than source image we may have situation where 
	// PartialStepCountBefore and PartialStepCountAfter will overlap.
	int CentralPartWidth = ResW - PartialStepCountBeforeX - PartialStepCountAfterX;
	int CentralPartHeight = ResH - PartialStepCountBeforeY - PartialStepCountAfterY;

	std::array<int, 9> windowOffsets;
	if( CentralPartHeight >= 0 && CentralPartWidth >= 0 ) {
		windowOffsets = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	} else {
		// Correct PartialStepCounts
		PartialStepCountBeforeX += CentralPartWidth < 0 ? CentralPartWidth : 0;
		PartialStepCountAfterX += CentralPartWidth < 0 ? CentralPartWidth : 0;
		PartialStepCountBeforeY += CentralPartHeight < 0 ? CentralPartHeight : 0;
		PartialStepCountAfterY += CentralPartHeight < 0 ? CentralPartHeight : 0;

		if( CentralPartWidth < 0 ) {
			if( CentralPartHeight < 0 ) {
				windowOffsets = { 0, 11, 2, 12, 4, 9, 6, 10, 15 };
			} else {
				windowOffsets = { 0, 11, 2, 12, 4, 9, 6, 10, 14 };
			}
		} else {
			windowOffsets = { 0, 11, 2, 12, 4, 9, 6, 10, 13 };

		}
		CentralPartWidth = std::min( ResW, std::abs( CentralPartWidth ) );
		CentralPartHeight = std::min( ResH, std::abs( CentralPartHeight ) );
	}

	// FilterH == FilterW == 3
	const int srcXOffset = 0 + ( DilationW - PaddingW );
	const int srcYOffset = 0 + ( DilationH - PaddingH );
	const float* realSrcStart = src + srcXOffset * ChCnt;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int ryStart;
		int ryCount;
		if( OmpGetTaskIndexAndCount( ResH, ryStart, ryCount ) ) {

			// Iterate through result, left->right, top->bottom
			const int currentRH = min( ResH, ryStart + ryCount );
			int ry = ryStart;

			int ryEnd = min( PartialStepCountBeforeY, currentRH );
			while( ry < ryEnd ) {
				// Top part of image
				const float* srcPtr = realSrcStart + ( srcYOffset + ry ) * SrcYStep;
				float* resPtr = res + ry * ResLineStride;
				bool useNarrowProcessing = ( ryEnd ) - ry >= NarrowBatchProcessSize.Height;

				processConvolutionLoop( PartialStepCountBeforeX, useNarrowProcessing, srcPtr, resPtr, windowOffsets[0] );
				processConvolutionLoop( CentralPartWidth, useNarrowProcessing, srcPtr, resPtr, windowOffsets[1] );
				processConvolutionLoop( PartialStepCountAfterX, useNarrowProcessing, srcPtr, resPtr, windowOffsets[2] );
				ry += useNarrowProcessing ? NarrowBatchProcessSize.Height : WideBatchProcessSize.Height;
			}

			ryEnd = min( ResH - PartialStepCountAfterY, currentRH );
			while( ry < ryEnd ) {
				// Middle part of image
				const float* srcPtr = realSrcStart + ( srcYOffset + ry ) * SrcYStep;
				float* resPtr = res + ry * ResLineStride;
				bool useNarrowProcessing = (ryEnd)-ry >= NarrowBatchProcessSize.Height;

				processConvolutionLoop( PartialStepCountBeforeX, useNarrowProcessing, srcPtr, resPtr, windowOffsets[7] );
				processConvolutionLoop( CentralPartWidth, useNarrowProcessing, srcPtr, resPtr, windowOffsets[8] );
				processConvolutionLoop( PartialStepCountAfterX, useNarrowProcessing, srcPtr, resPtr, windowOffsets[3] );
				ry += useNarrowProcessing ? NarrowBatchProcessSize.Height : WideBatchProcessSize.Height;
			}

			ryEnd = min( ResH, currentRH );
			while( ry < ryEnd ) {
				// Bottom part of image
				const float* srcPtr = realSrcStart + ( srcYOffset + ry ) * SrcYStep;
				float* resPtr = res + ry * ResLineStride;
				bool useNarrowProcessing = (ryEnd)-ry >= NarrowBatchProcessSize.Height;

				processConvolutionLoop( PartialStepCountBeforeX, useNarrowProcessing, srcPtr, resPtr, windowOffsets[6] );
				processConvolutionLoop( CentralPartWidth, useNarrowProcessing, srcPtr, resPtr, windowOffsets[5] );
				processConvolutionLoop( PartialStepCountAfterX, useNarrowProcessing, srcPtr, resPtr, windowOffsets[4] );
				ry += useNarrowProcessing ? NarrowBatchProcessSize.Height : WideBatchProcessSize.Height;
			}
		}
	}
}

template<int FltCnt>
inline typename CBlobConvolution<FltCnt>::CSize CBlobConvolution<FltCnt>::getNarrowBatchProcessSize()
{
	// Disable narrow processing by default
	return { INT_MAX, INT_MAX };
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::singleProcessNarrow( const float*, float*,  int )
{
	// dummy function
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::processConvolutionLoop( int rxSize, bool useNarrowProcessing, const float*& srcPtr, float*& resPtr, int windowIndex )
{
	const int batchStep = useNarrowProcessing ? NarrowBatchProcessSize.Width : WideBatchProcessSize.Width;
	for( ; rxSize >= batchStep; rxSize -= batchStep ) {
		batchProcess( srcPtr, resPtr, windowIndex, useNarrowProcessing );
		srcPtr += batchStep * SrcXStep;
		resPtr += batchStep * FltCnt;
	}

	if( useNarrowProcessing ) {
		for( ; rxSize > 0; rxSize-- ) {
			singleProcessNarrow( srcPtr, resPtr, windowIndex );
			srcPtr += SrcXStep;
			resPtr += FltCnt;
		}
	} else {
		for( ; rxSize > 0; rxSize-- ) {
			singleProcess( srcPtr, resPtr, windowIndex );
			srcPtr += SrcXStep;
			resPtr += FltCnt;
		}
	}
}

template<int FltCnt>
const float* CBlobConvolution<FltCnt>::rearrangeFilter( const float* filterData, CFloatHandleStackVar& filterTempBuffer )
{
	// Rearrange filter data.
	// Initial packing:
	// Filter[0] Pixel[0] Channel[0-23]
	// Filter[0] Pixel[1] Channel[0-23]
	// ...
	// Filter[0] Pixel[8] Channel[0-23]
	// Filter[1] Pixel[0] Channel[0-23]
	// ...
	// Filter[23] Pixel[8] Channel[0-23]
	//
	// 1. Result packing for case when FltCnt == FltCntM8 (for example: 24):
	// Pixel[0] Channel[0] Filter[0-23]
	// Pixel[0] Channel[1] Filter[0-23]
	// ...
	// Pixel[0] Channel[23] Filter[0-23]
	// Pixel[1] Channel[0] Filter[0-23]
	// ...
	// Pixel[8] Channel[23] Filter[0-23]
	//
	// 2. Result packing for case when FltCnt != FltCntM8 (for example: 18):
	// Pixel[0] Channel[0] Filter[0-17] Filter[0-5]
	// Pixel[0] Channel[1] Filter[0-23] Filter[0-5]
	// ...
	// Pixel[0] Channel[23] Filter[0-23] Filter[0-5]
	// Pixel[1] Channel[0] Filter[0-23] Filter[0-5]
	// ...
	// Pixel[8] Channel[23] Filter[0-23] Filter[0-5]

	float* resFilterStartPtr = static_cast<float*>( mathEngine->GetBuffer( filterTempBuffer.GetHandle(), 0, filterTempBuffer.Size() * sizeof( float ) ) );
	float* resFilter = resFilterStartPtr;
	ASSERT_EXPR( reinterpret_cast<uintptr_t>( resFilter ) % AvxAlignment == 0 );
	for( int y = 0; y < FltH; y++ ) {
		for( int x = 0; x < FltW; x++ ) {
			for( int c = 0; c < ChCnt; c++ ) {
				const float* srcFilter = filterData + ( x + y * FltW ) * ChCnt + c;
				for( int f = 0; f < FltCnt; f++ ) {
					*resFilter++ = *srcFilter;
					srcFilter += FltW * FltH * ChCnt;
				}
				if( FltCntM8 != FltCnt ) {
					srcFilter = filterData + ( x + y * FltW ) * ChCnt + c;
					for( int f = 0; f < FltCntM8 - FltCnt; f++ ) {
						*resFilter++ = *srcFilter;
						srcFilter += FltW * FltH * ChCnt;
					}
				}
			}
		}
	}

	return resFilterStartPtr;
}

template<int FltCnt>
const float* CBlobConvolution<FltCnt>::rearrangeFreeTerm( const float* freeTermData, CFloatHandleStackVar& freeTermTempBuffer )
{
	if( freeTermData == nullptr ) {
		return nullptr;
	}

	float* resFreeTermStartPtr = static_cast<float*>( mathEngine->GetBuffer( freeTermTempBuffer.GetHandle(), 0, freeTermTempBuffer.Size() * sizeof( float ) ) );
	float* resFreeTerm = resFreeTermStartPtr;
	ASSERT_EXPR( reinterpret_cast<uintptr_t>( resFreeTerm ) % AvxAlignment == 0 );

	for( int f = 0; f < FltCnt; f++ ) {
		*resFreeTerm++ = *freeTermData++;
	}
	if( FltCnt != FltCntM8 ) {
		freeTermData -= FltCnt;
		for( int f = 0; f < FltCnt; f++ ) {
			*resFreeTerm++ = *freeTermData++;
		}
	}
	return resFreeTermStartPtr;
}

// Filter window offset
// 0 1 2
// 3 4 5
// 6 7 8
template<int FltCnt>
const std::array<std::vector<int>, 16> CBlobConvolution<FltCnt>::fillSrcPixelOffset()
{
	const int SrcLineStride = SrcW * ChCnt;
	const int SrcYDilation = DilationH * SrcLineStride;
	const int SrcXDilation = DilationW * ChCnt;
	return {
		std::vector<int>{ 0, SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation }, // 0) 4 5 7 8
		std::vector<int>{ -SrcXDilation, 0, SrcXDilation, SrcYDilation - SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation }, // 1) 3 4 5 6 7 8
		std::vector<int>{ -SrcXDilation, 0, SrcYDilation - SrcXDilation, SrcYDilation }, // 2) 3 4 6 7
		std::vector<int>{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcXDilation, 0, SrcYDilation - SrcXDilation, SrcYDilation }, // 3) 0 1 3 4 6 7
		std::vector<int>{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcXDilation, 0 }, // 4) 0 1 3 4
		std::vector<int>{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcYDilation + SrcXDilation, -SrcXDilation, 0, SrcXDilation }, // 5) 0 1 2 3 4 5
		std::vector<int>{ -SrcYDilation, -SrcYDilation + SrcXDilation, 0, SrcXDilation }, // 6) 1 2 4 5
		std::vector<int>{ -SrcYDilation, -SrcYDilation + SrcXDilation, 0, SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation }, // 7) 1 2 4 5 7 8
		std::vector<int>{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcYDilation + SrcXDilation,
			-SrcXDilation, 0, SrcXDilation,
			SrcYDilation - SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation}, // 8) whole filter

		std::vector<int>{ -SrcYDilation, 0 }, // 9) 1 4
		std::vector<int>{ 0, SrcXDilation }, // 10) 4 5
		std::vector<int>{ 0, SrcYDilation }, // 11) 4 7
		std::vector<int>{ -SrcXDilation, 0 }, // 12) 3 4
		std::vector<int>{ -SrcXDilation, 0, SrcXDilation }, // 13) 3 4 5
		std::vector<int>{ -SrcYDilation, 0, SrcYDilation }, // 14) 1 4 7
		std::vector<int>{ 0 } // 15) 4
	};
}

template<int FltCnt>
const std::array<std::vector<int>, 16>  CBlobConvolution<FltCnt>::fillFltPixelOffset()
{
	return {
		std::vector<int>{ 4 * ChCnt * FltCntM8, 5 * ChCnt * FltCntM8, 7 * ChCnt * FltCntM8, 8 * ChCnt * FltCntM8 }, // 0) 4 5 7 8
		std::vector<int>{ 3 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 5 * ChCnt * FltCntM8, 6 * ChCnt * FltCntM8, 7 * ChCnt * FltCntM8, 8 * ChCnt * FltCntM8 }, // 1) 3 4 5 6 7 8
		std::vector<int>{ 3 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 6 * ChCnt * FltCntM8, 7 * ChCnt * FltCntM8 }, // 2) 3 4 6 7
		std::vector<int>{ 0 * ChCnt * FltCntM8, 1 * ChCnt * FltCntM8, 3 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 6 * ChCnt * FltCntM8, 7 * ChCnt * FltCntM8 }, // 3) 0 1 3 4 6 7
		std::vector<int>{ 0 * ChCnt * FltCntM8, 1 * ChCnt * FltCntM8, 3 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8 }, // 4) 0 1 3 4
		std::vector<int>{ 0 * ChCnt * FltCntM8, 1 * ChCnt * FltCntM8, 2 * ChCnt * FltCntM8, 3 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 5 * ChCnt * FltCntM8 }, // 5) 0 1 2 3 4 5
		std::vector<int>{ 1 * ChCnt * FltCntM8, 2 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 5 * ChCnt * FltCntM8 }, // 6) 1 2 4 5
		std::vector<int>{ 1 * ChCnt * FltCntM8, 2 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 5 * ChCnt * FltCntM8, 7 * ChCnt * FltCntM8, 8 * ChCnt * FltCntM8 }, // 7) 1 2 4 5 7 8
		std::vector<int>{ 0 * ChCnt * FltCntM8, 1 * ChCnt * FltCntM8, 2 * ChCnt * FltCntM8,
			3 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 5 * ChCnt * FltCntM8,
			6 * ChCnt * FltCntM8, 7 * ChCnt * FltCntM8, 8 * ChCnt * FltCntM8 }, // 8) whole filter

		
		std::vector<int>{ 1 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8 }, // 9) 1 4
		std::vector<int>{ 4 * ChCnt * FltCntM8, 5 * ChCnt * FltCntM8 }, // 10) 4 5
		std::vector<int>{ 4 * ChCnt * FltCntM8, 7 * ChCnt * FltCntM8 }, // 11) 4 7
		std::vector<int>{ 3 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8 }, // 12) 3 4
		std::vector<int>{ 3 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 5 * ChCnt * FltCntM8 }, // 13) 3 4 5
		std::vector<int>{ 1 * ChCnt * FltCntM8, 4 * ChCnt * FltCntM8, 7 * ChCnt * FltCntM8 }, // 14) 1 4 7
		std::vector<int>{ 4 * ChCnt * FltCntM8 } // 15) 4
	};
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::rotateLeft6( __m256& y0, __m256& y1, __m256& y2 )
{   //   y0        y1        y2
	// 0 1 2 3 - 4 5 6 7 - 8 0 1 2
	// 3 4 5 6 - 7 8 0 1 - 2 3 4 5
	// 6 7 8 0 - 1 2 3 4 - 5 6 7 8

	// before: 0 1 2 3
	// after:  2 3 0 1
	__m256 yt0 = _mm256_permute2f128_ps( y0, y0, _MM_SHUFFLE( 0, 0, 0, 1 ) );
	// before: 4 5 6 7
	// after:  6 7 4 5
	__m256 yt1 = _mm256_permute2f128_ps( y1, y1, _MM_SHUFFLE( 0, 0, 0, 1 ) );
	// before: 6 7 4 5|8 0 1 2
	// after:      7 8 5 1
	__m256 yt2 = _mm256_shuffle_ps( yt1, y2, _MM_SHUFFLE( 1, 0, 3, 2 ) );
	// before: 2 3 0 1|6 7 4 5
	// after:      2 3 4 5
	y2 = _mm256_blend_ps( yt0, yt1, 0xf0 );
	// before: 2 3 4 5|4 5 6 7
	// after:      3 4 5 6
	y0 = _mm256_shuffle_ps( y2, y1, _MM_SHUFFLE( 1, 0, 3, 2 ) );
	// before: 7 8 5 1|2 3 0 1
	// after:      7 8 0 1
	y1 = _mm256_blend_ps( yt2, yt0, 0xf0 );
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::rotateLeft2( __m256& y )
{
	// 0 1 2 0
	// 1 2 0 1
	// 2 0 1 2
	// before: 0 1 2 0
	// after:  2 0 0 1
	__m256 yt = _mm256_permute2f128_ps( y, y, _MM_SHUFFLE( 0, 0, 0, 1 ) );
	// before: 0 1 2 0|2 0 0 1
	// after:      1 2 0 0
	y = _mm256_shuffle_ps( y, yt, _MM_SHUFFLE( 1, 0, 3, 2 ) );
	// before:  1 2 0 0|2 0 0 1
	// after:      1 2 0 1
	y = _mm256_blend_ps( y, yt, 0xf0 );
}

} // namespace NeoML

// Class specializations
#include <BlobConvolutionImpl.h>
