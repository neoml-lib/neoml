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
#include <algorithm>
#include <utility>

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
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth, int resObjCnt );
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
	const int ResObjCnt;

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

	// When we move filter window over the source image we have different combination of intersection this window with
	// source image. Filters window moves left to right and up to bottom, "PixelOffsetResStepsWidth..." - is number 
	// of steps which window moved till its intersection with source image changed.We will calculate steps over width and heigh.
	std::vector<int> PixelOffsetResStepsWidthX;
	std::vector<int> PixelOffsetResStepsWidthY;

	// Choose proper pixels in source and filter:
	// 0  1  2
	// 3  4  5
	// 6  7  8 (example for 3x3)
	// Offset is relative to central pixel of source window
	std::vector<std::vector<int>> SrcPixelsOffset;
	// Offset is relative to center pixel of filter window
	std::vector<std::vector<int>>  FltPixelsOffset;
	// In some cases when the width of the image is nearly equals to the width of optimized batch processing window,
	// we may faced to situation ( when dilation is higth ) when no one optimized batch ptocessing can be
	// applied. For such cases we will use optimized batch processing with narrower window but height greater then one.
	const CSize NarrowBatchProcessSize;
	const CSize WideBatchProcessSize;

	// Initialize NarrowBatchProcessSize and WideBatchProcessSize
	CSize getNarrowBatchProcessSize();
	CSize getWideBatchProcessSize();

	// Process one line of image. In case of narrow processing we will step through several lines.
	void processConvolutionLoop( int rxSize, bool useNarrowProcessing, const float*& srcPtr, float*& resPtr, size_t windowIndex );

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
	// Process convolution for single result pixel.
	void singleProcess( const float* srcPtr, float* resPtr, size_t windowIndex );
	void singleProcessNarrow( const float* srcPtr, float* resPtr, size_t windowIndex );

	// Rearrange filter and fill 'Filter' and 'FreeTerm' members.
	const float* rearrangeFilter( const float* filterData, CFloatHandleStackVar& Filter );
	const float* rearrangeFreeTerm( const float* freeTermData, CFloatHandleStackVar& FreeTerm );
	// Function calculates offsets of center of filter window over the source image, where intersection over
	// them is changed. This function helps to calculate further PixelOffsetResStepsWidthX/Y, SrcPixelsOffset and FltPixelsOffset.
	// Src (source), F(filter), D(dilation), S(stride) and P(padding) linear dimention by X or Y axis.
	std::vector<int> getPixelOffsetSrcSteps( int SrcDim, int FDim, int DDim, int SDim, int PDim );

	// Initialize PixelOffsetResStepsX, PixelOffsetResStepsY, SrcPixelsOffset and FltPixelsOffset
	void fillPixelOffset();

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
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth, int resObjCnt );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CBlobConvolutionFabric::IsBlobConvolutionAvailable( int FltCnt, int FltH, int FltW )
{
	if( FltH % 2 == 0 || FltW % 2 == 0 ) {
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
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth, int resObjCnt )
{
	switch( filterCount ) {
	case 24:
		return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<24>( mathEngine,
			channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
			paddingHeight, paddingWidth, strideHeight, strideWidth,
			dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
	case 18:
		return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<18>( mathEngine,
			channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
			paddingHeight, paddingWidth, strideHeight, strideWidth,
			dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
	case 6:
		return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<6>( mathEngine,
			channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
			paddingHeight, paddingWidth, strideHeight, strideWidth,
			dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
	default:
		return nullptr;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int FltCnt>
CBlobConvolution<FltCnt>::CBlobConvolution( IMathEngine* _mathEngine, int channelCount, int filterHeight, int filterWidth,
	int sourceHeight, int sourceWidth, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth, int resObjCnt ) :
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
	ResObjCnt( resObjCnt ),
	src( nullptr ),
	flt( nullptr ),
	freeTerm( nullptr ),
	res( nullptr ),
	SrcLineStride( SrcW* ChCnt ),
	SrcXStep( StrideW* ChCnt ),
	SrcYStep( StrideH* SrcLineStride ),
	SrcXDilation( DilationW* ChCnt ),
	SrcYDilation( DilationH* SrcLineStride ),
	SrcXWindowSize( FltW* SrcXDilation ),
	ResLineStride( ResW* FltCnt ),
	NarrowBatchProcessSize( getNarrowBatchProcessSize() ),
	WideBatchProcessSize( getWideBatchProcessSize() )
{
	// // Initialize PixelOffsetResStepsX, PixelOffsetResStepsY, SrcPixelsOffset and FltPixelsOffset
	fillPixelOffset();
}

template<int FltCnt>
void CBlobConvolution<FltCnt>::ProcessConvolution( int threadCount,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
{
	CFloatHandleStackVar filterTempBuffer( *mathEngine, FltW * FltH * FltCntM8 * ChCnt );
	CFloatHandleStackVar freeTermTempBuffer( *mathEngine, FltCntM8 );

	src = sourceData;
	// Filter offset also are calculated from center
	flt = rearrangeFilter( filterData, filterTempBuffer ) + ( FltW * FltH ) / 2 * ChCnt * FltCntM8;
	freeTerm = rearrangeFreeTerm( freeTermData, freeTermTempBuffer );
	res = resultData;

	const int SrcObjSize = SrcW * SrcH * ChCnt;
	const int ResObjSize = ResW * ResH * FltCnt;
	const int ResRowCount = ResObjCnt * ResH;
	const int curThreadCount = IsOmpRelevant( ResRowCount, ResRowCount * ResW * FltCnt * FltW * FltH * ChCnt ) ? threadCount : 1;

	// Coordinates of the most top and left position of the center of the filter over the source image.
	const int srcXOffset = FltW / 2 * DilationW - PaddingW;
	const int srcYOffset = FltH / 2 * DilationH - PaddingH;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		// Index of row in whole result array
		int rowIdx;
		// Count of rows for current thread
		int rowCount;
		if( OmpGetTaskIndexAndCount( ResRowCount, rowIdx, rowCount ) ) {

			while( rowCount > 0 ) {
				// Index of result image in output batch
				int resIdx = rowIdx / ResH;
				// Offset in current result image 
				int ryStart = rowIdx % ResH;
				// Number of rows for processing ( or number of rows till the end of current result image ).
				int ryCount = min( ResH - ryStart, rowCount );
				rowIdx += ryCount;
				rowCount -= ryCount;

				// Pointers to src and res for current thread
				const float* realSrcStart = src + resIdx * SrcObjSize + srcXOffset * ChCnt;
				float* realResStart = res + resIdx * ResObjSize;

				// Iterate through result, left->right, top->bottom
				const int currentRH = min( ResH, ryStart + ryCount );
				int ry = ryStart;
				int yStep = ryStart;

				// Iterate through all combination of intersections
				for( int yStepIdx = 0; yStepIdx < PixelOffsetResStepsWidthY.size(); yStepIdx++ ) {

					// Last index of res for current intersection.
					yStep += PixelOffsetResStepsWidthY[yStepIdx];
					// Process up to current step or up to and of current butch 
					int ryEnd = min( yStep, currentRH );
					for( ; ry < ryEnd; ) {
						const float* srcPtr = realSrcStart + srcYOffset * SrcLineStride + ry * SrcYStep;
						float* resPtr = realResStart + ry * ResLineStride;
						bool useNarrowProcessing = ryEnd - ry >= NarrowBatchProcessSize.Height;

						size_t pixelsOffsetIdx = yStepIdx * PixelOffsetResStepsWidthX.size();

						for( const auto& xStep : PixelOffsetResStepsWidthX ) {
							processConvolutionLoop( xStep, useNarrowProcessing, srcPtr, resPtr, pixelsOffsetIdx );
							pixelsOffsetIdx++;
						}
						ry += useNarrowProcessing ? NarrowBatchProcessSize.Height : WideBatchProcessSize.Height;
					}
				}
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
inline void CBlobConvolution<FltCnt>::singleProcessNarrow( const float*, float*, size_t )
{
	// dummy function
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::processConvolutionLoop( int rxSize, bool useNarrowProcessing, const float*& srcPtr, float*& resPtr, size_t windowIndex )
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

	float* resFilterStartPtr = static_cast< float* >( mathEngine->GetBuffer( filterTempBuffer.GetHandle(), 0, filterTempBuffer.Size() * sizeof( float ), false ) );
	float* resFilter = resFilterStartPtr;
	ASSERT_EXPR( reinterpret_cast< uintptr_t >( resFilter ) % AvxAlignment == 0 );
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

	float* resFreeTermStartPtr = static_cast< float* >( mathEngine->GetBuffer( freeTermTempBuffer.GetHandle(), 0, freeTermTempBuffer.Size() * sizeof( float ), false ) );
	float* resFreeTerm = resFreeTermStartPtr;
	ASSERT_EXPR( reinterpret_cast< uintptr_t >( resFreeTerm ) % AvxAlignment == 0 );

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

template<int FltCnt>
std::vector<int> CBlobConvolution<FltCnt>::getPixelOffsetSrcSteps( int srcDim, int fDim, int dDim, int sDim, int pDim )
{
	vector<int> ret( fDim );
	const int halfFDim = fDim / 2;

	// First offset of center of the filter window (Take in consideration paddings)
	const int firstOffset = halfFDim * dDim - pDim;
	const int lastSrcPixelIdx = srcDim - 1;
	// Last offset of center of the filter window (Take in consideration paddings)
	// (lastSrcPixelIdx - 2 * firstOffset) - width of window
	const int lastOffset = firstOffset + ( lastSrcPixelIdx - 2 * firstOffset ) / sDim * sDim;
	ret[0] = firstOffset;

	for( int i = 1; i <= halfFDim; i++ ) {
		// up to middle
		ret[i] = firstOffset + ( i * dDim - firstOffset + sDim - 1 ) / sDim * sDim;
	}


	for( int i = fDim - 1, j = 1; i > fDim / 2; i--, j++ ) {
		// from last to next to middle
		ret[i] = lastOffset - ( lastOffset + j * dDim - lastSrcPixelIdx ) / sDim * sDim + sDim;
	}

	sort( ret.begin(), ret.end() );

	// Remove out of range and repeated items
	auto start = ret.begin();
	while( *start < 0 ) start++;
	auto end = start;
	auto tempIt = end + 1;
	int lastSrcDim = srcDim - firstOffset - 1;
	while( tempIt != ret.end() && *tempIt <= lastSrcDim ) {
		if( *tempIt != *end ) {
			int temp = *tempIt;
			*( ++end ) = temp;
		}
		tempIt++;
	}
	end++;

	return vector<int>( start, end );
}

template<int FltCnt>
void CBlobConvolution<FltCnt>::fillPixelOffset()
{
	using namespace std;
	vector<int> pixelOffsetSrcStepsX = getPixelOffsetSrcSteps( SrcW, FltW, DilationW, StrideW, PaddingW );
	vector<int> pixelOffsetSrcStepsY = getPixelOffsetSrcSteps( SrcH, FltH, DilationH, StrideH, PaddingH );

	// Calculate offset on the source image where intersection of filter and image is changed.
	auto getPixelOffsetResStepsWidth = []( const std::vector<int>& pixelOffsetSrcSteps, int srcDim, int fDim, int dDim, int sDim, int pDim )
	{
		vector<int> ret( pixelOffsetSrcSteps.size() );
		const int firstOffset = fDim / 2 * dDim - pDim;
		const int lastSrcPixelIdx = srcDim - 1;
		const int lastOffset = firstOffset + ( lastSrcPixelIdx - 2 * firstOffset ) / sDim * sDim;

		int i = 0;
		for( ; i < ret.size() - 1; i++ ) {
			ret[i] = ( pixelOffsetSrcSteps[i + 1] - pixelOffsetSrcSteps[i] ) / sDim;
		}
		ret[i] = ( lastOffset - pixelOffsetSrcSteps[i] ) / sDim + 1;

		return ret;
	};

	PixelOffsetResStepsWidthX = getPixelOffsetResStepsWidth( pixelOffsetSrcStepsX, SrcW, FltW, DilationW, StrideW, PaddingW );
	PixelOffsetResStepsWidthY = getPixelOffsetResStepsWidth( pixelOffsetSrcStepsY, SrcH, FltH, DilationH, StrideH, PaddingH );

	// Get size of intersection of filter window and source image
	auto getFilterWindowSize = []( const vector<int>& pixelOffsetSrcSteps, int srcDim, int fDim, int dDim ) -> vector<pair<int, int>> {
		// first - count of items in filter from center to top
		// second - count of items in filter from center to bottom
		vector<pair<int, int>> ret( pixelOffsetSrcSteps.size() );
		for( int i = 0; i < pixelOffsetSrcSteps.size(); i++ ) {
			const int halfFDim = fDim / 2;
			ret[i] = make_pair(
				min( pixelOffsetSrcSteps[i] / dDim, halfFDim ),
				min( ( ( srcDim - 1 ) - pixelOffsetSrcSteps[i] ) / dDim, halfFDim ) );
		}
		return ret;
	};

	vector<pair<int, int>> offsetSizeX = getFilterWindowSize( pixelOffsetSrcStepsX, SrcW, FltW, DilationW );
	vector<pair<int, int>> offsetSizeY = getFilterWindowSize( pixelOffsetSrcStepsY, SrcH, FltH, DilationH );

	// Calculate resulted offsets of pixels in window.
	auto fillPixelOffset = [&]( int hStride, int wStride ) ->vector<vector<int>> {
		vector<vector<int>> offsets( offsetSizeX.size() * offsetSizeY.size() );
		auto it = offsets.begin();

		for( const auto& y : offsetSizeY ) {
			for( const auto& x : offsetSizeX ) {
				it->resize( ( x.first + x.second + 1 ) * ( y.first + y.second + 1 ) );
				auto it_offt = it->begin();
				for( int i = -y.first; i <= y.second; i++ ) {
					for( int j = -x.first; j <= x.second; j++ ) {
						*it_offt++ = i * hStride + j * wStride;
					}
				}
				it++;
			}
		}
		return offsets;
	};

	SrcPixelsOffset = fillPixelOffset( SrcYDilation, SrcXDilation );
	FltPixelsOffset = fillPixelOffset( FltW * ChCnt * FltCntM8, ChCnt * FltCntM8 );

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
