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
#include <climits>

namespace NeoML {

class CBlobConvolutionBase {
public:
	virtual ~CBlobConvolutionBase() = default;
	virtual void ProcessConvolution( int threadCount ) = 0;

	// We should specify maximum available values of C, FC, FH and FW in order to allocate Filter and FreeTerm variables on stack.
	static constexpr int Cmax = 24;
	static constexpr int FCmax = 24;
	static constexpr int FHmax = 3;
	static constexpr int FWmax = 3;
};

template<int FC>
class CBlobConvolution : public CBlobConvolutionBase {
public:
	CBlobConvolution(
		int channelCount, int filterHeight, int filterWidth,
		int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
		const float* sourceData, const float* filterData, const float* freeTermData, float* resultData );
	~CBlobConvolution() override = default;

	void ProcessConvolution( int threadCount ) override;

private:
	struct CSize {
		int Height;
		int Width;
	};
	const int C;
	const int FH;
	const int FW;
	const int SrcH;
	const int SrcW;
	const int SH;
	const int SW;
	const int DH;
	const int DW;
	const int RH;
	const int RW;

	const float* const src;
	const float* const flt;
	const float* const freeTerm;
	float* const dst;

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
	const int DstLineStride;

	// For some cases we will use FC, rounded up to nearest integer multiple of 8
	static constexpr int FCm8 = ( FC + 8 - 1 ) / 8 * 8;
	// Filter should be alligned to 16 bytes
	static constexpr size_t FileterSize = FWmax * FHmax * FCmax * Cmax ;
	static constexpr size_t AvxAlignment = 32;
	float Filter[FileterSize + AvxAlignment];

	// Free term should be alligned to 16 bytes
	float FreeTerm[FCm8 + AvxAlignment];

	// Choose proper pixels in source and filter:
	// 0  1  2
	// 3  4  5
	// 6  7  8
	// Offset is relative to central pixel of source window
	const std::array<std::vector<int>, 8> SrcPixelsOffset;
	// Offset is relative to top left pixel of filter window
	const std::array<std::vector<int>, 8>  FltPixelsOffset;
	// In some cases when the width of the image is nearly equals to the width of optimized batch processing window,
	// we may faced to situation ( when dilation is higth ) when no one optimized batch ptocessing can be
	// applied. For such cases we will use optimized batch processing with narrower window but height greater then one.
	const CSize NarrowBatchProcessSize;
	const CSize WideBatchProcessSize;

	// Initialize NarrowBatchProcessSize and WideBatchProcessSize
	CSize getNarrowBatchProcessSize();
	CSize getWideBatchProcessSize();

	// Process one line of image. In case of narrow processing we will step through several lines.
	void processConvolutionLoop( int rxSize, bool useNarrowProcessing, const float*& srcPtr, float*& dstPtr, int windowIndex = -1 );

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
	void batchProcess( const float* srcPtr, float* dstPtr, int windowIndex, bool useNarrowProcessing );
	// Process convolution for single result pixel.
	void singleProcess( const float* srcPtr, float* dstPtr, int windowIndex );
	void singleProcessNarrow( const float* srcPtr, float* dstPtr, int windowIndex );

	// Rearrange filter and fill 'Filter' and 'FreeTerm' members.
	const float* rearrangeFileter( const float* filterData );
	const float* rearrangeFreeTerm( const float* freeTermData );
	const std::array<std::vector<int>, 8> fillSrcPixelOffset();
	const std::array<std::vector<int>, 8>  fillFltPixelOffset();

	// Circular rotation of three ymm registers to the left, step equals to six floats.
	static void RotateLeft6( __m256& y0, __m256& y1, __m256& y2 );
	// Circular rotation of three ymm registers to the left, step equals to two floats.
	static void RotateLeft2( __m256& y );

};

class CBlobConvolutionFabric {
public:
	static bool IsBlobConvolutionAvailable( int FC, int C, int FH, int FW );
	static std::unique_ptr<CBlobConvolutionBase> GetProperInstance( int FC,
		int channelCount, int filterHeight, int filterWidth,
		int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
		const float* sourceData, const float* filterData, const float* freeTermData, float* resultData );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CBlobConvolutionFabric::IsBlobConvolutionAvailable( int FC, int C, int FH, int FW )
{
	if( FC * C * FH * FW >
		CBlobConvolutionBase::FCmax * CBlobConvolutionBase::Cmax * CBlobConvolutionBase::FHmax * CBlobConvolutionBase::FWmax ) {
		return false;
	}

	if( FC == 24 ||
		FC == 18 ||
		FC == 6 ) {
		return true;
	}
	return false;
}

std::unique_ptr<CBlobConvolutionBase> CBlobConvolutionFabric::GetProperInstance( int FC,
	int channelCount, int filterHeight, int filterWidth,
	int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
{
	switch( FC ) {
		case 24:
			return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<24>(
				channelCount, filterHeight, filterWidth,
				sourceHeight, sourceWidth, strideHeight, strideWidth,
				dilationHeight, dilationWidth, resultHeight, resultWidth,
				sourceData, filterData, freeTermData, resultData ) );
		case 18:
			return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<18>(
				channelCount, filterHeight, filterWidth,
				sourceHeight, sourceWidth, strideHeight, strideWidth,
				dilationHeight, dilationWidth, resultHeight, resultWidth,
				sourceData, filterData, freeTermData, resultData ) );
		case 6:
			return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<6>(
				channelCount, filterHeight, filterWidth,
				sourceHeight, sourceWidth, strideHeight, strideWidth,
				dilationHeight, dilationWidth, resultHeight, resultWidth,
				sourceData, filterData, freeTermData, resultData ) );
		default:
			return nullptr;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int FC>
CBlobConvolution<FC>::CBlobConvolution( int channelCount, int filterHeight, int filterWidth,
	int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) :
	C( channelCount ),
	FH( filterHeight ),
	FW( filterWidth ),
	SrcH( sourceHeight ),
	SrcW( sourceWidth ),
	SH( strideHeight ),
	SW( strideWidth ),
	DH( dilationHeight ),
	DW( dilationWidth ),
	RH( resultHeight ),
	RW( resultWidth ),
	src( sourceData ),
	flt( rearrangeFileter( filterData ) ),
	freeTerm( rearrangeFreeTerm( freeTermData ) ),
	dst( resultData ),
	SrcLineStride( SrcW * C ),
	SrcXStep( SW * C ),
	SrcYStep( SH * SrcLineStride ),
	SrcXDilation( DW * C ),
	SrcYDilation( DH * SrcLineStride ),
	SrcXWindowSize( FW * SrcXDilation ),
	DstLineStride( RW * FC ),
	SrcPixelsOffset( fillSrcPixelOffset() ),
	FltPixelsOffset( fillFltPixelOffset() ),
	NarrowBatchProcessSize( getNarrowBatchProcessSize() ),
	WideBatchProcessSize( getWideBatchProcessSize() )
{

}

template<int FC>
void CBlobConvolution<FC>::ProcessConvolution( int threadCount )
{
	const int curThreadCount = IsOmpRelevant( RH, RH * RW * FC * FW * FH * C ) ? threadCount : 1;

	// Number of steps for each side of image, where filter is applied partially
	const int PartialStepCountBeforeX = static_cast<const int>( std::ceil( static_cast<float>( DW ) / SW ) );
	const int PartialStepCountAfterX = static_cast<const int>( std::ceil( ( SW * ( std::ceil( static_cast<float>( SrcW ) / SW ) - 1 ) - SrcW + DW + 1 ) / SW ) );
	const int PartialStepCountBeforeY = static_cast<const int>( std::ceil( static_cast<float>( DH ) / SH ) );
	const int PartialStepCountAfterY = static_cast<const int>( std::ceil( ( SH * ( std::ceil( static_cast<float>( SrcH ) / SH ) - 1 ) - SrcH + DH + 1 ) / SH ) );
	const int CentralPartWidth = RW - PartialStepCountBeforeX - PartialStepCountAfterX;


	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int yStart;
		int yCount;
		if( OmpGetTaskIndexAndCount( RH, yStart, yCount ) ) {

			// Iterate through result, left->right, top->bottom
			const int currentRH = min( RH, yStart + yCount );
			int ry = yStart;

			int ryEnd = min( PartialStepCountBeforeY, currentRH );
			while( ry < ryEnd ) {
				// Top part of image
				const float* srcPtr = src + ry * SrcYStep;
				float* dstPtr = dst + ry * DstLineStride;
				bool useNarrowProcessing = ( ryEnd ) - ry >= NarrowBatchProcessSize.Height;

				processConvolutionLoop( PartialStepCountBeforeX, useNarrowProcessing, srcPtr, dstPtr, 0 );
				processConvolutionLoop( CentralPartWidth, useNarrowProcessing, srcPtr, dstPtr, 1 );
				processConvolutionLoop( PartialStepCountAfterX, useNarrowProcessing, srcPtr, dstPtr, 2 );
				ry += useNarrowProcessing ? NarrowBatchProcessSize.Height : WideBatchProcessSize.Height;
			}


			ryEnd = min( RH - PartialStepCountAfterY, currentRH );
			while( ry < ryEnd ) {
				// Middle part of image
				const float* srcPtr = src + ry * SrcYStep;
				float* dstPtr = dst + ry * DstLineStride;
				bool useNarrowProcessing = (ryEnd)-ry >= NarrowBatchProcessSize.Height;

				processConvolutionLoop( PartialStepCountBeforeX, useNarrowProcessing, srcPtr, dstPtr, 7 );

				// Move to the top left pixel of window from central one
				srcPtr -= ( SrcYDilation + SrcXDilation );
				processConvolutionLoop( CentralPartWidth, useNarrowProcessing, srcPtr, dstPtr );

				// Move back to the central pixel again
				srcPtr += ( SrcYDilation + SrcXDilation );
				processConvolutionLoop( PartialStepCountAfterX, useNarrowProcessing, srcPtr, dstPtr, 3 );
				ry += useNarrowProcessing ? NarrowBatchProcessSize.Height : WideBatchProcessSize.Height;
			}

			ryEnd = min( RH, currentRH );
			while( ry < ryEnd ) {
				// Bottom part of image
				const float* srcPtr = src + ry * SrcYStep;
				float* dstPtr = dst + ry * DstLineStride;
				bool useNarrowProcessing = (ryEnd)-ry >= NarrowBatchProcessSize.Height;

				processConvolutionLoop( PartialStepCountBeforeX, useNarrowProcessing, srcPtr, dstPtr, 6 );
				processConvolutionLoop( CentralPartWidth, useNarrowProcessing, srcPtr, dstPtr, 5 );
				processConvolutionLoop( PartialStepCountAfterX, useNarrowProcessing, srcPtr, dstPtr, 4 );
				ry += useNarrowProcessing ? NarrowBatchProcessSize.Height : WideBatchProcessSize.Height;
			}
		}
	}
}

template<int FC>
inline typename CBlobConvolution<FC>::CSize CBlobConvolution<FC>::getNarrowBatchProcessSize()
{
	// Disable narrow processing by default
	return { INT_MAX, INT_MAX };
}

template<int FC>
inline void CBlobConvolution<FC>::singleProcessNarrow( const float*, float*,  int )
{
	// dummy function
}

template<int FC>
inline void CBlobConvolution<FC>::processConvolutionLoop( int rxSize, bool useNarrowProcessing, const float*& srcPtr, float*& dstPtr, int windowIndex )
{
	const int batchStep = useNarrowProcessing ? NarrowBatchProcessSize.Width : WideBatchProcessSize.Width;
	for( ; rxSize >= batchStep; rxSize -= batchStep ) {
		batchProcess( srcPtr, dstPtr, windowIndex, useNarrowProcessing );
		srcPtr += batchStep * SrcXStep;
		dstPtr += batchStep * FC;
	}

	if( useNarrowProcessing ) {
		for( ; rxSize > 0; rxSize-- ) {
			singleProcessNarrow( srcPtr, dstPtr, windowIndex );
			srcPtr += SrcXStep;
			dstPtr += FC;
		}
	} else {
		for( ; rxSize > 0; rxSize-- ) {
			singleProcess( srcPtr, dstPtr, windowIndex );
			srcPtr += SrcXStep;
			dstPtr += FC;
		}
	}
}

template<int FC>
const float* CBlobConvolution<FC>::rearrangeFileter( const float* filterData )
{
	size_t filterBufferSize = FileterSize + AvxAlignment;
	void* alignedFltPtr = const_cast<float*>( Filter );
	std::align( AvxAlignment, FileterSize, alignedFltPtr, filterBufferSize );

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
	// 1. Result packing for case when FC == FCm8 (for example: 24):
	// Pixel[0] Channel[0] Filter[0-23]
	// Pixel[0] Channel[1] Filter[0-23]
	// ...
	// Pixel[0] Channel[23] Filter[0-23]
	// Pixel[1] Channel[0] Filter[0-23]
	// ...
	// Pixel[8] Channel[23] Filter[0-23]
	//
	// 2. Result packing for case when FC != FCm8 (for example: 18):
	// Pixel[0] Channel[0] Filter[0-17] Filter[0-5]
	// Pixel[0] Channel[1] Filter[0-23] Filter[0-5]
	// ...
	// Pixel[0] Channel[23] Filter[0-23] Filter[0-5]
	// Pixel[1] Channel[0] Filter[0-23] Filter[0-5]
	// ...
	// Pixel[8] Channel[23] Filter[0-23] Filter[0-5]



	float* dstFilter = static_cast<float*>( alignedFltPtr );
	for( int y = 0; y < FH; y++ ) {
		for( int x = 0; x < FW; x++ ) {
			for( int c = 0; c < C; c++ ) {
				const float* srcFilter = filterData + ( x + y * FW ) * C + c;
				for( int f = 0; f < FC; f++ ) {
					*dstFilter++ = *srcFilter;
					srcFilter += FW * FH * C;
				}
				if( FCm8 != FC ) {
					srcFilter = filterData + ( x + y * FW ) * C + c;
					for( int f = 0; f < FCm8 - FC; f++ ) {
						*dstFilter++ = *srcFilter;
						srcFilter += FW * FH * C;
					}
				}
			}
		}
	}

	return static_cast<float*>( alignedFltPtr );
}

template<int FC>
const float* CBlobConvolution<FC>::rearrangeFreeTerm( const float* freeTermData )
{
	size_t freeTermBufferSize = FCm8 + AvxAlignment;
	void* alignedFreeTermPtr = const_cast<float*>( FreeTerm );
	std::align( AvxAlignment, FCm8, alignedFreeTermPtr, freeTermBufferSize );
	float* dstFreeTerm = static_cast<float*>( alignedFreeTermPtr );

	for( int f = 0; f < FC; f++ ) {
		*dstFreeTerm++ = *freeTermData++;
	}
	if( FC != FCm8 ) {
		freeTermData -= FC;
		for( int f = 0; f < FC; f++ ) {
			*dstFreeTerm++ = *freeTermData++;
		}
	}
	return static_cast<float*>( alignedFreeTermPtr );
}

template<int FC>
const std::array<std::vector<int>, 8> CBlobConvolution<FC>::fillSrcPixelOffset()
{
	const int SrcLineStride = SrcW * C;
	const int SrcYDilation = DH * SrcLineStride;
	const int SrcXDilation = DW * C;
	return {
		std::vector<int>{ 0, SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation }, // 4 5 7 8
		std::vector<int>{ -SrcXDilation, 0, SrcXDilation, SrcYDilation - SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation }, // 3 4 5 6 7 8
		std::vector<int>{ -SrcXDilation, 0, SrcYDilation - SrcXDilation, SrcYDilation }, // 3 4 6 7
		std::vector<int>{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcXDilation, 0, SrcYDilation - SrcXDilation, SrcYDilation }, // 0 1 3 4 6 7
		std::vector<int>{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcXDilation, 0 }, // 0 1 3 4
		std::vector<int>{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcYDilation + SrcXDilation, -SrcXDilation, 0, SrcXDilation }, // 0 1 2 3 4 5
		std::vector<int>{ -SrcYDilation, -SrcYDilation + SrcXDilation, 0, SrcXDilation }, // 1 2 4 5
		std::vector<int>{ -SrcYDilation, -SrcYDilation + SrcXDilation, 0, SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation } // 1 2 4 5 7 8
	};
}

template<int FC>
const std::array<std::vector<int>, 8>  CBlobConvolution<FC>::fillFltPixelOffset()
{
	return {
		std::vector<int>{ 4 * C * FCm8, 5 * C * FCm8, 7 * C * FCm8, 8 * C * FCm8 }, // 4 5 7 8
		std::vector<int>{ 3 * C * FCm8, 4 * C * FCm8, 5 * C * FCm8, 6 * C * FCm8, 7 * C * FCm8, 8 * C * FCm8 }, // 3 4 5 6 7 8
		std::vector<int>{ 3 * C * FCm8, 4 * C * FCm8, 6 * C * FCm8, 7 * C * FCm8 }, // 3 4 6 7
		std::vector<int>{ 0 * C * FCm8, 1 * C * FCm8, 3 * C * FCm8, 4 * C * FCm8, 6 * C * FCm8, 7 * C * FCm8 }, // 0 1 3 4 6 7
		std::vector<int>{ 0 * C * FCm8, 1 * C * FCm8, 3 * C * FCm8, 4 * C * FCm8 }, // 0 1 3 4
		std::vector<int>{ 0 * C * FCm8, 1 * C * FCm8, 2 * C * FCm8, 3 * C * FCm8, 4 * C * FCm8, 5 * C * FCm8 }, // 0 1 2 3 4 5
		std::vector<int>{ 1 * C * FCm8, 2 * C * FCm8, 4 * C * FCm8, 5 * C * FCm8 }, // 1 2 4 5
		std::vector<int>{ 1 * C * FCm8, 2 * C * FCm8, 4 * C * FCm8, 5 * C * FCm8, 7 * C * FCm8, 8 * C * FCm8 } // 1 2 4 5 7 8
	};
}

template<int FC>
inline void CBlobConvolution<FC>::RotateLeft6( __m256& y0, __m256& y1, __m256& y2 )
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

template<int FC>
inline void CBlobConvolution<FC>::RotateLeft2( __m256& y )
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
#include <CpuMathEngineDnnConvAvxImpl.h>
