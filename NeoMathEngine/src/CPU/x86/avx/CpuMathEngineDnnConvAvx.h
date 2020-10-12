/* Copyright � 2017-2020 ABBYY Production LLC

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

namespace NeoML {

class CBlobConvolutionBase {
public:
	virtual ~CBlobConvolutionBase() = default;
	virtual void ProcessConvolution( int threadCount ) = 0;
};

template<int C, int FC, int FH, int FW>
class CBlobConvolution : public CBlobConvolutionBase {
public:
	CBlobConvolution( int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
		const float* sourceData, const float* filterData, const float* freeTermData, float* resultData );
	~CBlobConvolution() override = default;

	void ProcessConvolution( int threadCount ) override;

private:
	// Create this variables in order to use them in class specialization
	static constexpr int C_ = C;
	static constexpr int FC_ = FC;
	static constexpr int FH_ = FH;
	static constexpr int FW_ = FW;

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

	const int SrcXStep;
	const int SrcLineStride;
	const int SrcXDilation;
	const int SrcYDilation;
	const int SrcXWindowSize;
	
	// For some cases we will use FC, rounded up to nearest integer multiple of 8
	static constexpr int FCm8 = ( FC + 8 - 1 ) / 8 * 8;
	// Filter should be alligned to 16 bytes
	static constexpr size_t FileterSize = FW * FH * FCm8 * C ;
	static constexpr size_t AvxAlignment = 32;
	float Filter[FileterSize + AvxAlignment];

	// Free term should be alligned to 16 bytes
	float FreeTerm[FCm8 + AvxAlignment];

	// Choose proper pixels in source and filter:
	// 0  1  2
	// 3  4  5
	// 6  7  8
	// Offset is relative to central pixel
	const std::array<std::vector<int>, 8> SrcPixelsOffset;
	// Offset is relative to top left pixel
	const std::array<std::vector<int>, 8>  FltPixelsOffset;
	// Number of pixels processed with partialBatchProcess() function
	const int BatchProcessSize;

	static constexpr int getBatchProcessSize();

	template<int idx>
	void partialBatchProcessLoop( int& rx, int rxEnd, const float* &srcPtr, float* &dstPtr );
	template<int idx>
	void partialSingleProcessLoop( int& rx, int rxEnd, const float* &srcPtr, float* &dstPtr );
	void wholeBatchProcessLoop( int& rx, int rxEnd, const float* &srcPtr, float* &dstPtr );
	void wholeSingleProcessLoop( int& rx, int rxEnd, const float* &srcPtr, float* &dstPtr );

	void batchProcessChannels( const float* srcPtr, const float* fltPtr,
		__m256& r00, __m256& r01, __m256& r02,
		__m256& r10, __m256& r11, __m256& r12,
		__m256& r20, __m256& r21, __m256& r22 );
	void singleProcessChannels( const float* srcPtr, const float* fltPtr, __m256& r0, __m256& r1, __m256& r2 );

	void partialBatchProcess( const float* srcPtr, const std::vector<int>& srcPixelsOffset,
		const float* fltPtr, const vector<int>& fltPixelsOffset, float* dstPtr );
	void partialSingleProcess( const float* srcPtr, const std::vector<int>& srcPixelsOffset,
		const float* fltPtr, const std::vector<int>& fltPixelsOffset, float* dstPtr );
	void wholeBatchProcess( const float* srcPtr, const float* fltPtr, float* dstPtr );
	void wholeSingleProcess( const float* srcPtr, const float* fltPtr, float* dstPtr );

	const float* rearrangeFileter( const float* filterData );
	const float* rearrangeFreeTerm( const float* freeTermData );
	const std::array<std::vector<int>, 8> fillSrcPixelOffset();
	const std::array<std::vector<int>, 8>  fillFltPixelOffset();

	// Circular rotation of three ymm registers to the left, step equals to six floats.
	static void RotateLeft6( __m256& y0, __m256& y1, __m256& y2 );

};

class CBlobConvolutionFabric {
public:
	static bool IsBlobConvolutionAvailable( int C, int FC, int FH, int FW );
	static std::unique_ptr<CBlobConvolutionBase> GetProperInstance( int C, int FC, int FH, int FW,
		int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
		int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
		const float* sourceData, const float* filterData, const float* freeTermData, float* resultData );
};

bool CBlobConvolutionFabric::IsBlobConvolutionAvailable( int C, int FC, int FH, int FW )
{
	if( ( C == 24 && FC == 24 && FH == 3 && FW == 3 ) ||
		( C == 18 && FC == 18 && FH == 3 && FW == 3 ) ){
		return true;
	}
	return false;
}

std::unique_ptr<CBlobConvolutionBase> CBlobConvolutionFabric::GetProperInstance( int C, int FC, int FH, int FW,
	int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
{
	if( C == 24 && FC == 24 && FH == 3 && FW == 3 ) {
		return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<24, 24, 3, 3>(
			sourceHeight, sourceWidth, strideHeight, strideWidth,
			dilationHeight, dilationWidth, resultHeight, resultWidth,
			sourceData, filterData, freeTermData, resultData ) );
	} else if( C == 18 && FC == 18 && FH == 3 && FW == 3 ) {
		return std::unique_ptr<CBlobConvolutionBase>( new CBlobConvolution<18, 18, 3, 3>(
			sourceHeight, sourceWidth, strideHeight, strideWidth,
			dilationHeight, dilationWidth, resultHeight, resultWidth,
			sourceData, filterData, freeTermData, resultData ) );
	}
	return nullptr;
}

template<int C, int FC, int FH, int FW>
CBlobConvolution<C, FC, FH, FW>::CBlobConvolution( int sourceHeight, int sourceWidth, int strideHeight, int strideWidth,
	int dilationHeight, int dilationWidth, int resultHeight, int resultWidth,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) :
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
	SrcXStep( SW * C ),
	SrcLineStride( SrcW* C ),
	SrcXDilation( DW * C ),
	SrcYDilation( DH * SrcLineStride ),
	SrcXWindowSize( FW* SrcXDilation ),
	SrcPixelsOffset( fillSrcPixelOffset() ),
	FltPixelsOffset( fillFltPixelOffset() ),
	BatchProcessSize( getBatchProcessSize() )
{

}

template<int C, int FC, int FH, int FW>
void CBlobConvolution<C, FC, FH, FW>::ProcessConvolution( int threadCount )
{
	const int curThreadCount = IsOmpRelevant( RH, RH * RW * FC * FW * FH * C ) ? threadCount : 1;

	// Number of steps for each side of image, where filter is applied partially
	const int PartialStepCountBeforeX = static_cast<const int>( std::ceil( static_cast<float>( DW ) / SW ) );
	const int PartialStepCountAfterX = static_cast<const int>( std::ceil( ( SW * ( std::ceil( static_cast<float>( SrcW ) / SW ) - 1 ) - SrcW + DW + 1 ) / SW ) );
	const int PartialStepCountBeforeY = static_cast<const int>( std::ceil( static_cast<float>( DH ) / SH ) );
	const int PartialStepCountAfterY = static_cast<const int>( std::ceil( ( SH * ( std::ceil( static_cast<float>( SrcH ) / SH ) - 1 ) - SrcH + DH + 1 ) / SH ) );

	const int SrcYStep = SH * SrcLineStride;
	const int DstYDilation = RW * FC;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int yStart;
		int yCount;
		if( OmpGetTaskIndexAndCount( RH, yStart, yCount ) ) {

			// Iterate through result, left->right, top->bottom
			// Top edge ( cut top part of filter )
			const float* fltPtr = flt;
			float* dstPtr = dst + yStart * DstYDilation;
			const int currentRH = min( RH, yStart + yCount );
			int ry = yStart;


			for( ; ry < min( PartialStepCountBeforeY, currentRH ); ry++ ) {
				// Top part of image
				const float* srcPtr = src + ry * SrcYStep;
				int rx = 0;

				partialSingleProcessLoop<0>( rx, PartialStepCountBeforeX, srcPtr, dstPtr );
				partialBatchProcessLoop<1>( rx, RW - PartialStepCountAfterX, srcPtr, dstPtr );
				partialSingleProcessLoop<1>( rx, RW - PartialStepCountAfterX, srcPtr, dstPtr );
				partialSingleProcessLoop<2>( rx, RW, srcPtr, dstPtr );
			}

			for( ; ry < min( RH - PartialStepCountAfterY, currentRH ); ry++ ) {
				// Middle part of image
				const float* srcPtr = src + ry * SrcYStep;
				int rx = 0;

				partialSingleProcessLoop<7>( rx, PartialStepCountBeforeX, srcPtr, dstPtr );

				// Move to the top left pixel of window from central one
				srcPtr -= ( SrcYDilation + SrcXDilation );
				wholeBatchProcessLoop( rx, RW - PartialStepCountAfterX, srcPtr, dstPtr );
				wholeSingleProcessLoop( rx, RW - PartialStepCountAfterX, srcPtr, dstPtr );

				// Move back to the central pixel again
				srcPtr += ( SrcYDilation + SrcXDilation );
				partialSingleProcessLoop<3>( rx, RW, srcPtr, dstPtr );
			}

			for( ; ry < min( RH, currentRH ); ry++ ) {
				// Bottom part of image
				const float* srcPtr = src + ry * SrcYStep;
				int rx = 0;

				partialSingleProcessLoop<6>( rx, PartialStepCountBeforeX, srcPtr, dstPtr );
				partialBatchProcessLoop<5>( rx, RW - PartialStepCountAfterX, srcPtr, dstPtr );
				partialSingleProcessLoop<5>( rx, RW - PartialStepCountAfterX, srcPtr, dstPtr );
				partialSingleProcessLoop<4>( rx, RW, srcPtr, dstPtr );
			}
		}
	}
}

template<int C, int FC, int FH, int FW>
template<int idx>
inline void CBlobConvolution<C, FC, FH, FW>::partialBatchProcessLoop( int& rx, int rxEnd, const float* &srcPtr, float* &dstPtr )
{
	for( ; rx <= rxEnd - BatchProcessSize; rx += BatchProcessSize ) {
		partialBatchProcess( srcPtr, SrcPixelsOffset[idx], flt, FltPixelsOffset[idx], dstPtr );
		srcPtr += BatchProcessSize * SrcXStep;
		dstPtr += BatchProcessSize * FC;
	}
}

template<int C, int FC, int FH, int FW>
template<int idx>
inline void CBlobConvolution<C, FC, FH, FW>::partialSingleProcessLoop( int& rx, int rxEnd, const float* &srcPtr, float* &dstPtr )
{
	for( ; rx < rxEnd; rx++ ) {
		partialSingleProcess( srcPtr, SrcPixelsOffset[idx], flt, FltPixelsOffset[idx], dstPtr );
		srcPtr += SrcXStep;
		dstPtr += FC;
	}
}

template<int C, int FC, int FH, int FW>
inline void CBlobConvolution<C, FC, FH, FW>::wholeBatchProcessLoop( int& rx, int rxEnd, const float* &srcPtr, float* &dstPtr )
{
	for( ; rx <= rxEnd - BatchProcessSize; rx += BatchProcessSize ) {
		wholeBatchProcess( srcPtr, flt, dstPtr );
		srcPtr += BatchProcessSize * SrcXStep;
		dstPtr += BatchProcessSize * FC;
	}
}

template<int C, int FC, int FH, int FW>
inline void CBlobConvolution<C, FC, FH, FW>::wholeSingleProcessLoop( int& rx, int rxEnd, const float* &srcPtr, float* &dstPtr )
{
	for( ; rx < rxEnd; rx++ ) {
		wholeSingleProcess( srcPtr, flt, dstPtr );
		srcPtr += SrcXStep;
		dstPtr += FC;
	}
}

template<int C, int FC, int FH, int FW>
const float* CBlobConvolution<C, FC, FH, FW>::rearrangeFileter( const float* filterData )
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

template<int C, int FC, int FH, int FW>
const float* CBlobConvolution<C, FC, FH, FW>::rearrangeFreeTerm( const float* freeTermData )
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

template<int C, int FC, int FH, int FW>
const std::array<std::vector<int>, 8> CBlobConvolution<C, FC, FH, FW>::fillSrcPixelOffset()
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

template<int C, int FC, int FH, int FW>
const std::array<std::vector<int>, 8>  CBlobConvolution<C, FC, FH, FW>::fillFltPixelOffset()
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

template<int C, int FC, int FH, int FW>
inline void CBlobConvolution<C, FC, FH, FW>::RotateLeft6( __m256& y0, __m256& y1, __m256& y2 )
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

} // namespace NeoML

// Class specializations
#include <CpuMathEngineDnnConvAvxImpl.h>
