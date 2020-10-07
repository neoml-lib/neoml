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

namespace NeoML {

template<int imm8, int FC>
inline void Process_avx( __m256& r0, __m256& r1, __m256& r2,
						 __m128& s0, const float* &fltPtr ) {
	__m128 st0_m128 =  _mm_permute_ps( s0, imm8 );
	__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );

	__m256 f0 = _mm256_loadu_ps( fltPtr );
	__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
	__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );

	r0 = _mm256_fmadd_ps( f0, st0, r0 );
	r1 = _mm256_fmadd_ps( f1, st0, r1 );
	r2 = _mm256_fmadd_ps( f2, st0, r2 );

	fltPtr += FC;
}

template<int imm8, int FC>
inline void Process_avx_x2( __m256& r00, __m256& r01, __m256& r02,
							__m256& r10, __m256& r11, __m256& r12,
							__m128& s0, __m128& s1, const float* &fltPtr ) {
	__m128 st0_m128 =  _mm_permute_ps( s0, imm8 );
	__m256 st0 = _mm256_set_m128( st0_m128, st0_m128 );
	__m128 st1_m128 =  _mm_permute_ps( s1, imm8 );
	__m256 st1 = _mm256_set_m128( st1_m128, st1_m128 );

	__m256 f0 = _mm256_loadu_ps( fltPtr );
	__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
	__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );

	r00 = _mm256_fmadd_ps( f0, st0, r00 );
	r01 = _mm256_fmadd_ps( f1, st0, r01 );
	r02 = _mm256_fmadd_ps( f2, st0, r02 );

	r10 = _mm256_fmadd_ps( f0, st1, r10 );
	r11 = _mm256_fmadd_ps( f1, st1, r11 );
	r12 = _mm256_fmadd_ps( f2, st1, r12 );

	fltPtr += FC;
}

extern "C"
FME_DLL_EXPORT void BlobConvolution_f3x3_c24_fc24( int sourceWidth, int resultHeight, int resultWidth, int stride, int dilation,
	int threadCount, const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
{
	const int SW = sourceWidth;
	static constexpr int C = 24; // Channel count
	const int S = stride;
	const int D = dilation;
	static constexpr int FC = 24; // Filter count
	static constexpr int FH = 3; // Filter height
	static constexpr int FW = 3; // Filter width
	const int RH = resultHeight;
	const int RW = resultWidth;

	// Create temporary filter for convolution, data will be aligned.
	constexpr size_t FilterBufferSize = FH * FW * C * FC + 16;
	size_t filterBufferSize = FilterBufferSize;
	float Filter[FilterBufferSize];
	void* alignedFltPtr = Filter;
	std::align( 16, FH * FW * C * FC, alignedFltPtr, filterBufferSize );
	float* flt = static_cast<float*>( alignedFltPtr );

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
	// Result packing:
	// Pixel[0] Channel[0] Filter[0-23]
	// Pixel[0] Channel[1] Filter[0-23]
	// ...
	// Pixel[0] Channel[23] Filter[0-23]
	// Pixel[1] Channel[0] Filter[0-23]
	// ...
	// Pixel[8] Channel[23] Filter[0-23]

	float* dstFilter = flt;
	for( int y = 0; y < FH; y++ ) {
		for( int x = 0; x < FW; x++ ) {
			for( int c = 0; c < C; c++ ) {
				const float* srcFilter = filterData + ( x + y * FW ) * C + c;
				for( int f = 0; f < FC; f++ ) {
					*dstFilter++ = *srcFilter;
					srcFilter += FW * FH * C;
				}
			}
		}
	}

	const int SrcLineStride = SW * C;
	// Number of steps for each side of image, where filter is applied partially
	const int PartialStepCountBefore = static_cast<const int>( std::ceil( static_cast<float>( D )/ S ) );
	const int PartialStepCountAfter = static_cast<const int>( std::ceil( ( S * ( std::ceil( static_cast<float>( SW ) / S ) - 1 ) - SW + D + 1 ) / S ) );

	const float* src = sourceData;
	float* dst = resultData;

	const int SrcYStep = S * SrcLineStride;
	const int SrcXStep = S * C;


	auto ProcessChannels_avx = [&]( const float* srcPtr, const float* fltPtr, __m256& r0, __m256& r1, __m256& r2 ) {

		for( int c = 0; c < C; c += 4 ) {
			__m128 s = _mm_loadu_ps( srcPtr + c );
			Process_avx<0x00, FC>( r0, r1, r2, s, fltPtr );
			Process_avx<0x55, FC>( r0, r1, r2, s, fltPtr );
			Process_avx<0xaa, FC>( r0, r1, r2, s, fltPtr );
			Process_avx<0xff, FC>( r0, r1, r2, s, fltPtr );
		}

	};

	auto ProcessChannels_avx_x2 = [&]( const float* srcPtr, const float* fltPtr,
			__m256& r00, __m256& r01, __m256& r02,
			__m256& r10, __m256& r11, __m256& r12 ) {

		for( int c = 0; c < C; c += 4 ) {
			__m128 s0 = _mm_loadu_ps( srcPtr + c );
			__m128 s1 = _mm_loadu_ps( srcPtr + SrcXStep + c );

			Process_avx_x2<0x00, FC>( r00, r01, r02, r10, r11, r12, s0, s1, fltPtr );
			Process_avx_x2<0x55, FC>( r00, r01, r02, r10, r11, r12, s0, s1, fltPtr );
			Process_avx_x2<0xaa, FC>( r00, r01, r02, r10, r11, r12, s0, s1, fltPtr );
			Process_avx_x2<0xff, FC>( r00, r01, r02, r10, r11, r12, s0, s1, fltPtr );
		}
	};

	auto ProcessChannels_avx_x3 = [&]( const float* srcPtr, const float* fltPtr,
			__m256& r00, __m256& r01, __m256& r02,
			__m256& r10, __m256& r11, __m256& r12,
			__m256& r20, __m256& r21, __m256& r22 ) {

		auto ProcessNext = [&]() {
			__m256 st0 = _mm256_broadcast_ss( srcPtr );
			__m256 st1 = _mm256_broadcast_ss( srcPtr + SrcXStep );
			__m256 st2 = _mm256_broadcast_ss( srcPtr + 2 * SrcXStep );
			__m256 f0 = _mm256_loadu_ps( fltPtr );
			__m256 f1 = _mm256_loadu_ps( fltPtr + 8 );
			__m256 f2 = _mm256_loadu_ps( fltPtr + 16 );
			r00 = _mm256_fmadd_ps( f0, st0, r00 );
			r01 = _mm256_fmadd_ps( f1, st0, r01 );
			r02 = _mm256_fmadd_ps( f2, st0, r02 );
			r10 = _mm256_fmadd_ps( f0, st1, r10 );
			r11 = _mm256_fmadd_ps( f1, st1, r11 );
			r12 = _mm256_fmadd_ps( f2, st1, r12 );
			r20 = _mm256_fmadd_ps( f0, st2, r20 );
			r21 = _mm256_fmadd_ps( f1, st2, r21 );
			r22 = _mm256_fmadd_ps( f2, st2, r22 );

			fltPtr += FC;
			srcPtr++;
		};

		for( int c = 0; c < C; c ++ ) {
			ProcessNext();
		}
	};


	// Choose proper pixels in source and filter:
	// 0  1  2
	// 3  4  5
	// 6  7  8
	const int SrcYDilation = D * SrcLineStride;
	const int SrcXDilation  = D * C;
	const int SrcXWindowSize = FW * SrcXDilation;
	const int DstYDilation = RW * FC;
	// Offset is relative to central pixel
	const vector<int> SrcPixelOffset[8] = {
		{ 0, SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation }, // 4 5 7 8
		{ -SrcXDilation, 0, SrcXDilation, SrcYDilation - SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation }, // 3 4 5 6 7 8
		{ -SrcXDilation, 0, SrcYDilation - SrcXDilation, SrcYDilation }, // 3 4 6 7
		{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcXDilation, 0, SrcYDilation - SrcXDilation, SrcYDilation }, // 0 1 3 4 6 7
		{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcXDilation, 0 }, // 0 1 3 4
		{ -SrcYDilation - SrcXDilation, -SrcYDilation, -SrcYDilation + SrcXDilation, -SrcXDilation, 0, SrcXDilation }, // 0 1 2 3 4 5
		{ -SrcYDilation, -SrcYDilation + SrcXDilation, 0, SrcXDilation }, // 1 2 4 5
		{ -SrcYDilation, -SrcYDilation + SrcXDilation, 0, SrcXDilation, SrcYDilation, SrcYDilation + SrcXDilation } // 1 2 4 5 7 8
	};
	// Offset is relative to top left pixel
	const vector<int> FltPixels[8] = {
		{ 4 * C * FC, 5 * C * FC, 7 * C * FC, 8 * C * FC }, // 4 5 7 8
		{ 3 * C * FC, 4 * C * FC, 5 * C * FC, 6 * C * FC, 7 * C * FC, 8 * C * FC }, // 3 4 5 6 7 8
		{ 3 * C * FC, 4 * C * FC, 6 * C * FC, 7 * C * FC }, // 3 4 6 7
		{ 0 * C * FC, 1 * C * FC, 3 * C * FC, 4 * C * FC, 6 * C * FC, 7 * C * FC }, // 0 1 3 4 6 7
		{ 0 * C * FC, 1 * C * FC, 3 * C * FC, 4 * C * FC }, // 0 1 3 4
		{ 0 * C * FC, 1 * C * FC, 2 * C * FC, 3 * C * FC, 4 * C * FC, 5 * C * FC }, // 0 1 2 3 4 5
		{ 1 * C * FC, 2 * C * FC, 4 * C * FC, 5 * C * FC }, // 1 2 4 5
		{ 1 * C * FC, 2 * C * FC, 4 * C * FC, 5 * C * FC, 7 * C * FC, 8 * C * FC } // 1 2 4 5 7 8
	};

	const __m256 ft0 = freeTermData != 0 ? _mm256_loadu_ps( freeTermData ) : _mm256_setzero_ps();
	const __m256 ft1 = freeTermData != 0 ? _mm256_loadu_ps( freeTermData + 8) : _mm256_setzero_ps();
	const __m256 ft2 = freeTermData != 0 ? _mm256_loadu_ps( freeTermData + 16 ) : _mm256_setzero_ps();

	auto ApplyPartitialFilter3x3_24ch = [&]( const float* srcPtr, const vector<int>& srcPixelOffset,
			const float* fltPtr, const vector<int>& fltPixels, float* dstPtr ) {

		__m256 r0 = ft0;
		__m256 r1 = ft1;
		__m256 r2 = ft2;

		auto srcIt = srcPixelOffset.cbegin();
		auto fltIt = fltPixels.cbegin();
		for( ; srcIt != srcPixelOffset.cend(); srcIt++, fltIt++ ) {
			ProcessChannels_avx( srcPtr + *srcIt, fltPtr + *fltIt , r0, r1, r2  );
		}

		// Store result of convolution for (fx,fy) pixel of f-th channel
		_mm256_storeu_ps( dstPtr, r0 );
		_mm256_storeu_ps( dstPtr + 8, r1 );
		_mm256_storeu_ps( dstPtr + 16, r2 );
	};

	auto ApplyPartitialFilter3x3_24ch_x2 = [&]( const float* srcPtr, const vector<int>& srcPixelOffset,
			const float* fltPtr, const vector<int>& fltPixels, float* dstPtr ) {

		__m256 r00 = ft0;
		__m256 r01 = ft1;
		__m256 r02 = ft2;
		__m256 r10 = ft0;
		__m256 r11 = ft1;
		__m256 r12 = ft2;

		auto srcIt = srcPixelOffset.cbegin();
		auto fltIt = fltPixels.cbegin();
		for( ; srcIt != srcPixelOffset.cend(); srcIt++, fltIt++ ) {
			ProcessChannels_avx_x2( srcPtr + *srcIt, fltPtr + *fltIt, r00, r01, r02, r10, r11, r12 );
		}

		// Store result of convolution for (fx,fy) pixel of f-th channel
		_mm256_storeu_ps( dstPtr, r00 );
		_mm256_storeu_ps( dstPtr + 8, r01 );
		_mm256_storeu_ps( dstPtr + 16, r02 );
		_mm256_storeu_ps( dstPtr + 24, r10 );
		_mm256_storeu_ps( dstPtr + 32, r11 );
		_mm256_storeu_ps( dstPtr + 40, r12 );
	};

	auto ApplyPartitialFilter3x3_24ch_x3 = [&]( const float* srcPtr, const vector<int>& srcPixelOffset,
			const float* fltPtr, const vector<int>& fltPixels, float* dstPtr ) {

		__m256 r00 = ft0;
		__m256 r01 = ft1;
		__m256 r02 = ft2;
		__m256 r10 = ft0;
		__m256 r11 = ft1;
		__m256 r12 = ft2;
		__m256 r20 = ft0;
		__m256 r21 = ft1;
		__m256 r22 = ft2;

		auto srcIt = srcPixelOffset.cbegin();
		auto fltIt = fltPixels.cbegin();
		for( ; srcIt != srcPixelOffset.cend(); srcIt++, fltIt++ ) {
			ProcessChannels_avx_x3( srcPtr + *srcIt, fltPtr + *fltIt, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
		}

		// Store result of convolution for (fx,fy) pixel of f-th channel
		_mm256_storeu_ps( dstPtr, r00 );
		_mm256_storeu_ps( dstPtr + 8, r01 );
		_mm256_storeu_ps( dstPtr + 16, r02 );
		_mm256_storeu_ps( dstPtr + 24, r10 );
		_mm256_storeu_ps( dstPtr + 32, r11 );
		_mm256_storeu_ps( dstPtr + 40, r12 );
		_mm256_storeu_ps( dstPtr + 48, r20 );
		_mm256_storeu_ps( dstPtr + 56, r21 );
		_mm256_storeu_ps( dstPtr + 64, r22 );
	};

	auto ApplyWholeFilter3x3_24ch = [&]( const float* srcPtr, const float* fltPtr, float* dstPtr  ) {

		__m256 r0 = ft0;
		__m256 r1 = ft1;
		__m256 r2 = ft2;

			for( int fy = 0; fy < FH; fy++ ) {
				for( int fx = 0; fx < FW; fx++ ) {
					ProcessChannels_avx( srcPtr, fltPtr, r0, r1, r2 );
					// Move to next pixel in source image on the SAME line
					srcPtr += SrcXDilation;
					fltPtr += C * FC;
				}
				// Move to next pixel in source image on the NEXT line
				srcPtr += SrcYDilation - SrcXWindowSize;
			}
			// Store result of convolution for (fx,fy) pixel of f-th channel
			_mm256_storeu_ps( dstPtr, r0 );
			_mm256_storeu_ps( dstPtr + 8, r1 );
			_mm256_storeu_ps( dstPtr + 16, r2 );
	};

	auto ApplyWholeFilter3x3_24ch_x2 = [&]( const float* srcPtr, const float* fltPtr, float* dstPtr  ) {

		__m256 r00 = ft0;
		__m256 r01 = ft1;
		__m256 r02 = ft2;
		__m256 r10 = ft0;
		__m256 r11 = ft1;
		__m256 r12 = ft2;

			for( int fy = 0; fy < FH; fy++ ) {
				for( int fx = 0; fx < FW; fx++ ) {
					ProcessChannels_avx_x2( srcPtr, fltPtr, r00, r01, r02, r10, r11, r12 );
					// Move to next pixel in source image on the SAME line
					srcPtr += SrcXDilation;
					fltPtr += C * FC;
				}
				// Move to next pixel in source image on the NEXT line
				srcPtr += SrcYDilation - SrcXWindowSize;
			}
			// Store result of convolution for (fx,fy) pixel of f-th channel
			_mm256_storeu_ps( dstPtr, r00 );
			_mm256_storeu_ps( dstPtr + 8, r01 );
			_mm256_storeu_ps( dstPtr + 16, r02 );
			_mm256_storeu_ps( dstPtr + 24, r10 );
			_mm256_storeu_ps( dstPtr + 32, r11 );
			_mm256_storeu_ps( dstPtr + 40, r12 );
	};

	auto ApplyWholeFilter3x3_24ch_x3 = [&]( const float* srcPtr, const float* fltPtr, float* dstPtr  ) {

		__m256 r00 = ft0;
		__m256 r01 = ft1;
		__m256 r02 = ft2;
		__m256 r10 = ft0;
		__m256 r11 = ft1;
		__m256 r12 = ft2;
		__m256 r20 = ft0;
		__m256 r21 = ft1;
		__m256 r22 = ft2;

			for( int fy = 0; fy < FH; fy++ ) {
				for( int fx = 0; fx < FW; fx++ ) {
					ProcessChannels_avx_x3( srcPtr, fltPtr, r00, r01, r02, r10, r11, r12, r20, r21, r22 );
					// Move to next pixel in source image on the SAME line
					srcPtr += SrcXDilation;
					fltPtr += C * FC;
				}
				// Move to next pixel in source image on the NEXT line
				srcPtr += SrcYDilation - SrcXWindowSize;
			}
			// Store result of convolution for (fx,fy) pixel of f-th channel
			_mm256_storeu_ps( dstPtr, r00 );
			_mm256_storeu_ps( dstPtr + 8, r01 );
			_mm256_storeu_ps( dstPtr + 16, r02 );
			_mm256_storeu_ps( dstPtr + 24, r10 );
			_mm256_storeu_ps( dstPtr + 32, r11 );
			_mm256_storeu_ps( dstPtr + 40, r12 );
			_mm256_storeu_ps( dstPtr + 48, r20 );
			_mm256_storeu_ps( dstPtr + 56, r21 );
			_mm256_storeu_ps( dstPtr + 64, r22 );
	};

	const int curThreadCount = IsOmpRelevant( RH, RH * RW * FC * FW * FH * C ) ? threadCount : 1;

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

			// We process all central pixels by groups for increasing performance.
			bool ProcessLastOnePixel = ( RW - PartialStepCountAfter - PartialStepCountBefore ) % 3 == 1;
			bool ProcessLastTwoPixels = ( RW - PartialStepCountAfter - PartialStepCountBefore ) % 3 == 2;

			for( ; ry < min( PartialStepCountBefore, currentRH ); ry++ ) {
				// Top part of image
				const float* srcPtr =  src + ry * SrcYStep;

				// Partial applying filter
				for( int rx = 0; rx < PartialStepCountBefore; rx++ ) {
					// Top left corner, // 4 5 7 8
					ApplyPartitialFilter3x3_24ch( srcPtr, SrcPixelOffset[0], fltPtr, FltPixels[0], dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}
				for( int rx = PartialStepCountBefore; rx <= RW - PartialStepCountAfter - 3; rx += 3 ) {
					// Top edge, 3 4 5 6 7 8
					ApplyPartitialFilter3x3_24ch_x3( srcPtr, SrcPixelOffset[1], fltPtr, FltPixels[1], dstPtr );
					srcPtr += 3 * SrcXStep;
					dstPtr += 3 * FC;
				}
				if( ProcessLastTwoPixels ) {
					ApplyPartitialFilter3x3_24ch_x2( srcPtr, SrcPixelOffset[1], fltPtr, FltPixels[1], dstPtr );
					srcPtr += 2 * SrcXStep;
					dstPtr += 2 * FC;
				}
				if( ProcessLastOnePixel ) {
					ApplyPartitialFilter3x3_24ch( srcPtr, SrcPixelOffset[1], fltPtr, FltPixels[1], dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}

				for( int rx = RW - PartialStepCountAfter; rx < RW; rx++ ) {
					// Top right corner, 3 4 6 7
					ApplyPartitialFilter3x3_24ch( srcPtr, SrcPixelOffset[2], fltPtr, FltPixels[2], dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}
			}

			for( ; ry < min( RH - PartialStepCountAfter, currentRH ); ry++ ) {
				// Middle part of image
				const float* srcPtr =  src + ry * SrcYStep;

				// Partial applying filter
				for( int rx = 0; rx < PartialStepCountBefore; rx++ ) {
					// Top left corner, // 4 5 7 8
					ApplyPartitialFilter3x3_24ch( srcPtr, SrcPixelOffset[7], fltPtr, FltPixels[7], dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}


				// Move to the top left pixel of window from central one
				srcPtr -= ( SrcYDilation + SrcXDilation);
				for( int rx = PartialStepCountBefore; rx <= RW - PartialStepCountAfter - 3; rx += 3 ) {
					// Top edge, 3 4 5 6 7 8
					ApplyWholeFilter3x3_24ch_x3( srcPtr, fltPtr, dstPtr );
					srcPtr += 3 * SrcXStep;
					dstPtr += 3 * FC;
				}
				if( ProcessLastTwoPixels ) {
					ApplyWholeFilter3x3_24ch_x2( srcPtr, fltPtr, dstPtr );
					srcPtr += 2 * SrcXStep;
					dstPtr += 2 * FC;
				}
				if( ProcessLastOnePixel ) {
					ApplyWholeFilter3x3_24ch( srcPtr, fltPtr, dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}


				// Move back to the central pixel again
				srcPtr += ( SrcYDilation + SrcXDilation);
				for( int rx = RW - PartialStepCountAfter; rx < RW; rx++ ) {
					// Top right corner, 3 4 6 7
					ApplyPartitialFilter3x3_24ch( srcPtr, SrcPixelOffset[3], fltPtr, FltPixels[3], dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}
			}

			for( ; ry < min( RH, currentRH ); ry++ ) {
				// Bottom part of image
				const float* srcPtr =  src + ry * SrcYStep;

				// Partial applying filter
				for( int rx = 0; rx < PartialStepCountBefore; rx++ ) {
					// Top left corner, // 4 5 7 8
					ApplyPartitialFilter3x3_24ch( srcPtr, SrcPixelOffset[6], fltPtr, FltPixels[6], dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}
				for( int rx = PartialStepCountBefore; rx <= RW - PartialStepCountAfter - 3; rx += 3 ) {
					// Top edge, 3 4 5 6 7 8
					ApplyPartitialFilter3x3_24ch_x3( srcPtr, SrcPixelOffset[5], fltPtr, FltPixels[5], dstPtr );
					srcPtr += 3 * SrcXStep;
					dstPtr += 3 * FC;
				}
				if( ProcessLastTwoPixels ) {
					ApplyPartitialFilter3x3_24ch_x2( srcPtr, SrcPixelOffset[5], fltPtr, FltPixels[5], dstPtr );
					srcPtr += 2 * SrcXStep;
					dstPtr += 2 * FC;
				}
				if( ProcessLastOnePixel ) {
					ApplyPartitialFilter3x3_24ch( srcPtr, SrcPixelOffset[5], fltPtr, FltPixels[5], dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}

				for( int rx = RW - PartialStepCountAfter; rx < RW; rx++ ) {
					// Top right corner, 3 4 6 7
					ApplyPartitialFilter3x3_24ch( srcPtr, SrcPixelOffset[4], fltPtr, FltPixels[4], dstPtr );
					srcPtr += SrcXStep;
					dstPtr += FC;
				}
			}
		}
	}
}

}
